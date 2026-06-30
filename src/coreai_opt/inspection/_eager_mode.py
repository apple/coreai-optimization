# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause


"""Eager mode op discovery implementation.

Runs a forward pass with ``TorchFunctionMode`` interception to discover
operations in an ``nn.Module`` without exporting to a graph.
"""

import itertools
import linecache
import sys
import weakref
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch.overrides import TorchFunctionMode

from coreai_opt._utils.insertion.torch_function.modes import _is_interceptable_func
from coreai_opt._utils.insertion.torch_function.module_boundary_tracker import (
    TensorIdVersion,
)
from coreai_opt._utils.insertion.torch_function.utils import (
    get_func_base_name,
    get_func_name,
)
from coreai_opt._utils.python_utils import fqn as _fqn
from coreai_opt._utils.torch_utils import NamedModule, flatten_tensors_to_list
from coreai_opt.base_model_compressor import _BaseModelCompressor
from coreai_opt.palettization import KMeansPalettizer
from coreai_opt.quantization import Quantizer
from coreai_opt.quantization._eager import EagerQuantizer
from coreai_opt.quantization.config.quantization_config import ExecutionMode

from ._common import (
    FORWARD_FUNCTION_NAME,
    build_module_tree,
    filter_module_tree,
)
from .types import (
    BoundaryEdge,
    InputEdge,
    ModelSummary,
    ModuleContext,
    ModuleInfo,
    OpInfo,
    SourceFrame,
)


class _EagerOpDiscoveryMode(TorchFunctionMode):
    """TorchFunctionMode that discovers ops during a forward pass.

    Args:
        model (nn.Module): The model to inspect.
        op_type_resolver (Callable[[Callable], str | None] | None):
            When provided, maps a torch function to its compressor-defined
            op type. A non-None return marks the op as compressor-supported
            (for post-hoc filtering) and uses the returned string as op_type.
            When None, all ops use base_name as op_type and no filtering
            metadata is collected.
    """

    def __init__(
        self,
        model: nn.Module,
        op_type_resolver: Callable[[Callable], str | None] | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self._op_type_resolver = op_type_resolver
        self.parents: list[NamedModule] = []
        self.traversed_modules: set[nn.Module] = set()
        self.hooks: list[torch.utils.hooks.RemovableHook] = []

        # Per-module function call counts: module_name → base_name → count
        self.func_counts: defaultdict[str, defaultdict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Discovered ops
        self.all_ops: list[OpInfo] = []
        self.supported_op_names: set[str] = set()
        self._seen_op_names: set[str] = set()

        # Tensor connectivity: (id, version) → InputEdge (producing op + output slot).
        # Tracks activation tensors only; state tensors are tracked separately below.
        self._tensor_producers: dict[TensorIdVersion, InputEdge] = {}

        self._states_to_names = {
            id(state): name
            for name, state in itertools.chain(
                self.model.named_parameters(),
                self.model.named_buffers(),
            )
        }
        # State tensors are looked up purely by object id — no version tracking. Version
        # is irrelevant for states: they are not produced by ops and their identity is
        # stable for the model's lifetime regardless of in-place mutations.
        self._state_op_infos: dict[int, OpInfo] = {}

        # Ephemeral OpInfos for tensors that are neither registered states nor produced
        # by any intercepted op (e.g. raw tensor attributes, global tensors). These are
        # created on demand in _resolve_inputs and keyed by id(tensor). They are NOT added
        # to all_ops — they exist only as input references so they appear in the consuming
        # op's inputs tuple and therefore in the formatted op inputs display.
        self._ephemeral_op_infos: dict[int, OpInfo] = {}
        self._ephemeral_counter: int = 0

        self._module_input_producers: dict[str, list[InputEdge | None]] = {}
        self._module_output_producers: dict[str, list[InputEdge | None]] = {}

        # Capture model-level input tensors BEFORE the module-loop hooks so that
        # ``_capture_input_tensors`` fires before ``_enter_module("")``, ensuring
        # root input tensors are registered in ``_tensor_producers`` by the time
        # ``_enter_module("")`` resolves their producers.
        self.hooks.append(model.register_forward_pre_hook(self._capture_input_tensors))

        for name, module in model.named_modules(remove_duplicate=True):
            pre_hook = module.register_forward_pre_hook(self._enter_module(name))
            post_hook = module.register_forward_hook(self._exit_module(name), always_call=True)
            self.hooks.append(pre_hook)
            self.hooks.append(post_hook)

        # Registered after the module-loop hooks so that ``_exit_module("")``
        # fires before output capture.
        self.hooks.append(
            model.register_forward_hook(self._capture_output_tensors, always_call=True)
        )

    @property
    def current_module_name(self) -> str:
        return self.parents[-1].name if self.parents else ""

    @property
    def current_module(self) -> nn.Module:
        return self.parents[-1].module if self.parents else self.model

    def _add_op(self, op_info: OpInfo) -> None:
        assert op_info.op_name not in self._seen_op_names, f"duplicate op_name {op_info.op_name}"
        self._seen_op_names.add(op_info.op_name)
        self.all_ops.append(op_info)

    def _get_or_create_ephemeral(self, tensor_id: int) -> InputEdge:
        """Return the ephemeral InputEdge for this tensor id, creating one if needed."""
        op_info = self._ephemeral_op_infos.get(tensor_id)
        if op_info is None:
            op_info = OpInfo(
                op_name=f"untracked_{self._ephemeral_counter}",
                op_type=None,
                module_stack=(),
                source_frames=(),
                inputs=(),
                outputs={},
                is_state=False,
            )
            self._ephemeral_op_infos[tensor_id] = op_info
            self._ephemeral_counter += 1
        return InputEdge(op=op_info, output_idx=None)

    def _resolve_boundary_tensor(self, t: torch.Tensor) -> InputEdge | None:
        """Resolve a module-boundary tensor to its producer entry.

        Mirrors the three-way lookup in ``_resolve_inputs`` so module boundary
        detection is consistent with op-level input detection:

        - Known activation: returns the existing ``InputEdge``.
        - Registered state: returns ``None`` (states are filtered in
          ``_populate_boundary_ops_eager`` and handled via ``module_state_spec``).
        - Untracked tensor (raw attribute, global): creates or reuses an ephemeral
          ``OpInfo`` so the tensor appears in ``module inputs`` for the consuming module.
        """
        key = TensorIdVersion(id(t), t._version)
        entry = self._tensor_producers.get(key)
        if entry is not None:
            return entry
        if id(t) in self._states_to_names:
            return None
        return self._get_or_create_ephemeral(id(t))

    def _enter_module(self, name: str) -> Callable:
        def hook(module: nn.Module, inputs: Any) -> None:
            self.parents.append(NamedModule(name, module))
            if module not in self.traversed_modules:
                # First visit only — mirrors the quantizer's enter_module guard.
                # Resolve each input tensor now while it is fresh, using the same
                # three-way lookup as _resolve_inputs so that untracked tensors
                # (e.g. raw attributes passed as forward arguments) appear in
                # module inputs rather than being silently dropped.
                self._module_input_producers[name] = [
                    self._resolve_boundary_tensor(t) for t in flatten_tensors_to_list(inputs)
                ]

        return hook

    def _exit_module(self, name: str) -> Callable:
        def hook(module: nn.Module, inputs: Any, outputs: Any) -> None:
            assert self.parents[-1].name == name
            if module not in self.traversed_modules:
                # First visit: store the full _TensorProducerEntry (op_info + output_idx)
                # so that _populate_boundary_ops_eager can identify which output slot of
                # each producing op corresponds to each module output tensor.
                self._module_output_producers[name] = [
                    self._tensor_producers.get(TensorIdVersion(id(t), t._version))
                    for t in flatten_tensors_to_list(outputs)
                ]
            else:
                # Re-traversal: __torch_function__ was skipped so the output tensors were
                # never registered in _tensor_producers. Register them now, pointing back
                # to the same entries from the first traversal, so that downstream modules
                # (e.g. a module consuming this re-traversal's output) can resolve them.
                # strict=False: a shared module should always return the same tensor
                # count across traversals (same forward signature), but if it doesn't
                # we truncate silently rather than crash — the downstream module will
                # fall through to creating ephemeral ops, which is recoverable.
                for entry, tensor in zip(
                    self._module_output_producers.get(name, []),
                    flatten_tensors_to_list(outputs),
                    strict=False,
                ):
                    if entry is not None:
                        key = TensorIdVersion(id(tensor), tensor._version)
                        self._tensor_producers[key] = entry
                        weakref.finalize(tensor, self._tensor_producers.pop, key, None)
            self.parents.pop()
            self.traversed_modules.add(module)

        return hook

    def _capture_input_tensors(self, module: nn.Module, inputs: Any) -> None:
        """Create placeholder-like ops for each module-level input tensor."""
        for i, tensor in enumerate(flatten_tensors_to_list(inputs)):
            op_info = OpInfo(
                op_name=f"input_{i}",
                op_type=None,
                module_stack=(),
                source_frames=(),
                inputs=(),
                outputs={},
                is_state=False,
            )
            self._add_op(op_info)
            key = TensorIdVersion(id(tensor), tensor._version)
            self._tensor_producers[key] = InputEdge(op=op_info, output_idx=None)
            weakref.finalize(tensor, self._tensor_producers.pop, key, None)

    def _capture_output_tensors(self, module: nn.Module, inputs: Any, outputs: Any) -> None:
        """Create output-like ops for each module-level output tensor."""
        for i, tensor in enumerate(flatten_tensors_to_list(outputs)):
            key = TensorIdVersion(id(tensor), tensor._version)
            entry = self._tensor_producers.get(key)
            op_info = OpInfo(
                op_name=f"output_{i}",
                op_type=None,
                module_stack=(),
                source_frames=(),
                inputs=(entry,) if entry is not None else (),
                outputs={},
                is_state=False,
            )
            self._add_op(op_info)
            if entry is not None and entry.output_idx is not None:
                existing = entry.outputs.get(entry.output_idx, ())
                entry.op.outputs[entry.output_idx] = existing + (op_info,)

    def _get_module_stack(self) -> tuple[ModuleContext, ...]:
        return tuple(
            ModuleContext(module_name=named_mod.name, module_type=_fqn(type(named_mod.module)))
            for named_mod in self.parents
        )

    def _extract_source_frames(self) -> tuple[SourceFrame, ...]:
        # inspect.stack() would work here but reads source context for every
        # frame on the stack; walking frames manually and calling linecache only
        # for the forward() frames we keep is significantly faster on large models.
        frames: list[SourceFrame] = []
        frame = sys._getframe()
        while frame is not None:
            if frame.f_code.co_name == FORWARD_FUNCTION_NAME:
                code_context = linecache.getline(frame.f_code.co_filename, frame.f_lineno).strip()
                frames.append(
                    SourceFrame(
                        filename=frame.f_code.co_filename,
                        lineno=frame.f_lineno,
                        function_name=frame.f_code.co_name,
                        code_context=code_context,
                    )
                )
            frame = frame.f_back
        # Reverse: outermost forward first (matching graph mode order)
        return tuple(reversed(frames))

    def _resolve_inputs(
        self, input_tensor_keys: tuple[TensorIdVersion, ...]
    ) -> tuple[InputEdge, ...]:
        """Look up which previously-recorded ops produced the input tensors.

        Returns an ordered tuple of :class:`InputEdge` objects (duplicates
        preserved), each carrying the producing op and its output slot.
        """
        input_edges: list[InputEdge] = []
        for key in input_tensor_keys:
            if key.id in self._states_to_names:
                # State tensor: look up or create by id only — version is irrelevant.
                op_info = self._state_op_infos.get(key.id)
                if op_info is None:
                    op_info = OpInfo(
                        op_name=self._states_to_names[key.id],
                        op_type=None,
                        module_stack=(),
                        source_frames=(),
                        inputs=(),
                        outputs={},
                        is_state=True,
                    )
                    self._add_op(op_info)
                    self._state_op_infos[key.id] = op_info
                input_edges.append(InputEdge(op=op_info, output_idx=None))
            else:
                entry = self._tensor_producers.get(key)
                if entry is not None:
                    input_edges.append(entry)
                else:
                    # Unknown tensor: not a registered state and not produced by any
                    # intercepted op (e.g. a raw tensor attribute or a global tensor).
                    # Create an ephemeral OpInfo so the consuming op's inputs tuple is
                    # complete and the correct arg index appears in the formatted output.
                    # Ephemeral ops are NOT added to all_ops — they never appear as their
                    # own nodes in the summary tree.
                    input_edges.append(self._get_or_create_ephemeral(key.id))

        return tuple(input_edges)

    def _record_outputs(self, out: Any, op_info: OpInfo) -> None:
        """Record that op_info produced these output tensors."""
        for idx, tensor in enumerate(flatten_tensors_to_list(out)):
            key = TensorIdVersion(id(tensor), tensor._version)
            self._tensor_producers[key] = InputEdge(op=op_info, output_idx=idx)
            weakref.finalize(tensor, self._tensor_producers.pop, key, None)

    def _register_as_consumer(self, inputs: tuple[InputEdge, ...], consumer: OpInfo) -> None:
        """Append consumer to each producer's outputs dict at the given slot.

        Deduplicates per (producer, output_idx) pair so that passing the same
        tensor twice as arguments to one op only registers the consumer once per
        slot. A consumer that uses both output[0] and output[1] of the same
        producer is still registered in both slots.
        """
        seen: set[tuple[str, int | None]] = set()
        for slot in inputs:
            pair = (slot.op_name, slot.output_idx)
            if pair in seen:
                continue
            seen.add(pair)
            if slot.output_idx is not None:
                existing = slot.outputs.get(slot.output_idx, ())
                slot.op.outputs[slot.output_idx] = existing + (consumer,)

    def __torch_function__(
        self,
        func: Callable,
        types: list,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        # Snapshot input tensor keys BEFORE func() executes.
        # Critical for in-place ops: func() mutates _version, but
        # the producer was recorded at the pre-mutation version.
        input_tensor_keys = tuple(
            TensorIdVersion(id(t), t._version)
            for t in flatten_tensors_to_list((*args, *kwargs.values()))
        )

        out = func(*args, **kwargs)

        if self.current_module in self.traversed_modules:
            return out

        if not _is_interceptable_func(func):
            return out

        module_name = self.current_module_name

        # Compute op name with per-module counter
        base_name = get_func_base_name(func)
        count = self.func_counts[module_name][base_name]
        local_name = get_func_name(func, count)
        self.func_counts[module_name][base_name] = count + 1

        op_name = f"{module_name}.{local_name}" if module_name else local_name

        # Determine op_type and track supported ops.
        # In the case of compressor being None, simply use base_name as the op_type.
        # If the compressor is not None and a registry is in play, the op_type for the same op may
        # differ based on how it the function is associated in the registry.
        # Ops not registered will later be filtered out and will not appear as part of the module
        # tree, but can still appear as op inputs or outputs for other ops and as module
        # input and output ops.
        op_type = base_name
        if self._op_type_resolver is not None:
            resolved_type = self._op_type_resolver(func)
            if resolved_type is not None:
                op_type = resolved_type
                self.supported_op_names.add(op_name)

        module_stack = self._get_module_stack()
        source_frames = self._extract_source_frames()
        input_edges = self._resolve_inputs(input_tensor_keys)

        op_info = OpInfo(
            op_name=op_name,
            op_type=op_type,
            module_stack=module_stack,
            source_frames=source_frames,
            inputs=input_edges,
            outputs={},
            is_state=False,
        )

        self._add_op(op_info)

        self._register_as_consumer(input_edges, op_info)
        self._record_outputs(out, op_info)

        return out

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def _get_op_type_resolver(
    compressor: type[_BaseModelCompressor] | None,
) -> Callable[[Callable], str | None] | None:
    """Get the op type resolver for a compressor in eager mode.

    The resolver maps a torch function to its compressor-defined op type.
    Returns None if no compressor is specified.
    """
    if compressor is None:
        return None
    if issubclass(compressor, Quantizer):
        return EagerQuantizer.get_op_type_resolver()
    if issubclass(compressor, KMeansPalettizer):
        return KMeansPalettizer.get_op_type_resolver()
    msg = f"No eager mode op type resolver for compressor {compressor.__name__}."
    raise ValueError(msg)


def _populate_boundary_ops_eager(
    module: ModuleInfo,
    mode: _EagerOpDiscoveryMode,
    subtree_ops_by_module: dict[str, list[OpInfo]],
) -> None:
    """Populate input_ops and output_ops for all modules using hook-captured tensor boundaries.

    Recurses depth-first, then processes each module. Both lists are ordered by
    module spec index (position in the flattened module input/output tensor list),
    matching the index semantics of ``module_input_spec`` / ``module_output_spec``
    in the eager quantizer.

    Args:
        module (ModuleInfo): The module to populate.
        mode (_EagerOpDiscoveryMode): The discovery mode instance, used to access
            ``_module_input_producers`` and ``_module_output_producers``.
        subtree_ops_by_module (dict[str, list[OpInfo]]): Pre-partitioned map from
            module name to all ops in that module's subtree, in execution order.
            Built once in :func:`parse_ops_for_eager` from ``mode.all_ops`` using
            each op's ``module_stack`` to avoid an O(num_modules × num_ops) scan.

    Note:
        Compression's ``RegisterEagerOptimizationMode`` uses ``ModuleBoundaryTracker`` for
        boundary detection. The inspector uses inline resolution in ``_enter_module`` /
        ``_exit_module`` hooks instead. Both share the same conceptual methodology (capture
        tensors at module entry/exit), but the inspector resolves ``TensorIdVersion`` keys to
        ``OpInfo`` objects immediately while they are live, avoiding the need for a stable counter
        and a deferred lookup layer. Adopting ``ModuleBoundaryTracker`` here would align the hook
        call sites but add indirection (counter → ``OpInfo``) without removing the
        inspector-specific translation layer (state filtering, module→producers reconstruction).
    """
    for child in module.child_modules.values():
        _populate_boundary_ops_eager(child, mode, subtree_ops_by_module)

    subtree_ops_list = subtree_ops_by_module.get(module.module_name, [])
    # OpInfo defines __eq__ and __hash__ on op_name, so set membership is op_name-based.
    subtree_ops = set(subtree_ops_list)

    # Build a map: (external producer op_name, output_idx) → [(consuming_op, input_slot), ...]
    # covering all ops inside this subtree that have at least one external non-state input.
    # Keying by (op_name, output_idx) distinguishes e.g. the two outputs of a chunk/split op
    # when both flow into this module as separate forward args.
    external_to_consumers: dict[tuple[str, int | None], list[tuple[OpInfo, int]]] = {}
    for op in subtree_ops_list:
        for i, inp in enumerate(op.inputs):
            if not inp.is_state and inp.op not in subtree_ops:
                external_to_consumers.setdefault((inp.op_name, inp.output_idx), []).append((op, i))

    # input_ops: keyed by module input spec index (position in _module_input_producers).
    # Each key maps to all (op, input_slot) pairs that the tensor at that position feeds
    # into — a module input tensor can fan out to multiple ops inside the module.
    # Positions occupied by states or tensors with no consuming op in the subtree are absent.
    module.input_ops = {}
    for spec_idx, entry in enumerate(mode._module_input_producers.get(module.module_name, [])):
        if entry is None or entry.is_state:
            continue
        consumers = external_to_consumers.get((entry.op_name, entry.output_idx), [])
        if not consumers:
            continue
        module.input_ops[spec_idx] = [
            BoundaryEdge(op=op, index=input_idx) for op, input_idx in consumers
        ]

    # output_ops: keyed by module output spec index (position in _module_output_producers).
    # Each key maps to the single (op, output_slot) pair that produces the tensor at that
    # position. Positions occupied by states or untracked tensors are absent.
    module.output_ops = {}
    for spec_idx, entry in enumerate(mode._module_output_producers.get(module.module_name, [])):
        if entry is None:
            continue
        op_info, output_idx = entry.op, entry.output_idx
        if op_info not in subtree_ops or op_info.is_state or output_idx is None:
            continue
        module.output_ops[spec_idx] = BoundaryEdge(op=op_info, index=output_idx)


def parse_ops_for_eager(
    model: nn.Module,
    example_inputs: tuple[Any, ...],
    compressor: type[_BaseModelCompressor] | None = None,
) -> ModelSummary:
    """Discover ops by running a forward pass with torch function interception.

    Args:
        model (nn.Module): The model to inspect.
        example_inputs (tuple[Any, ...]): Example inputs for the forward pass.
        compressor (type[_BaseModelCompressor] | None): A compressor class to
            filter ops to only those supported by that compression algorithm.
            When None, all interceptable ops are included.

    Returns:
        ModelSummary: The discovered operations nested in a module hierarchy.
    """
    op_type_resolver = _get_op_type_resolver(compressor)
    mode = _EagerOpDiscoveryMode(model, op_type_resolver)
    try:
        with torch.no_grad(), mode:
            model(*example_inputs)
    finally:
        mode.remove_hooks()

    root = build_module_tree(_fqn(type(model)), mode.all_ops)

    # Pre-partition mode.all_ops by module subtree in a single O(N × D) pass, where N is
    # the total op count and D is the average module stack depth. Each op is appended to
    # every ancestor module's list, so each per-module list is already in execution order.
    subtree_ops_by_module: dict[str, list[OpInfo]] = defaultdict(list)
    for op in mode.all_ops:
        for ctx in op.module_stack:
            subtree_ops_by_module[ctx.module_name].append(op)

    _populate_boundary_ops_eager(root, mode, subtree_ops_by_module)

    if op_type_resolver is not None:
        root = filter_module_tree(root, mode.supported_op_names)

    return ModelSummary(model=root, mode=ExecutionMode.EAGER)
