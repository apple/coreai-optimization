# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Data types for model operation inspection."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, ClassVar

from coreai_opt._utils.fx_utils import get_local_state_name as _get_local_state_name
from coreai_opt.quantization.config.quantization_config import ExecutionMode


@dataclass(frozen=True)
class SourceFrame:
    """A single frame in the source call stack leading to an operation.

    Represents one level of the call hierarchy, typically a ``forward()``
    method in the user's model code.

    Attributes:
        filename (str): Absolute or relative path to the source file.
        lineno (int): Line number in the source file.
        function_name (str): Name of the function (e.g., ``"forward"``).
        code_context (str): The source code text on that line, stripped of
            leading/trailing whitespace.
    """

    filename: str
    lineno: int
    function_name: str
    code_context: str


@dataclass(frozen=True)
class ModuleContext:
    """One level of the ``nn.Module`` nesting hierarchy.

    Attributes:
        module_name (str): Fully-qualified module name as it appears in
            ``model.named_modules()`` (e.g., ``"encoder.layer1"``).
            This is the string used by ``module_name_configs`` in
            :class:`~coreai_opt.quantization.config.QuantizerConfig`.
        module_type (str): Fully-qualified class name of the module (e.g.,
            ``"torch.nn.modules.linear.Linear"``). This is the string
            used by ``module_type_configs``.
    """

    module_name: str
    module_type: str


@dataclass(frozen=True)
class InputEdge:
    """One input edge into an op, pairing the producing op with its output slot.

    Used as the element type of :attr:`OpInfo.inputs`. Delegation properties
    forward the most-accessed :class:`OpInfo` attributes so that code iterating
    ``op.inputs`` does not need to go through ``.op`` for routine checks.

    Attributes:
        op (OpInfo): The op that produced this input tensor.
        output_idx (int | None): Which output slot of ``op`` this tensor came from,
            or ``None`` for synthetic ops (registered states, ephemeral/untracked
            tensors) that have no meaningful output slot.
    """

    op: OpInfo
    output_idx: int | None

    # --- delegation properties -------------------------------------------------
    @property
    def op_name(self) -> str:
        return self.op.op_name

    @property
    def op_type(self) -> str | None:
        return self.op.op_type

    @property
    def is_state(self) -> bool:
        return self.op.is_state

    @property
    def module_stack(self) -> tuple[ModuleContext, ...]:
        return self.op.module_stack

    @property
    def inputs(self) -> tuple[InputEdge, ...]:
        return self.op.inputs

    @property
    def outputs(self) -> dict[int, tuple[OpInfo, ...]]:
        return self.op.outputs

    @property
    def _display_name(self) -> str:
        return self.op._display_name


@dataclass(eq=False)
class OpInfo:
    """Information about a single operation discovered in a model.

    Attributes:
        op_name (str): The operation name that ``op_name_config`` regex patterns
            match against (e.g., ``"add_1"``, ``"linear"``).
        op_type (str | None): The operation type that ``op_type_config`` keys match
            against (e.g., ``"add"``, ``"linear"``). ``None`` if the
            type could not be determined.
        module_stack (tuple[ModuleContext, ...]): The ``nn.Module`` nesting hierarchy
            from outermost to innermost. The innermost entry's ``module_name`` is the
            string that ``module_name_configs`` would match, and its
            ``module_type`` is the string that ``module_type_configs``
            would match.
        source_frames (tuple[SourceFrame, ...]): Source code locations from outermost
            ``forward()`` to innermost, showing the call chain that produced this op.
            May be empty if source information is unavailable.
        inputs (tuple[InputEdge, ...]): Ordered input edges. Each :class:`InputEdge`
            carries the producing op and the output slot of that op the tensor came from.
        outputs (dict[int, tuple[OpInfo, ...]]): Dictionary mapping op outputs to a tuple of ops
            consuming the output.
        is_state (bool): ``True`` if this op represents a model parameter or
            buffer rather than a computation. State ops have an empty
            ``module_stack`` and do not appear in module tree or boundary lists.
    """

    op_name: str
    op_type: str | None
    module_stack: tuple[ModuleContext, ...]
    source_frames: tuple[SourceFrame, ...]
    inputs: tuple[InputEdge, ...]
    outputs: dict[int, tuple[OpInfo, ...]]
    is_state: bool

    _IMMUTABLE_FIELDS: ClassVar[frozenset[str]] = frozenset({"op_name"})

    @property
    def _display_name(self) -> str:
        """Name for user-facing output such as ``format_summary``.

        Equal to :attr:`op_name` except for state ops, where only the last
        dotted component is returned because state-matching configs are
        currently keyed on that suffix (e.g., a parameter with name ``"conv.weight"``
        can only be matched with ``"weight"`` in the config). Remove this property
        once full-FQN state matching lands (rdar://177076777).
        """
        if self.is_state:
            return _get_local_state_name(self.op_name) or self.op_name
        return self.op_name

    def __repr__(self) -> str:
        return f"OpInfo(op_name={self.op_name!r}, op_type={self.op_type!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, OpInfo):
            return NotImplemented
        return self.op_name == other.op_name

    def __hash__(self) -> int:
        return hash(self.op_name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._IMMUTABLE_FIELDS and name in self.__dict__:
            msg = (
                f"OpInfo.{name} is immutable after initialization "
                "(would invalidate the hash/equality contract)"
            )
            raise AttributeError(msg)
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in self._IMMUTABLE_FIELDS:
            msg = (
                f"OpInfo.{name} is immutable and cannot be deleted "
                "(would invalidate the hash/equality contract)"
            )
            raise AttributeError(msg)
        super().__delattr__(name)


@dataclass(frozen=True)
class BoundaryEdge:
    """A single data-flow edge crossing a module boundary.

    Each entry in :attr:`ModuleInfo.input_ops` or :attr:`ModuleInfo.output_ops`
    corresponds to one quantizer configuration point at the boundary.

    Attributes:
        op (OpInfo): The op inside the module at the boundary.
        index (int): For input boundaries: the input slot of ``op`` that
            receives external data, matching the index used in
            ``module_input_spec``. For output boundaries: the output slot of
            ``op`` whose tensor leaves the module, matching the index used in
            ``module_output_spec``.
    """

    op: OpInfo
    index: int


@dataclass
class ModuleInfo:
    """A node in the ``nn.Module`` hierarchy with its directly-owned ops.

    Mirrors the ``nn.Module`` nesting structure: each ``ModuleInfo``
    holds the ops that belong directly to that module and references its
    child modules as nested ``ModuleInfo`` instances.

    Attributes:
        module_name (str): Fully-qualified module name (e.g.,
            ``"encoder.conv1"``).  Empty string for the root module.
        module_type (str): Fully-qualified class name of the module (e.g.,
            ``"torch.nn.modules.conv.Conv2d"``).
        child_modules (dict[str, ModuleInfo]): Child modules keyed by
            ``module_name``, in insertion order.
        ops (list[OpInfo]): Ops directly owned by this module, in
            graph order.
        input_ops (dict[int, list[BoundaryEdge]]): Boundary edges where data enters this
            module from outside. Keys are module input spec indices (positions in the
            flattened module forward arguments). Values are lists of all
            ``(op, input_slot)`` pairs that the tensor at that position feeds into inside
            the module — a single module input tensor can fan out to multiple ops. Keys
            are absent for positions occupied by state tensors, untracked tensors, or
            unused arguments. The key is what the user passes to ``module_input_spec``.
        output_ops (dict[int, BoundaryEdge]): Boundary edges where data leaves this
            module. Keys are module output spec indices (positions in the flattened
            module return value). Each key maps to the ``(op, output_slot)`` pair that
            produces the tensor at that position. Keys are absent for positions occupied
            by state tensors or untracked tensors. The key is what the user passes to
            ``module_output_spec``.
    """

    module_name: str
    module_type: str
    child_modules: dict[str, ModuleInfo]
    ops: list[OpInfo]
    input_ops: dict[int, list[BoundaryEdge]]
    output_ops: dict[int, BoundaryEdge]

    def children(self) -> Iterator[ModuleInfo]:
        """Yield direct child modules in insertion order."""
        yield from self.child_modules.values()

    def named_children(self) -> Iterator[tuple[str, ModuleInfo]]:
        """Yield ``(module_name, ModuleInfo)`` for direct child modules."""
        for child in self.child_modules.values():
            yield child.module_name, child

    def modules(self) -> Iterator[ModuleInfo]:
        """Yield this module and all descendant modules in depth-first order."""
        yield self
        for child in self.child_modules.values():
            yield from child.modules()

    def named_modules(self) -> Iterator[tuple[str, ModuleInfo]]:
        """Yield ``(module_name, ModuleInfo)`` for this module and all descendants."""
        yield self.module_name, self
        for child in self.child_modules.values():
            yield from child.named_modules()

    def get_submodule(self, module_name: str) -> ModuleInfo:
        """Return a descendant module by its fully-qualified name.

        Args:
            module_name (str): Fully-qualified module name (e.g.,
                ``"encoder.conv1"``).

        Raises:
            KeyError: If no module with the given name exists in this subtree.
        """
        if module_name == self.module_name:
            return self
        # Strip this module's prefix to get the relative suffix, then walk
        # down one level at a time, rebuilding absolute FQNs for child_modules.
        if self.module_name:
            if not module_name.startswith(self.module_name + "."):
                raise KeyError(f"No submodule with name {module_name!r}")
            remaining = module_name[len(self.module_name) + 1 :]
        else:
            remaining = module_name
        parts = remaining.split(".")
        current = self
        fqn = self.module_name
        for part in parts:
            fqn = f"{fqn}.{part}" if fqn else part
            if fqn not in current.child_modules:
                raise KeyError(f"No submodule with name {module_name!r}")
            current = current.child_modules[fqn]
        return current

    def all_ops(self) -> list[OpInfo]:
        """Return all ops within this module and its submodules in graph order."""
        ops: list[OpInfo] = []
        for m in self.modules():
            ops.extend(m.ops)
        return ops


@dataclass(frozen=True)
class ModelSummary:
    """Complete listing of operations discovered in a model.

    Attributes:
        model (ModuleInfo): Top level module summary of the module hierarchy tree containing
            all discovered operations nested within their owning modules.
        mode (ExecutionMode): Which discovery mode was used: ``ExecutionMode.GRAPH`` for exported
            ``GraphModule`` models, ``ExecutionMode.EAGER`` for ``nn.Module`` models
            traced via a forward pass.
    """

    model: ModuleInfo
    mode: ExecutionMode
