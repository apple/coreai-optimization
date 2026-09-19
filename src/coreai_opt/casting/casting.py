# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
FP and INT 16-bit cast for torch exported programs.

FP32→FP16: Convert parameters/inputs upfront, walk nodes topologically inserting
casts only when ops should run in higher precision,
let cleanup collapse roundtrips between consecutive casted ops.

INT32→INT16: Convert inputs where all uses are safe args, walk nodes
topologically inserting int16 casts at designated ops with cast-backs after,
let cleanup collapse roundtrips between consecutive casted ops.

Public API:
    cast_fp32_to_fp16(exported_program) -> ExportedProgram
    cast_int32_to_int16(exported_program) -> ExportedProgram
    cast_to_16_bit_precision(exported_program) -> ExportedProgram (performs FP and then INT casting)
"""

import operator
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterable
from typing import Any

import torch

from coreai_opt._utils.casting_utils import (
    CAST_OPS as _CAST_OPS,
    CREATION_OPS as _CREATION_OPS,
    FLOAT_DTYPES as _FLOAT_DTYPES,
    INT16_MAX as _INT16_MAX,
    INT16_MIN as _INT16_MIN,
    INT_DTYPES as _INT_DTYPES,
    INT_DTYPES_EXTENDED as _INT_DTYPES_EXTENDED,
    anchor_int16_reshape_inputs as _anchor_int16_reshape_inputs,
    build_castable_int16_nodes as _build_castable_int16_nodes,
    build_unsafe_to_cast_nodes as _build_unsafe_to_cast_nodes,
    build_user_reachable_nodes as _build_user_reachable_nodes,
    cast_scalar_to_fp16 as _cast_scalar_to_fp16,
    cast_tensor_to_fp16 as _cast_tensor_to_fp16,
    check_tensor_overflow_fp16 as _check_tensor_overflow_fp16,
    classify_float_args as _classify_float_args,
    cleanup_casts as _cleanup_casts,
    get_node_dtype as _get_node_dtype,
    get_placeholder_store_and_tensor as _get_placeholder_store_and_tensor,
    get_to_op_dtype as _get_to_op_dtype,
    insert_cast as _insert_cast,
    insert_cast_after as _insert_cast_after,
    is_ignored_op as _is_ignored_op,
    maybe_update_assert_dtype as _maybe_update_assert_dtype,
    set_to_op_dtype as _set_to_op_dtype,
    store_converted_placeholder as _store_converted_placeholder,
    update_meta_dtype as _update_meta_dtype,
)

# =============================================================================
# Base Class
# =============================================================================


class _CastPassBase(ABC):
    """Abstract base for FP16 and INT16 cast passes."""

    def __init__(self, exported_program: torch.export.ExportedProgram) -> None:
        self._ep = exported_program
        self._pass_inserted: set[torch.fx.Node] = set()
        self._target_to_placeholder = self.get_param_name_to_placeholder_node()
        self._user_input_placeholders = self.get_user_input_placeholders()

    def __call__(self) -> torch.export.ExportedProgram:
        self.convert_placeholders()
        self.convert_user_inputs()
        self.iterate_nodes_and_insert_casts()
        self.cleanup()
        return self._ep

    def get_user_input_placeholders(self) -> list[torch.fx.Node]:
        """Get a list of nodes corresponding to user inputs"""
        user_input_names = {
            spec.arg.name
            for spec in self._ep.graph_signature.input_specs
            if spec.kind.name == "USER_INPUT"
        }
        return [
            node
            for node in self._ep.graph.nodes
            if node.op == "placeholder" and node.name in user_input_names
        ]

    def get_param_name_to_placeholder_node(self) -> dict[str, torch.fx.Node]:
        """Get a dictionary mapping param names to placeholder nodes.

        This is a two step process where we
        1. Use the graph input specs to get a mapping of target names to param names. Param names
           are the typical Pytorch dot convention, e.g. "conv.weight", whereas target names tend to
           have underscores replacing the names along with a prefix, e.g. "p_conv_weight".
        2. Iterate through nodes in the graph and check if it is a placeholder with a target name
           seen in step 1. If so, associate the param name with the node.
        """
        # "Target" here refers to the name that node.target will return in step 2, not spec.target
        # which is actually the param name with dot notation.
        # Ex. spec.target = conv.weight, spec.arg.name = node.target = p_conv_weight
        _target_name_to_param_name = {
            spec.arg.name: spec.target
            for spec in self._ep.graph_signature.input_specs
            if spec.target is not None
        }
        return {
            param_name: node
            for node in self._ep.graph.nodes
            if node.op == "placeholder"
            and (param_name := _target_name_to_param_name.get(node.name)) is not None
        }

    @abstractmethod
    def cleanup(self) -> None: ...

    @abstractmethod
    def convert_placeholders(self) -> None: ...

    @abstractmethod
    def convert_user_inputs(self) -> None: ...

    @abstractmethod
    def iterate_nodes_and_insert_casts(self) -> None: ...


# =============================================================================
# FP16Casting
# =============================================================================


class _FP16Casting(_CastPassBase):
    """FP32 to FP16 conversion for torch.export graphs.

    High level idea:
    FP32 to FP16 conversion tries to cast as much of the model to FP16 as
    possible. All parameters/buffers are casted up front unless their values
    cannot be casted. Operations which can be run in FP16 have their output
    dtypes updated. Otherwise, FP16 -> FP32 casts are inserted for whichever
    inputs need to be upcasted, the output dtype is set to FP32, and a
    FP32 -> FP16 downcast is added after the operation to bring the output
    tensor back to FP16.

    Cast op cleanup at the end of the flow helps to remove unnecessary back to
    back casts.

    Algorithm:
    1. Convert parameters and buffers: check all placeholder nodes which are
       not user inputs. If all elements of the node's value satisfy
       abs(value) < FP16_MAX, cast the node's value to FP16. Otherwise leave
       as FP32.

       Corner case (tensor underflow):
       - Certain tensor values are 0 < abs(value) < FP16_MIN (~5.96e-08). In
         this case, set those values to either FP16_MIN or -FP16_MIN depending
         on sign.

    2. Convert user inputs: Mark all FP32 user input placeholders as FP16 dtype.

    3. Walk graph topologically: iterate through graph nodes in topological
       order. If the node has no FP node or scalar input, skip it. Otherwise,
       determine if the node should run in FP16 or FP32:
       - Node runs in FP32 if any node inputs are FP32 dtype, or if any scalars
         are FP16_MAX < abs(scalar) < FP32_INF_THRESHOLD
       - Otherwise node runs in FP16

       If the node is to be run in FP16,
       - for scalar inputs that are abs(value) > FP32_INF_THRESHOLD, update
         them to be -FP16_MAX or FP16_MAX depending on sign
       - for scalar inputs that are 0 < abs(value) < FP16_MIN, update them to
         be -FP16_MIN or FP16_MIN depending on sign
       - cast all other scalar inputs to FP16 values using
         float(np.float16(val))
       - mark the output dtype as FP16

       If the node is to be run in FP32,
       - insert FP16 -> FP32 upcasts for any node inputs with FP16 dtype
       - add a FP32 -> FP16 downcast to the node's output

    4. Cleanup casts: Run various passes through the model continuously until
       graph convergence. Passes include:
       - Removal of casts with same input and output dtypes
       - Condensing and/or eliminating chains of 2 or more casts
       - Condensing branches where a cast has multiple children, one or more of
         which is a cast

       Note: when removing casts, we distinguish between casts inserted by the
       algorithm vs. existing casts in the model. For casts inserted by the
       algorithm we can be more lenient in removing them. For example,
       typically a back to back narrowing cast chain like FP32 -> FP16 -> FP32
       cannot be removed since there is a potential lowering of precision,
       while a back to back widening cast chain like FP16 -> FP32 -> FP16 can
       always be removed. In the narrowing cast chain case, we only eliminate
       it if all casts involved are ones inserted by the algorithm. In the
       widening cast chain case, existing model casts can also be removed.
    """

    def __init__(
        self,
        exported_program: torch.export.ExportedProgram,
        ignored_ops: (
            Collection[torch._ops.OpOverload | torch._ops.OpOverloadPacket]
            | torch._ops.OpOverload
            | torch._ops.OpOverloadPacket
            | Callable[[torch.fx.Node], bool]
            | None
        ) = None,
    ) -> None:
        super().__init__(exported_program)
        if isinstance(ignored_ops, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)):
            self._ignored_ops: set[Any] | Callable[[torch.fx.Node], bool] | None = {ignored_ops}
        elif isinstance(ignored_ops, (set, frozenset)):
            self._ignored_ops = ignored_ops
        elif isinstance(ignored_ops, Iterable) and not callable(ignored_ops):
            self._ignored_ops = set(ignored_ops)
        else:
            self._ignored_ops = ignored_ops

    # -------------------------------------------------------------------------
    # Step 1: Convert parameters
    # -------------------------------------------------------------------------
    def convert_placeholder(self, param_name: str) -> bool:
        """Convert a single FP32 state_dict or constants entry to FP16.
        Returns True if converted."""
        result = _get_placeholder_store_and_tensor(param_name, self._ep)
        if result is None:
            return False
        store, tensor = result

        if tensor.dtype != torch.float32 or _check_tensor_overflow_fp16(tensor):
            return False

        tensor_data = tensor.detach() if isinstance(tensor, torch.nn.Parameter) else tensor
        tensor_fp16 = _cast_tensor_to_fp16(tensor_data)
        _store_converted_placeholder(store, param_name, tensor, tensor_fp16)
        return True

    def convert_placeholders(self) -> None:
        """Convert all FP32 parameters/constants to FP16 and update placeholder metadata."""
        for target, placeholder in self._target_to_placeholder.items():
            if self.convert_placeholder(target):
                if "val" in placeholder.meta:
                    placeholder.meta["val"] = placeholder.meta["val"].to(torch.float16)

    # -------------------------------------------------------------------------
    # Step 2: Convert user inputs
    # -------------------------------------------------------------------------
    def convert_user_inputs(self) -> None:
        """Change FP32 user input placeholders to FP16."""
        for node in self._user_input_placeholders:
            val = node.meta.get("val")
            if hasattr(val, "dtype") and val.dtype == torch.float32:
                node.meta["val"] = val.to(torch.float16)

    # -------------------------------------------------------------------------
    # Step 3: Iterate nodes and insert casts

    # Walk the graph topologically. For each node, take certain actions:
    # - Special nodes: (getitem, assert, creation ops, etc.) have specific
    #   handlers
    # - All other nodes:
    #     - If there are no float args or node inputs, move on to next node
    #     - Decide if node is "overflow" and should run in FP32 by looking at
    #       float args and node inputs
    #     - Node is determined to run in FP32 if any node input has FP32 dtype,
    #       or if any float scalar arg value is
    #       FP16_MAX < value < FP32_INF_THRESHOLD
    #     - If node should run in FP32, insert FP16 -> FP32 casts for any node
    #       inputs in FP16 and insert FP32 -> FP16 cast after the node
    #     - Otherwise cast any float scalar args to FP16 values and update the
    #       node's meta val dtype to FP16
    # -------------------------------------------------------------------------
    def handle_creation_op(self, node: torch.fx.Node) -> bool:
        """Set dtype=fp16 on creation ops. Returns True if handled."""
        val = node.meta.get("val")
        if not (hasattr(val, "dtype") and val.dtype == torch.float32):
            return False

        # Check and clamp float scalar args/kwargs; abort if any overflow fp16 range.
        # Build new values first so node is not partially mutated on early return.
        new_args = list(node.args)
        for i, arg in enumerate(new_args):
            if not isinstance(arg, float):
                continue
            clamped = _cast_scalar_to_fp16(arg)
            if clamped is None:
                return False
            new_args[i] = clamped

        new_kwargs = dict(node.kwargs)
        for key, kwarg in new_kwargs.items():
            if not isinstance(kwarg, float):
                continue
            clamped = _cast_scalar_to_fp16(kwarg)
            if clamped is None:
                return False
            new_kwargs[key] = clamped

        node.args = tuple(new_args)
        node.kwargs = {**new_kwargs, "dtype": torch.float16}
        _update_meta_dtype(node, torch.float16, _FLOAT_DTYPES)
        return True

    def iterate_nodes_and_insert_casts(self) -> None:
        """Walk graph topologically inserting casts only where needed."""
        graph = self._ep.graph

        for node in list(graph.nodes):
            if node.op != "call_function":
                continue

            # --- getitem: reuse dtype from container ---
            if node.target == operator.getitem:
                assert len(node.args) == 2
                container_val = node.args[0].meta.get("val")
                index = node.args[1]
                element = container_val[index]
                if hasattr(element, "dtype"):
                    node.meta["val"] = element
                continue

            # --- assert nodes: update dtype to match input ---
            if node.target == torch.ops.aten._assert_tensor_metadata.default:
                _maybe_update_assert_dtype(node)
                continue

            # --- cast ops: retarget fp32 output to fp16 ---
            if node.target in _CAST_OPS:
                target_dtype = _get_to_op_dtype(node)
                if target_dtype == torch.float32:
                    _set_to_op_dtype(node, torch.float16)
                    _update_meta_dtype(node, torch.float16, _FLOAT_DTYPES)
                    self._pass_inserted.add(node)
                continue

            # --- ignored ops: keep in FP32 ---
            if _is_ignored_op(node, self._ignored_ops):
                self.handle_overflow_op(node)
                continue

            # --- creation ops: set dtype directly ---
            if node.target in _CREATION_OPS:
                self.handle_creation_op(node)
                continue

            # --- general compute ops ---
            self.handle_compute_op(node)

    def cleanup(self) -> None:
        """Eliminate redundant casts and dead code."""
        _cleanup_casts(self._ep.graph, self._pass_inserted)
        self._ep.graph.eliminate_dead_code()
        self._ep.graph_module.recompile()

    def handle_compute_op(self, node: torch.fx.Node) -> None:
        """Handle a general compute op."""
        has_float, has_overflow_or_fp32_node = _classify_float_args(node)
        if not has_float:
            return

        if has_overflow_or_fp32_node:
            self.handle_overflow_op(node)
        else:
            self.handle_non_overflow_op(node)

    def handle_overflow_op(self, node: torch.fx.Node) -> None:
        """Op stays FP32: insert fp16->fp32 casts before, fp32->fp16 after."""

        # Cast any fp16 Node inputs to fp32
        def _cast_up(arg: object) -> object:
            if isinstance(arg, torch.fx.Node) and _get_node_dtype(arg) == torch.float16:
                return _insert_cast(
                    self._ep.graph, arg, torch.float32, node, pass_inserted=self._pass_inserted
                )
            if isinstance(arg, (list, tuple)):
                return type(arg)(_cast_up(item) for item in arg)
            return arg

        node.args = tuple(_cast_up(a) for a in node.args)
        node.kwargs = {k: _cast_up(v) for k, v in node.kwargs.items()}

        # Output stays fp32 — but insert fp16 cast-back for downstream consumers
        out_val = node.meta.get("val")
        if not (hasattr(out_val, "dtype") and out_val.dtype in _FLOAT_DTYPES):
            return

        _insert_cast_after(self._ep.graph, node, torch.float16, pass_inserted=self._pass_inserted)

    def handle_non_overflow_op(self, node: torch.fx.Node) -> None:
        """No overflow: inputs are fp16, clamp inf-like scalars, update metadata."""

        def _update_arg(arg: object) -> object:
            if isinstance(arg, float):
                casted_scalar = _cast_scalar_to_fp16(arg)
                # If _cast_scalar_to_fp16 were to return None, we would have taken the
                # _handle_overflow_op case instead.
                assert casted_scalar is not None
                return casted_scalar
            # Rewrite dtype arguments (e.g. output dtype kwargs) from fp32 to fp16.
            # This is safe in the non-overflow path because all float inputs are already
            # fp16, so any dtype arg specifying fp32 is an output-type that should match.
            if isinstance(arg, torch.dtype) and arg == torch.float32:
                return torch.float16
            return arg

        node.args = tuple(_update_arg(a) for a in node.args)
        node.kwargs = {k: _update_arg(v) for k, v in node.kwargs.items()}
        _update_meta_dtype(node, torch.float16, _FLOAT_DTYPES)


# =============================================================================
# INT16Casting
# =============================================================================


class _INT16Casting(_CastPassBase):
    """INT32/INT64 to INT16 conversion for torch.export graphs.

    High level idea:
    INT32 to INT16 casting shares the same high level workflow as FP32 to
    FP16 casting in that it casts placeholders first, then walks the graph
    topologically to insert casts, and finally cleans up casts at the end.
    However INT32 to INT16 is more conservative in casting compared to FP32
    to FP16 casting.

    Prior to cast insertion, we run multiple passes through the graph to
    identify nodes which are castable. Castable nodes are nodes dealing with
    integer tensors which are computation ops or adjacent to computation
    ops, which are not eventually used as an index input or residing in
    a constant foldable part of the graph.

    No parameters/buffers are casted up front, and a user input is only
    cast to INT16 if it is present in the castable nodes set.

    When walking the graph topologically, casts are only inserted for nodes
    which are in the castable set. INT32 -> INT16 downcasts are inserted
    for all integer node inputs, and INT16 -> INT32 upcasts are inserted
    after the node to bring the tensor back to INT32.

    Finally, cast op cleanup at the end of the flow helps to remove
    unnecessary back to back casts (same logic as FP32->FP16 cast cleanup).

    Algorithm:
    1. Identify uncastable ops:
       - Identify constant foldable nodes: the coreai-torch converter can
         typically constant-fold parts of the graph which do not depend on
         dynamic user inputs, but cannot perform folding if INT32 -> INT16
         casts are present. Identify all parts of the graph which do not
         depend on user inputs by traversing the graph starting from user
         inputs and tracking all nodes which are not encountered. Add all
         such ops to uncastable ops set.

       - Identify unsafe ops leading to indices: we do not want to
         downcast any nodes whose outputs eventually are used as index
         inputs for certain ops (ops in INDEX_ARG1_OPS, INDEX_ARG2_OPS,
         and INDEX_LIST_ARG1_OPS). Iterate through the graph and identify
         all such ops. For each one, traverse the graph upwards through
         their index input and mark all nodes visited as uncastable ops.

         The ops with indices as inputs themselves are not marked as
         uncastable. If their data tensor input is int dtype, they may
         still be cast if they are part of a group of ops connected to an
         op in INT16_COMPUTATION_OPS (see step #2 below).

    2. Identify castable ops:
       Iterate through the graph and identify all INT16_COMPUTATION_OPS
       (add/mul/sub) ops which take tensors as inputs. (Computation ops
       with scalar inputs cannot run in INT16 as per the coreai-torch
       converter).

       For all matching ops, propagate upwards through parent nodes and
       downwards through child nodes following only int dtype inputs and
       outputs. Any node traversed which is not in the uncastable ops set
       is marked as castable. The traversal ends when there are no more
       integer inputs/outputs to follow, or if the parent/child node
       encountered is an uncastable op.

    3. Convert user inputs: Mark user input placeholders as int16 dtype if
       the placeholder is in the castable set.

    4. Walk graph topologically: iterate through graph nodes in topological
       order. If the node has no integer node or scalar input, skip it.
       Otherwise, if the node is in the castable set, insert INT32 -> INT16
       casts to all integer inputs and insert a INT16 -> INT32 cast after
       the node.

    5. Cleanup casts: Run various passes through the model continuously until
       graph convergence. Passes include:
       - Removal of casts with same input and output dtypes
       - Condensing and/or eliminating chains of 2 or more casts
       - Condensing branches where a cast has multiple children, one or more of
         which is a cast

       The same logic as that of FP32->FP16 casting regarding when chains of
       casts can be eliminated applies.
    """

    def __init__(self, exported_program: torch.export.ExportedProgram) -> None:
        super().__init__(exported_program)
        self._unsafe_nodes = _build_unsafe_to_cast_nodes(self._ep.graph)
        user_reachable = _build_user_reachable_nodes(self._user_input_placeholders)
        self._castable_nodes = _build_castable_int16_nodes(
            self._ep.graph, user_reachable, self._unsafe_nodes
        )

    def cleanup(self) -> None:
        """Eliminate redundant casts, anchor reshape inputs, then recompile."""
        _cleanup_casts(self._ep.graph, self._pass_inserted)
        _anchor_int16_reshape_inputs(self._ep.graph, self._pass_inserted)
        self._ep.graph.eliminate_dead_code()
        self._ep.graph_module.recompile()

    def is_placeholder_in_range(self, target: str) -> bool:
        result = _get_placeholder_store_and_tensor(target, self._ep)
        if result is None:
            return False
        _, tensor = result
        return (
            tensor.dtype in _INT_DTYPES
            and _INT16_MIN <= tensor.min().item()
            and tensor.max().item() <= _INT16_MAX
        )

    # -------------------------------------------------------------------------
    # Step 1: Convert parameters (no-op)
    # -------------------------------------------------------------------------

    def convert_placeholders(self) -> None:
        # INT16 parameters are not converted upfront. Instead, int32/int64 inputs
        # are downcast when applicable, and cleanup collapses roundtrip casts
        # between consecutive ops.
        pass

    # -------------------------------------------------------------------------
    # Step 2: Convert user inputs
    # -------------------------------------------------------------------------

    def convert_user_inputs(self) -> None:
        """Change int32/int64 user input placeholders to int16 where castable."""
        for node in self._user_input_placeholders:
            val = node.meta.get("val")
            if hasattr(val, "dtype") and val.dtype in _INT_DTYPES:
                if node in self._castable_nodes:
                    node.meta["val"] = val.to(torch.int16)

    # -------------------------------------------------------------------------
    # Step 3: Iterate nodes and insert casts

    # Walk the graph topologically. For each node, take certain actions:
    # - Special nodes: (assert, cast_ops) have specific handlers
    # - All other nodes:
    #     - Skip the node if it is not in the castable set
    #     - For castable integer node inputs, insert an INT32 -> INT16 cast
    #     - Insert an INT16 -> INT32 cast after the node
    # -------------------------------------------------------------------------
    def iterate_nodes_and_insert_casts(self) -> None:
        """Walk topologically, inserting int16 casts at castable nodes."""
        graph = self._ep.graph

        for node in list(graph.nodes):
            if node.op != "call_function":
                continue

            # Assert nodes: update dtype to match input
            if node.target == torch.ops.aten._assert_tensor_metadata.default:
                _maybe_update_assert_dtype(node)
                continue

            # Cast ops targeting int16: retarget to int32 to normalize the graph. If users of
            # this cast op are castable nodes, another cast int32 -> 16 will be inserted, and both
            # this cast as well as the second cast will be removed during _cleanup().
            if node.target in _CAST_OPS:
                if _get_to_op_dtype(node) == torch.int16:
                    _set_to_op_dtype(node, torch.int32)
                    _update_meta_dtype(node, torch.int32, _INT_DTYPES_EXTENDED)
                    self._pass_inserted.add(node)
                continue

            # Castable nodes: cast int32 data args to int16, insert cast-back
            if node in self._castable_nodes:
                self.handle_cast_to_int16_op(node)

    def handle_cast_to_int16_op(self, node: torch.fx.Node) -> None:
        """For a castable node, cast int32 data args to int16 and insert cast-back."""
        graph = self._ep.graph

        # Check if any Node input or output is int16/int32/int64 (worth processing)
        has_int = (
            any(_get_node_dtype(inp) in _INT_DTYPES_EXTENDED for inp in node.all_input_nodes)
            or _get_node_dtype(node) in _INT_DTYPES_EXTENDED
        )
        if not has_int:
            return

        # Only cast int32/int64 inputs which are safe and castable (or non-placeholder)
        def _cast_down(arg: object) -> object:
            if isinstance(arg, (list, tuple)):
                return type(arg)(_cast_down(item) for item in arg)
            if not isinstance(arg, torch.fx.Node):
                return arg
            if _get_node_dtype(arg) not in _INT_DTYPES:
                return arg
            if arg in self._unsafe_nodes:
                return arg
            if arg.op == "placeholder" and arg not in self._castable_nodes:
                return arg
            return _insert_cast(graph, arg, torch.int16, node, pass_inserted=self._pass_inserted)

        node.args = tuple(_cast_down(a) for a in node.args)
        node.kwargs = {k: _cast_down(v) for k, v in node.kwargs.items()}

        # Only update metadata if at least one input actually ended up as int16.
        # If every int input was skipped (e.g. all in _unsafe_nodes), or if an input placeholder
        # was not cast due to not being castable, the output dtype should remain unchanged.
        if not any(_get_node_dtype(inp) == torch.int16 for inp in node.all_input_nodes):
            return

        # Update output metadata to int16
        _update_meta_dtype(node, torch.int16, _INT_DTYPES_EXTENDED)

        # Insert int16→int32 cast-back for downstream consumers
        out_val = node.meta.get("val")
        if not (hasattr(out_val, "dtype") and out_val.dtype == torch.int16):
            return

        _insert_cast_after(graph, node, torch.int32, pass_inserted=self._pass_inserted)


# =============================================================================
# Public API
# =============================================================================
def cast_fp32_to_fp16(
    exported_program: torch.export.ExportedProgram,
    ignored_ops: (
        Collection[torch._ops.OpOverload | torch._ops.OpOverloadPacket]
        | torch._ops.OpOverload
        | torch._ops.OpOverloadPacket
        | Callable[[torch.fx.Node], bool]
        | None
    ) = None,
) -> torch.export.ExportedProgram:
    """Convert a torch exported program from FP32 to FP16 where applicable.

    Converts parameters, user inputs, and compute ops to FP16, inserting
    casts only where values would overflow FP16 range.

    Args:
        exported_program: Exported program to convert.
        ignored_ops: Optional operations or predicate to exclude from FP16 casting.
            Can be a collection/set of ``OpOverload`` or ``OpOverloadPacket``
            instances (e.g. ``{torch.ops.aten.exp, torch.ops.aten.exp.default}``),
            a single op instance, or a predicate callable taking a ``torch.fx.Node``
            and returning a boolean (e.g. ``lambda node: node.name == "exp_1"``).
            Ignored ops are kept in FP32 with boundary casts inserted.

    Returns:
        The modified exported program.
    """
    return _FP16Casting(exported_program, ignored_ops=ignored_ops)()


def cast_int32_to_int16(
    exported_program: torch.export.ExportedProgram,
) -> torch.export.ExportedProgram:
    """Convert INT32/INT64 tensors to INT16 in a torch exported program.

    Only designated ops (data operations) are converted. Positional values
    (indices, strides, dimensions) are left as int32/int64.
    """
    return _INT16Casting(exported_program)()


def cast_to_16_bit_precision(
    exported_program: torch.export.ExportedProgram,
    ignored_ops: (
        Collection[torch._ops.OpOverload | torch._ops.OpOverloadPacket]
        | torch._ops.OpOverload
        | torch._ops.OpOverloadPacket
        | Callable[[torch.fx.Node], bool]
        | None
    ) = None,
) -> torch.export.ExportedProgram:
    """Convert a torch exported program to 16-bit precision: FP32→FP16 and INT32/64→INT16.

    Runs both cast passes sequentially:
    1. cast_fp32_to_fp16: FP32→FP16
    2. cast_int32_to_int16: INT32/INT64→INT16

    Args:
        exported_program: Exported program to convert.
        ignored_ops: Optional operations or predicate to exclude from FP16 casting.
            Forwarded to :func:`cast_fp32_to_fp16`.

    Returns:
        The modified exported program.
    """
    cast_fp32_to_fp16(exported_program, ignored_ops=ignored_ops)
    cast_int32_to_int16(exported_program)
    return exported_program
