# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Shared constants and utility functions for 16-bit casting passes.

Constants, op sets, and helper functions used by the FP16 and INT16 propagation
passes in ``casting``.
"""

from collections.abc import Callable, Collection, Iterable

import numpy as np
import torch

# =============================================================================
# Constants
# =============================================================================
# FP constants
_FP16_MAX = float(np.finfo(np.float16).max)  # 65504.0
_FP16_MIN = float(np.nextafter(0.0, 1.0, dtype=np.float16))  # ~5.96e-08
_FP16_NEG_MIN = float(np.nextafter(0.0, -1.0, dtype=np.float16))  # ~-5.96e-08
_FP32_INF_THRESHOLD = 1e38
FLOAT_DTYPES = frozenset({torch.float16, torch.float32, torch.float64, torch.bfloat16})

# INT16 constants
INT16_MAX = 32767
INT16_MIN = -32768
INT_DTYPES = frozenset({torch.int32, torch.int64})
# Extended int dtype set used by INT16 meta updates
INT_DTYPES_EXTENDED = frozenset(INT_DTYPES | {torch.int16})

# Dtype categories and bit-widths for safe-collapse analysis.
# A chain of casts is numerically safe to collapse when all dtypes share the
# same category and every intermediate is at least as wide as both endpoints.
_DTYPE_CATEGORY: dict[torch.dtype, str] = {
    torch.bfloat16: "float",
    torch.bool: "bool",
    torch.float8_e4m3fn: "float",
    torch.float8_e5m2: "float",
    torch.float16: "float",
    torch.float32: "float",
    torch.float64: "float",
    torch.int8: "signed_int",
    torch.int16: "signed_int",
    torch.int32: "signed_int",
    torch.int64: "signed_int",
    torch.uint8: "unsigned_int",
}

_DTYPE_BIT_WIDTH: dict[torch.dtype, int] = {
    torch.bfloat16: 16,
    torch.bool: 1,
    torch.float8_e4m3fn: 8,
    torch.float8_e5m2: 8,
    torch.float16: 16,
    torch.float32: 32,
    torch.float64: 64,
    torch.int8: 8,
    torch.int16: 16,
    torch.int32: 32,
    torch.int64: 64,
    torch.uint8: 8,
}


# =============================================================================
# Op Sets
# =============================================================================
# All dtype-casting ops (handled separately from unsupported ops)
CAST_OPS = frozenset(
    {
        torch.ops.aten._to_copy.default,
        torch.ops.aten.to.dtype,
        torch.ops.aten.to.dtype_layout,
    }
)

# Tensor creation ops
CREATION_OPS = frozenset(
    {
        torch.ops.aten.arange.default,
        torch.ops.aten.arange.start,
        torch.ops.aten.arange.start_step,
        torch.ops.aten.empty,
        torch.ops.aten.empty.memory_format,
        torch.ops.aten.full.default,
        torch.ops.aten.full_like.default,
        torch.ops.aten.ones.default,
        torch.ops.aten.ones_like.default,
        torch.ops.aten.rand.default,
        torch.ops.aten.randn.default,
        torch.ops.aten.scalar_tensor.default,
        torch.ops.aten.zeros.default,
        torch.ops.aten.zeros_like.default,
    }
)

# Integer computation ops for INT16 casting
INT16_COMPUTATION_OPS = frozenset(
    {
        torch.ops.aten.add.Tensor,
        torch.ops.aten.mul.Tensor,
        torch.ops.aten.sub.Tensor,
    }
)

# The below 3 sets contain ops which take integer inputs representing indices. For such ops, we
# ensure that the input nodes along with any parent nodes in the graph leading to the input node
# are not downcast.

# Ops where arg[1] is the index tensor
INDEX_ARG1_OPS = frozenset(
    {
        torch.ops.aten.embedding.default,
        torch.ops.aten.take_along_dim.default,
    }
)

# Ops where arg[2] is the index tensor
INDEX_ARG2_OPS = frozenset(
    {
        torch.ops.aten.gather.default,
        torch.ops.aten.index_select.default,
        torch.ops.aten.scatter.src,
        torch.ops.aten.scatter.value,
        torch.ops.aten.scatter_add.default,
        torch.ops.aten.scatter_reduce.two,
        torch.ops.aten.index_copy.default,
        torch.ops.aten.index_add.default,
        torch.ops.aten.index_fill.int_Scalar,
        torch.ops.aten.index_fill.int_Tensor,
    }
)

# Ops where arg[1] is a list of index tensors
INDEX_LIST_ARG1_OPS = frozenset(
    {
        torch.ops.aten.index.Tensor,
        torch.ops.aten.index_put.default,
        torch.ops.aten.index_put_.default,
    }
)


# =============================================================================
# Shared Helper Functions
# =============================================================================
def get_placeholder_store_and_tensor(
    param_name: str, exported_program: torch.export.ExportedProgram
) -> tuple[dict, torch.Tensor] | None:
    """Look up a parameter in state_dict or constants.
    Returns (store_dict, tensor) or None if not found or not a Tensor."""
    if param_name in exported_program.state_dict:
        store = exported_program.state_dict
    elif param_name in exported_program.constants:
        store = exported_program.constants
    else:
        return None
    tensor = store[param_name]
    if not isinstance(tensor, torch.Tensor):
        return None
    return store, tensor


def store_converted_placeholder(
    store: dict, param_name: str, original: torch.Tensor, converted: torch.Tensor
) -> None:
    """Store a converted tensor, preserving Parameter wrapper if needed."""
    if isinstance(original, torch.nn.Parameter):
        converted = torch.nn.Parameter(converted, requires_grad=original.requires_grad)
    store[param_name] = converted


def maybe_update_assert_dtype(node: torch.fx.Node) -> None:
    """Normalize and sync dtype on an _assert_tensor_metadata node.

    Handles a torch.export deserialization bug where dtype appears in both
    args[3] and kwargs['dtype'] after loading a .pt2 file. Drops kwargs['dtype']
    in that case to avoid "expected at most 6 argument(s) but received 7".
    Also updates the asserted dtype to match its input's current dtype.
    """
    # Torch deserialization bug handling
    if len(node.args) >= 4 and isinstance(node.args[3], torch.dtype) and "dtype" in node.kwargs:
        node.kwargs = {k: v for k, v in node.kwargs.items() if k != "dtype"}

    input_dtype = get_node_dtype(node.args[0])
    if input_dtype is not None:
        current = _get_assert_dtype(node)
        if current is not None and current != input_dtype:
            _set_assert_dtype(node, input_dtype)


def _get_assert_dtype(node: torch.fx.Node) -> torch.dtype | None:
    """Get the dtype checked by an _assert_tensor_metadata node."""
    dt = node.kwargs.get("dtype")
    if dt is not None:
        return dt
    if len(node.args) >= 4 and isinstance(node.args[3], torch.dtype):
        return node.args[3]
    return None


def _set_assert_dtype(node: torch.fx.Node, new_dtype: torch.dtype) -> None:
    """Set the dtype on an _assert_tensor_metadata node."""
    if "dtype" in node.kwargs or (len(node.args) < 4 or not isinstance(node.args[3], torch.dtype)):
        node.kwargs = {**node.kwargs, "dtype": new_dtype}
    else:
        node.args = node.args[:3] + (new_dtype,) + node.args[4:]


def get_to_op_dtype(node: torch.fx.Node) -> torch.dtype | None:
    """Extract the target dtype from any aten.to.* node."""
    if len(node.args) >= 2 and isinstance(node.args[1], torch.dtype):
        return node.args[1]
    return node.kwargs.get("dtype")


def set_to_op_dtype(node: torch.fx.Node, new_dtype: torch.dtype) -> None:
    """Set the target dtype on any aten.to.* node (args[1] or kwargs['dtype'])."""
    if len(node.args) >= 2 and isinstance(node.args[1], torch.dtype):
        node.args = (node.args[0], new_dtype) + node.args[2:]
    elif "dtype" in node.kwargs:
        node.kwargs = {**node.kwargs, "dtype": new_dtype}
    else:
        error_msg = f"Unable to set to_op dtype for node {node.name}"
        raise RuntimeError(error_msg)


def _to_op_has_copy_flag(node: torch.fx.Node) -> bool:
    """Return True if the to.* node forces a copy even when dtypes match."""
    if len(node.args) >= 4 and node.args[3]:
        return True
    return bool(node.kwargs.get("copy", False))


def get_node_dtype(node: torch.fx.Node) -> torch.dtype | None:
    """Get the dtype from a node's meta['val']."""
    val = node.meta.get("val")
    return val.dtype if hasattr(val, "dtype") else None


def insert_cast(
    graph: torch.fx.Graph,
    input_node: torch.fx.Node,
    target_dtype: torch.dtype,
    insert_before: torch.fx.Node,
    pass_inserted: set[torch.fx.Node] | None = None,
) -> torch.fx.Node:
    """Create a _to_copy cast node with correct metadata."""
    with graph.inserting_before(insert_before):
        cast_node = graph.create_node(
            "call_function",
            torch.ops.aten._to_copy.default,
            args=(input_node,),
            kwargs={"dtype": target_dtype},
            name=f"{input_node.name}_to_{target_dtype}".replace(".", "_"),
        )
        if "val" in input_node.meta:
            cast_node.meta["val"] = input_node.meta["val"].to(target_dtype)
    if pass_inserted is not None:
        pass_inserted.add(cast_node)
    return cast_node


def insert_cast_after(
    graph: torch.fx.Graph,
    node: torch.fx.Node,
    cast_dtype: torch.dtype,
    pass_inserted: set[torch.fx.Node] | None = None,
) -> None:
    """Insert a cast node after `node` and rewire all current users to it."""

    def _replace(item: object) -> object:
        if item is node:
            return cast_back
        if isinstance(item, (list, tuple)):
            return type(item)(_replace(elem) for elem in item)
        return item

    users = list(node.users.keys())
    if not users:
        return
    set_users = set(users)
    insert_point = next((n for n in graph.nodes if n in set_users), None)
    if insert_point is None:
        return
    cast_back = insert_cast(graph, node, cast_dtype, insert_point, pass_inserted=pass_inserted)
    for user in users:
        user.args = tuple(_replace(a) for a in user.args)
        user.kwargs = {k: _replace(v) for k, v in user.kwargs.items()}


def update_meta_dtype(
    node: torch.fx.Node,
    new_dtype: torch.dtype,
    valid_dtypes: frozenset,
) -> None:
    """Update meta['val'] dtype for tensors whose current dtype is in valid_dtypes."""
    if "val" not in node.meta:
        return
    val = node.meta["val"]
    if hasattr(val, "dtype") and val.dtype in valid_dtypes:
        node.meta["val"] = val.to(new_dtype)
    elif isinstance(val, (tuple, list)):
        node.meta["val"] = type(val)(
            v.to(new_dtype) if hasattr(v, "dtype") and v.dtype in valid_dtypes else v for v in val
        )


def _remove_noop_casts(graph: torch.fx.Graph) -> bool:
    """Remove cast ops where input dtype already matches target dtype."""
    changed = False
    for node in list(graph.nodes):
        if node.target not in CAST_OPS:
            continue
        if _to_op_has_copy_flag(node):
            continue
        input_node = node.all_input_nodes[0]
        target_dtype = get_to_op_dtype(node)
        input_dtype = get_node_dtype(input_node)
        if input_dtype is not None and input_dtype == target_dtype:
            node.replace_all_uses_with(input_node)
            graph.erase_node(node)
            changed = True
    return changed


def _bypass_branching_cast(graph: torch.fx.Graph, pass_inserted: set[torch.fx.Node]) -> bool:
    """Bypass a cast whose parent cast has multiple users.

    For outer_cast(B→C) whose input parent_cast(A→B) has >1 user:
      - A == C: rewire outer_cast's users to parent_cast's source, remove outer_cast
      - A != C: insert new cast(A→C) from source, rewire outer_cast's users, remove outer_cast

    parent_cast is left untouched for its other users. If all its cast children are
    bypassed in a single pass, parent_cast ends up with 0 users and is also erased.
    Single-user chains are left to _fuse_cast_chains.

    Both casts must be eligible: pass-inserted, or an original-model widening cast
    in the same dtype category.
    """
    changed = False
    for node in list(graph.nodes):
        if node.target not in CAST_OPS:
            continue
        parent = node.all_input_nodes[0]
        if parent.target not in CAST_OPS:
            continue
        if len(parent.users) <= 1:
            continue  # single-user chain: belongs to _fuse_cast_chains
        source = parent.all_input_nodes[0]
        source_dtype = get_node_dtype(source)
        target_dtype = get_to_op_dtype(node)
        if source_dtype is None or target_dtype is None:
            continue
        chain_category = _DTYPE_CATEGORY.get(source_dtype)
        if chain_category is None:
            continue
        if not _can_include_in_chain(parent, chain_category, pass_inserted):
            continue
        if not _can_include_in_chain(node, chain_category, pass_inserted):
            continue
        if source_dtype == target_dtype:
            replacement = source
        else:
            replacement = insert_cast(
                graph, source, target_dtype, insert_before=node, pass_inserted=pass_inserted
            )
        node.replace_all_uses_with(replacement)
        pass_inserted.discard(node)
        graph.erase_node(node)
        if len(parent.users) == 0:
            pass_inserted.discard(parent)
            graph.erase_node(parent)
        changed = True
    return changed


def _is_widening_cast(node: torch.fx.Node) -> bool:
    """Return True if the cast widens (target bit-width ≥ input bit-width)."""
    input_dtype = get_node_dtype(node.all_input_nodes[0])
    target_dtype = get_to_op_dtype(node)
    if input_dtype is None or target_dtype is None:
        return False

    # Use -1 and 0 as default return types for unrecognized dtypes to return False to be
    # conservative.
    return _DTYPE_BIT_WIDTH.get(target_dtype, -1) >= _DTYPE_BIT_WIDTH.get(input_dtype, 0)


def _can_include_in_chain(
    node: torch.fx.Node,
    chain_category: str,
    pass_inserted: set[torch.fx.Node],
) -> bool:
    """Return True if a cast node can be added to a chain.

    A cast is includable if it stays in the same dtype category AND either:
    - it was inserted by a cast pass (can always be collapsed), or
    - it is an original-model cast that widens (lossless, safe to skip) and
      does not force a copy (copy=True casts are intentional even when
      the dtype is unchanged).
    """
    target_dtype = get_to_op_dtype(node)
    if target_dtype is None:
        return False
    if _DTYPE_CATEGORY.get(target_dtype) != chain_category:
        return False
    if node in pass_inserted:
        return True
    if _to_op_has_copy_flag(node):
        return False
    return _is_widening_cast(node)


def _fuse_cast_chains(graph: torch.fx.Graph, pass_inserted: set[torch.fx.Node]) -> bool:
    """Fuse a linear chain of casts into a single cast (or eliminate entirely).

    For any chain  A → cast(A→B) → cast(B→C) → ... → cast(...→Z)  where every
    node except the last has exactly one user (the next cast), the chain collapses
    to:
      - A single cast  A → cast(A→Z)  if A != Z
      - Nothing (chain removed) if A == Z

    A cast is included in a chain if it stays within the same dtype category and
    either was inserted by a cast pass (always eligible) or is an original-model
    cast that widens (lossless and safe to skip). Original-model narrowing casts
    act as chain barriers.
    """
    changed = False
    erased: set[torch.fx.Node] = set()
    for node in list(graph.nodes):
        if node in erased:
            continue
        if node.target not in CAST_OPS:
            continue

        # Determine the chain's category from the first cast's input dtype.
        first_input_dtype = get_node_dtype(node.all_input_nodes[0])
        if first_input_dtype is None:
            continue
        chain_category = _DTYPE_CATEGORY.get(first_input_dtype)
        if chain_category is None:
            continue
        if not _can_include_in_chain(node, chain_category, pass_inserted):
            continue

        # Walk forward while the current node has exactly one user that is also
        # an includable cast op.
        chain: list[torch.fx.Node] = [node]
        current = node
        while True:
            users = list(current.users.keys())
            if len(users) != 1:
                break
            user = users[0]
            if user.target not in CAST_OPS:
                break
            if not _can_include_in_chain(user, chain_category, pass_inserted):
                break
            chain.append(user)
            current = user

        if len(chain) < 2:
            continue

        chain_input = chain[0].all_input_nodes[0]
        final_dtype = get_to_op_dtype(chain[-1])
        final_node = chain[-1]

        if first_input_dtype == final_dtype:
            # Round-trip: eliminate the entire chain
            final_node.replace_all_uses_with(chain_input)
            to_erase = chain
        else:
            # Collapse to a single cast: retarget chain[0] to final_dtype
            set_to_op_dtype(chain[0], final_dtype)
            val = chain_input.meta.get("val")
            if hasattr(val, "dtype"):
                chain[0].meta["val"] = val.to(final_dtype)
            final_node.replace_all_uses_with(chain[0])
            to_erase = chain[1:]

        for c in reversed(to_erase):
            erased.add(c)
            pass_inserted.discard(c)
            graph.erase_node(c)
        changed = True
    return changed


def _deduplicate_casts(graph: torch.fx.Graph, pass_inserted: set[torch.fx.Node]) -> None:
    r"""Merge identical cast nodes that share the same input and target dtype.

    When multiple casts from the same source to the same dtype exist (e.g. two
    sibling ops that both needed int32→int16 from the same parent), replace all
    but the first with the canonical cast node.  Iterating in topological order
    guarantees the first encountered is the earliest valid position.

                  / cast_1 (pass inserted fp32 -> 16) -> B (some op)
    Ex. A (some op)
                  \ cast_2 (pass inserted fp32 -> 16) -> C (some op)

    is turned into

                                                     / B (some op)
        A (some op) -> cast (pass inserted fp32 -> 16)
                                                     \ C (some op)

    Only casts that are chain-includable (pass-inserted, or widening originals
    without copy flag) are eligible.  Original narrowing or copy-flag casts are
    left untouched.
    """
    canonical: dict[tuple[torch.fx.Node, torch.dtype], torch.fx.Node] = {}
    for node in list(graph.nodes):
        if node.target not in CAST_OPS:
            continue
        target_dtype = get_to_op_dtype(node)
        if target_dtype is None:
            continue
        input_node = node.all_input_nodes[0]
        input_dtype = get_node_dtype(input_node)
        if input_dtype is None:
            continue
        chain_category = _DTYPE_CATEGORY.get(input_dtype)
        if chain_category is None:
            continue
        if not _can_include_in_chain(node, chain_category, pass_inserted):
            continue
        key = (input_node, target_dtype)
        if key in canonical:
            node.replace_all_uses_with(canonical[key])
            pass_inserted.discard(node)
            graph.erase_node(node)
        else:
            canonical[key] = node


def cleanup_casts(graph: torch.fx.Graph, pass_inserted: set[torch.fx.Node]) -> None:
    """Fixpoint loop: fuse/eliminate cast chains until no further simplification."""
    # Use bitwise OR (not `or`) so all three passes run every iteration,
    # allowing each to expose new simplification opportunities for the others.
    while (
        _fuse_cast_chains(graph, pass_inserted)
        | _remove_noop_casts(graph)
        | _bypass_branching_cast(graph, pass_inserted)
    ):
        pass
    _deduplicate_casts(graph, pass_inserted)


# =============================================================================
# FP Helper Functions
# =============================================================================
def check_tensor_overflow_fp16(tensor: torch.Tensor) -> bool:
    """Return True if any finite FP32 value exceeds FP16_MAX."""
    if tensor.dtype != torch.float32:
        return False
    abs_val = torch.abs(tensor)
    finite_mask = abs_val < _FP32_INF_THRESHOLD
    return bool(finite_mask.any() and torch.max(abs_val[finite_mask]) > _FP16_MAX)


def cast_tensor_to_fp16(tensor: torch.Tensor) -> torch.Tensor:
    """Cast an FP32 tensor to FP16, snapping near-zero values to ±FP16_MIN.

    Values with 0 < abs(val) < FP16_MIN would underflow to zero in fp16.
    This snaps them to the smallest representable fp16 value instead.
    """
    abs_val = tensor.abs()
    underflow_mask = (tensor != 0) & (abs_val < _FP16_MIN)
    if underflow_mask.any():
        tensor = tensor.clone()
        tensor[underflow_mask & (tensor > 0)] = _FP16_MIN
        tensor[underflow_mask & (tensor < 0)] = _FP16_NEG_MIN
    return tensor.to(torch.float16)


def cast_scalar_to_fp16(val: float) -> float | None:
    """Return the fp16-safe equivalent of a float scalar, or None if it overflows.

    - Values in (FP16_MAX, FP32_INF_THRESHOLD) are unrepresentable in fp16: return None.
    - Values >= FP32_INF_THRESHOLD are treated as inf and clamped to ±FP16_MAX.
    - Near-zero values (0 < abs < FP16_MIN) are snapped to ±FP16_MIN to prevent underflow.
    - All other values are rounded to fp16 precision.
    """
    abs_val = abs(val)
    if abs_val > _FP16_MAX and abs_val < _FP32_INF_THRESHOLD:
        return None
    if abs_val >= _FP32_INF_THRESHOLD:
        return _FP16_MAX if val > 0 else -_FP16_MAX
    if val != 0.0 and abs_val < _FP16_MIN:
        return _FP16_MIN if val > 0 else _FP16_NEG_MIN
    return float(np.float16(val))


def classify_float_args(node: torch.fx.Node) -> tuple[bool, bool]:
    """Single pass over args/kwargs to detect float inputs and overflow/input nodes still in fp32.

    Returns (has_float_input, has_overflow_or_fp32_node):
    - has_float_input: any arg is a float-dtype Node or a Python float scalar
    - has_overflow_or_fp32_node: any Node arg is still fp32 or any scalar float is between FP16_MAX
      and FP32_INF_THRESHOLD.
      Values >= FP32_INF_THRESHOLD are treated as inf and don't count as overflow.
    """

    def _check(arg: object) -> tuple[bool, bool]:
        if isinstance(arg, torch.fx.Node):
            dt = get_node_dtype(arg)
            if dt is not None and dt in FLOAT_DTYPES:
                return True, dt == torch.float32
        elif isinstance(arg, float):
            abs_val = abs(arg)
            return True, abs_val > _FP16_MAX and abs_val < _FP32_INF_THRESHOLD
        elif isinstance(arg, (list, tuple)):
            has_float, has_overflow_or_fp32_node = False, False
            for item in arg:
                f, o = _check(item)
                has_float |= f
                has_overflow_or_fp32_node |= o
                if has_float and has_overflow_or_fp32_node:
                    break
            return has_float, has_overflow_or_fp32_node
        return False, False

    return _check((*node.args, *node.kwargs.values()))


def is_ignored_op(
    node: torch.fx.Node,
    ignored_ops: (
        Collection[torch._ops.OpOverload | torch._ops.OpOverloadPacket]
        | torch._ops.OpOverload
        | torch._ops.OpOverloadPacket
        | Callable[[torch.fx.Node], bool]
        | None
    ),
) -> bool:
    """Check whether an FX node should be excluded from casting.

    Supported inputs:
    1. A set/collection of ``OpOverload`` or ``OpOverloadPacket`` instances
       (e.g. ``{torch.ops.aten.exp, torch.ops.aten.exp.default}``) or a single op instance.
    2. A predicate function taking a ``torch.fx.Node`` and returning ``bool``
       (e.g. ``lambda node: node.name == "exp_1"``).
    """
    if ignored_ops is None:
        return False

    if callable(ignored_ops) and not isinstance(
        ignored_ops, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)
    ):
        return bool(ignored_ops(node))

    if isinstance(ignored_ops, (torch._ops.OpOverload, torch._ops.OpOverloadPacket)):
        ignored_set = {ignored_ops}
    elif isinstance(ignored_ops, (set, frozenset)):
        ignored_set = ignored_ops
    elif isinstance(ignored_ops, Iterable):
        ignored_set = set(ignored_ops)
    else:
        return False

    target = node.target
    return target in ignored_set or getattr(target, "overloadpacket", None) in ignored_set


# =============================================================================
# INT16 Helper Functions
# =============================================================================
def build_unsafe_to_cast_nodes(graph: torch.fx.Graph) -> set[torch.fx.Node]:
    """Find all nodes whose values feed into an index argument.

    Walks backward from every index-arg of index-consuming ops, marking all
    upstream nodes as unsafe for int16 conversion.
    """
    unsafe: set[torch.fx.Node] = set()

    def _mark_upstream_as_unsafe(start: torch.fx.Node) -> None:
        stack = [start]
        while stack:
            node = stack.pop()
            if node in unsafe:
                continue
            unsafe.add(node)
            stack.extend(node.all_input_nodes)

    for node in graph.nodes:
        if node.target in INDEX_ARG1_OPS:
            if len(node.args) >= 2 and isinstance(node.args[1], torch.fx.Node):
                _mark_upstream_as_unsafe(node.args[1])

        elif node.target in INDEX_ARG2_OPS:
            if len(node.args) >= 3 and isinstance(node.args[2], torch.fx.Node):
                _mark_upstream_as_unsafe(node.args[2])

        elif node.target in INDEX_LIST_ARG1_OPS:
            if len(node.args) >= 2:
                for idx in node.args[1]:
                    if isinstance(idx, torch.fx.Node):
                        _mark_upstream_as_unsafe(idx)

    return unsafe


def build_user_reachable_nodes(
    user_input_placeholders: list[torch.fx.Node],
) -> set[torch.fx.Node]:
    """Forward-walk from user input placeholders, returning all reachable nodes.

    Nodes not in the returned set have no user-input in their ancestry and are
    constant-foldable (their values are fully determined at compile time by
    parameters, buffers, and creation ops).
    """
    reachable: set[torch.fx.Node] = set()
    stack = list(user_input_placeholders)
    while stack:
        node = stack.pop()
        if node in reachable:
            continue
        reachable.add(node)
        stack.extend(node.users.keys())
    return reachable


def build_castable_int16_nodes(
    graph: torch.fx.Graph,
    user_reachable: set[torch.fx.Node],
    unsafe: set[torch.fx.Node],
) -> set[torch.fx.Node]:
    """Build the set of nodes eligible for int16 casting via computation ops + propagation.

    Iterate all graph nodes and pick integer computation ops
    (``INT16_COMPUTATION_OPS``) that are user-reachable and not unsafe.  From
    each computation op, propagate bidirectionally through any op as long as
    the connecting tensor is integer-typed.

    Computation ops with a scalar operand (fewer than 2 tensor inputs) cannot
    run in int16 and are excluded as both seeds and propagation targets.
    """

    def _has_scalar_operand(node: torch.fx.Node) -> bool:
        if node.target not in INT16_COMPUTATION_OPS:
            return False
        return any(not isinstance(arg, torch.fx.Node) for arg in node.args[:2])

    castable: set[torch.fx.Node] = set()

    computation_ops: list[torch.fx.Node] = []
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        if node.target not in INT16_COMPUTATION_OPS:
            continue
        if node not in user_reachable:
            continue
        if node in unsafe:
            continue
        if _has_scalar_operand(node):
            continue
        has_int = (
            any(get_node_dtype(inp) in INT_DTYPES_EXTENDED for inp in node.all_input_nodes)
            or get_node_dtype(node) in INT_DTYPES_EXTENDED
        )
        if not has_int:
            continue
        computation_ops.append(node)

    stack = list(computation_ops)
    while stack:
        node = stack.pop()
        if node in castable:
            continue
        castable.add(node)

        for user in node.users:
            if user in castable or user in unsafe or user not in user_reachable:
                continue
            if user.op == "call_function" and get_node_dtype(user) in INT_DTYPES_EXTENDED:
                if _has_scalar_operand(user):
                    continue
                stack.append(user)

        for inp in node.all_input_nodes:
            if inp in castable or inp in unsafe or inp not in user_reachable:
                continue
            if get_node_dtype(inp) not in INT_DTYPES_EXTENDED:
                continue
            if _has_scalar_operand(inp):
                continue
            stack.append(inp)

    return castable


# ==================================================================================
# INT16 Legalization Workaround (TODO: check if converter can handle this correctly)
# ==================================================================================
# Ops that lower to coreai.reshape (strict same-element-type constraint)
_RESHAPE_OPS = {
    torch.ops.aten._unsafe_view.default,
    torch.ops.aten.clone.default,
    torch.ops.aten.contiguous.default,
    torch.ops.aten.expand.default,
    torch.ops.aten.narrow.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.reshape.default,
    torch.ops.aten.select.int,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.squeeze.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.squeeze.dims,
    torch.ops.aten.transpose.int,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.view.default,
}


def anchor_int16_reshape_inputs(graph: torch.fx.Graph, pass_inserted: set[torch.fx.Node]) -> int:
    """Insert explicit _to_copy(int16) before reshape ops fed by compute ops.

    The coreai legalization pass re-infers element types for compute ops,
    potentially overriding FX metadata (e.g. mul(int16, scalar) -> si32).
    Reshape ops enforce result type == operand type, so a re-inferred si32
    input with si16 output triggers a verification failure.  An explicit cast
    forces the compiler to emit coreai.cast(si32, si16) before the reshape.
    """
    count = 0
    for node in list(graph.nodes):
        if node.target not in _RESHAPE_OPS:
            continue
        out_val = node.meta.get("val")
        if not hasattr(out_val, "dtype") or out_val.dtype != torch.int16:
            continue
        if not node.args or not isinstance(node.args[0], torch.fx.Node):
            continue

        input_node = node.args[0]
        in_val = input_node.meta.get("val")
        if not hasattr(in_val, "dtype") or in_val.dtype != torch.int16:
            continue
        # Already an explicit int16 cast — no anchor needed
        if (
            input_node.op == "call_function"
            and input_node.target == torch.ops.aten._to_copy.default
            and input_node.kwargs.get("dtype") == torch.int16
        ):
            continue
        # Placeholders have explicitly declared types the compiler honours
        if input_node.op == "placeholder":
            continue
        # Another reshape op — those propagate si16 correctly in coreai
        if input_node.target in _RESHAPE_OPS:
            continue

        with graph.inserting_before(node):
            anchor = graph.create_node(
                "call_function",
                torch.ops.aten._to_copy.default,
                args=(input_node,),
                kwargs={"dtype": torch.int16},
                name=f"{input_node.name}_to_int16_anchor",
            )
            anchor.meta["val"] = in_val.to(torch.int16)
        pass_inserted.add(anchor)

        node.args = (anchor,) + node.args[1:]
        count += 1

    return count
