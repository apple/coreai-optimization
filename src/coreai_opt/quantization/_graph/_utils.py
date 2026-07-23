# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import logging
from typing import Any

import torch
from torch.fx import GraphModule, Node

from coreai_opt.quantization.spec.fake_quantize import FakeQuantizeImplBase
from coreai_opt.quantization.spec.granularity import (
    PerChannelGranularity,
    PerTensorGranularity,
    QuantizationGranularity,
)

logger = logging.getLogger(__name__)
# torchao < 0.16.0 does not allow kwargs for annotated nodes during prepare_qat_pt2e call
# Kwargs containing value types other than _TORCHAO_DISALLOWED_NODE_KWARG_TYPES
# (e.g str, int) are pure metadata and safe to strip and restore around the prepare_qat_pt2e call.
_TORCHAO_DISALLOWED_NODE_KWARG_TYPES = (torch.fx.Node, torch.Tensor)


# Aten ops that alter channel axis semantics, making per-channel/per-block granularity
# invalid for shared observers. These ops remap or collapse dimensions so the
# channel axis in the input has no meaningful counterpart in the output; also
# includes pooling ops (max/avg/adaptive-avg pool, mean), which preserve rank but
# can shrink the specific axis a per-channel granularity points at (see
# force_per_tensor_for_channel_altering_ops for how this is handled precisely).
_CHANNEL_ALTERING_ATEN_OPS = {
    torch.ops.aten.flatten.using_ints,
    torch.ops.aten.reshape.default,
    torch.ops.aten.permute.default,
    torch.ops.aten.transpose.int,
    torch.ops.aten.t.default,
    torch.ops.aten.view.default,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.max_pool1d.default,
    torch.ops.aten.max_pool2d.default,
    torch.ops.aten.max_pool3d.default,
    torch.ops.aten.avg_pool1d.default,
    torch.ops.aten.avg_pool2d.default,
    torch.ops.aten.avg_pool3d.default,
    torch.ops.aten.adaptive_avg_pool1d.default,
    torch.ops.aten.adaptive_avg_pool2d.default,
    torch.ops.aten.adaptive_avg_pool3d.default,
    # AvgPoolPattern also registers "mean" as a shared-observer op (some global
    # average pooling is traced as x.mean(dim=[...]) rather than AvgPool/
    # AdaptiveAvgPool), so it needs the same axis-size-mismatch protection.
    torch.ops.aten.mean.dim,
}


def resolve_attr(model: GraphModule, target: str) -> torch.Tensor:
    """Resolve attribute from graph module by target path.

    Args:
        model: The graph module containing the attribute
        target: Dot-separated path to the attribute (e.g., "layer1.weight")

    Returns:
        The resolved tensor attribute

    Raises:
        AttributeError: If any component in the path cannot be resolved

    """
    obj = model
    for atom in target.split("."):
        if not hasattr(obj, atom):
            msg = f"Failed to resolve attribute '{atom}' in path '{target}'"
            raise AttributeError(msg)
        obj = getattr(obj, atom)
    return obj  # type: ignore[return-value]


def assign_attr(
    model: GraphModule,
    target: str,
    value: torch.Tensor,
) -> None:
    """Assign a tensor to a nested attribute path in the model.

    Navigates through nested attributes using dot-separated paths and assigns
    the tensor value to the final attribute as a parameter. This is the setter
    counterpart to resolve_attr.

    Args:
        model: The graph module containing the attribute
        target: Dot-separated attribute path (e.g., "layer1.weight")
        value: Tensor to assign as a parameter

    Raises:
        AttributeError: If any component in the path cannot be resolved

    """
    *path, attr_name = target.split(".")

    # Navigate to the parent module
    parent = model
    for part in path:
        if not hasattr(parent, part):
            msg = f"Failed to resolve attribute '{part}' in path '{target}'"
            raise AttributeError(msg)
        parent = getattr(parent, part)

    # Set the attribute as a parameter
    parent.register_parameter(attr_name, torch.nn.Parameter(value))


def get_source_module_name(node: Node) -> str | None:
    """Extract the deepest (most specific) module name from a node's metadata.

    Reads the ``nn_module_stack`` metadata attached to the node and
    returns the fully-qualified name of the innermost module.

    Args:
        node: An FX graph node with ``nn_module_stack`` metadata.

    Returns:
        The module FQN, or ``None`` if the node has no
        ``nn_module_stack`` metadata.
    """
    stack = getattr(node, "meta", {}).get("nn_module_stack")
    if not stack:
        return None
    module_fqn, _ = next(reversed(stack.values()))
    return module_fqn


def _is_aten_op(node: torch.fx.Node) -> bool:
    """Check if a node targets an aten operator."""
    return isinstance(node.target, torch._ops.OpOverload) and node.target.namespace == "aten"


def _has_no_disallowed_kwargs(node: torch.fx.Node) -> bool:
    """
    Check that no kwargs values are of a type disallowed by torchao annotation.
    """
    return not any(
        isinstance(v, _TORCHAO_DISALLOWED_NODE_KWARG_TYPES) for v in node.kwargs.values()
    )


def strip_non_aten_metadata_kwargs(
    graph: torch.fx.Graph,
) -> dict[str, dict[str, Any]]:
    """
    Strip metadata kwargs from non-aten custom ops.

    torchao < 0.16.0 asserts that annotated nodes have empty kwargs.
    Non-aten custom ops (e.g., CompositeOps.label_tensor_as_input) carry
    metadata kwargs (name, op_name, id, index) that trigger this assert
    but are irrelevant to observer insertion, which only operates on node.args.

    Args:
        graph: The FX graph to process.

    Returns:
        Dictionary mapping node names to their saved kwargs for later
        restoration via restore_kwargs.
    """
    saved_kwargs: dict[str, dict[str, Any]] = {}
    for node in graph.nodes:
        if (
            node.op == "call_function"
            and not _is_aten_op(node)
            and node.kwargs
            and _has_no_disallowed_kwargs(node)
        ):
            saved_kwargs[node.name] = dict(node.kwargs)
            node.kwargs = {}
    return saved_kwargs


def restore_kwargs(graph: torch.fx.Graph, saved_kwargs: dict[str, dict[str, Any]]) -> None:
    """
    Restore previously stripped kwargs to their nodes.

    Must be called after prepare_qat_pt2e completes to ensure
    the graph retains its original kwargs for downstream passes (e.g.,
    MLIR lowering, composite op recognition by the COREAI compiler).

    Args:
        graph: The FX graph to restore kwargs to.
        saved_kwargs: Dictionary mapping node names to saved kwargs,
            as returned by strip_non_aten_metadata_kwargs.
    """
    for node in graph.nodes:
        if node.name in saved_kwargs:
            node.kwargs = saved_kwargs[node.name]


def force_per_tensor_for_channel_altering_ops(model: GraphModule) -> None:
    """Force per-tensor granularity on shared fake quantize modules where the
    channel-altering op they straddle (flatten, reshape, transpose, permute,
    pooling, etc.) invalidates the shared axis.

    These ops change tensor dimensions, making per-channel/per-block axis
    semantics invalid when the input and output quantizers are the same shared
    object. When the quantizers are separate objects, their granularity is left
    unchanged since each side has independent axis semantics.

    This pass runs after prepare_qat_pt2e when fake quantize modules are fully
    instantiated.

    Args:
        model: The prepared graph module after prepare_qat_pt2e.
    """
    modules = dict(model.named_modules(remove_duplicate=False))

    for node in model.graph.nodes:
        if node.op != "call_function" or node.target not in _CHANNEL_ALTERING_ATEN_OPS:
            continue

        # Collect input fake quantize modules, keyed by id, alongside the graph
        # node that feeds each one (needed to read the pre-op input shape).
        input_fq_nodes_by_id: dict[int, Node] = {}
        for input_node in node.all_input_nodes:
            if input_node.op == "call_module":
                mod = modules.get(str(input_node.target))
                if isinstance(mod, FakeQuantizeImplBase):
                    input_fq_nodes_by_id[id(mod)] = input_node

        output_fqs: list[FakeQuantizeImplBase] = []
        for user_node in node.users:
            if user_node.op == "call_module":
                mod = modules.get(str(user_node.target))
                if isinstance(mod, FakeQuantizeImplBase):
                    output_fqs.append(mod)

        # Only force per-tensor when input and output quantizers are the same
        # shared object — meaning they share observer parameters across the
        # channel-altering op, which breaks axis semantics.
        for output_fq in output_fqs:
            input_fq_node = input_fq_nodes_by_id.get(id(output_fq))
            if input_fq_node is None:
                continue
            if not _shared_granularity_axis_is_safe(output_fq, input_fq_node, node):
                _force_fake_quant_to_per_tensor(output_fq, node)


def _shared_granularity_axis_is_safe(
    fake_quant: FakeQuantizeImplBase,
    input_fq_node: Node,
    op_node: Node,
) -> bool:
    """Return True only if it's proven safe to keep ``fake_quant``'s
    granularity shared across ``op_node``; unproven cases default to unsafe.

    Per-channel activation quantization on a shared observer is assumed
    unsafe by default — every branch below must explicitly prove safety to
    return True, rather than assume safety and look for reasons to reject
    it. This fail-safe direction matters because a new channel-altering op
    that this function doesn't yet know how to reason about should silently
    fall back to per-tensor (safe, if imprecise), not silently stay
    per-channel (fast, but potentially wrong or crashing).
    """
    granularity = fake_quant.granularity
    # No axis to violate — trivially safe.
    if isinstance(granularity, PerTensorGranularity):
        return True
    # Anything other than PerChannelGranularity (e.g. PerBlockGranularity)
    # has no proof below, so it's not safe.
    if not isinstance(granularity, PerChannelGranularity):
        return False

    output_shape = op_node.meta["val"].shape
    input_shape = input_fq_node.all_input_nodes[0].meta["val"].shape
    # Rank changed (e.g. flatten): positional axis comparison is meaningless,
    # since broadcasting aligns dims from the trailing side, not by raw
    # index, so there's no proof to offer here.
    if len(input_shape) != len(output_shape):
        return False
    axis = QuantizationGranularity._resolve_axis(granularity, len(input_shape))
    if axis is None:
        return False

    # Proof 1: the op doesn't relabel which physical dimension sits at
    # `axis` (e.g. transpose/permute swapping it with another axis of the
    # same size would pass a size check below despite being wrong).
    if not _op_preserves_axis_identity(op_node, axis):
        return False
    # Proof 2: the op doesn't change that dimension's size (e.g. pooling
    # shrinking a spatial axis).
    return input_shape[axis] == output_shape[axis]


def _op_preserves_axis_identity(op_node: Node, axis: int) -> bool:
    """Return True if ``op_node`` never moves the physical dimension at
    ``axis`` to a different position between its input and output.

    Ops that only shrink/grow dimensions in place (pooling, mean, etc.)
    always preserve axis identity — they're not in the branches below, so
    they fall through to True. Ops that reorder dimensions (transpose,
    permute, t) must be checked explicitly: a dimension that gets moved
    can coincidentally have the same size as whatever replaces it, which
    would otherwise pass a pure size check while quietly applying the wrong
    per-index scale.
    """
    if op_node.target == torch.ops.aten.transpose.int:
        _, dim0, dim1 = op_node.args
        ndim = len(op_node.meta["val"].shape)
        # Convert negative axis to positive axis by adding ndim
        if dim0 < 0:
            dim0 += ndim
        if dim1 < 0:
            dim1 += ndim
        return axis not in (dim0, dim1)
    if op_node.target == torch.ops.aten.t.default:
        return axis not in (0, 1)
    if op_node.target == torch.ops.aten.permute.default:
        _, dims = op_node.args
        return dims[axis] == axis
    return True


def _force_fake_quant_to_per_tensor(fake_quant: FakeQuantizeImplBase, op_node: Node) -> None:
    """Update a fake quantize module's granularity to per-tensor if needed."""
    granularity = fake_quant.granularity
    if isinstance(granularity, PerTensorGranularity):
        return
    if isinstance(granularity, PerChannelGranularity):
        detail = (
            ": this op either changes the size of the quantization axis "
            "between its input and output, or moves it to a different "
            "physical dimension (e.g. via transpose/permute), so a "
            "per-channel scale can't safely apply to both sides. To keep "
            "per-channel activation quantization here, choose a different "
            "axis that this op leaves untouched."
        )
    else:
        detail = ""
    logger.warning(
        "Forcing per-tensor granularity for the shared observer around '%s' (was %s)%s",
        op_node.name,
        granularity,
        detail,
    )
    fake_quant.granularity = PerTensorGranularity()


def _validate_fake_quant_node(model: GraphModule, node: Node) -> None:
    """Validate that a node is a FakeQuantize call_module node.

    Args:
        model (GraphModule): The graph module containing the node.
        node (Node): The graph node to validate.

    Raises:
        ValueError: If the node is not a ``call_module`` node or does not
            reference a FakeQuantize module.

    """
    if node.op != "call_module":
        raise ValueError(f"Expected call_module node, got {node.op}")

    modules = dict(model.named_modules(remove_duplicate=False))
    module = modules.get(str(node.target))
    if not isinstance(module, FakeQuantizeImplBase):
        raise ValueError(
            f"Expected FakeQuantize node (instance of FakeQuantizeImplBase), "
            f"got '{type(module).__name__}' for target '{node.target}'"
        )


def remove_fake_quant_module(model: GraphModule, node: Node) -> None:
    """Remove a FakeQuantize module from a GraphModule.

    Args:
        model (GraphModule): The graph module containing the FakeQuantize module.
        node (Node): A ``call_module`` node referencing the FakeQuantize module.

    """
    module_name = str(node.target)
    if hasattr(model, module_name):
        delattr(model, module_name)


def bypass_fake_quant_node(model: GraphModule, node: Node) -> None:
    """Replace a FakeQuantize node with its input and erase it from the graph.

    The node's output is redirected to its first input (the tensor being
    quantized) and the node is erased from the graph. The backing module
    is NOT removed — callers must handle module cleanup separately.

    Args:
        model (GraphModule): The graph module containing the node.
        node (Node): The FakeQuantize ``call_module`` node to bypass.

    """
    input_node = node.args[0]
    node.replace_all_uses_with(input_node)
    model.graph.erase_node(node)


def remove_fake_quant_nodes(
    model: GraphModule,
    nodes_to_remove: set[Node],
) -> int:
    """Remove specified FakeQuantize nodes from an FX graph.

    Each node is replaced by its input (bypassed), the corresponding module
    is deleted, and the node is erased from the graph. After all removals,
    dead code is eliminated and the graph is recompiled.

    Args:
        model (GraphModule): The graph module to modify.
        nodes_to_remove (set[Node]): Set of FakeQuantize ``call_module``
            graph nodes to remove.

    Returns:
        int: Number of nodes removed.

    """
    num_removed = 0
    for node in list(model.graph.nodes):
        if node in nodes_to_remove:
            _validate_fake_quant_node(model, node)
            bypass_fake_quant_node(model, node)
            remove_fake_quant_module(model, node)
            num_removed += 1

    if num_removed > 0:
        model.graph.eliminate_dead_code()
        model.recompile()

    return num_removed
