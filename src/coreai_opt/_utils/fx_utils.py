# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Utilities for working with torch.fx graphs and nodes."""

import logging
import re

import torch
from torch.fx import Node

logger = logging.getLogger(__name__)


def get_node_type(node: Node, warn_on_failure: bool = True) -> str | None:
    """Extract the op type string from an FX node's ``torch_fn`` metadata.

    The ``torch_fn`` metadata entry is a two-element tuple where the second
    element encodes the ATen operator in *namespace.op_name* form.  This
    function returns the *op_name* part (after the dot).

    Args:
        node (Node): An FX graph node.
        warn_on_failure (bool): If True, log a warning if node type could not be found.

    Returns:
        str | None: The op type string, or ``None`` if unavailable.
    """
    try:
        _, torch_fn = node.meta.get("torch_fn")
        return torch_fn.split(".")[1]
    except (AttributeError, IndexError, TypeError, ValueError):
        if warn_on_failure:
            warning_msg = f"Unable to determine node type for node {node.name}. Skipping the node."
            logger.warning(warning_msg)
    return None


def normalize_module_fqn(path: str) -> str:
    """Normalize module path from nn_module_stack to match named_modules format.

    Handles various torch.export contexts including decorators (@torch.no_grad,
    @wraps), array indexing, and nested _modules['X'] patterns.

    Examples:
        "model.layers.0.norm" -> "model.layers.0.norm"
        "L['self'].model" -> "model"
        "L['fn'].model" -> "model"
        "L['args'][0].model.layers[0]" -> "model.layers.0"
        "_modules['model']._modules['layers']._modules['0']" -> "model.layers.0"
    """
    # Remove torch.export prefixes (self, fn, args[N])
    path = re.sub(r"^(?:L\['(?:self|fn)'\]\.|L\['args'\]\[\d+\]\.)", "", path)

    # Convert _modules['X'] and array indexing [N] to dot notation in one pass
    path = re.sub(
        r"_modules\['([^']+)'\]|\[(\d+)\]", lambda m: "." + (m.group(1) or m.group(2)), path
    )

    # Collapse multiple dots and strip leading/trailing dots
    return re.sub(r"\.+", ".", path).strip(".")


def get_local_state_name(state: torch.fx.Node | str) -> str | None:
    """Return the local state name (the last dotted component of the state identifier).

    For ``get_attr`` nodes the identifier is ``node.target``; for string inputs
    (e.g., an ``OpInfo.op_name``) the string itself is used directly.
    Returns ``None`` for ``call_function`` nodes (e.g., ``lut_to_dense``),
    which represent already-compressed state and have no traditional parameter name.

    Args:
        state (torch.fx.Node | str): An FX ``get_attr`` node, a ``call_function``
            state node, or a state name string.

    Returns:
        str | None: The last dotted component, or ``None`` for call_function nodes.

    Example:
        >>> get_local_state_name("model.mod1.mod2.weight")
        'weight'
        >>> get_local_state_name("model_weight")
        'model_weight'
    """
    if isinstance(state, str):
        return state.rsplit(".", 1)[-1]
    if state.op != "get_attr":
        # call_function nodes identified as state (e.g., lut_to_dense from palettization)
        # don't have a traditional state name - they are already compressed
        return None
    return state.target.rsplit(".", 1)[-1]


def is_coreai_compressed_state_node(node: Node) -> bool:
    """Return True if node represents model state (not a computation).

    A node is state if it is:

    1. A ``get_attr`` node (model parameter or buffer access).
    2. A ``call_function`` node targeting a coreai state-producing op:

       - ``coreai.lut_to_dense``: palettized weight decompression.
       - ``coreai.constexpr_blockwise_shift_scale``: block shift/scale on weights.
       - ``coreai.sparse_to_dense``: sparse weight decompression (pruning).
       - ``coreai.sparse_with_bitmask_to_dense``: bitmask-based sparse decompression (pruning).

    Note:
        Update this function if new coreai ops are introduced that produce state tensors
        from compressed representations, or if existing op names change.

    Args:
        node (Node): An FX graph node.

    Returns:
        bool: True if the node is a state node, False otherwise.
    """
    if node.op == "get_attr":
        return True
    if node.op != "call_function":
        return False
    target = node.target
    if not isinstance(target, torch._ops.OpOverload) or target.namespace != "coreai":
        return False
    return target._opname in (
        "lut_to_dense",
        "constexpr_blockwise_shift_scale",
        "sparse_to_dense",
        "sparse_with_bitmask_to_dense",
    )


def get_module_boundary_nodes(
    nodes_in_module: list[Node],
) -> tuple[list[tuple[Node, Node]], list[Node]]:
    """Return the input and output boundary nodes for a set of nodes in a module.

    Args:
        nodes_in_module (list[Node]): All FX nodes belonging to the module's subtree,
            in topological order.

    Returns:
        tuple: A pair ``(input_consumer_tuples, output_nodes)`` where:

        - ``input_consumer_tuples``: ``(external_node, consumer_node)`` pairs in which
          ``external_node`` is outside the module and ``consumer_node`` (inside the module)
          consumes it. State nodes are excluded from ``external_node``.
        - ``output_nodes``: Nodes inside the module that have at least one user outside it,
          in topological order.
    """
    input_consumer_tuples: list[tuple[Node, Node]] = []
    output_nodes: list[Node] = []
    nodes_in_module_set = set(nodes_in_module)

    for node in nodes_in_module:
        for input_node in node.all_input_nodes:
            if (
                not is_coreai_compressed_state_node(input_node)
                and input_node not in nodes_in_module_set
            ):
                input_consumer_tuples.append((input_node, node))
        for user in node.users:
            if user not in nodes_in_module_set:
                output_nodes.append(node)
                break

    return input_consumer_tuples, output_nodes
