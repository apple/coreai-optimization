# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Driver for graph-mode quantization annotation.

:func:`annotate_via_reconciliation` seeds a provisional qspec per quantizable
tensor, drains a queue of constraints over them to a fixed point, and writes the
result onto the graph. Each of those steps lives in a ``_qspec_*`` sibling.

See ``docs/architecture_notes/graph_annotation.md`` for the reasoning.
"""

from __future__ import annotations

from collections import deque

import torch.fx as fx

from ._provisional_qspec_generation import build_initial_provisional_qspecs
from ._qspec_constraint_generation import (
    _AnnotationContext,
    _generate_constraints_for_node,
    _nodes_covered_by,
)
from ._qspec_constraints import Constraint
from ._qspec_resolution import resolve_qspecs

# Re-export so callers depend only on _qspec_reconcile.
__all__ = [
    "_AnnotationContext",
    "_nodes_covered_by",
    "annotate_via_reconciliation",
]

_MAX_ITERS_PER_SLOT = 100
"""Drain-loop iteration budget per slot. See :func:`_drain_iteration_bound`."""


def annotate_via_reconciliation(
    model: fx.GraphModule, ctx: _AnnotationContext
) -> None:
    """Annotate ``model`` in place using the constraint-queue reconciler."""
    qspecs = build_initial_provisional_qspecs(
        winning_configs=ctx.winning_configs,
        node_priorities=ctx.node_priorities,
        pattern_groups=ctx.pattern_groups,
        module_name_to_state_names_map=ctx.module_name_to_state_names_map,
        primary_nodes=ctx.primary_nodes,
    )

    queue: deque[Constraint] = deque()
    for node in model.graph.nodes:
        queue.extend(_generate_constraints_for_node(node, qspecs, ctx))

    max_iters = _drain_iteration_bound(model)
    iters = 0
    while queue:
        constraint = queue.popleft()
        changed = constraint.apply(qspecs)
        iters += 1
        if iters > max_iters:
            raise RuntimeError(
                f"Constraint reconciliation exceeded budget of {max_iters} iterations"
            )
        touched_nodes = {slot.node for slot in changed}
        for touched in touched_nodes:
            queue.extend(_generate_constraints_for_node(touched, qspecs, ctx))

    resolve_qspecs(model, qspecs)


def _drain_iteration_bound(model: fx.GraphModule) -> int:
    """Iteration budget for the drain loop: ``100 * slots``.

    The theoretical bound is O(#nodes * #slots * #fields), but measured usage is
    about one iteration per slot, so this takes the empirical shape with a
    generous constant. Exceeding it means something is wrong.
    """

    slots = sum(1 + len(node.all_input_nodes) for node in model.graph.nodes)
    return _MAX_ITERS_PER_SLOT * max(slots, 1)
