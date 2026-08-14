# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Constraint generation for the qspec reconciliation drain loop.

:func:`_generate_constraints_for_node` runs once per fx node to seed the queue,
then again whenever a node's slots change, so a newly-populated slot can trigger
constraints that did not apply while it was empty. The sources are adjacent
edges, shared-observer patterns, shared-state consumers, and shape-only ops.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.fx as fx
from torch.fx.passes.utils.matcher_utils import InternalMatch
from torch.fx.passes.utils.source_matcher_utils import SourcePartition

from coreai_opt._utils.fx_utils import is_coreai_compressed_state_node as _is_state_node
from coreai_opt.quantization.config import OpQuantizerConfig

from ._annotation_pattern_registry import (
    _OPERAND_NODE_OPS,
    SharedObserverModulePattern,
)
from ._annotation_utils import _get_call_function_node_from_partition
from ._qspec_constraints import Constraint, InheritFields, ShareObserverInstance
from ._qspec_types import FieldName, NodeSlot, ProvisionalQSpecMap, SlotKind

# Shape-only ops: their output is numerically their input. Split on whether
# they preserve tensor rank, which decides how much can be shared across them.
#
# Rank-preserving — safe to share one observer instance.
_RANK_PRESERVING_PASSTHROUGH_OPS: frozenset = frozenset(
    {
        torch.ops.aten.clone,
        torch.ops.aten.dropout,
        torch.ops.aten.feature_dropout,
        torch.ops.aten.permute,
        torch.ops.aten.slice,
        torch.ops.aten.t,
        torch.ops.aten.transpose,
    }
)

# Rank-changing — cannot share an instance, since a QParamsCalculator sizes its
# buffers to the first tensor it sees (see QParamsCalculatorBase._resolve_axis).
# ``expand`` belongs here: Tensor.expand can prepend leading dimensions.
_RANK_CHANGING_PASSTHROUGH_OPS: frozenset = frozenset(
    {
        torch.ops.aten.expand,
        torch.ops.aten.reshape,
        torch.ops.aten.select,
        torch.ops.aten.squeeze,
        torch.ops.aten.unsqueeze,
        torch.ops.aten.view,
    }
)

# Every shape-only op, regardless of what can be shared across it.
# tests/quantization/test_annotation_utils.py checks each member lowers to the
# ATen packet named here.
_PASSTHROUGH_OP_OVERLOADS: frozenset = (
    _RANK_PRESERVING_PASSTHROUGH_OPS | _RANK_CHANGING_PASSTHROUGH_OPS
)


def _passthrough_kind(node: fx.Node) -> frozenset | None:
    """Return which passthrough set ``node`` belongs to, or ``None``."""
    if node.op != "call_function":
        return None
    packet = getattr(node.target, "overloadpacket", None)
    if packet in _RANK_PRESERVING_PASSTHROUGH_OPS:
        return _RANK_PRESERVING_PASSTHROUGH_OPS
    if packet in _RANK_CHANGING_PASSTHROUGH_OPS:
        return _RANK_CHANGING_PASSTHROUGH_OPS
    return None


def _is_passthrough(node: fx.Node) -> bool:
    """Whether ``node`` is any shape-only op."""
    return _passthrough_kind(node) is not None


# ---------------------------------------------------------------------------
# Context bundle.
# ---------------------------------------------------------------------------


@dataclass
class _AnnotationContext:
    """Inputs to the reconciliation pipeline.

    Attributes:
        winning_configs (dict[fx.Node, OpQuantizerConfig]): Highest-priority
            config per covered node.
        node_priorities (dict[fx.Node, int]): Position in the annotation sort
            order; lower means higher priority.
        pattern_groups (dict[fx.Node, frozenset[fx.Node]]): The nodes in each
            covered node's winning pattern match.
        primary_nodes (set[fx.Node]): The node each pattern is anchored on — the
            conv of a conv-bn-relu, not the bn or the relu. Only these consume
            the config's input and state specs; the rest of a pattern is covered
            for grouping and for carrying the output spec.
        shared_observer_nodes (dict[fx.Node, type[SharedObserverModulePattern]]):
            The pattern class owning each shared-observer node's semantics.
        module_name_to_state_names_map (Mapping[str, Mapping[str, list[str]]]):
            Per module, each reachable state FQN mapped to its local names. One
            shared parameter has several, and each consumer's config may key on
            a different one.
    """

    winning_configs: dict[fx.Node, OpQuantizerConfig]
    node_priorities: dict[fx.Node, int]
    pattern_groups: dict[fx.Node, frozenset[fx.Node]]
    primary_nodes: set[fx.Node]
    shared_observer_nodes: dict[fx.Node, type[SharedObserverModulePattern]]
    module_name_to_state_names_map: Mapping[str, Mapping[str, list[str]]]


# ---------------------------------------------------------------------------
# Per-node constraint dispatch.
# ---------------------------------------------------------------------------


def _generate_constraints_for_node(
    node: fx.Node, qspecs: ProvisionalQSpecMap, ctx: _AnnotationContext
) -> list[Constraint]:
    """Return every constraint whose scope includes ``node``.

    Generators read the current ``qspecs``, so an already-decided neighboring
    slot can make a rule emit a constraint it otherwise wouldn't.
    """
    constraints: list[Constraint] = []

    constraints.extend(_adjacent_edge_constraints(node, qspecs, ctx))

    pattern_class = ctx.shared_observer_nodes.get(node)
    if pattern_class is not None:
        constraints.extend(pattern_class.generate_qspec_sharing_constraints(node, qspecs))

    if _is_state_node(node) and len(node.users) >= 2:
        constraints.extend(_shared_state_constraints(node, qspecs, ctx))

    passthrough_kind = _passthrough_kind(node)
    if passthrough_kind is _RANK_PRESERVING_PASSTHROUGH_OPS:
        constraints.extend(_passthrough_op_constraints(node, qspecs, ctx))
    elif passthrough_kind is _RANK_CHANGING_PASSTHROUGH_OPS:
        constraints.extend(_rank_changing_passthrough_constraints(node, qspecs, ctx))

    return constraints


# ---------------------------------------------------------------------------
# Adjacent-edge sharing.
# ---------------------------------------------------------------------------


def _adjacent_edge_constraints(
    node: fx.Node, qspecs: ProvisionalQSpecMap, ctx: _AnnotationContext
) -> list[Constraint]:
    """Tie a tensor's producer OUTPUT slot to its consumer INPUT slot so one
    observer serves both, per :func:`_edge_should_share`.

    A decline suppresses the constraint rather than the whole group: the consumer
    can observe its own input while the producer observes nothing, and either way
    the edge carries one observer.

    Constraints are pairwise, but :meth:`ShareObserverInstance.apply` widens each
    group, so a node's whole fan-out ends up on one observer. Edges internal to a
    pattern group are skipped.
    """
    if node not in ctx.winning_configs:
        # A covered neighbor that cares about this edge emits it from its side.
        return []

    constraints: list[Constraint] = []
    pattern = ctx.pattern_groups.get(node, frozenset({node}))

    # Incoming edges.
    for arg_index, producer in enumerate(node.all_input_nodes):
        if producer in pattern:
            continue
        if producer not in ctx.winning_configs:
            continue
        producer_output = NodeSlot(node=producer, kind=SlotKind.OUTPUT, arg_index=0)
        consumer_input = NodeSlot(node=node, kind=SlotKind.INPUT, arg_index=arg_index)
        if _edge_should_share(producer_output, consumer_input, qspecs):
            constraints.append(ShareObserverInstance(frozenset({producer_output, consumer_input})))

    # Outgoing edges.
    for consumer in node.users:
        if consumer in pattern:
            continue
        if consumer not in ctx.winning_configs:
            continue
        consumer_input = _find_input_slot_for_producer(consumer, node)
        if consumer_input is None:
            continue
        producer_output = NodeSlot(node=node, kind=SlotKind.OUTPUT, arg_index=0)
        if _edge_should_share(producer_output, consumer_input, qspecs):
            constraints.append(ShareObserverInstance(frozenset({producer_output, consumer_input})))

    return constraints


def _edge_should_share(first: NodeSlot, second: NodeSlot, qspecs: ProvisionalQSpecMap) -> bool:
    """Whether an edge's slots should share an observer: neither declined, and at
    least one holding fields.

    "At least one" is what populates an otherwise-empty INPUT slot, which a
    downstream shared-observer pattern then needs in order to fire.
    """
    first_qspec, second_qspec = qspecs.get(first), qspecs.get(second)
    if any(qspec is not None and qspec.declined for qspec in (first_qspec, second_qspec)):
        return False
    return any(qspec is not None and bool(qspec.fields) for qspec in (first_qspec, second_qspec))


def _find_input_slot_for_producer(consumer: fx.Node, producer: fx.Node) -> NodeSlot | None:
    """Return the consumer's INPUT slot reading from ``producer``, if any."""
    for arg_index, actual_producer in enumerate(consumer.all_input_nodes):
        if actual_producer is producer:
            return NodeSlot(node=consumer, kind=SlotKind.INPUT, arg_index=arg_index)
    return None


# ---------------------------------------------------------------------------
# Shared-state (shared-weight) sharing.
# ---------------------------------------------------------------------------


def _shared_state_constraints(
    state_node: fx.Node, qspecs: ProvisionalQSpecMap, ctx: _AnnotationContext
) -> list[Constraint]:
    """Tie the INPUT slot of every covered consumer of ``state_node`` onto one
    observer.

    One copy of the tensor, quantized once, so every consumer must agree on the
    spec — and one consumer declining it leaves it unquantized for all of them.
    """
    consumer_slots: list[NodeSlot] = []
    for consumer in state_node.users:
        if consumer not in ctx.winning_configs:
            continue
        consumer_input = _find_input_slot_for_producer(consumer, state_node)
        if consumer_input is None:
            continue
        consumer_slots.append(consumer_input)

    # Declined consumers participate; slots nothing spoke for stay out.
    known_slots = [slot for slot in consumer_slots if slot in qspecs]
    if len(known_slots) < 2:
        return []
    # Each consumer's config independently says whether it quantizes the tensor,
    # so ordinary config precedence decides — including when it declines.
    return [ShareObserverInstance(frozenset(known_slots), priority_decides_decline=True)]


# ---------------------------------------------------------------------------
# Passthrough-op propagation.
# ---------------------------------------------------------------------------


def _passthrough_op_constraints(
    node: fx.Node, qspecs: ProvisionalQSpecMap, ctx: _AnnotationContext
) -> list[Constraint]:
    """Tie the slots around a rank-preserving shape-only op onto one observer.

    Both sides see identical values at identical rank, so a second observer would
    compute the same thing.
    """
    if not node.all_input_nodes:
        return []

    slots: set[NodeSlot] = {
        NodeSlot(node=node, kind=SlotKind.INPUT, arg_index=0),
        NodeSlot(node=node, kind=SlotKind.OUTPUT, arg_index=0),
    }

    producer = node.all_input_nodes[0]
    if producer in ctx.winning_configs or _is_passthrough(producer):
        slots.add(NodeSlot(node=producer, kind=SlotKind.OUTPUT, arg_index=0))

    for consumer in node.users:
        if consumer in ctx.winning_configs:
            consumer_input = _find_input_slot_for_producer(consumer, node)
            if consumer_input is not None:
                slots.add(consumer_input)
        elif _is_passthrough(consumer):
            slots.add(NodeSlot(node=consumer, kind=SlotKind.INPUT, arg_index=0))

    if len(slots) < 2:
        return []
    return [ShareObserverInstance(_slots=frozenset(slots))]


# Facts about the data rather than choices about it, so they hold on the far
# side of a shape-only op. DTYPE is excluded: it is a config decision.
_DATA_FACT_FIELDS: frozenset = frozenset({FieldName.QSCHEME, FieldName.FLOAT_RANGE})


def _rank_changing_passthrough_constraints(
    node: fx.Node, qspecs: ProvisionalQSpecMap, ctx: _AnnotationContext
) -> list[Constraint]:
    """Copy data facts from one end of a rank-changing shape-only chain to the other.

    These ops alter no values, so a fact proven upstream holds downstream; but
    they change rank, so the ends cannot share an observer. Resolved chain-wise
    because the intermediate ops hold no fields to carry a fact hop by hop.
    """
    source = _chain_source_slot(node, qspecs)
    if source is None:
        return []
    targets = _chain_target_slots(node, qspecs)
    if not targets:
        return []
    return [InheritFields(source=source, targets=frozenset(targets), fields=_DATA_FACT_FIELDS)]


def _chain_source_slot(node: fx.Node, qspecs: ProvisionalQSpecMap) -> NodeSlot | None:
    """Walk upstream past rank-changing passthroughs to the slot holding the facts,
    or ``None`` if nothing on the chain has any yet.
    """
    current = node
    seen: set[fx.Node] = set()
    while current.all_input_nodes and current not in seen:
        seen.add(current)
        producer = current.all_input_nodes[0]
        candidate = NodeSlot(node=producer, kind=SlotKind.OUTPUT, arg_index=0)
        if candidate in qspecs:
            return candidate
        if _passthrough_kind(producer) is not _RANK_CHANGING_PASSTHROUGH_OPS:
            # Not a passthrough and not seeded — the chain has no source.
            return None
        current = producer
    return None


def _chain_target_slots(node: fx.Node, qspecs: ProvisionalQSpecMap) -> set[NodeSlot]:
    """Walk downstream past rank-changing passthroughs to the slots that hold observers."""
    targets: set[NodeSlot] = set()
    frontier = [node]
    seen: set[fx.Node] = set()
    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        for consumer in current.users:
            consumer_input = _find_input_slot_for_producer(consumer, current)
            if consumer_input is not None and consumer_input in qspecs:
                targets.add(consumer_input)
            elif _passthrough_kind(consumer) is _RANK_CHANGING_PASSTHROUGH_OPS:
                frontier.append(consumer)
    return targets


# ---------------------------------------------------------------------------
# Pattern coverage helper.
# ---------------------------------------------------------------------------


def _nodes_covered_by(match_info: Any) -> list[fx.Node]:
    """Return every fx node an annotator match annotates: the ops it spans.

    Reads ``nodes_map`` rather than ``name_node_map``, which is a projection of it
    through role names: an in-place activation aliases several roles onto one node,
    leaving the op it overwrote with no role and so unreachable.

    ``nodes_map`` keys are pattern nodes, so selecting on them excludes operands
    however they resolve in the model — a compressed weight resolves to a
    ``call_function`` — and skips the ``None`` standing for an absent one.
    """
    match = match_info.annotator_match

    if isinstance(match, InternalMatch):
        return [
            model_node
            for pattern_node, model_node in match.nodes_map.items()
            if pattern_node.op not in _OPERAND_NODE_OPS
        ]

    if isinstance(match, tuple) and all(
        isinstance(partition, SourcePartition) for partition in match
    ):
        return [_get_call_function_node_from_partition(partition) for partition in match]

    raise TypeError(
        f"Unknown annotator match type: {type(match).__name__}. Update "
        f"_nodes_covered_by when adding a new pattern family."
    )
