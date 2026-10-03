# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Write the settled :class:`ProvisionalQSpecMap` onto the fx graph as
:class:`QuantizationAnnotation` entries.

Sharing is already expressed as slots pointing at one
:class:`ProvisionalQSpec`, so the work is projecting that onto torchao's per-node
model: group by spec identity, pick a topologically-first anchor per group, give
the anchor a concrete spec and the rest a :class:`SharedQuantizationSpec`
pointing at it, then bucket per node and write.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch.fx as fx
from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec as TorchAOQuantizationSpec,
    SharedQuantizationSpec as _SharedQuantizationSpec,
)
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY

from coreai_opt.quantization.spec import QuantizationSpec

from ._annotation_config import AnnotationConfig
from ._qspec_types import (
    FieldName,
    NodeSlot,
    ProvisionalQSpec,
    ProvisionalQSpecMap,
    ReconciliationError,
    SlotKind,
)

# Reused verbatim, so a reconciled spec converts by the same path as a
# configured one.
_convert_to_pt2e_spec = AnnotationConfig._convert_to_pt2e_spec

# A concrete spec on a group's anchor slot; a SharedQuantizationSpec pointing at
# the anchor on every other slot.
_SlotSpec = TorchAOQuantizationSpec | _SharedQuantizationSpec

# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def resolve_qspecs(model: fx.GraphModule, qspecs: ProvisionalQSpecMap) -> None:
    """Project reconciled per-slot state onto ``model``'s annotations, in place."""
    groups = _group_slots_by_qspec_identity(qspecs)
    topo_index = _topo_index(model)

    slot_assignments: dict[NodeSlot, _SlotSpec] = {}
    for group in groups:
        _assign_group_specs(group, topo_index, slot_assignments)

    per_node_inputs, per_node_outputs = _bucket_per_node(slot_assignments)
    _write_annotations(per_node_inputs, per_node_outputs)


# ---------------------------------------------------------------------------
# Grouping.
# ---------------------------------------------------------------------------


@dataclass
class _QSpecGroup:
    """Slots that all point at one :class:`ProvisionalQSpec`."""

    qspec: ProvisionalQSpec
    slots: list[NodeSlot]


def _group_slots_by_qspec_identity(qspecs: ProvisionalQSpecMap) -> list[_QSpecGroup]:
    """Bucket slots by ``id(ProvisionalQSpec)`` — same object = same group."""
    groups_by_id: dict[int, _QSpecGroup] = {}
    for slot, qspec in qspecs.items():
        group = groups_by_id.get(id(qspec))
        if group is None:
            group = _QSpecGroup(qspec=qspec, slots=[])
            groups_by_id[id(qspec)] = group
        group.slots.append(slot)
    return list(groups_by_id.values())


# ---------------------------------------------------------------------------
# Anchor selection.
# ---------------------------------------------------------------------------


@dataclass(order=True, frozen=True)
class SlotOrderKey:
    """Sort key for picking a shared-observer group's anchor.

    Attributes:
        topo_index (int): Position in ``model.graph.nodes``. Torchao walks nodes
            in this order, so the anchor's observer must be registered before
            anything can reference it.
        kind_order (int): INPUT (0) before OUTPUT (1), matching the order
            ``_get_edge_or_node_to_qspec`` iterates a node's slots.
        arg_index (int): Tiebreaker among INPUT slots.
    """

    topo_index: int
    kind_order: int
    arg_index: int


def _topo_index(model: fx.GraphModule) -> dict[fx.Node, int]:
    return {node: i for i, node in enumerate(model.graph.nodes)}


def _slot_order_key(slot: NodeSlot, topo_index: dict[fx.Node, int]) -> SlotOrderKey:
    return SlotOrderKey(
        topo_index=topo_index.get(slot.node, len(topo_index)),
        kind_order=0 if slot.kind is SlotKind.INPUT else 1,
        arg_index=slot.arg_index,
    )


def _pick_anchor(group: _QSpecGroup, topo_index: dict[fx.Node, int]) -> NodeSlot:
    """Return the topologically-first slot in ``group``, per :class:`SlotOrderKey`."""
    return min(group.slots, key=lambda slot: _slot_order_key(slot, topo_index))


# ---------------------------------------------------------------------------
# Spec assignment.
# ---------------------------------------------------------------------------


def _assign_group_specs(
    group: _QSpecGroup,
    topo_index: dict[fx.Node, int],
    slot_assignments: dict[NodeSlot, _SlotSpec],
) -> None:
    """Fill ``slot_assignments`` with the spec each slot in ``group`` carries: the
    concrete spec on the anchor, a :class:`SharedQuantizationSpec` on the rest.
    """
    concrete = _build_concrete_spec(group.qspec)
    if concrete is None:
        # Nothing to annotate.
        return
    if len(group.slots) == 1:
        slot_assignments[group.slots[0]] = concrete
        return

    anchor = _pick_anchor(group, topo_index)
    shared_ref = _shared_spec_pointing_at(anchor)
    for slot in group.slots:
        slot_assignments[slot] = concrete if slot == anchor else shared_ref


def _shared_spec_pointing_at(anchor: NodeSlot) -> _SharedQuantizationSpec:
    """Build a :class:`SharedQuantizationSpec` in the form torchao registered
    ``anchor``'s observer under: by node for an output, by edge for an input.
    """
    if anchor.kind is SlotKind.OUTPUT:
        return _SharedQuantizationSpec(anchor.node)
    producer = anchor.node.all_input_nodes[anchor.arg_index]
    return _SharedQuantizationSpec((producer, anchor.node))


# The other half of _provisional_qspec_generation._FIELD_FROM_SPEC_ATTR: the two
# make up the round trip between a spec and its decomposed fields.
_SPEC_KWARG_FROM_FIELD: dict[FieldName, str] = {
    FieldName.DTYPE: "dtype",
    FieldName.QSCHEME: "qscheme",
    FieldName.QFORMULATION: "qformulation",
    FieldName.GRANULARITY: "granularity",
    FieldName.FLOAT_RANGE: "float_range",
    FieldName.FAKE_QUANTIZE_CLS: "fake_quantize_cls",
    FieldName.QPARAM_CALCULATOR_CLS: "qparam_calculator_cls",
    FieldName.RANGE_CALCULATOR_CLS: "range_calculator_cls",
    FieldName.SCALE_DTYPE: "scale_dtype",
    FieldName.SPARSITY: "_sparsity",
}


def _build_concrete_spec(qspec: ProvisionalQSpec) -> TorchAOQuantizationSpec | None:
    """Reassemble a :class:`QuantizationSpec` from reconciled fields and convert it
    for torchao. ``None`` when the group is declined or holds no fields.

    Building the observer here, from settled values, is what makes the field map
    the source of truth — the qscheme that takes effect is the one inside the
    observer's qparams calculator.
    """
    if qspec.declined:
        # No annotation, so torchao leaves the op in float. Upstream producers
        # keep their observers, so their branches dequantize into it.
        return None
    if not qspec.fields:
        return None
    missing = [
        field_name.name for field_name in _SPEC_KWARG_FROM_FIELD if field_name not in qspec.fields
    ]
    if missing or FieldName.QUANTIZATION_TARGET not in qspec.fields:
        # Every observed slot is seeded from a whole QuantizationSpec, so a
        # partial group is a generation bug. Don't paper over it with defaults.
        raise ReconciliationError(
            f"Cannot rebuild a QuantizationSpec: reconciled group is missing "
            f"{missing or ['QUANTIZATION_TARGET']}. Present: "
            f"{sorted(f.name for f in qspec.fields)}."
        )

    spec = QuantizationSpec(
        **{
            kwarg: qspec.fields[field_name].value
            for field_name, kwarg in _SPEC_KWARG_FROM_FIELD.items()
        }
    )
    return _convert_to_pt2e_spec(spec, qspec.fields[FieldName.QUANTIZATION_TARGET].value)


# ---------------------------------------------------------------------------
# Per-node bucketing + write.
# ---------------------------------------------------------------------------


def _bucket_per_node(
    slot_assignments: dict[NodeSlot, _SlotSpec],
) -> tuple[dict[fx.Node, dict[fx.Node, _SlotSpec]], dict[fx.Node, _SlotSpec]]:
    """Split per-slot assignments into per-node ``input_qspec_map`` / ``output_qspec``."""
    per_node_inputs: dict[fx.Node, dict[fx.Node, _SlotSpec]] = defaultdict(dict)
    per_node_outputs: dict[fx.Node, _SlotSpec] = {}
    for slot, spec in slot_assignments.items():
        if slot.kind is SlotKind.OUTPUT:
            per_node_outputs[slot.node] = spec
        else:
            producer = slot.node.all_input_nodes[slot.arg_index]
            per_node_inputs[slot.node][producer] = spec
    return per_node_inputs, per_node_outputs


def _write_annotations(
    per_node_inputs: dict[fx.Node, dict[fx.Node, _SlotSpec]],
    per_node_outputs: dict[fx.Node, _SlotSpec],
) -> None:
    """Mutate each touched node's meta with a :class:`QuantizationAnnotation`.

    Input maps are backfilled with ``None`` for unspecified positions, because
    torchao's later passes index them positionally and IndexError on gaps. An
    interop concession, not a reconciliation decision.
    """
    touched: set[fx.Node] = set(per_node_inputs) | set(per_node_outputs)
    for node in touched:
        annotation = node.meta.get(Q_ANNOTATION_KEY, QuantizationAnnotation())
        input_map = per_node_inputs.get(node)
        if input_map:
            annotation.input_qspec_map = _backfill_input_qspec_map(node, input_map)
        if node in per_node_outputs:
            annotation.output_qspec = per_node_outputs[node]
        annotation._annotated = True
        node.meta[Q_ANNOTATION_KEY] = annotation


def _backfill_input_qspec_map(
    node: fx.Node, input_map: dict[fx.Node, _SlotSpec]
) -> dict[fx.Node, _SlotSpec | None]:
    """An ``input_qspec_map`` with an entry per positional input, ``None`` where unset."""
    return {producer: input_map.get(producer, None) for producer in node.all_input_nodes}
