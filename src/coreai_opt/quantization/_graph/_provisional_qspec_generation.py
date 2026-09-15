# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Seed a :class:`ProvisionalQSpecMap` from the winning configs.

Each covered node seeds its OUTPUT slot, plus every INPUT slot whose producer
lies outside its pattern group. Ops with a known output distribution (sigmoid,
tanh, hardsigmoid, relu, relu6, hardtanh) also propose ``QSCHEME`` and
``FLOAT_RANGE`` at :data:`OP_INTRINSIC_PRIORITY`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch.fx as fx

from coreai_opt._utils.config_utils import ALL_TENSORS, get_last_matching_spec
from coreai_opt._utils.fx_utils import (
    get_local_state_name,
    is_coreai_compressed_state_node as _is_state_node,
)
from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.quantization.config import OpQuantizerConfig
from coreai_opt.quantization.spec import (
    QuantizationScheme,
    QuantizationSpec,
)

from ._annotation_utils import (
    _fixed_q_params_ops,
    _get_state_aliases,
    _hardtanh_ops,
    _is_fx_node_floating_point,
    _warn_non_quantizable_tensor_setting,
)
from ._qspec_constraints import _get_or_create
from ._qspec_types import (
    OP_INTRINSIC_PRIORITY,
    FieldName,
    FieldValue,
    NodeSlot,
    ProvisionalQSpecMap,
    ReconciliationError,
    SlotKind,
)

logger = logging.getLogger(__name__)


# Exactly the settable inputs of QuantizationSpec, which _qspec_resolution
# reassembles one from. Adding an attribute is a line here and a line there.
_FIELD_FROM_SPEC_ATTR: dict[FieldName, str] = {
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


_NO_MATCH = object()
"""Sentinel: no key in the spec map addressed this slot.

Distinct from a key that matched and held ``None``, which is a decline.
"""


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def build_initial_provisional_qspecs(
    winning_configs: dict[fx.Node, OpQuantizerConfig],
    node_priorities: dict[fx.Node, int],
    pattern_groups: dict[fx.Node, frozenset[fx.Node]],
    module_name_to_state_names_map: Mapping[str, Mapping[str, list[str]]],
    primary_nodes: set[fx.Node],
) -> ProvisionalQSpecMap:
    """Build the initial map from the winning configs and op intrinsics."""
    qspecs: ProvisionalQSpecMap = {}
    for node, cfg in winning_configs.items():
        priority = node_priorities[node]
        pattern = pattern_groups.get(node, frozenset({node}))
        if node in primary_nodes:
            # A pattern's input and state specs belong to the op it is anchored
            # on. Seeding them on the other covered nodes would quantize their
            # parameters too — a conv-bn would fake-quantize the bn's affine
            # scale alongside the folded conv weight.
            _populate_input_slots(
                node, cfg, priority, pattern, qspecs, module_name_to_state_names_map
            )
        _populate_output_slot(node, cfg, priority, pattern, qspecs)

    return qspecs


# ---------------------------------------------------------------------------
# Input-slot population.
# ---------------------------------------------------------------------------


def _populate_input_slots(
    node: fx.Node,
    cfg: OpQuantizerConfig,
    priority: int,
    pattern: frozenset[fx.Node],
    qspecs: ProvisionalQSpecMap,
    module_name_to_state_names_map: Mapping[str, Mapping[str, list[str]]],
) -> None:
    """Populate every non-internal INPUT slot on ``node`` from its config."""
    for arg_index, producer in enumerate(node.all_input_nodes):
        if producer in pattern:
            continue  # internal edge — skip
        if not _is_fx_node_floating_point(producer):
            # Unobservable, not declined: a fact about the graph rather than a
            # user decision, so it must not suppress a group it joins.
            _warn_if_non_quantizable_input_configured(producer, arg_index, cfg)
            continue
        slot = NodeSlot(node=node, kind=SlotKind.INPUT, arg_index=arg_index)
        if _is_state_node(producer):
            _validate_state_not_referenced_via_input_spec(producer, arg_index, cfg)
            spec = _lookup_state_spec(cfg, producer, module_name_to_state_names_map)
            target = CompressionTargetTensor.WEIGHT
        else:
            spec = _lookup_by_key(cfg.op_input_spec, arg_index)
            target = CompressionTargetTensor.ACTIVATION
        if spec is None:
            # A key named this slot and held None: an explicit opt-out.
            _mark_declined(qspecs, slot, priority)
            continue
        if spec is _NO_MATCH:
            # No key addressed this slot, so the config has no opinion on it.
            continue
        _populate_fields_from_spec(qspecs, slot, spec, priority, target)


def _warn_if_non_quantizable_input_configured(
    producer: fx.Node, arg_index: int, cfg: OpQuantizerConfig
) -> None:
    """Warn when the user named a non-floating-point input explicitly.

    A ``*`` spec sweeping up an int tensor is routine and stays silent.
    """
    if arg_index in cfg.op_input_spec:
        _warn_non_quantizable_tensor_setting(producer, "input", arg_index, cfg.op_input_spec)
    if _is_state_node(producer):
        state_name = get_local_state_name(producer)
        if state_name is not None and state_name in cfg.op_state_spec:
            _warn_non_quantizable_tensor_setting(producer, "state", state_name, cfg.op_state_spec)


def _validate_state_not_referenced_via_input_spec(
    state_producer: fx.Node, arg_index: int, cfg: OpQuantizerConfig
) -> None:
    """Raise if the user's ``op_input_spec`` targets an arg index whose
    producer is a state tensor. State tensors must be configured via
    ``op_state_spec``.
    """
    if arg_index in cfg.op_input_spec:
        raise RuntimeError(
            f"Config is attempting to set op_input_spec idx {arg_index}, "
            f"but the input is a state tensor (node: {state_producer.name}). "
            f"Use op_state_spec to configure state inputs instead.\n"
            f"op_input_spec: {cfg.op_input_spec}"
        )


def _lookup_state_spec(
    consumer_cfg: OpQuantizerConfig,
    state_node: fx.Node,
    module_name_to_state_names_map: Mapping[str, Mapping[str, list[str]]],
) -> QuantizationSpec | None:
    """Resolve a state consumer's spec from its ``op_state_spec``.

    A shared parameter has several local names — ``linear1.my_weight`` and
    ``linear2.other_weight`` may be one tensor — and each consumer's config keys
    on the one it knows, so match against the whole alias set.
    """
    if get_local_state_name(state_node) is None:
        # Already-compressed state (e.g. a palettization lut_to_dense output)
        # has no state name and isn't quantized.
        return None
    state_names = _get_state_aliases(state_node, module_name_to_state_names_map)
    spec, matched = get_last_matching_spec(state_names, consumer_cfg.op_state_spec)
    if spec is None and not matched:
        # Nothing in op_state_spec addressed this tensor — no opinion.
        return _NO_MATCH
    return spec


# ---------------------------------------------------------------------------
# Output-slot population + op-intrinsic override.
# ---------------------------------------------------------------------------


def _populate_output_slot(
    node: fx.Node,
    cfg: OpQuantizerConfig,
    priority: int,
    pattern: frozenset[fx.Node],
    qspecs: ProvisionalQSpecMap,
) -> None:
    """Populate ``node``'s OUTPUT slot from its config, then apply any
    op-intrinsic override.

    Raises on two pattern shapes the rules can't handle: an op-intrinsic node
    fully internal to a pattern, whose contribution would have nowhere to land,
    and a covered node with consumers both inside and outside its pattern.
    """
    if not node.users:
        return

    if not _is_fx_node_floating_point(node):
        # Not a tensor — e.g. the SymInt shape-assertion `mul`s torch.export
        # synthesizes. A qspec here would send an int to the qparams calculator.
        if 0 in cfg.op_output_spec:
            _warn_non_quantizable_tensor_setting(node, "output", 0, cfg.op_output_spec)
        return

    consumers_in_pattern = [consumer for consumer in node.users if consumer in pattern]
    consumers_outside_pattern = [consumer for consumer in node.users if consumer not in pattern]

    has_intrinsic = _get_op_intrinsic(node) is not None

    if consumers_in_pattern and consumers_outside_pattern:
        in_pattern_names = sorted(consumer.name for consumer in consumers_in_pattern)
        external_names = sorted(consumer.name for consumer in consumers_outside_pattern)
        raise ReconciliationError(
            f"Node {node.name!r} (target={node.target!r}) has consumers both "
            f"inside and outside its pattern group, so an observer added for the "
            f"external ones would also be visible to the internal ones.\n"
            f"  pattern group: {sorted(covered.name for covered in pattern)}\n"
            f"  in-pattern consumers: {in_pattern_names}\n"
            f"  external consumers: {external_names}"
        )

    fully_internal = not consumers_outside_pattern
    if fully_internal:
        if has_intrinsic:
            raise ReconciliationError(
                f"Op-intrinsic node {node.name!r} (target={node.target!r}) "
                f"has all consumers inside its pattern group "
                f"{sorted(covered.name for covered in pattern)}"
            )
        return

    output_slot = NodeSlot(node=node, kind=SlotKind.OUTPUT, arg_index=0)
    out_spec = _lookup_by_key(cfg.op_output_spec, 0)
    if out_spec is None:
        _mark_declined(qspecs, output_slot, priority)
        return
    if out_spec is _NO_MATCH:
        return

    _populate_fields_from_spec(
        qspecs, output_slot, out_spec, priority, CompressionTargetTensor.ACTIVATION
    )
    if has_intrinsic:
        _apply_op_intrinsic_override(node, output_slot, out_spec, qspecs)


def _apply_op_intrinsic_override(
    node: fx.Node,
    output_slot: NodeSlot,
    user_spec: QuantizationSpec,
    qspecs: ProvisionalQSpecMap,
) -> None:
    """Propose ``QSCHEME`` and ``FLOAT_RANGE`` from the op's known output
    distribution, at :data:`OP_INTRINSIC_PRIORITY` so they outrank user config.

    Nothing else: the op has no opinion on the other fields.

    The ``QSCHEME`` half is skipped for a floating-point dtype, which admits only
    the symmetric scheme — see ``QuantizationSpec.validate_qscheme_for_fp_quant``.
    An op like ``relu`` proposes an asymmetric scheme to exploit its one-sided
    range, but that is a representational choice the dtype forbids, so proposing it
    would make the group unbuildable rather than merely lower quality. The
    ``FLOAT_RANGE`` half still applies: the bound is a fact about the data and
    holds whatever the dtype.
    """
    intrinsic = _get_op_intrinsic(node)
    assert intrinsic is not None, "caller must gate on _get_op_intrinsic"
    scheme, float_range = intrinsic

    if not user_spec.dtype.is_floating_point:
        _override_field(
            qspecs,
            output_slot,
            FieldName.QSCHEME,
            scheme,
            OP_INTRINSIC_PRIORITY,
            node,
            user_spec.qscheme,
            "qscheme",
        )
    _override_field(
        qspecs,
        output_slot,
        FieldName.FLOAT_RANGE,
        list(float_range),
        OP_INTRINSIC_PRIORITY,
        node,
        user_spec.float_range,
        "float_range",
    )


def _override_field(
    qspecs: ProvisionalQSpecMap,
    slot: NodeSlot,
    field_name: FieldName,
    intrinsic_value: Any,
    priority: int,
    node: fx.Node,
    user_value: Any,
    human_field_name: str,
) -> None:
    """Write ``intrinsic_value`` at ``field_name`` on ``slot``, logging only if
    ``user_value`` asked for something different.
    """
    qspec = _get_or_create(qspecs, slot)
    qspec.fields[field_name] = FieldValue(value=intrinsic_value, priority=priority)
    if user_value is not None and user_value != intrinsic_value:
        logger.info(
            "Op-intrinsic override on %s (target=%s): user %s=%r overridden by intrinsic %s=%r.",
            node.name,
            node.target,
            human_field_name,
            user_value,
            human_field_name,
            intrinsic_value,
        )


def _get_op_intrinsic(
    node: fx.Node,
) -> tuple[QuantizationScheme, tuple[float | None, float | None]] | None:
    """Return ``(qscheme, float_range)`` for ops with a known output distribution.

    Fixed-bound ops come from :data:`_fixed_q_params_ops`; ``hardtanh`` takes its
    bounds from the node's args and is symmetric iff ``min == -max``.
    """
    if node.target in _fixed_q_params_ops:
        return _fixed_q_params_ops[node.target]
    if node.target in _hardtanh_ops and len(node.args) >= 3:
        min_val, max_val = node.args[1], node.args[2]
        scheme = (
            QuantizationScheme.SYMMETRIC if min_val == -max_val else QuantizationScheme.ASYMMETRIC
        )
        return scheme, (float(min_val), float(max_val))
    return None


# ---------------------------------------------------------------------------
# Spec-population helpers.
# ---------------------------------------------------------------------------


def _mark_declined(qspecs: ProvisionalQSpecMap, slot: NodeSlot, priority: int) -> None:
    """Record that ``slot``'s config declined to observe it, keeping the strongest
    declining priority.
    """
    qspec = _get_or_create(qspecs, slot)
    if qspec.declined_by is None or priority < qspec.declined_by:
        qspec.declined_by = priority


def _populate_fields_from_spec(
    qspecs: ProvisionalQSpecMap,
    slot: NodeSlot,
    spec: QuantizationSpec,
    priority: int,
    target: CompressionTargetTensor,
) -> None:
    """Seed ``slot``'s fields from a coreai_opt :class:`QuantizationSpec`.

    Reads the spec's settable inputs rather than a converted torchao spec, so
    each property the user configured stays independently reconcilable.
    """
    qspec = _get_or_create(qspecs, slot)
    for field_name, attr in _FIELD_FROM_SPEC_ATTR.items():
        qspec.fields[field_name] = FieldValue(value=getattr(spec, attr), priority=priority)
    qspec.fields[FieldName.QUANTIZATION_TARGET] = FieldValue(value=target, priority=priority)


def _lookup_by_key(spec_map: dict[Any, Any], key: Any) -> Any:
    """Look up ``key`` in an ``op_input/op_output/op_state`` map with ``*``
    fallback, returning :data:`_NO_MATCH` when nothing addressed ``key``."""
    if key in spec_map:
        return spec_map[key]
    if ALL_TENSORS in spec_map:
        return spec_map[ALL_TENSORS]
    return _NO_MATCH
