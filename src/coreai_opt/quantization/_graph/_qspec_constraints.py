# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Constraint types, and the per-field policies used to reconcile them."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ._qspec_types import (
    FieldName,
    FieldValue,
    NodeSlot,
    ProvisionalQSpec,
    ProvisionalQSpecMap,
    ReconciliationError,
)

logger = logging.getLogger(__name__)


_ALL_FIELDS: frozenset[FieldName] = frozenset(FieldName)


# ---------------------------------------------------------------------------
# Constraint ABC + implementations.
# ---------------------------------------------------------------------------


class Constraint(ABC):
    """A relation over slots.

    :meth:`apply` enforces the relation and returns the slots it changed. An
    empty set means it was already satisfied, which is how the drain loop knows
    when to stop.
    """

    @property
    @abstractmethod
    def slots(self) -> frozenset[NodeSlot]: ...

    @abstractmethod
    def apply(self, qspecs: ProvisionalQSpecMap) -> set[NodeSlot]: ...


@dataclass(frozen=True)
class ShareFields(Constraint):
    """Every slot in ``slots`` must agree on each of ``fields``.

    Reconciled per field (:func:`_reconcile_field`), then the winner is written
    to every slot. The result carries the highest priority among the proposals,
    which only ever rises — that is why the drain terminates.
    """

    _slots: frozenset[NodeSlot]
    fields: frozenset[FieldName]

    @property
    def slots(self) -> frozenset[NodeSlot]:
        return self._slots

    def apply(self, qspecs: ProvisionalQSpecMap) -> set[NodeSlot]:
        changed: set[NodeSlot] = set()
        for field_name in self.fields:
            reconciled = _reconcile_field(field_name, self._slots, qspecs)
            if reconciled is None:
                continue
            for slot in self._slots:
                qspec = _get_or_create(qspecs, slot)
                current = qspec.fields.get(field_name)
                if _field_value_stronger(reconciled, current):
                    qspec.fields[field_name] = reconciled
                    changed.add(slot)
        return changed


@dataclass(frozen=True)
class InheritFields(Constraint):
    """``targets`` take ``source``'s value for each of ``fields``, verbatim.

    A one-way copy, for slots on the same tensor that cannot share an observer
    because their ranks differ. :class:`ShareFields` reconciles bidirectionally,
    so a downstream default would widen the upstream value instead of adopting
    it. ``fields`` should name only facts about the data, never config choices.

    Copies at unchanged priority, so it terminates on the fx graph being a DAG
    rather than on priority rising.
    """

    source: NodeSlot
    targets: frozenset[NodeSlot]
    fields: frozenset[FieldName]

    @property
    def slots(self) -> frozenset[NodeSlot]:
        return self.targets | {self.source}

    def apply(self, qspecs: ProvisionalQSpecMap) -> set[NodeSlot]:
        source_qspec = qspecs.get(self.source)
        if source_qspec is None or source_qspec.declined:
            # Nothing established to carry.
            return set()

        changed: set[NodeSlot] = set()
        for field_name in self.fields:
            source_value = source_qspec.fields.get(field_name)
            if source_value is None:
                continue
            for target in self.targets:
                target_qspec = qspecs.get(target)
                if target_qspec is None or target_qspec.declined:
                    # Only slots that already hold fields, and are not opted
                    # out, can inherit.
                    continue
                if target_qspec.fields.get(field_name) == source_value:
                    continue
                target_qspec.fields[field_name] = source_value
                changed.add(target)
        return changed


@dataclass(frozen=True)
class ShareObserverInstance(Constraint):
    """Every slot in ``slots`` shares one :class:`ProvisionalQSpec`, and so one
    observer at runtime.

    Applying widens the group to anything already sharing a spec with a member,
    reconciles every field across it, and points all of them at one spec.

    ``priority_decides_decline`` selects how a declined member is resolved:

    * ``False`` (default) — a decline only takes effect if every member declines.
      For slots tied by op semantics, which are one tensor of one op: disabling
      an op's output while its input is quantized (or the reverse) is not a
      coherent instruction, so quantization wins either way.
    * ``True`` — the highest-priority proposal decides, decline or not. For one
      tensor read by several ops, where each consumer's config independently
      says whether it quantizes, so ordinary config precedence applies.
    """

    _slots: frozenset[NodeSlot]
    priority_decides_decline: bool = False

    @property
    def slots(self) -> frozenset[NodeSlot]:
        return self._slots

    def apply(self, qspecs: ProvisionalQSpecMap) -> set[NodeSlot]:
        # Widen to anything already sharing a spec with a member.
        target_ids = {id(qspecs[slot]) for slot in self._slots if slot in qspecs}
        widened: set[NodeSlot] = set(self._slots)
        for slot, qspec in qspecs.items():
            if id(qspec) in target_ids:
                widened.add(slot)

        group_declined_by = self._resolve_decline(widened, qspecs)

        reconciled_fields: dict[FieldName, FieldValue] = {}
        for field_name in _ALL_FIELDS:
            reconciled = _reconcile_field(field_name, frozenset(widened), qspecs)
            if reconciled is not None:
                reconciled_fields[field_name] = reconciled

        # Reuse a member's spec when it already matches, so a settled group
        # reports no change.
        shared = next(
            (
                qspecs[slot]
                for slot in widened
                if slot in qspecs
                and qspecs[slot].fields == reconciled_fields
                and qspecs[slot].declined_by == group_declined_by
            ),
            None,
        )
        if shared is None:
            shared = ProvisionalQSpec(
                fields=dict(reconciled_fields), declined_by=group_declined_by
            )

        changed = {slot for slot in widened if qspecs.get(slot) is not shared}
        for slot in changed:
            qspecs[slot] = shared
        return changed

    def _resolve_decline(
        self, widened: set[NodeSlot], qspecs: ProvisionalQSpecMap
    ) -> int | None:
        """Priority of the decline that governs the group, or ``None`` if observed."""
        declines = [
            qspecs[slot].declined_by
            for slot in widened
            if slot in qspecs and qspecs[slot].declined
        ]
        if not declines:
            return None
        best_decline = min(declines)

        active_priorities = [
            value.priority
            for slot in widened
            if slot in qspecs
            for value in qspecs[slot].fields.values()
        ]
        if not active_priorities:
            return best_decline
        if not self.priority_decides_decline:
            return None
        # Ties go to the decline, matching the pre-reconciler behavior where the
        # highest-priority consumer decided for the whole shared tensor.
        return best_decline if best_decline <= min(active_priorities) else None


# ---------------------------------------------------------------------------
# Per-field reconciliation.
# ---------------------------------------------------------------------------


def _reconcile_field(
    field_name: FieldName, slots: frozenset[NodeSlot], qspecs: ProvisionalQSpecMap
) -> FieldValue | None:
    """Reconcile one field across ``slots``, or ``None`` if no slot proposed it."""
    proposals = [
        qspecs[slot].fields[field_name]
        for slot in slots
        if slot in qspecs and field_name in qspecs[slot].fields
    ]
    if not proposals:
        return None
    policy = _FIELD_POLICY[field_name]
    return policy(proposals)


def _priority_min(proposals: Sequence[FieldValue]) -> int:
    """Highest priority (lowest priority-number) among the proposals."""
    return min(proposal.priority for proposal in proposals)


def _tie_break_key(proposal: FieldValue) -> tuple[int, str]:
    """Sort key picking the highest-priority proposal.

    Proposals are gathered from a ``frozenset``, whose order follows identity
    hashes, so breaking ties on ``str(value)`` is what makes the same model
    quantize the same way twice.
    """
    return (proposal.priority, str(proposal.value))


def _policy_priority_wins(proposals: Sequence[FieldValue]) -> FieldValue:
    """Highest-priority proposal wins; ties broken by :func:`_tie_break_key`."""
    winner = min(proposals, key=_tie_break_key)
    return FieldValue(value=winner.value, priority=_priority_min(proposals))


def _policy_must_agree(proposals: Sequence[FieldValue]) -> FieldValue:
    """All values must be equal, else :class:`ReconciliationError`."""
    values = {proposal.value for proposal in proposals}
    if len(values) > 1:
        raise ReconciliationError(
            f"Incompatible values across slots: {sorted(str(value) for value in values)}. "
            f"Proposals: {[(proposal.value, proposal.priority) for proposal in proposals]}"
        )
    return FieldValue(value=next(iter(values)), priority=_priority_min(proposals))


def _policy_qscheme_priority_wins(proposals: Sequence[FieldValue]) -> FieldValue:
    """Highest-priority proposal wins; ``None`` proposals are ignored.

    Not a join over "looseness", which would override config precedence: for
    ``A -> B`` with symmetric on ``A``'s output and asymmetric on ``B``'s input,
    a join picks asymmetric however the user ranked them. Op semantics still win
    by carrying :data:`OP_INTRINSIC_PRIORITY`.
    """
    stated = [proposal for proposal in proposals if proposal.value is not None]
    if not stated:
        return FieldValue(value=None, priority=_priority_min(proposals))
    winner = min(stated, key=_tie_break_key)
    return FieldValue(value=winner.value, priority=_priority_min(proposals))


def _policy_float_range_union(proposals: Sequence[FieldValue]) -> FieldValue:
    """Union of the proposed ranges, where ``None`` means "learn from data" and
    so beats any concrete bound.

        [0.0, 1.0]  ⊔  [None, None]  =  [None, None]
        [0.0, 1.0]  ⊔  [-1.0, 1.0]   =  [-1.0, 1.0]
        [0.0, None] ⊔  [0.0, None]   =  [0.0, None]

    Priority is ignored: one observer has to cover every slot sharing it.
    """
    ranges = [proposal.value for proposal in proposals if proposal.value is not None]
    if not ranges:
        return FieldValue(value=None, priority=_priority_min(proposals))

    lows = [r[0] for r in ranges]
    highs = [r[1] for r in ranges]
    # An unpinned bound is wider than any pinned one, so it wins.
    low = None if any(v is None for v in lows) else min(lows)
    high = None if any(v is None for v in highs) else max(highs)
    united = [low, high]

    relaxed = [r for r in ranges if list(r) != united]
    if relaxed:
        logger.info(
            "FLOAT_RANGE: pinned range(s) %r relaxed to %r so one observer "
            "covers every slot sharing it.",
            relaxed,
            united,
        )
    return FieldValue(value=united, priority=_priority_min(proposals))


_FIELD_POLICY: dict[FieldName, Any] = {
    # Config precedence decides every field except the analytic range.
    FieldName.DTYPE: _policy_priority_wins,
    FieldName.QSCHEME: _policy_qscheme_priority_wins,
    FieldName.QFORMULATION: _policy_priority_wins,
    FieldName.GRANULARITY: _policy_priority_wins,
    FieldName.FAKE_QUANTIZE_CLS: _policy_priority_wins,
    FieldName.QPARAM_CALCULATOR_CLS: _policy_priority_wins,
    FieldName.RANGE_CALCULATOR_CLS: _policy_priority_wins,
    FieldName.SCALE_DTYPE: _policy_priority_wins,
    # Covering every member's values is a correctness constraint, not a
    # preference, so this one unions instead of deferring to priority.
    FieldName.FLOAT_RANGE: _policy_float_range_union,
    # Weight and activation slots do share observers — flatten(param) -> add
    # ties a WEIGHT input slot to an ACTIVATION output slot — so this needs
    # resolving. Priority keeps every field coming from one winning config.
    FieldName.QUANTIZATION_TARGET: _policy_priority_wins,
}


def _field_value_stronger(new: FieldValue, current: FieldValue | None) -> bool:
    """Whether writing ``new`` would change the value or raise its priority."""
    if current is None:
        return True
    if new.value != current.value:
        return True
    return new.priority < current.priority


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _get_or_create(qspecs: ProvisionalQSpecMap, slot: NodeSlot) -> ProvisionalQSpec:
    """Return the :class:`ProvisionalQSpec` for ``slot``, creating one if absent."""
    if slot not in qspecs:
        qspecs[slot] = ProvisionalQSpec()
    return qspecs[slot]
