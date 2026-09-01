# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Unit tests for the constraint-queue qspec reconciliation pipeline.

Tests here are pure — they construct :class:`ProvisionalQSpec`
:class:`ProvisionalQSpecMap` values and :class:`Constraint`s by hand and
exercise ``.apply()`` directly. No fx graph construction. End-to-end
pipeline behavior is covered by trace-driven checks against the
walkthrough toy.
"""

from unittest.mock import Mock

import pytest
import torch

from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.quantization._graph._annotation_pattern_registry import (
    ConcatPattern,
    _same_axis,
)
from coreai_opt.quantization._graph._qspec_constraints import (
    InheritFields,
    ShareFields,
    ShareObserverInstance,
    _reconcile_field,
)
from coreai_opt.quantization._graph._qspec_resolution import _build_concrete_spec
from coreai_opt.quantization._graph._qspec_types import (
    OP_INTRINSIC_PRIORITY,
    FieldName,
    FieldValue,
    NodeSlot,
    ProvisionalQSpec,
    ProvisionalQSpecMap,
    ReconciliationError,
    SlotKind,
)
from coreai_opt.quantization.spec import (
    PerChannelGranularity,
    PerTensorGranularity,
    QuantizationFormulation,
    QuantizationScheme,
)
from coreai_opt.quantization.spec.fake_quantize import _DefaultFakeQuantizeImpl
from coreai_opt.quantization.spec.qparams_calculator import (
    MovingAverageQParamsCalculator,
    StaticQParamsCalculator,
    _DefaultQParamsCalculator,
)
from coreai_opt.quantization.spec.range_calculator import MinMaxRangeCalculator

# ---------------------------------------------------------------------------
# Test helpers.
# ---------------------------------------------------------------------------


def _slot(name: str = "s", kind: SlotKind = SlotKind.OUTPUT, arg_index: int = 0) -> NodeSlot:
    """Opaque ``NodeSlot`` — reconciler never inspects the fx node."""
    return NodeSlot(node=Mock(name=name), kind=kind, arg_index=arg_index)


def _pspec(**fields: FieldValue) -> ProvisionalQSpec:
    """Build a ProvisionalQSpec by keyword: DTYPE=FieldValue(int8, 0), ..."""
    field_map = {FieldName[key]: value for key, value in fields.items()}
    return ProvisionalQSpec(fields=field_map)


def _fv(value, priority: int = 0) -> FieldValue:
    return FieldValue(value=value, priority=priority)


# ---------------------------------------------------------------------------
# _reconcile_field policies.
# ---------------------------------------------------------------------------


class TestReconcileField:
    def test_dtype_priority_wins(self) -> None:
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: _pspec(DTYPE=_fv(torch.int4, priority=1)),
            b: _pspec(DTYPE=_fv(torch.int8, priority=5)),
        }
        result = _reconcile_field(FieldName.DTYPE, frozenset({a, b}), state)
        assert result.value == torch.int4  # priority 1 beats 5
        assert result.priority == 1

    def test_dtype_tie_first_encountered(self) -> None:
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: _pspec(DTYPE=_fv(torch.int8, priority=3)),
            b: _pspec(DTYPE=_fv(torch.int4, priority=3)),
        }
        result = _reconcile_field(FieldName.DTYPE, frozenset({a, b}), state)
        # min() with equal keys returns the first — encounter order
        assert result.value in (torch.int8, torch.int4)

    def test_qscheme_priority_wins_over_looser_scheme(self) -> None:
        """Higher-priority symmetric beats lower-priority affine.

        Regression guard for the priority-vs-lattice decision: an earlier
        revision lattice-joined qschemes so affine ("looser") always won,
        which silently overrode whichever config the user had ranked
        higher. Config precedence now decides.
        """
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: _pspec(QSCHEME=_fv(torch.per_tensor_symmetric, priority=0)),
            b: _pspec(QSCHEME=_fv(torch.per_tensor_affine, priority=5)),
        }
        result = _reconcile_field(FieldName.QSCHEME, frozenset({a, b}), state)
        assert result.value == torch.per_tensor_symmetric  # priority 0 wins
        assert result.priority == 0  # min of {0, 5}

    def test_qscheme_priority_wins_reversed(self) -> None:
        """Same pair, priorities swapped — now affine wins. The user's
        ranking is what changes the answer, not the qscheme itself."""
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: _pspec(QSCHEME=_fv(torch.per_tensor_symmetric, priority=5)),
            b: _pspec(QSCHEME=_fv(torch.per_tensor_affine, priority=0)),
        }
        result = _reconcile_field(FieldName.QSCHEME, frozenset({a, b}), state)
        assert result.value == torch.per_tensor_affine
        assert result.priority == 0

    def test_qscheme_op_intrinsic_priority_outranks_user_config(self) -> None:
        """An op-intrinsic proposal at OP_INTRINSIC_PRIORITY beats any
        user-config proposal, however highly the user ranked it. This is
        what keeps the sigmoid-into-cat case resolving to affine now that
        the lattice join is gone."""
        conv_out, sigmoid_out = _slot("conv_out"), _slot("sigmoid_out")
        state: ProvisionalQSpecMap = {
            # Best possible user priority.
            conv_out: _pspec(QSCHEME=_fv(torch.per_tensor_symmetric, priority=0)),
            # Op semantics, not user preference.
            sigmoid_out: _pspec(
                QSCHEME=_fv(torch.per_tensor_affine, priority=OP_INTRINSIC_PRIORITY)
            ),
        }
        result = _reconcile_field(FieldName.QSCHEME, frozenset({conv_out, sigmoid_out}), state)
        assert result.value == torch.per_tensor_affine
        assert result.priority == OP_INTRINSIC_PRIORITY

    def test_qscheme_per_channel_no_longer_auto_widens(self) -> None:
        """Mixed granularity resolves by priority, not by widening to
        per-channel. The old lattice returned per_channel whenever any
        member proposed it; that silently upgraded granularity for slots
        that had no valid ch_axis."""
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: _pspec(QSCHEME=_fv(torch.per_tensor_symmetric, priority=0)),
            b: _pspec(QSCHEME=_fv(torch.per_channel_symmetric, priority=5)),
        }
        result = _reconcile_field(FieldName.QSCHEME, frozenset({a, b}), state)
        assert result.value == torch.per_tensor_symmetric

    def test_qscheme_none_proposals_ignored(self) -> None:
        """A slot with no qscheme opinion doesn't get to win the field."""
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: _pspec(QSCHEME=_fv(None, priority=0)),
            b: _pspec(QSCHEME=_fv(torch.per_tensor_affine, priority=5)),
        }
        result = _reconcile_field(FieldName.QSCHEME, frozenset({a, b}), state)
        assert result.value == torch.per_tensor_affine

    def test_float_range_union_relaxes_a_pin_against_an_unpinned_member(self) -> None:
        """A pinned range loses to an unpinned one, per bound.

        A shared observer measures every member of its group, so its range has
        to cover them all. Keeping sigmoid's ``[0, 1]`` pin when another member
        is unbounded would clip that member's values.
        """
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: _pspec(FLOAT_RANGE=_fv([0.0, 1.0], priority=0)),
            b: _pspec(FLOAT_RANGE=_fv([None, None], priority=5)),
        }
        result = _reconcile_field(FieldName.FLOAT_RANGE, frozenset({a, b}), state)
        assert result.value == [None, None]

    def test_float_range_union_ignores_priority(self) -> None:
        """Same as above with priorities swapped — covering the data is a
        correctness constraint, not a preference, so priority doesn't rescue
        the pin."""
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: _pspec(FLOAT_RANGE=_fv([0.0, 1.0], priority=5)),
            b: _pspec(FLOAT_RANGE=_fv([None, None], priority=0)),
        }
        result = _reconcile_field(FieldName.FLOAT_RANGE, frozenset({a, b}), state)
        assert result.value == [None, None]

    def test_float_range_union_widens_two_concrete_pins(self) -> None:
        """sigmoid [0,1] sharing with tanh [-1,1] widens to cover both."""
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: _pspec(FLOAT_RANGE=_fv([0.0, 1.0], priority=0)),
            b: _pspec(FLOAT_RANGE=_fv([-1.0, 1.0], priority=0)),
        }
        result = _reconcile_field(FieldName.FLOAT_RANGE, frozenset({a, b}), state)
        assert result.value == [-1.0, 1.0]

    def test_float_range_union_is_per_bound(self) -> None:
        """relu's pinned min survives while the max stays data-driven."""
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: _pspec(FLOAT_RANGE=_fv([0.0, None], priority=0)),
            b: _pspec(FLOAT_RANGE=_fv([0.0, None], priority=5)),
        }
        result = _reconcile_field(FieldName.FLOAT_RANGE, frozenset({a, b}), state)
        assert result.value == [0.0, None]

    def test_granularity_priority_wins(self) -> None:
        """Granularity carries the axis, so it resolves as one unit rather than
        letting a separate axis field disagree with it."""
        a, b = _slot("a"), _slot("b")
        per_tensor = PerTensorGranularity()
        per_channel = PerChannelGranularity(axis=0)
        state: ProvisionalQSpecMap = {
            a: _pspec(GRANULARITY=_fv(per_channel, priority=0)),
            b: _pspec(GRANULARITY=_fv(per_tensor, priority=5)),
        }
        result = _reconcile_field(FieldName.GRANULARITY, frozenset({a, b}), state)
        assert result.value is per_channel

    def test_quantization_target_priority_wins(self) -> None:
        """Weight and activation slots do share observers — a shared-observer op
        fed by a state node ties them. The higher-priority config's target wins,
        so the rebuilt observer stays coherent with its other fields."""
        weight_slot = _slot("flatten_in", kind=SlotKind.INPUT)
        act_slot = _slot("flatten_out", kind=SlotKind.OUTPUT)
        state: ProvisionalQSpecMap = {
            weight_slot: _pspec(
                QUANTIZATION_TARGET=_fv(CompressionTargetTensor.WEIGHT, priority=0)
            ),
            act_slot: _pspec(
                QUANTIZATION_TARGET=_fv(CompressionTargetTensor.ACTIVATION, priority=1)
            ),
        }
        result = _reconcile_field(
            FieldName.QUANTIZATION_TARGET, frozenset({weight_slot, act_slot}), state
        )
        assert result.value is CompressionTargetTensor.WEIGHT

    def test_missing_field_returns_none(self) -> None:
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {a: _pspec(), b: _pspec()}
        assert _reconcile_field(FieldName.DTYPE, frozenset({a, b}), state) is None


# ---------------------------------------------------------------------------
# ShareFields.
# ---------------------------------------------------------------------------


class TestShareFields:
    def test_broadcasts_winner_to_all_slots(self) -> None:
        a, b, c = _slot("a"), _slot("b"), _slot("c")
        state: ProvisionalQSpecMap = {
            a: _pspec(DTYPE=_fv(torch.int4, priority=0)),
            b: _pspec(DTYPE=_fv(torch.int8, priority=5)),
            c: _pspec(),  # no proposal
        }
        con = ShareFields(_slots=frozenset({a, b, c}), fields=frozenset({FieldName.DTYPE}))
        changed = con.apply(state)
        assert changed == {b, c}  # a already had int4; b and c gain it
        assert state[a].fields[FieldName.DTYPE].value == torch.int4
        assert state[b].fields[FieldName.DTYPE].value == torch.int4
        assert state[c].fields[FieldName.DTYPE].value == torch.int4
        # All at priority 0 after reconciliation.
        assert state[a].fields[FieldName.DTYPE].priority == 0
        assert state[b].fields[FieldName.DTYPE].priority == 0

    def test_noop_when_already_reconciled(self) -> None:
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: _pspec(DTYPE=_fv(torch.int8, priority=0)),
            b: _pspec(DTYPE=_fv(torch.int8, priority=0)),
        }
        con = ShareFields(_slots=frozenset({a, b}), fields=frozenset({FieldName.DTYPE}))
        assert con.apply(state) == set()

    def test_multiple_fields(self) -> None:
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: _pspec(
                DTYPE=_fv(torch.int4, priority=0),
                QSCHEME=_fv(torch.per_tensor_symmetric, priority=0),
            ),
            b: _pspec(
                DTYPE=_fv(torch.int8, priority=5),
                QSCHEME=_fv(torch.per_tensor_affine, priority=5),
            ),
        }
        con = ShareFields(
            _slots=frozenset({a, b}),
            fields=frozenset({FieldName.DTYPE, FieldName.QSCHEME}),
        )
        con.apply(state)
        # Both fields resolve to a's values — a is higher priority on both.
        assert state[a].fields[FieldName.DTYPE].value == torch.int4
        assert state[b].fields[FieldName.DTYPE].value == torch.int4
        assert state[a].fields[FieldName.QSCHEME].value == torch.per_tensor_symmetric
        assert state[b].fields[FieldName.QSCHEME].value == torch.per_tensor_symmetric


# ---------------------------------------------------------------------------
# ShareObserverInstance.
# ---------------------------------------------------------------------------


class TestShareObserverInstance:
    def test_merges_two_slots_into_one_instance(self) -> None:
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: _pspec(DTYPE=_fv(torch.int8, priority=0)),
            b: _pspec(DTYPE=_fv(torch.int8, priority=0)),
        }
        assert state[a] is not state[b]
        con = ShareObserverInstance(_slots=frozenset({a, b}))
        con.apply(state)
        assert state[a] is state[b]  # identity, not just value equality

    def test_field_mutation_after_share_propagates(self) -> None:
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {a: _pspec(), b: _pspec()}
        ShareObserverInstance(_slots=frozenset({a, b})).apply(state)
        state[a].fields[FieldName.DTYPE] = _fv(torch.int8, 0)
        # b's ProvisionalQSpec is the same object → sees the new field.
        assert state[b].fields[FieldName.DTYPE].value == torch.int8

    def test_transitive_merge_pulls_in_prior_sharers(self) -> None:
        a, b, c = _slot("a"), _slot("b"), _slot("c")
        state: ProvisionalQSpecMap = {a: _pspec(), b: _pspec(), c: _pspec()}
        # First merge a and b.
        ShareObserverInstance(_slots=frozenset({a, b})).apply(state)
        assert state[a] is state[b]
        # Now merge a with c — b should be pulled in transitively.
        ShareObserverInstance(_slots=frozenset({a, c})).apply(state)
        assert state[a] is state[b] is state[c]

    def test_reconciles_fields_across_merged_group(self) -> None:
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: _pspec(DTYPE=_fv(torch.int4, priority=0)),
            b: _pspec(DTYPE=_fv(torch.int8, priority=5)),
        }
        ShareObserverInstance(_slots=frozenset({a, b})).apply(state)
        assert state[a].fields[FieldName.DTYPE].value == torch.int4
        assert state[b].fields[FieldName.DTYPE].value == torch.int4
        assert state[a] is state[b]

    def test_noop_when_already_shared_with_correct_fields(self) -> None:
        a, b = _slot("a"), _slot("b")
        shared = _pspec(DTYPE=_fv(torch.int8, priority=0))
        state: ProvisionalQSpecMap = {a: shared, b: shared}
        changed = ShareObserverInstance(_slots=frozenset({a, b})).apply(state)
        assert changed == set()
        assert state[a] is shared
        assert state[b] is shared


# ---------------------------------------------------------------------------
# Convergence / no-op behavior.
# ---------------------------------------------------------------------------


class TestConvergence:
    def test_repeated_share_fields_stabilizes(self) -> None:
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: _pspec(DTYPE=_fv(torch.int4, priority=1)),
            b: _pspec(DTYPE=_fv(torch.int8, priority=5)),
        }
        con = ShareFields(_slots=frozenset({a, b}), fields=frozenset({FieldName.DTYPE}))
        first = con.apply(state)
        assert first  # something changed
        # Immediate re-apply is a no-op — proves the priority-inheritance
        # rule prevents oscillation.
        assert con.apply(state) == set()

    def test_repeated_share_observer_stabilizes(self) -> None:
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {a: _pspec(), b: _pspec()}
        con = ShareObserverInstance(_slots=frozenset({a, b}))
        first = con.apply(state)
        assert first
        assert con.apply(state) == set()


# ---------------------------------------------------------------------------
# YOLOX-shape scenario reconciliation.
# ---------------------------------------------------------------------------


def test_yolox_scenario_end_to_end_reconciliation() -> None:
    """cat-of-sigmoid: two sigmoid branches plus a symmetric conv branch, all
    merged into one observer group.

    The sigmoids carry ``OP_INTRINSIC_PRIORITY`` on QSCHEME and FLOAT_RANGE
    (what ``_apply_op_intrinsic_override`` writes), so:

    * QSCHEME resolves to ASYMMETRIC — the intrinsic outranks user config.
    * FLOAT_RANGE relaxes to unbounded — the group also contains the conv
      output, which is not confined to [0, 1], so keeping the pin would clip
      it. This is the union policy overriding even the reserved priority,
      because covering the data is a correctness constraint.
    """
    conv_out = _slot("conv_out")
    sig_a_out = _slot("sig_a_out")
    sig_b_out = _slot("sig_b_out")
    cat_out = _slot("cat_out")

    def _user_activation(**overrides):
        base = dict(
            DTYPE=_fv(torch.int8, priority=5),
            QSCHEME=_fv(QuantizationScheme.SYMMETRIC, priority=5),
            FLOAT_RANGE=_fv([None, None], priority=5),
            GRANULARITY=_fv(PerTensorGranularity(), priority=5),
            QUANTIZATION_TARGET=_fv(CompressionTargetTensor.ACTIVATION, priority=5),
        )
        base.update(overrides)
        return _pspec(**base)  # _pspec maps str -> FieldName itself

    state: ProvisionalQSpecMap = {
        conv_out: _user_activation(),
        # Op-intrinsic contributions land out-of-band on both fields.
        sig_a_out: _user_activation(
            QSCHEME=_fv(QuantizationScheme.ASYMMETRIC, priority=OP_INTRINSIC_PRIORITY),
            FLOAT_RANGE=_fv([0.0, 1.0], priority=OP_INTRINSIC_PRIORITY),
        ),
        sig_b_out: _user_activation(
            QSCHEME=_fv(QuantizationScheme.ASYMMETRIC, priority=OP_INTRINSIC_PRIORITY),
            FLOAT_RANGE=_fv([0.0, 1.0], priority=OP_INTRINSIC_PRIORITY),
        ),
        cat_out: _user_activation(),
    }
    ShareObserverInstance(_slots=frozenset({conv_out, sig_a_out, sig_b_out, cat_out})).apply(state)

    # All four share one ProvisionalQSpec now.
    assert state[conv_out] is state[sig_a_out] is state[sig_b_out] is state[cat_out]
    shared = state[conv_out].fields
    assert shared[FieldName.QSCHEME].value is QuantizationScheme.ASYMMETRIC
    assert shared[FieldName.DTYPE].value == torch.int8
    # The sigmoid pin is relaxed so the shared observer still covers the conv.
    assert shared[FieldName.FLOAT_RANGE].value == [None, None]


def test_adjacent_edge_share_resolves_dtype_conflict_by_priority() -> None:
    """test_op_level_precedence-shape: linear1.OUTPUT wants int4 at lower
    priority; linear2.INPUT wants int8 at higher priority. Adjacent-edge
    sharing merges them; priority resolves dtype to int8."""
    linear1_out = _slot("linear1_out", kind=SlotKind.OUTPUT)
    linear2_in = _slot("linear2_in", kind=SlotKind.INPUT)
    state: ProvisionalQSpecMap = {
        linear1_out: _pspec(
            DTYPE=_fv(torch.int4, priority=5),
            QSCHEME=_fv(QuantizationScheme.SYMMETRIC, priority=5),
            GRANULARITY=_fv(PerTensorGranularity(), priority=5),
            QUANTIZATION_TARGET=_fv(CompressionTargetTensor.ACTIVATION, priority=5),
        ),
        linear2_in: _pspec(
            DTYPE=_fv(torch.int8, priority=0),
            QSCHEME=_fv(QuantizationScheme.SYMMETRIC, priority=0),
            GRANULARITY=_fv(PerTensorGranularity(), priority=0),
            QUANTIZATION_TARGET=_fv(CompressionTargetTensor.ACTIVATION, priority=0),
        ),
    }
    ShareObserverInstance(_slots=frozenset({linear1_out, linear2_in})).apply(state)
    assert state[linear1_out] is state[linear2_in]
    assert state[linear1_out].fields[FieldName.DTYPE].value == torch.int8


def test_share_fields_dtype_only_leaves_other_fields_independent() -> None:
    """``ShareFields`` agrees on the named fields without merging the
    ProvisionalQSpec objects, so everything else stays per-slot."""
    a_out = _slot("a_out")
    b_out = _slot("b_out")
    state: ProvisionalQSpecMap = {
        a_out: _pspec(DTYPE=_fv(torch.int8, priority=0)),
        b_out: _pspec(DTYPE=_fv(torch.int4, priority=5)),
    }
    ShareFields(_slots=frozenset({a_out, b_out}), fields=frozenset({FieldName.DTYPE})).apply(state)
    assert state[a_out] is not state[b_out]
    assert state[a_out].fields[FieldName.DTYPE].value == torch.int8
    assert state[b_out].fields[FieldName.DTYPE].value == torch.int8


class TestConcatSharingConstraints:
    """``ConcatPattern.generate_qspec_sharing_constraints`` picks between tying the
    inputs to one observer and merely agreeing on dtype."""

    @staticmethod
    def _cat_node(rank: int, dim: int, n_inputs: int = 2):
        node = Mock()
        node.args = (None, dim)
        node.kwargs = {}
        node.meta = {"val": Mock(shape=tuple(range(1, rank + 1)))}
        node.all_input_nodes = [Mock() for _ in range(n_inputs)]
        return node

    def _state(self, node, granularity) -> ProvisionalQSpecMap:
        return {
            NodeSlot(node=node, kind=SlotKind.OUTPUT, arg_index=0): _pspec(
                DTYPE=_fv(torch.int8, priority=0),
                GRANULARITY=_fv(granularity, priority=0),
            ),
            NodeSlot(node=node, kind=SlotKind.INPUT, arg_index=0): _pspec(
                DTYPE=_fv(torch.int8, priority=0)
            ),
        }

    def test_per_channel_along_concat_axis_shares_dtype_only(self) -> None:
        """Each input keeps its own scale, so they need only agree on dtype."""
        node = self._cat_node(rank=4, dim=1)
        state = self._state(node, PerChannelGranularity(axis=1))
        constraints = ConcatPattern.generate_qspec_sharing_constraints(node, state)
        assert len(constraints) == 1
        assert isinstance(constraints[0], ShareFields)
        assert constraints[0].fields == frozenset({FieldName.DTYPE})

    def test_per_channel_along_other_axis_shares_observer(self) -> None:
        node = self._cat_node(rank=4, dim=1)
        state = self._state(node, PerChannelGranularity(axis=0))
        constraints = ConcatPattern.generate_qspec_sharing_constraints(node, state)
        assert [type(c) for c in constraints] == [ShareObserverInstance]

    def test_per_tensor_shares_observer(self) -> None:
        node = self._cat_node(rank=4, dim=1)
        state = self._state(node, PerTensorGranularity())
        constraints = ConcatPattern.generate_qspec_sharing_constraints(node, state)
        assert [type(c) for c in constraints] == [ShareObserverInstance]

    def test_no_populated_input_slot_emits_nothing(self) -> None:
        node = self._cat_node(rank=4, dim=1)
        assert ConcatPattern.generate_qspec_sharing_constraints(node, {}) == []


# ---------------------------------------------------------------------------
# Concat axis normalization.
# ---------------------------------------------------------------------------


class TestConcatAxisNormalization:
    """``_same_axis`` must treat negative and positive axis indices that name
    the same dimension as equal.

    ``torch.cat(..., dim=-3)`` survives ``torch.export`` as ``-3`` rather than
    being normalized, while a spec's ``ch_axis`` is stored verbatim from the
    user's config. Comparing them as raw integers made
    ``per_channel_along_concat_axis`` wrongly ``False``, which forces every
    concat input onto one observer even though per-channel-along-concat
    semantics require each input to keep its own scale.
    """

    @staticmethod
    def _node_with_rank(rank: int):
        node = Mock()
        node.meta = {"val": Mock(shape=tuple(range(1, rank + 1)))}
        return node

    @pytest.mark.parametrize(
        "ch_axis, concat_dim, rank, expected",
        [
            (1, 1, 4, True),  # both positive, same dim
            (1, -3, 4, True),  # positive vs negative, same dim (NCHW channel)
            (-3, 1, 4, True),  # reversed
            (-3, -3, 4, True),  # both negative, same dim
            (0, -1, 4, False),  # genuinely different dims
            (1, 2, 4, False),  # genuinely different dims
            (0, -4, 4, True),  # batch dim via either sign
        ],
    )
    def test_same_axis(self, ch_axis, concat_dim, rank, expected) -> None:
        assert _same_axis(ch_axis, concat_dim, self._node_with_rank(rank)) is expected

    def test_same_axis_without_metadata_falls_back_to_raw_compare(self) -> None:
        """No ``val`` metadata means rank is unknown, so only an exact match
        counts. Better a missed match than a crash mid-annotation."""
        node = Mock()
        node.meta = {}
        assert _same_axis(1, 1, node) is True
        assert _same_axis(1, -3, node) is False


# ---------------------------------------------------------------------------
# Field map -> observer round trip.
# ---------------------------------------------------------------------------


class TestBuildConcreteSpec:
    """``_build_concrete_spec`` must carry every reconciled field into the
    observer it builds.

    This is the property the field/partial inversion exists to guarantee. In
    the previous design the observer partial was reconciled as one opaque
    field, so a reconciled ``QSCHEME`` was written to
    ``TorchAOQuantizationSpec.qscheme`` — which nothing downstream reads —
    while the qscheme that actually took effect stayed baked inside the
    partial's qparams calculator, untouched by reconciliation.
    """

    @staticmethod
    def _fields(**overrides):
        base = {
            FieldName.DTYPE: _fv(torch.int8),
            FieldName.QSCHEME: _fv(QuantizationScheme.SYMMETRIC),
            FieldName.QFORMULATION: _fv(QuantizationFormulation.ZP),
            FieldName.GRANULARITY: _fv(PerTensorGranularity()),
            FieldName.FLOAT_RANGE: _fv([None, None]),
            FieldName.FAKE_QUANTIZE_CLS: _fv(_DefaultFakeQuantizeImpl),
            FieldName.QPARAM_CALCULATOR_CLS: _fv(MovingAverageQParamsCalculator),
            FieldName.RANGE_CALCULATOR_CLS: _fv(MinMaxRangeCalculator),
            FieldName.SCALE_DTYPE: _fv(None),
            FieldName.SPARSITY: _fv(None),
            FieldName.QUANTIZATION_TARGET: _fv(CompressionTargetTensor.ACTIVATION),
        }
        # Keyword keys arrive as strings; map them onto FieldName so they
        # replace entries rather than adding new string-keyed ones.
        base.update({FieldName[key]: value for key, value in overrides.items()})
        return ProvisionalQSpec(fields=base)

    @staticmethod
    def _calculator(torchao_spec):
        return torchao_spec.observer_or_fake_quant_ctr.callable_args["qparams_calculator"]()

    def test_reconciled_qscheme_reaches_the_calculator(self) -> None:
        """The regression guard for the original defect."""
        spec = _build_concrete_spec(self._fields(QSCHEME=_fv(QuantizationScheme.ASYMMETRIC)))
        assert self._calculator(spec).qscheme is QuantizationScheme.ASYMMETRIC

    def test_reconciled_float_range_reaches_the_calculator(self) -> None:
        spec = _build_concrete_spec(self._fields(FLOAT_RANGE=_fv([0.0, 1.0])))
        assert tuple(self._calculator(spec).float_range) == (0.0, 1.0)

    def test_reconciled_dtype_and_derived_range(self) -> None:
        """quant_min/quant_max are not reconciled — they fall out of dtype and
        qscheme when the spec is reassembled."""
        spec = _build_concrete_spec(self._fields(DTYPE=_fv(torch.int8)))
        assert spec.dtype == torch.int8
        assert (spec.quant_min, spec.quant_max) == (-128, 127)

    def test_target_selects_the_default_calculator(self) -> None:
        """A ``"default"`` qparam calculator resolves via QUANTIZATION_TARGET:
        Static for weights (derive once at prepare), MovingAverage for
        activations (learn over calibration). This is why the target has to be
        reconciled rather than guessed at resolution time."""
        weight = _build_concrete_spec(
            self._fields(
                QPARAM_CALCULATOR_CLS=_fv(_DefaultQParamsCalculator),
                QUANTIZATION_TARGET=_fv(CompressionTargetTensor.WEIGHT),
            )
        )
        activation = _build_concrete_spec(
            self._fields(
                QPARAM_CALCULATOR_CLS=_fv(_DefaultQParamsCalculator),
                QUANTIZATION_TARGET=_fv(CompressionTargetTensor.ACTIVATION),
            )
        )
        assert isinstance(self._calculator(weight), StaticQParamsCalculator)
        assert isinstance(self._calculator(activation), MovingAverageQParamsCalculator)

    def test_empty_field_map_is_not_observed(self) -> None:
        assert _build_concrete_spec(ProvisionalQSpec(fields={})) is None

    def test_partial_field_map_raises(self) -> None:
        """A group carrying only some fields means a constraint invented fields
        for a slot that was never seeded from a config — surface it instead of
        filling in defaults."""
        with pytest.raises(ReconciliationError, match="Cannot rebuild"):
            _build_concrete_spec(ProvisionalQSpec(fields={FieldName.DTYPE: _fv(torch.int8)}))


# ---------------------------------------------------------------------------
# Explicitly-declined slots.
# ---------------------------------------------------------------------------


class TestDeclinedSlots:
    """A declined slot is distinct from one nothing has spoken for.

    Absence means "no opinion"; declined means a config that covers this slot
    resolved no spec for it. Collapsing the two let a decline be silently
    discarded, because absence cannot outrank anything.
    """

    @staticmethod
    def _observed(priority: int = 0) -> ProvisionalQSpec:
        return ProvisionalQSpec(
            fields={
                FieldName.DTYPE: _fv(torch.int8, priority),
                FieldName.QSCHEME: _fv(QuantizationScheme.SYMMETRIC, priority),
                FieldName.GRANULARITY: _fv(PerTensorGranularity(), priority),
                FieldName.QUANTIZATION_TARGET: _fv(CompressionTargetTensor.ACTIVATION, priority),
            }
        )

    @staticmethod
    def _declined(priority: int = 0) -> ProvisionalQSpec:
        return ProvisionalQSpec(declined_by=priority)

    def test_declined_is_not_the_same_as_empty(self) -> None:
        assert self._declined().declined is True
        assert ProvisionalQSpec().declined is False

    def test_an_observing_member_outranks_a_decline(self) -> None:
        """A decline loses to any member that wants observation.

        The members are one tensor, so excluding one reference to it while
        another asks for quantization is not a coherent instruction. For
        ``cat(A, declined_B, C)`` the cat is still observed.
        """
        a, b, c = _slot("a"), _slot("b"), _slot("c")
        state: ProvisionalQSpecMap = {
            a: self._observed(priority=0),
            b: self._declined(priority=7),
            c: self._observed(priority=0),
        }
        ShareObserverInstance(_slots=frozenset({a, b, c})).apply(state)
        assert state[a] is state[b] is state[c]
        assert state[a].declined is False

    def test_decline_loses_regardless_of_priority(self) -> None:
        """Priority does not rescue a decline: it loses even when it holds the
        best priority and the observing peer the worst."""
        observed, declined = _slot("observed"), _slot("declined")
        state: ProvisionalQSpecMap = {
            observed: self._observed(priority=99),  # worst
            declined: self._declined(priority=0),  # best
        }
        ShareObserverInstance(_slots=frozenset({observed, declined})).apply(state)
        assert state[observed].declined is False

    def test_group_declined_only_when_every_member_declines(self) -> None:
        """With nothing asking for observation, the decline stands."""
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            a: self._declined(priority=3),
            b: self._declined(priority=7),
        }
        ShareObserverInstance(_slots=frozenset({a, b})).apply(state)
        assert state[a] is state[b]
        assert state[a].declined is True
        assert state[a].declined_by == 3

    def test_group_without_declines_stays_observed(self) -> None:
        a, b = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {a: self._observed(), b: self._observed()}
        ShareObserverInstance(_slots=frozenset({a, b})).apply(state)
        assert state[a].declined is False
        assert state[a].fields[FieldName.DTYPE].value == torch.int8

    def test_declined_group_emits_no_annotation(self) -> None:
        """Resolution skips a declined group, which is what leaves the op in
        float for torchao."""
        declined = self._observed()
        declined.declined_by = 3
        assert _build_concrete_spec(declined) is None


# ---------------------------------------------------------------------------
# Same-tensor propagation.
# ---------------------------------------------------------------------------


class TestInheritFields:
    """``InheritFields`` copies data facts one direction; it does not merge.

    Its reason to exist: slots either side of a rank-changing passthrough
    observe the *same tensor* but cannot share an observer instance, because a
    ``QParamsCalculator`` is sized to the rank of the first tensor it sees. A
    fact proven about the data upstream therefore has to be copied, not
    reconciled — ``ShareFields`` would route it through
    ``_policy_float_range_union``, which relaxes the very pin being carried.
    """

    @staticmethod
    def _with(**overrides) -> ProvisionalQSpec:
        base = {
            FieldName.DTYPE: _fv(torch.int8),
            FieldName.QSCHEME: _fv(QuantizationScheme.SYMMETRIC),
            FieldName.FLOAT_RANGE: _fv([None, None]),
        }
        base.update({FieldName[key]: value for key, value in overrides.items()})
        return ProvisionalQSpec(fields=base)

    _FACTS = frozenset({FieldName.QSCHEME, FieldName.FLOAT_RANGE})

    def test_copies_facts_downstream(self) -> None:
        src, dst = _slot("relu_out"), _slot("linear_in", kind=SlotKind.INPUT)
        state: ProvisionalQSpecMap = {
            src: self._with(
                QSCHEME=_fv(QuantizationScheme.ASYMMETRIC, OP_INTRINSIC_PRIORITY),
                FLOAT_RANGE=_fv([0.0, None], OP_INTRINSIC_PRIORITY),
            ),
            dst: self._with(),
        }
        changed = InheritFields(source=src, targets=frozenset({dst}), fields=self._FACTS).apply(
            state
        )
        assert changed == {dst}
        assert state[dst].fields[FieldName.QSCHEME].value is QuantizationScheme.ASYMMETRIC
        assert state[dst].fields[FieldName.FLOAT_RANGE].value == [0.0, None]

    def test_does_not_merge_objects(self) -> None:
        """Unlike ShareObserverInstance — each side keeps its own observer, which
        is the whole point when rank differs."""
        src, dst = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {src: self._with(), dst: self._with()}
        InheritFields(source=src, targets=frozenset({dst}), fields=self._FACTS).apply(state)
        assert state[src] is not state[dst]

    def test_leaves_untargeted_fields_alone(self) -> None:
        """DTYPE is a user choice, not a fact about the data, so it must not be
        copied — doing so would let a passthrough override a higher-priority
        config's dtype."""
        src, dst = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            src: self._with(DTYPE=_fv(torch.int4)),
            dst: self._with(DTYPE=_fv(torch.int8)),
        }
        InheritFields(source=src, targets=frozenset({dst}), fields=self._FACTS).apply(state)
        assert state[dst].fields[FieldName.DTYPE].value == torch.int8

    def test_inherits_the_relaxed_value_not_a_stale_pin(self) -> None:
        """If the source's own group relaxed its range first, targets must get
        the relaxed value. Guards against propagating a pin that reconciliation
        already decided was unsafe."""
        src, dst = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            src: self._with(FLOAT_RANGE=_fv([None, None])),  # already relaxed
            dst: self._with(FLOAT_RANGE=_fv([0.0, 1.0])),
        }
        InheritFields(source=src, targets=frozenset({dst}), fields=self._FACTS).apply(state)
        assert state[dst].fields[FieldName.FLOAT_RANGE].value == [None, None]

    def test_second_apply_is_a_noop(self) -> None:
        """Convergence: this constraint copies at unchanged priority rather than
        lowering one, so idempotence is what keeps the drain loop terminating."""
        src, dst = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            src: self._with(FLOAT_RANGE=_fv([0.0, None])),
            dst: self._with(),
        }
        con = InheritFields(source=src, targets=frozenset({dst}), fields=self._FACTS)
        assert con.apply(state)
        assert con.apply(state) == set()

    def test_skips_declined_target(self) -> None:
        """A decline is an explicit opt-out and outranks a propagated fact."""
        src, dst = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            src: self._with(FLOAT_RANGE=_fv([0.0, None])),
            dst: ProvisionalQSpec(declined_by=0),
        }
        assert (
            InheritFields(source=src, targets=frozenset({dst}), fields=self._FACTS).apply(state)
            == set()
        )

    def test_declined_source_carries_nothing(self) -> None:
        src, dst = _slot("a"), _slot("b")
        state: ProvisionalQSpecMap = {
            src: ProvisionalQSpec(declined_by=0),
            dst: self._with(),
        }
        assert (
            InheritFields(source=src, targets=frozenset({dst}), fields=self._FACTS).apply(state)
            == set()
        )

    def test_absent_target_is_not_created(self) -> None:
        """Propagation must not invent observation for a slot nothing asked to
        observe — that would produce a partial field map, which
        ``_build_concrete_spec`` rejects."""
        src, absent = _slot("a"), _slot("absent")
        state: ProvisionalQSpecMap = {src: self._with(FLOAT_RANGE=_fv([0.0, None]))}
        assert (
            InheritFields(source=src, targets=frozenset({absent}), fields=self._FACTS).apply(state)
            == set()
        )
        assert absent not in state
