# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for QAT schedule runtime behavior (step-based training control).

Config-level tests (QATSchedule validation, from_dict, from_yaml) live in
test_quantization_config.py. This file tests the runtime behavior of
step(), training_mode(), and low-level enable/disable APIs.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch.nn as nn

from coreai_opt.quantization import (
    ModuleQuantizerConfig,
    Quantizer,
    QuantizerConfig,
)
from coreai_opt.quantization.config import ExecutionMode, QATSchedule
from coreai_opt.quantization.spec import (
    QuantizationSpec,
    default_activation_quantization_spec,
    default_weight_quantization_spec,
)
from coreai_opt.quantization.spec.fake_quantize import (
    FakeQuantizeImplBase,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleModel(nn.Module):
    """Conv2d → ReLU → flatten → Linear."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, 3, padding=1)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(4 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)


class NestedModel(nn.Module):
    """Model with a nested block: block.conv → block.relu → flatten → linear."""

    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),
            nn.ReLU(),
        )
        self.linear = nn.Linear(4 * 8 * 8, 10)

    def forward(self, x):
        x = self.block(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)


def _make_example_input():
    return (torch.randn(1, 1, 8, 8),)


def _weight_only_config(execution_mode=ExecutionMode.EAGER, qat_schedule=None):
    """Weight-only quantizer config, optionally with a QATSchedule."""
    return QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={"weight": default_weight_quantization_spec()},
            op_input_spec=None,
            op_output_spec=None,
            qat_schedule=qat_schedule,
        ),
        execution_mode=execution_mode,
    )


def _get_fake_quant_modules(model):
    return [m for m in model.modules() if isinstance(m, FakeQuantizeImplBase)]


def _resolve_fqs(model_or_fqs):
    """Accept a model (extracts FQ modules) or a list of FQ modules."""
    if isinstance(model_or_fqs, list):
        return model_or_fqs
    return _get_fake_quant_modules(model_or_fqs)


def _all_observers_enabled(model_or_fqs):
    fqs = _resolve_fqs(model_or_fqs)
    return len(fqs) > 0 and all(fq.observer_enabled.item() == 1 for fq in fqs)


def _all_observers_disabled(model_or_fqs):
    fqs = _resolve_fqs(model_or_fqs)
    return len(fqs) > 0 and all(fq.observer_enabled.item() == 0 for fq in fqs)


def _all_fake_quant_enabled(model_or_fqs):
    fqs = _resolve_fqs(model_or_fqs)
    return len(fqs) > 0 and all(fq.fake_quant_enabled.item() == 1 for fq in fqs)


def _all_fake_quant_disabled(model_or_fqs):
    fqs = _resolve_fqs(model_or_fqs)
    return len(fqs) > 0 and all(fq.fake_quant_enabled.item() == 0 for fq in fqs)


def _get_fqs_for_module(quantizer, module_name):
    """Get FQ modules for a specific original module name."""
    fq_map = quantizer._quantizer._get_fake_quantize_modules()
    return fq_map.get(module_name, [])


# ---------------------------------------------------------------------------
# _get_fake_quantize_modules correctness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("execution_mode", [ExecutionMode.EAGER, ExecutionMode.GRAPH])
def test_get_fake_quantize_modules(execution_mode):
    """FQ map has correct total count and no duplicate instances."""
    config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={"weight": default_weight_quantization_spec()},
            op_input_spec={"*": default_activation_quantization_spec()},
            op_output_spec={"*": default_activation_quantization_spec()},
        ),
        execution_mode=execution_mode,
    )
    quantizer = Quantizer(SimpleModel(), config)
    quantizer.prepare(_make_example_input())

    fq_map = quantizer._get_fake_quantize_modules()
    all_fqs = [fq for fqs in fq_map.values() for fq in fqs]
    all_fqs_from_model = _get_fake_quant_modules(quantizer._model)

    assert len(all_fqs) == len(all_fqs_from_model), (
        f"FQ map total ({len(all_fqs)}) != model total ({len(all_fqs_from_model)})"
    )
    assert len(set(id(fq) for fq in all_fqs)) == len(all_fqs), "Duplicate FQ instances in map"


class SharedWeightModel(nn.Module):
    """Two Linear layers sharing the same weight parameter."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 10, bias=False)
        self.linear2 = nn.Linear(10, 10, bias=False)
        self.linear2.weight = self.linear1.weight

    def forward(self, x):
        return self.linear2(self.linear1(x))


@pytest.mark.parametrize("execution_mode", [ExecutionMode.EAGER, ExecutionMode.GRAPH])
def test_shared_weight_schedule_owner(execution_mode):
    """When two modules share a weight, the shared FQ's schedule follows a
    single, consistent owner rather than an arbitrary mix of both configs.

    In Eager mode the same FakeQuantize instance is literally shared between
    both modules, so a warning is emitted and the first-assigned schedule
    (linear1's) wins. In graph mode the shared weight is deduplicated into
    one FQ node whose owning module is resolved by config priority — the
    same priority that decides the FQ's dtype (see
    ``GraphQuantizer._get_fake_quantize_modules``) — which for two
    ``module_name_configs`` entries is the later-declared one (linear2's).
    """
    config = QuantizerConfig(
        global_config=None,
        module_name_configs={
            "linear1": ModuleQuantizerConfig(
                op_state_spec={"weight": default_weight_quantization_spec()},
                op_input_spec=None,
                op_output_spec=None,
                qat_schedule=QATSchedule(enable_observer=0, enable_fake_quant=1),
            ),
            "linear2": ModuleQuantizerConfig(
                op_state_spec={"weight": default_weight_quantization_spec()},
                op_input_spec=None,
                op_output_spec=None,
                qat_schedule=QATSchedule(enable_observer=0, enable_fake_quant=5),
            ),
        },
        execution_mode=execution_mode,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quantizer = Quantizer(SharedWeightModel(), config)
        quantizer.prepare((torch.randn(1, 10),))

    schedule_warnings = [w for w in caught if "shared" in str(w.message).lower()]
    if execution_mode == ExecutionMode.EAGER:
        assert len(schedule_warnings) >= 1
    else:
        assert len(schedule_warnings) == 0

    assert len(quantizer._fq_to_schedule) == 1
    (schedule,) = quantizer._fq_to_schedule.values()
    # Eager mode picks linear1's schedule (first-assigned to the shared FQ
    # instance); graph mode picks linear2's, since it has higher config
    # priority (declared later) — the same priority that decides dtype.
    expected_step = 1 if execution_mode == ExecutionMode.EAGER else 5
    assert schedule.enable_fake_quant == expected_step

    # Verify the winning owner's schedule is actually applied
    with quantizer.training_mode():
        (fq_mod,) = quantizer._fq_to_schedule.keys()
        assert fq_mod.fake_quant_enabled.item() == 0
        for _ in range(expected_step):
            quantizer.step()
        assert fq_mod.fake_quant_enabled.item() == 1


def test_shared_weight_dtype_and_schedule_share_owner_graph_mode():
    """In graph mode, a shared weight's QATSchedule follows the same config
    priority used to resolve its dtype and the rest of its qspec.

    linear2 has higher priority than linear1 since it is declared later in
    ``module_name_configs``, so the shared FQ's dtype, qspec, and
    QATSchedule should all follow linear2's config — even though linear1 is
    the first consumer encountered in graph order.
    """
    config = QuantizerConfig(
        global_config=None,
        module_name_configs={
            "linear1": ModuleQuantizerConfig(
                op_state_spec={"weight": QuantizationSpec(dtype=torch.int8)},
                op_input_spec=None,
                op_output_spec=None,
                qat_schedule=QATSchedule(enable_observer=0, enable_fake_quant=1),
            ),
            "linear2": ModuleQuantizerConfig(
                op_state_spec={"weight": QuantizationSpec(dtype=torch.int4)},
                op_input_spec=None,
                op_output_spec=None,
                qat_schedule=QATSchedule(enable_observer=0, enable_fake_quant=5),
            ),
        },
        execution_mode=ExecutionMode.GRAPH,
    )
    quantizer = Quantizer(SharedWeightModel(), config)
    quantizer.prepare((torch.randn(1, 10),))

    assert len(quantizer._fq_to_schedule) == 1
    (fq_mod, schedule) = next(iter(quantizer._fq_to_schedule.items()))

    # linear2 (declared later, higher config priority) owns the dtype ...
    assert (fq_mod.quant_min, fq_mod.quant_max) == (-8, 7)  # int4
    # ... and must also own the schedule.
    assert schedule.enable_fake_quant == 5


# ---------------------------------------------------------------------------
# training_mode() behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("execution_mode", [ExecutionMode.EAGER, ExecutionMode.GRAPH])
class TestTrainingMode:
    """Tests for training_mode() entry/exit state with and without schedule."""

    def test_default_behavior_no_schedule(self, execution_mode):
        """Without schedule, training_mode enables both observer and fq."""
        model = SimpleModel()
        config = _weight_only_config(execution_mode=execution_mode, qat_schedule=None)
        quantizer = Quantizer(model, config)
        quantizer.prepare(_make_example_input())

        with quantizer.training_mode():
            assert _all_observers_enabled(quantizer._model)
            assert _all_fake_quant_enabled(quantizer._model)

    def test_entry_applies_schedule_state(self, execution_mode):
        """On entry, schedule overrides default: fq OFF when threshold not reached."""
        schedule = QATSchedule(enable_observer=0, enable_fake_quant=5)
        model = SimpleModel()
        config = _weight_only_config(execution_mode=execution_mode, qat_schedule=schedule)
        quantizer = Quantizer(model, config)
        quantizer.prepare(_make_example_input())

        with quantizer.training_mode():
            assert _all_observers_enabled(quantizer._model)
            assert _all_fake_quant_disabled(quantizer._model)

    def test_exit_resets_and_reentry_restores(self, execution_mode):
        """Simulates a multi-epoch training loop: each iteration enters
        training_mode, steps forward, exits (state resets), and re-enters
        (schedule state restored from accumulated step count).
        """
        schedule = QATSchedule(
            enable_observer=0,
            enable_fake_quant=2,
            disable_observer=5,
        )
        model = SimpleModel()
        config = _weight_only_config(execution_mode=execution_mode, qat_schedule=schedule)
        quantizer = Quantizer(model, config)
        quantizer.prepare(_make_example_input())

        for epoch in range(3):
            with quantizer.training_mode():
                # On entry: schedule state applied from step_count
                step = quantizer._step_count
                expect_obs = step < 5
                expect_fq = step >= 2
                if expect_obs:
                    assert _all_observers_enabled(quantizer._model)
                else:
                    assert _all_observers_disabled(quantizer._model)
                if expect_fq:
                    assert _all_fake_quant_enabled(quantizer._model)
                else:
                    assert _all_fake_quant_disabled(quantizer._model)

                # Advance 2 steps per epoch
                quantizer.step()
                quantizer.step()

            # After exit: always obs OFF, fq ON
            assert _all_observers_disabled(quantizer._model), (
                f"Observer should be OFF after exiting epoch {epoch}"
            )
            assert _all_fake_quant_enabled(quantizer._model), (
                f"FQ should be ON after exiting epoch {epoch}"
            )

        # After all epochs (step_count=6): re-enter one more time
        assert quantizer._step_count == 6
        with quantizer.training_mode():
            # obs OFF (6 >= disable_observer=5), fq ON (6 >= fq=2)
            assert _all_observers_disabled(quantizer._model)
            assert _all_fake_quant_enabled(quantizer._model)

    def test_nested_raises(self, execution_mode):
        """Nested training_mode() raises RuntimeError."""
        model = SimpleModel()
        config = _weight_only_config(execution_mode=execution_mode)
        quantizer = Quantizer(model, config)
        quantizer.prepare(_make_example_input())

        with pytest.raises(RuntimeError, match="Nested"):
            with quantizer.training_mode():
                with quantizer.training_mode():
                    pass


# ---------------------------------------------------------------------------
# step() API
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("execution_mode", [ExecutionMode.EAGER, ExecutionMode.GRAPH])
class TestStep:
    """Tests for step() — counter, transitions, edge cases."""

    def _make_quantizer(self, execution_mode):
        schedule = QATSchedule(
            enable_observer=1,
            enable_fake_quant=2,
            disable_observer=5,
        )
        model = SimpleModel()
        config = _weight_only_config(execution_mode=execution_mode, qat_schedule=schedule)
        quantizer = Quantizer(model, config)
        quantizer.prepare(_make_example_input())
        return quantizer

    def test_increments_counter(self, execution_mode):
        """Each step() call increments _step_count by 1."""
        quantizer = self._make_quantizer(execution_mode)
        with quantizer.training_mode():
            for _ in range(4):
                quantizer.step()
        assert quantizer._step_count == 4

    def test_enables_observer_at_threshold(self, execution_mode):
        """Observer turns ON when step_count reaches enable_observer=1."""
        quantizer = self._make_quantizer(execution_mode)
        with quantizer.training_mode():
            assert _all_observers_disabled(quantizer._model)
            quantizer.step()  # step_count = 1
            assert _all_observers_enabled(quantizer._model)

    def test_enables_fake_quant_at_threshold(self, execution_mode):
        """Fake quant turns ON when step_count reaches enable_fake_quant=2."""
        quantizer = self._make_quantizer(execution_mode)
        with quantizer.training_mode():
            assert _all_fake_quant_disabled(quantizer._model)
            quantizer.step()  # step_count = 1
            assert _all_fake_quant_disabled(quantizer._model)
            quantizer.step()  # step_count = 2
            assert _all_fake_quant_enabled(quantizer._model)

    def test_disables_observer_at_threshold(self, execution_mode):
        """Observer turns OFF when step_count reaches disable_observer=5."""
        quantizer = self._make_quantizer(execution_mode)
        with quantizer.training_mode():
            for _ in range(5):
                quantizer.step()
            assert _all_observers_disabled(quantizer._model)

    def test_counter_monotonic_across_loops(self, execution_mode):
        """Step counter is never reset between training_mode loops."""
        quantizer = self._make_quantizer(execution_mode)
        with quantizer.training_mode():
            for _ in range(3):
                quantizer.step()
        assert quantizer._step_count == 3

        with quantizer.training_mode():
            for _ in range(2):
                quantizer.step()
        assert quantizer._step_count == 5

    def test_warns_when_no_schedule(self, execution_mode):
        """step() warns if no qat_schedule is configured."""
        model = SimpleModel()
        config = _weight_only_config(execution_mode=execution_mode, qat_schedule=None)
        quantizer = Quantizer(model, config)
        quantizer.prepare(_make_example_input())
        with quantizer.training_mode():
            with pytest.warns(UserWarning, match="no qat_schedule"):
                quantizer.step()

    def test_noop_when_no_schedule(self, execution_mode):
        """Without schedule, step() does not change observer/fq state."""
        model = SimpleModel()
        config = _weight_only_config(execution_mode=execution_mode, qat_schedule=None)
        quantizer = Quantizer(model, config)
        quantizer.prepare(_make_example_input())
        with quantizer.training_mode():
            assert _all_observers_enabled(quantizer._model)
            assert _all_fake_quant_enabled(quantizer._model)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                quantizer.step()
            assert _all_observers_enabled(quantizer._model)
            assert _all_fake_quant_enabled(quantizer._model)

    def test_outside_training_mode_raises(self, execution_mode):
        """step() outside training_mode() raises RuntimeError."""
        quantizer = self._make_quantizer(execution_mode)
        with pytest.raises(RuntimeError, match="training_mode"):
            quantizer.step()


# ---------------------------------------------------------------------------
# Low-level enable/disable APIs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("execution_mode", [ExecutionMode.EAGER, ExecutionMode.GRAPH])
class TestLowLevelAPIs:
    """Tests for enable/disable observer/fake_quant APIs (no schedule)."""

    def _make_quantizer(self, execution_mode):
        model = SimpleModel()
        config = _weight_only_config(execution_mode=execution_mode, qat_schedule=None)
        quantizer = Quantizer(model, config)
        quantizer.prepare(_make_example_input())
        return quantizer

    def test_enable_disable_observer(self, execution_mode):
        """enable/disable_observer toggles observer state globally."""
        quantizer = self._make_quantizer(execution_mode)
        quantizer.disable_observer()
        assert _all_observers_disabled(quantizer._model)
        quantizer.enable_observer()
        assert _all_observers_enabled(quantizer._model)

    def test_enable_disable_fake_quant(self, execution_mode):
        """enable/disable_fake_quant toggles fake quant state globally."""
        quantizer = self._make_quantizer(execution_mode)
        quantizer.disable_fake_quant()
        assert _all_fake_quant_disabled(quantizer._model)
        quantizer.enable_fake_quant()
        assert _all_fake_quant_enabled(quantizer._model)

    def test_enable_disable_on_submodule(self, execution_mode):
        """enable/disable observer and fake_quant scope to a specific submodule."""
        quantizer = self._make_quantizer(execution_mode)
        conv = dict(quantizer._model.named_modules())["conv"]
        conv_fqs = _get_fqs_for_module(quantizer, "conv")
        all_fqs = _get_fake_quant_modules(quantizer._model)
        non_conv_fqs = list(set(all_fqs) - set(conv_fqs))

        # Observer: disable all, re-enable on submodule only
        quantizer.disable_observer()
        quantizer.enable_observer(conv)
        assert _all_observers_enabled(conv_fqs)
        assert _all_observers_disabled(non_conv_fqs)

        # Fake quant: disable all, re-enable on submodule only
        quantizer.disable_fake_quant()
        quantizer.enable_fake_quant(conv)
        assert _all_fake_quant_enabled(conv_fqs)
        assert _all_fake_quant_disabled(non_conv_fqs)

    def test_all_blocked_when_schedule_configured(self, execution_mode):
        """All low-level APIs raise RuntimeError when a schedule is set."""
        schedule = QATSchedule(enable_observer=0, enable_fake_quant=5)
        model = SimpleModel()
        config = _weight_only_config(execution_mode=execution_mode, qat_schedule=schedule)
        quantizer = Quantizer(model, config)
        quantizer.prepare(_make_example_input())

        for api in [
            quantizer.enable_observer,
            quantizer.disable_observer,
            quantizer.enable_fake_quant,
            quantizer.disable_fake_quant,
        ]:
            with pytest.raises(RuntimeError, match="qat_schedule"):
                api()


# ---------------------------------------------------------------------------
# Per-module schedules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("execution_mode", [ExecutionMode.EAGER, ExecutionMode.GRAPH])
class TestPerModuleSchedule:
    """Tests for per-module schedules: module_type, module_name,
    precedence, propagation to children, and weight+activation FQs.
    """

    def test_different_thresholds_per_module_type(self, execution_mode):
        """Conv FQ enables at step 1; Linear FQ enables at step 5."""
        conv_schedule = QATSchedule(enable_observer=0, enable_fake_quant=1)
        lin_schedule = QATSchedule(enable_observer=0, enable_fake_quant=5)
        config = QuantizerConfig(
            global_config=None,
            module_type_configs={
                "torch.nn.modules.conv.Conv2d": ModuleQuantizerConfig(
                    op_state_spec={"weight": default_weight_quantization_spec()},
                    op_input_spec=None,
                    op_output_spec=None,
                    qat_schedule=conv_schedule,
                ),
                "torch.nn.modules.linear.Linear": ModuleQuantizerConfig(
                    op_state_spec={"weight": default_weight_quantization_spec()},
                    op_input_spec=None,
                    op_output_spec=None,
                    qat_schedule=lin_schedule,
                ),
            },
            execution_mode=execution_mode,
        )
        quantizer = Quantizer(SimpleModel(), config)
        quantizer.prepare(_make_example_input())

        with quantizer.training_mode():
            conv_fqs = _get_fqs_for_module(quantizer, "conv")
            lin_fqs = _get_fqs_for_module(quantizer, "linear")

            # step_count=0: both off
            assert _all_fake_quant_disabled(conv_fqs)
            assert _all_fake_quant_disabled(lin_fqs)

            quantizer.step()  # step_count = 1
            assert _all_fake_quant_enabled(conv_fqs), "Conv FQ should be ON at step 1"
            assert _all_fake_quant_disabled(lin_fqs), "Linear FQ should still be OFF at step 1"

            for _ in range(4):
                quantizer.step()  # step_count = 5
            assert _all_fake_quant_enabled(lin_fqs), "Linear FQ should be ON at step 5"

    def test_module_name_schedule(self, execution_mode):
        """Schedule set via module_name_configs applies to the named module."""
        schedule = QATSchedule(enable_observer=0, enable_fake_quant=3)
        config = QuantizerConfig(
            global_config=None,
            module_name_configs={
                "conv": ModuleQuantizerConfig(
                    op_state_spec={"weight": default_weight_quantization_spec()},
                    op_input_spec=None,
                    op_output_spec=None,
                    qat_schedule=schedule,
                ),
            },
            execution_mode=execution_mode,
        )
        quantizer = Quantizer(SimpleModel(), config)
        quantizer.prepare(_make_example_input())

        with quantizer.training_mode():
            conv_fqs = _get_fqs_for_module(quantizer, "conv")
            assert len(conv_fqs) > 0
            assert _all_fake_quant_disabled(conv_fqs)
            for _ in range(3):
                quantizer.step()
            assert _all_fake_quant_enabled(conv_fqs), "Conv FQ should be ON at step 3"

    def test_module_type_overrides_global(self, execution_mode):
        """Module-type schedule takes precedence; non-overridden modules
        use the global schedule.
        """
        global_schedule = QATSchedule(enable_observer=0, enable_fake_quant=10)
        conv_schedule = QATSchedule(enable_observer=0, enable_fake_quant=2)
        config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec={"weight": default_weight_quantization_spec()},
                op_input_spec=None,
                op_output_spec=None,
                qat_schedule=global_schedule,
            ),
            module_type_configs={
                "torch.nn.modules.conv.Conv2d": ModuleQuantizerConfig(
                    op_state_spec={"weight": default_weight_quantization_spec()},
                    op_input_spec=None,
                    op_output_spec=None,
                    qat_schedule=conv_schedule,
                ),
            },
            execution_mode=execution_mode,
        )
        quantizer = Quantizer(SimpleModel(), config)
        quantizer.prepare(_make_example_input())

        with quantizer.training_mode():
            quantizer.step()  # step_count = 1
            quantizer.step()  # step_count = 2

            # Conv: module-type schedule (fq=2) → ON
            conv_fqs = _get_fqs_for_module(quantizer, "conv")
            assert _all_fake_quant_enabled(conv_fqs), (
                "Conv should follow module-type schedule (fq at 2)"
            )

            # Linear: global schedule (fq=10) → still OFF
            lin_fqs = _get_fqs_for_module(quantizer, "linear")
            assert _all_fake_quant_disabled(lin_fqs), (
                "Linear should follow global schedule (fq at 10)"
            )

    def test_schedule_propagates_to_children(self, execution_mode):
        """Schedule on a parent module (block) propagates to its children's FQs."""
        schedule = QATSchedule(enable_observer=0, enable_fake_quant=3)
        config = QuantizerConfig(
            global_config=None,
            module_name_configs={
                "block": ModuleQuantizerConfig(
                    op_state_spec={"weight": default_weight_quantization_spec()},
                    op_input_spec=None,
                    op_output_spec=None,
                    qat_schedule=schedule,
                ),
            },
            execution_mode=execution_mode,
        )
        quantizer = Quantizer(NestedModel(), config)
        quantizer.prepare(_make_example_input())

        # block.0 is the Conv2d child — its FQs should follow block's schedule
        conv_name = "block.0"
        with quantizer.training_mode():
            conv_fqs = _get_fqs_for_module(quantizer, conv_name)
            assert len(conv_fqs) > 0, f"Expected FQ modules for {conv_name}"
            assert _all_fake_quant_disabled(conv_fqs)
            for _ in range(3):
                quantizer.step()
            assert _all_fake_quant_enabled(conv_fqs), (
                "Child conv FQ should follow parent block's schedule"
            )

    def test_schedule_applies_to_weight_and_activation_fqs(self, execution_mode):
        """Schedule controls both weight and activation FQ modules."""
        schedule = QATSchedule(enable_observer=0, enable_fake_quant=2)
        config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec={"weight": default_weight_quantization_spec()},
                op_input_spec={"*": default_activation_quantization_spec()},
                op_output_spec=None,
                qat_schedule=schedule,
            ),
            execution_mode=execution_mode,
        )
        quantizer = Quantizer(SimpleModel(), config)
        quantizer.prepare(_make_example_input())

        # Should have more FQ modules than weight-only (activation FQs too)
        all_fqs = _get_fake_quant_modules(quantizer._model)
        assert len(all_fqs) > 2, "Expected weight + activation FQ modules"

        with quantizer.training_mode():
            # step_count=0: all FQs should be OFF
            assert _all_fake_quant_disabled(quantizer._model)

            quantizer.step()  # step_count = 1
            assert _all_fake_quant_disabled(quantizer._model)

            quantizer.step()  # step_count = 2
            # ALL FQs (weight + activation) should be ON
            assert _all_fake_quant_enabled(quantizer._model)


# ---------------------------------------------------------------------------
# PT2E deduplication edge cases
# ---------------------------------------------------------------------------


def test_pt2e_schedule_controls_all_fqs_with_output_quant():
    """QAT schedule must control every FQ, including output-edge FQs.

    With output quantization enabled, the FQ on the final operation's
    output feeds into the graph ``output`` node. All FQ modules in the
    model should respect the configured schedule.
    """
    schedule = QATSchedule(enable_observer=0, enable_fake_quant=5)
    config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={"weight": default_weight_quantization_spec()},
            op_input_spec={"*": default_activation_quantization_spec()},
            op_output_spec={"*": default_activation_quantization_spec()},
            qat_schedule=schedule,
        ),
        execution_mode=ExecutionMode.GRAPH,
    )
    quantizer = Quantizer(SimpleModel(), config)
    quantizer.prepare(_make_example_input())

    all_fqs = _get_fake_quant_modules(quantizer._model)
    assert len(all_fqs) > 0

    with quantizer.training_mode():
        # Step 0 < enable_fake_quant=5: every FQ should have fake_quant OFF
        assert _all_fake_quant_disabled(all_fqs), (
            "All FQ modules should have fake_quant OFF at step 0 "
            "(below enable_fake_quant=5 threshold)"
        )


# TODO: respect PT2E FQ-node deduplication when mapping QAT schedules to activation FQ nodes.
@pytest.mark.xfail(
    reason=(
        "Graph mode does not propagate the conv output schedule to the "
        "deduplicated activation FQ on the conv→linear edge — the FQ follows "
        "the consumer's (linear's) global schedule instead."
    ),
)
def test_pt2e_output_quant_schedule_follows_producer():
    """In PT2E, when conv has op_output_spec with a schedule and global config
    has op_input_spec with no schedule, the deduplicated FQ on the intermediate
    edge should ideally follow the conv output schedule.

    The FQ between conv's output and linear's input is deduplicated by PT2E
    into a single node. Our _get_fake_quantize_modules maps it to linear
    (the consumer) via nn_module_stack. So the conv output schedule is not
    applied to this activation FQ — it follows the global (no schedule)
    behavior instead.
    """
    conv_schedule = QATSchedule(enable_observer=0, enable_fake_quant=5)
    config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={"weight": default_weight_quantization_spec()},
            op_input_spec={"*": default_activation_quantization_spec()},
            op_output_spec=None,
        ),
        module_type_configs={
            "torch.nn.modules.conv.Conv2d": ModuleQuantizerConfig(
                op_state_spec={"weight": default_weight_quantization_spec()},
                op_input_spec=None,
                op_output_spec={"*": default_activation_quantization_spec()},
                qat_schedule=conv_schedule,
            ),
        },
        execution_mode=ExecutionMode.GRAPH,
    )
    quantizer = Quantizer(SimpleModel(), config)
    quantizer.prepare(_make_example_input())

    # Walk the graph to find the FQ node after conv: conv -> ... -> FQ
    model = quantizer._model
    modules = dict(model.named_modules())
    conv_fq = None
    for node in model.graph.nodes:
        if node.op == "call_function" and "conv" in str(node.target):
            # Walk forward from conv until we hit an FQ call_module node
            queue = list(node.users)
            while queue:
                user = queue.pop(0)
                if user.op == "call_module" and isinstance(
                    modules.get(user.target), FakeQuantizeImplBase
                ):
                    conv_fq = modules[user.target]
                    break
                queue.extend(user.users)
            break

    assert conv_fq is not None, "Expected to find an FQ node after conv"

    with quantizer.training_mode():
        # Conv's output schedule says enable_fake_quant=5, so at step 0
        # this FQ should be OFF if the schedule were applied. But since
        # it's mapped to linear (global, no schedule), FQ is ON from start.
        assert conv_fq.fake_quant_enabled.item() == 0, (
            "Conv output FQ should follow conv's schedule (OFF at step 0), "
            "but it follows the consumer's global schedule instead"
        )
