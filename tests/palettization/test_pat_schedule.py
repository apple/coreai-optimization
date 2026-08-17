# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for PAT (palettization-aware training) schedule runtime behavior.

Covers the PATSchedule config object plus the step-based training control on
KMeansPalettizer: training_mode(), step(), and the _mode state machine shared
with calibration_mode().
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrize as P
from pydantic import ValidationError

from coreai_opt.base_model_compressor import _CompressorLifecycle
from coreai_opt.palettization import (
    KMeansPalettizer,
    KMeansPalettizerConfig,
    ModuleKMeansPalettizerConfig,
)
from coreai_opt.palettization.config import PATSchedule
from coreai_opt.palettization.kmeans.kmeans_fake_palettize import _KMeansFakePalettize
from coreai_opt.palettization.spec import default_weight_palettization_spec


class ToyModel(nn.Module):
    """Single palettizable Linear layer."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 8)

    def forward(self, x):
        return self.linear(x)


def _example_input() -> tuple[torch.Tensor]:
    return (torch.randn(2, 16),)


def _prepared_palettizer(
    schedule: PATSchedule | None = None,
) -> tuple[KMeansPalettizer, nn.Module]:
    """Build a palettizer over ToyModel (optionally with a PAT schedule) and
    prepare it. Returns ``(palettizer, prepared_model)``.
    """
    config = KMeansPalettizerConfig(
        global_config=ModuleKMeansPalettizerConfig(
            op_state_spec={"weight": default_weight_palettization_spec()},
            pat_schedule=schedule,
        )
    )
    palettizer = KMeansPalettizer(ToyModel(), config)
    prepared = palettizer.prepare(_example_input())
    return palettizer, prepared


def _fake_palett_modules(model: nn.Module) -> list[_KMeansFakePalettize]:
    """Return every _KMeansFakePalettize parametrization in the model."""
    modules = []
    for _, module in model.named_modules():
        if not P.is_parametrized(module):
            continue
        for parametrizations in module.parametrizations.values():
            for param in parametrizations:
                if isinstance(param, _KMeansFakePalettize):
                    modules.append(param)
    return modules


def _all_enabled(model: nn.Module) -> bool:
    return all(m.fake_palett_enabled[0].item() == 1 for m in _fake_palett_modules(model))


def _all_disabled(model: nn.Module) -> bool:
    return all(m.fake_palett_enabled[0].item() == 0 for m in _fake_palett_modules(model))


class TestPATSchedule:
    """Pure-logic tests for the PATSchedule config object."""

    def test_default_enable_fake_palettize_is_zero(self):
        assert PATSchedule().enable_fake_palettize == 0

    @pytest.mark.parametrize(
        "threshold, step_count, expected",
        [
            (0, 0, True),  # active immediately when threshold is 0
            (5, 4, False),  # before threshold
            (5, 5, True),  # exactly at threshold
            (5, 6, True),  # after threshold
        ],
    )
    def test_compute_state(self, threshold, step_count, expected):
        schedule = PATSchedule(enable_fake_palettize=threshold)
        assert schedule._compute_state(step_count) is expected

    def test_negative_threshold_rejected(self):
        with pytest.raises(ValidationError):
            PATSchedule(enable_fake_palettize=-1)

    def test_frozen(self):
        schedule = PATSchedule(enable_fake_palettize=3)
        with pytest.raises(ValidationError):
            schedule.enable_fake_palettize = 5


class TestTrainingMode:
    """Runtime behavior of KMeansPalettizer.training_mode()."""

    def test_training_mode_requires_prepared_model(self):
        config = KMeansPalettizerConfig(
            global_config=ModuleKMeansPalettizerConfig(
                op_state_spec={"weight": default_weight_palettization_spec()},
            )
        )
        palettizer = KMeansPalettizer(ToyModel(), config)
        with pytest.raises(RuntimeError, match="Model must be prepared"):
            with palettizer.training_mode():
                pass

    def test_entry_trains_exit_evals(self):
        palettizer, prepared = _prepared_palettizer()
        prepared.eval()
        with palettizer.training_mode():
            assert prepared.training is True
        assert prepared.training is False

    def test_entry_from_train_stays_train_on_exit(self):
        palettizer, prepared = _prepared_palettizer()
        prepared.train()
        with palettizer.training_mode():
            assert prepared.training is True
        assert prepared.training is True

    def test_default_no_schedule_stays_enabled(self):
        palettizer, prepared = _prepared_palettizer()
        with palettizer.training_mode():
            assert _all_enabled(prepared)

    def test_schedule_below_threshold_gates_off_on_entry(self):
        palettizer, prepared = _prepared_palettizer(PATSchedule(enable_fake_palettize=5))
        # prepare() leaves fake palettization enabled...
        assert _all_enabled(prepared)
        # ...but entering training_mode applies the schedule at step 0 (< 5).
        with palettizer.training_mode():
            assert _all_disabled(prepared)

    def test_nested_training_mode_raises(self):
        palettizer, _ = _prepared_palettizer()
        with palettizer.training_mode():
            with pytest.raises(RuntimeError, match="Cannot enter training_mode"):
                with palettizer.training_mode():
                    pass

    def test_calibration_inside_training_raises(self):
        palettizer, _ = _prepared_palettizer()
        with pytest.raises(RuntimeError, match="Cannot enter calibration_mode"):
            with palettizer.training_mode():
                with palettizer.calibration_mode(loss_fn=F.mse_loss):
                    pass

    def test_training_inside_calibration_raises(self):
        palettizer, prepared = _prepared_palettizer()
        (example,) = _example_input()
        with pytest.raises(RuntimeError, match="Cannot enter training_mode"):
            with palettizer.calibration_mode(loss_fn=F.mse_loss) as skm:
                skm.step(prepared(example), torch.randn(2, 8))
                with palettizer.training_mode():
                    pass

    def test_mode_restored_to_idle_on_exception(self):
        palettizer, _ = _prepared_palettizer()
        with pytest.raises(ValueError, match="boom"):
            with palettizer.training_mode():
                assert palettizer._lifecycle is _CompressorLifecycle.TRAINING
                raise ValueError("boom")
        assert palettizer._lifecycle is _CompressorLifecycle.IDLE


class TestStep:
    """Runtime behavior of KMeansPalettizer.step()."""

    def test_increments_counter(self):
        palettizer, _ = _prepared_palettizer()
        with palettizer.training_mode():
            assert palettizer._step_count == 0
            palettizer.step()
            assert palettizer._step_count == 1
            palettizer.step()
            assert palettizer._step_count == 2

    def test_outside_training_mode_raises(self):
        palettizer, _ = _prepared_palettizer()
        with pytest.raises(RuntimeError, match="must be called inside a training_mode"):
            palettizer.step()

    def test_counter_monotonic_across_loops(self):
        palettizer, _ = _prepared_palettizer()
        with palettizer.training_mode():
            palettizer.step()
            palettizer.step()
        assert palettizer._step_count == 2
        # A second training_mode() loop continues counting, never resets.
        with palettizer.training_mode():
            palettizer.step()
        assert palettizer._step_count == 3

    def test_crossing_threshold_enables_at_exact_step(self):
        palettizer, prepared = _prepared_palettizer(PATSchedule(enable_fake_palettize=2))
        with palettizer.training_mode():
            assert _all_disabled(prepared)  # step 0 < 2
            palettizer.step()
            assert _all_disabled(prepared)  # step 1 < 2
            palettizer.step()
            assert _all_enabled(prepared)  # step 2 == threshold

    def test_noop_without_schedule(self):
        palettizer, prepared = _prepared_palettizer()
        with palettizer.training_mode():
            palettizer.step()  # no schedule configured -> does not raise
            assert _all_enabled(prepared)
        assert palettizer._step_count == 1


class MultiLayerModel(nn.Module):
    """Conv2d + two Linear layers, for exercising per-module schedule resolution."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, padding=1)
        self.linear1 = nn.Linear(2 * 8 * 8, 8)
        self.linear2 = nn.Linear(8, 4)

    def forward(self, x):
        x = self.conv(x).flatten(1)
        return self.linear2(self.linear1(x))


def _fp_for(model: nn.Module, name: str) -> _KMeansFakePalettize:
    """Return the fake-palettize module parametrizing ``<name>.weight``."""
    module = model.get_submodule(name)
    for parametrizations in module.parametrizations.values():
        for param in parametrizations:
            if isinstance(param, _KMeansFakePalettize):
                return param
    raise AssertionError(f"no fake-palettize module for {name}")


class TestPerModuleSchedule:
    """Schedule resolution across the config hierarchy."""

    def test_name_over_type_over_global(self):
        """One test exercising all three levels: module_name_configs beats
        module_type_configs beats global_config.
        """

        def _module_config(threshold: int) -> ModuleKMeansPalettizerConfig:
            return ModuleKMeansPalettizerConfig(
                op_state_spec={"weight": default_weight_palettization_spec()},
                pat_schedule=PATSchedule(enable_fake_palettize=threshold),
            )

        config = KMeansPalettizerConfig(
            global_config=_module_config(3),  # conv falls through to here
            module_type_configs={nn.Linear: _module_config(2)},  # linear2
            module_name_configs={"linear1": _module_config(0)},  # linear1
        )
        palettizer = KMeansPalettizer(MultiLayerModel(), config)
        prepared = palettizer.prepare((torch.randn(2, 1, 8, 8),))

        fp_conv = _fp_for(prepared, "conv")
        fp_l1 = _fp_for(prepared, "linear1")
        fp_l2 = _fp_for(prepared, "linear2")

        with palettizer.training_mode():
            palettizer.step()  # step_count == 1
            # linear1 (name, threshold 0) active; linear2 (type, 2) and conv
            # (global, 3) not yet.
            assert fp_l1.fake_palett_enabled[0].item() == 1
            assert fp_l2.fake_palett_enabled[0].item() == 0
            assert fp_conv.fake_palett_enabled[0].item() == 0

            palettizer.step()  # step_count == 2
            # linear2 crosses its type-level threshold; conv's global threshold
            # (3) still far off -> confirms each layer used its own level.
            assert fp_l1.fake_palett_enabled[0].item() == 1
            assert fp_l2.fake_palett_enabled[0].item() == 1
            assert fp_conv.fake_palett_enabled[0].item() == 0

            palettizer.step()  # step_count == 3
            # conv's global threshold is crossed.
            assert fp_l1.fake_palett_enabled[0].item() == 1
            assert fp_l2.fake_palett_enabled[0].item() == 1
            assert fp_conv.fake_palett_enabled[0].item() == 1


class MixedModel(nn.Module):
    """One palettized Linear feeding a non-palettized Linear head."""

    def __init__(self):
        super().__init__()
        self.palettized = nn.Linear(16, 8, bias=False)
        self.head = nn.Linear(8, 4)

    def forward(self, x):
        return self.head(self.palettized(x))


class TestDefaultStrategyTraining:
    """The default training strategy's behavior inside a training_mode() loop."""

    @pytest.mark.parametrize("use_training_mode_ctx", [True, False])
    def test_gradient_flow_with_and_without_training_mode_context(self, use_training_mode_ctx):
        """Training-time gradients must flow through the palettized layer whether
        or not the training_mode() context is active: the input and downstream
        params receive gradients while the frozen palettized weight receives none
        (no unintended gradient path).
        """
        config = KMeansPalettizerConfig(
            module_name_configs={
                "palettized": ModuleKMeansPalettizerConfig(
                    op_state_spec={"weight": default_weight_palettization_spec()}
                )
            },
            global_config=None,  # only the named layer is palettized
        )
        palettizer = KMeansPalettizer(MixedModel(), config)
        prepared = palettizer.prepare((torch.randn(2, 16),))

        x = torch.randn(2, 16, requires_grad=True)
        if use_training_mode_ctx:
            with palettizer.training_mode():
                prepared(x).sum().backward()
        else:
            prepared.train()
            prepared(x).sum().backward()

        # frozen palettized weight gets no gradient (no unintended path)
        assert prepared.palettized.parametrizations.weight.original.grad is None
        # graph intact: gradients reach the input (through the palettized layer) and downstream
        assert x.grad is not None
        assert prepared.head.weight.grad is not None
