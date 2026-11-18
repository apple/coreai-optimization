# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the checks ``Quantizer.prepare`` runs on the model it prepares."""

import pytest
import torch
from torch import fx, nn

from coreai_opt.quantization import Quantizer, QuantizerConfig, _prepare_checks
from coreai_opt.quantization._prepare_checks import register_prepare_check, run_prepare_checks
from coreai_opt.quantization.config.quantization_config import ExecutionMode

MODES = [ExecutionMode.EAGER, ExecutionMode.GRAPH]


@pytest.fixture(autouse=True)
def _restore_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give each test an empty registry, and restore the original afterwards.

    The registry is a module-level list, so a test that registers a check would otherwise
    change what every later test sees. Starting from empty also keeps these assertions
    independent of what earlier files, or an installed plugin, registered.
    """
    monkeypatch.setattr(_prepare_checks, "_PREPARE_CHECKS", [])


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4)).eval()


def _prepare(execution_mode: ExecutionMode) -> nn.Module:
    config = QuantizerConfig.presets.w8(execution_mode=execution_mode)
    return Quantizer(_model(), config).prepare((torch.randn(2, 16),))


@pytest.mark.parametrize("execution_mode", MODES)
def test_a_check_receives_the_prepared_model_and_its_config(
    execution_mode: ExecutionMode,
) -> None:
    seen: list[tuple[nn.Module, QuantizerConfig]] = []
    register_prepare_check(lambda model, config: seen.append((model, config)))
    config = QuantizerConfig.presets.w8(execution_mode=execution_mode)

    prepared = Quantizer(_model(), config).prepare((torch.randn(2, 16),))

    assert len(seen) == 1
    assert seen[0][0] is prepared and seen[0][1] is config
    assert isinstance(prepared, fx.GraphModule) == (execution_mode == ExecutionMode.GRAPH)


def test_checks_refuse_a_model_that_was_not_prepared() -> None:
    register_prepare_check(lambda model, config: None)

    with pytest.raises(RuntimeError, match="prepared model"):
        run_prepare_checks(_model(), QuantizerConfig())


@pytest.mark.parametrize("execution_mode", MODES)
def test_a_failing_check_makes_prepare_raise(execution_mode: ExecutionMode) -> None:
    def reject(model: nn.Module, config: QuantizerConfig) -> None:
        raise ValueError("rejected by the check")

    register_prepare_check(reject)

    with pytest.raises(ValueError, match="rejected by the check"):
        _prepare(execution_mode)


def test_checks_run_in_registration_order_and_stop_at_the_first_failure() -> None:
    calls: list[str] = []

    def first(model: nn.Module, config: QuantizerConfig) -> None:
        calls.append("first")
        raise ValueError("first")

    def second(model: nn.Module, config: QuantizerConfig) -> None:
        calls.append("second")

    register_prepare_check(first)
    register_prepare_check(second)

    with pytest.raises(ValueError, match="first"):
        _prepare(ExecutionMode.EAGER)
    assert calls == ["first"]


def test_a_check_registered_twice_runs_once() -> None:
    calls: list[nn.Module] = []

    def record(model: nn.Module, config: QuantizerConfig) -> None:
        calls.append(model)

    register_prepare_check(record)
    register_prepare_check(record)

    _prepare(ExecutionMode.EAGER)

    assert len(calls) == 1
