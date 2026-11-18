# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the checks ``Quantizer.prepare`` runs on the model it prepares."""

import pytest
import torch
from torch import fx, nn

from coreai_opt.quantization import Quantizer, QuantizerConfig, _prepare_checks
from coreai_opt.quantization._prepare_checks import register_prepare_check
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


def _prepare(execution_mode: ExecutionMode) -> nn.Module:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4)).eval()
    config = QuantizerConfig.presets.w8(execution_mode=execution_mode)
    return Quantizer(model, config).prepare((torch.randn(2, 16),))


@pytest.mark.parametrize("execution_mode", MODES)
def test_a_check_receives_the_model_prepare_returns(execution_mode: ExecutionMode) -> None:
    seen: list[nn.Module] = []
    register_prepare_check(seen.append)

    prepared = _prepare(execution_mode)

    assert seen == [prepared]
    assert isinstance(prepared, fx.GraphModule) == (execution_mode == ExecutionMode.GRAPH)


@pytest.mark.parametrize("execution_mode", MODES)
def test_a_failing_check_makes_prepare_raise(execution_mode: ExecutionMode) -> None:
    def reject(model: nn.Module) -> None:
        raise ValueError("rejected by the check")

    register_prepare_check(reject)

    with pytest.raises(ValueError, match="rejected by the check"):
        _prepare(execution_mode)


def test_checks_run_in_registration_order_and_stop_at_the_first_failure() -> None:
    calls: list[str] = []

    def first(model: nn.Module) -> None:
        calls.append("first")
        raise ValueError("first")

    def second(model: nn.Module) -> None:
        calls.append("second")

    register_prepare_check(first)
    register_prepare_check(second)

    with pytest.raises(ValueError, match="first"):
        _prepare(ExecutionMode.EAGER)
    assert calls == ["first"]


def test_a_check_registered_twice_runs_once() -> None:
    calls: list[nn.Module] = []
    register_prepare_check(calls.append)
    register_prepare_check(calls.append)

    _prepare(ExecutionMode.EAGER)

    assert len(calls) == 1
