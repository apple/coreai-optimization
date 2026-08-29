# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the activation export handler registries."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn

from coreai_opt.quantization import (
    _export_utils,
)
from coreai_opt.quantization._export_utils import (
    canonicalize_qparam_shape,
    get_activation_export_handler,
    register_eager_activation_export_handler,
    register_graph_activation_export_handler,
    validate_activation_export_supported,
)
from coreai_opt.quantization.config.quantization_config import ExecutionMode
from coreai_opt.quantization.spec.fake_quantize import FakeQuantizeImplBase
from coreai_opt.quantization.spec.granularity import (
    PerBlockGranularity,
    PerChannelGranularity,
    PerTensorGranularity,
)


@pytest.fixture(autouse=True)
def _restore_registries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give each test empty registries, and restore the originals afterwards.

    The registries are module-level dicts, so a test that registers a handler
    would otherwise change what every later test sees. Starting from empty rather
    than from the live dicts also keeps these assertions independent of what
    earlier files registered.
    """
    monkeypatch.setattr(_export_utils, "_GRAPH_ACTIVATION_EXPORT_HANDLERS", {})
    monkeypatch.setattr(_export_utils, "_EAGER_ACTIVATION_EXPORT_HANDLERS", {})


def test_graph_handler_is_returned_for_its_granularity_only() -> None:
    """Lookup matches on granularity type and leaves other granularities alone."""

    def handler(model: Any, node: Any, fake_quant_mod: Any) -> None:
        raise AssertionError("should not be called")

    register_graph_activation_export_handler(PerBlockGranularity, handler)

    graph = ExecutionMode.GRAPH
    assert get_activation_export_handler(PerBlockGranularity(block_size=32), graph) is handler
    assert get_activation_export_handler(PerTensorGranularity(), graph) is None
    assert get_activation_export_handler(PerChannelGranularity(axis=1), graph) is None


def test_eager_handler_is_returned_for_its_granularity_only() -> None:
    """The eager registry is independent of the graph one."""

    def handler(fake_quant_mod: Any) -> nn.Module:
        return nn.Identity()

    register_eager_activation_export_handler(PerBlockGranularity, handler)

    per_block = PerBlockGranularity(block_size=32)
    assert get_activation_export_handler(per_block, ExecutionMode.EAGER) is handler
    assert get_activation_export_handler(PerTensorGranularity(), ExecutionMode.EAGER) is None
    # Registering for eager must not register for graph.
    assert get_activation_export_handler(per_block, ExecutionMode.GRAPH) is None


def test_per_block_export_error_names_the_caller_alternatives() -> None:
    """With no handler registered, per-block activations fail with actionable options.

    This is reached from ``finalize``, after a model is prepared and calibrated, so the
    message has to tell a caller what to do -- switch granularity, switch backend, or
    activate an extension -- not only name the developer-facing registration hook.
    """
    # validate_activation_export_supported only reads dtype and granularity.
    fake_quant_mod = cast(
        FakeQuantizeImplBase,
        SimpleNamespace(dtype=torch.int8, granularity=PerBlockGranularity(block_size=32)),
    )

    with pytest.raises(ValueError, match="does not support PerBlockGranularity") as e:
        validate_activation_export_supported(fake_quant_mod)

    msg = str(e.value)
    for alternative in ("PerChannelGranularity", "ExportBackend._TORCH", "extension"):
        assert alternative in msg, f"the error should offer {alternative}: {msg}"


@pytest.mark.parametrize(
    ("qparam", "granularity", "expected_shape"),
    [
        (torch.ones(1, 1), PerTensorGranularity(), ()),
        (torch.ones(1, 4), PerChannelGranularity(axis=1), (4,)),
    ],
)
def test_canonicalize_still_handles_supported_granularities(
    qparam: torch.Tensor, granularity: Any, expected_shape: tuple[int, ...]
) -> None:
    """The built-in path is unchanged by the registry."""
    assert canonicalize_qparam_shape(qparam, granularity).shape == expected_shape
