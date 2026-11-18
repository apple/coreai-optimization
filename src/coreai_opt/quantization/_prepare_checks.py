# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from torch import nn

from coreai_opt.base_model_compressor import _BaseModelCompressor

if TYPE_CHECKING:
    from coreai_opt.quantization.config.quantization_config import QuantizerConfig

PrepareCheck = Callable[[nn.Module, "QuantizerConfig"], None]

_PREPARE_CHECKS: list[PrepareCheck] = []


def register_prepare_check(check: PrepareCheck) -> None:
    """Run ``check`` on every model ``Quantizer.prepare`` prepares from now on.

    The registry is process-global. Registering the same check twice has no effect.

    Args:
        check (PrepareCheck): Called with the prepared model and the ``QuantizerConfig`` it was
            prepared with; raises to reject the model.
    """
    if check not in _PREPARE_CHECKS:
        _PREPARE_CHECKS.append(check)


def run_prepare_checks(model: nn.Module, config: QuantizerConfig) -> None:
    """Run every registered check on the prepared ``model``, in registration order.

    In eager mode ``prepare`` has already modified ``model`` in place, so a rejected model must be
    prepared again from the original.

    Args:
        model (nn.Module): The prepared model (an ``fx.GraphModule`` in graph mode).
        config (QuantizerConfig): The config ``model`` was prepared with.

    Raises:
        RuntimeError: If ``model`` has not been prepared.
        Exception: Whatever the first failing check raises.
    """
    if not _BaseModelCompressor._is_model_prepared(model):
        raise RuntimeError("Model checks run on a prepared model; call Quantizer.prepare first.")
    for check in _PREPARE_CHECKS:
        check(model, config)
