# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Helpers for toggling fake quantization by quantization target."""

import torch

from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.quantization.spec.fake_quantize import FakeQuantizeImplBase


def disable_activation_fake_quant(module: torch.nn.Module) -> None:
    """Disable fake quantization on activation FakeQuantize modules only.

    Mirrors ``torchao.quantization.pt2e.fake_quantize.disable_fake_quant`` but
    skips weight FQ modules. Used by ``calibration_mode`` so activation
    observers see the effect of quantized weights when collecting statistics.

    Args:
        module (torch.nn.Module): Module to (possibly) toggle. No-op for any
            module that is not a ``FakeQuantizeImplBase`` whose
            ``quantization_target`` is ``ACTIVATION``.
    """
    if (
        isinstance(module, FakeQuantizeImplBase)
        and module.quantization_target == CompressionTargetTensor.ACTIVATION
    ):
        module.disable_fake_quant()


def enable_weight_fake_quant(module: torch.nn.Module) -> None:
    """Enable fake quantization on weight FakeQuantize modules only.

    Mirrors ``torchao.quantization.pt2e.fake_quantize.enable_fake_quant`` but
    skips activation FQ modules. Companion to
    :func:`disable_activation_fake_quant` used by ``calibration_mode``.

    Args:
        module (torch.nn.Module): Module to (possibly) toggle. No-op for any
            module that is not a ``FakeQuantizeImplBase`` whose
            ``quantization_target`` is ``WEIGHT``.
    """
    if (
        isinstance(module, FakeQuantizeImplBase)
        and module.quantization_target == CompressionTargetTensor.WEIGHT
    ):
        module.enable_fake_quant()
