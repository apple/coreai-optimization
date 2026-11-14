# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Helpers for toggling fake quantization by quantization target."""

import warnings

import torch

from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.quantization.spec.fake_quantize import FakeQuantizeImplBase
from coreai_opt.quantization.spec.qparams_calculator import RunningRangeMixin


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


def reset_activation_running_ranges(module: torch.nn.Module) -> None:
    """Mark the running range of activation running-range calculators as unset.

    ``prepare()`` observes ``example_inputs`` once to initialize qparams. Applied
    right after that pass, this keeps the resulting qparams usable but makes the
    first batch observed afterwards start the running range afresh, so
    ``example_inputs`` does not bias the ranges collected from calibration data.

    Args:
        module (torch.nn.Module): Module to (possibly) update. No-op for any
            module that is not an activation ``FakeQuantizeImplBase`` with a
            running-range qparams calculator, or whose ``float_range`` fixes both bounds.
    """
    if (
        isinstance(module, FakeQuantizeImplBase)
        and module.quantization_target == CompressionTargetTensor.ACTIVATION
        and isinstance(module.qparams_calculator, RunningRangeMixin)
        # With both float_range bounds fixed, the range never depends on the data and forward()
        # stops updating it once initialized, so a reset marker would never be cleared.
        and None in module.qparams_calculator.float_range
    ):
        module.qparams_calculator.reset_running_range()


def warn_unquantized_tensors(num_tensors: int) -> None:
    """Warn that ``prepare()`` left ``num_tensors`` tensors unquantized.

    Each such tensor is already logged when its FakeQuantize module is disabled; this
    one-line summary makes the outcome visible and points to those logs.
    """
    warnings.warn(
        f"prepare() left {num_tensors} tensor(s) unquantized because their shapes are "
        "incompatible with the block size configuration. See the WARNING logs of the "
        "'coreai_opt' logger for the affected tensors.",
        UserWarning,
        stacklevel=2,
    )
