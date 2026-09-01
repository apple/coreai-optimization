# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Shared utilities for quantization export preparation.

This module provides common utilities used across different execution modes (graph, eager) and
export formats (MIL, MLIR).
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.quantization.spec.fake_quantize import FakeQuantizeImplBase
from coreai_opt.quantization.spec.granularity import (
    PerBlockGranularity,
    PerTensorGranularity,
    QuantizationGranularity,
)

from .spec.qformulation import QuantizationFormulation

# Default zero point values for different dtypes
_DEFAULT_UINT8_ZERO_POINT = 128
_DEFAULT_INT8_ZERO_POINT = 0
_FP4_EXPORT_BLOCK_SIZE = 32


class _MILActivationQuantizeModule(nn.Module):
    """Activation quantize module for MIL export.

    Applies torch.quantize_per_tensor or torch.quantize_per_channel depending
    on whether axis is provided. Scale and zero_point are stored as
    non-trainable persistent buffers.

    Buffers:
        scale: Scale tensor for quantization (scalar for per-tensor, 1D for per-channel)
        zero_point: Zero point tensor for quantization

    Attributes:
        dtype: Quantized dtype (torch.qint8 or torch.quint8)
        axis: Channel axis for per-channel quantization, or None for per-tensor

    """

    scale: torch.Tensor
    zero_point: torch.Tensor
    dtype: torch.dtype
    axis: int | None

    def __init__(
        self,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        dtype: torch.dtype,
        axis: int | None = None,
    ) -> None:
        super().__init__()
        # For per-channel, scale/zero_point must be 1D tensors
        # https://docs.pytorch.org/docs/stable/generated/torch.quantize_per_channel.html
        if axis is not None:
            # Squeeze to 1D for per-channel quantization
            scale = scale.flatten()
            zero_point = zero_point.flatten()
        self.register_buffer("scale", scale.detach().clone())
        self.register_buffer("zero_point", zero_point.detach().clone())
        self.dtype = dtype
        self.axis = axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize input tensor.

        Args:
            x: Input tensor

        Returns:
            Quantized tensor

        """
        if self.axis is not None:
            return torch.quantize_per_channel(x, self.scale, self.zero_point, self.axis, self.dtype)
        return torch.quantize_per_tensor(x, self.scale, self.zero_point, self.dtype)


class _MILActivationDequantizeModule(nn.Module):
    """Activation dequantize module for MIL export.

    Applies .dequantize() to dequantize activations.

    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dequantize input tensor.

        Args:
            x: Quantized tensor

        Returns:
            Dequantized float tensor

        """
        return x.dequantize()


def create_mil_act_quant_seq(
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    dtype: torch.dtype,
    axis: int | None = None,
) -> nn.Sequential:
    """Create Sequential module with quantize and dequantize operations for MIL.

    Args:
        scale: Scale tensor for quantization
        zero_point: Zero point tensor for quantization
        dtype: Quantized dtype (torch.qint8 or torch.quint8)
        axis: Channel axis for per-channel quantization, or None for per-tensor

    Returns:
        Sequential module containing quantize and dequantize operations

    """
    return nn.Sequential(
        OrderedDict(
            [
                (
                    "quantize",
                    _MILActivationQuantizeModule(
                        scale=scale,
                        zero_point=zero_point,
                        dtype=dtype,
                        axis=axis,
                    ),
                ),
                ("dequantize", _MILActivationDequantizeModule()),
            ],
        ),
    )


def canonicalize_qparam_shape(
    qparam: torch.Tensor, granularity: QuantizationGranularity
) -> torch.Tensor:
    """Canonicalize quantization parameter to 0-D (per-tensor) or 1-D (per-channel).

    MLIR custom ops accept scale/zero_point tensors that are 0-D, 1-D, or
    whose rank matches the input tensor rank. Shared observers may produce
    higher-rank scales (e.g. ``[1,1,1,1]`` for per-tensor on a 4-D tensor) that
    don't match the rank of the actual input after reshape/flatten ops. This function squeezes
    them to canonical 0-D or 1-D shapes that are valid regardless of input rank.

    Args:
        qparam (torch.Tensor): Scale or zero_point tensor.
        granularity (QuantizationGranularity): Quantization granularity.

    Returns:
        torch.Tensor: 0-D scalar if per-tensor, 1-D if per-channel.

    Raises:
        ValueError: If granularity is per-block (multi-dimensional scales
            cannot be canonicalized to 0-D or 1-D).

    """
    if isinstance(granularity, PerBlockGranularity):
        msg = "Per-block granularity cannot be canonicalized to 0-D or 1-D."
        raise ValueError(msg)

    if isinstance(granularity, PerTensorGranularity):
        return qparam.squeeze()
    return qparam.flatten()


def extract_quantization_params(
    fake_quant_mod: FakeQuantizeImplBase,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Extract and prepare quantization parameters from fake quantization module.

    Args:
        fake_quant_mod: The fake quantization module

    Returns:
        Tuple of (scale, zero_point, minval) tensors.

    """
    scale, zero_point, minval = fake_quant_mod.calculate_qparams()

    if zero_point is not None:
        zero_point = zero_point.to(fake_quant_mod.target_dtype)

    return scale, zero_point, minval


def select_export_qparams_by_formulation(
    fake_quant_mod: FakeQuantizeImplBase,
    zero_point: torch.Tensor | None,
    minval: torch.Tensor | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Pick which of ``(zero_point, minval)`` the export carries for the
    active formulation.

    - ZP: keeps ``zero_point``, drops ``minval``. The runtime infers
      ``float_offset = 0``.
    - MINVAL: keeps ``minval``, drops ``zero_point``. The runtime derives
      ``quant_min`` automatically.

    For floating-point dtypes both will already be ``None``
    """

    if fake_quant_mod.qformulation == QuantizationFormulation.ZP:
        return zero_point, None
    if fake_quant_mod.qformulation == QuantizationFormulation.MINVAL:
        return None, minval
    raise NotImplementedError(f"Unknown qformulation: {fake_quant_mod.qformulation}")


def validate_qformulation_for_mil_export(fake_quant_mod: FakeQuantizeImplBase) -> None:
    """Reject CoreML export for non-ZP quantization formulations.

    CoreML only supports the zero-point formulation. Other
    formulations (e.g. MINVAL) must be exported through the CoreAI path.
    """

    if fake_quant_mod.qformulation != QuantizationFormulation.ZP:
        raise ValueError(
            f"CoreML export does not support qformulation="
            f"{fake_quant_mod.qformulation}. Set qformulation="
            f"QuantizationFormulation.ZP in the QuantizationSpec, or use the "
            f"CoreAI export path."
        )


def convert_dtype_for_torch_quantize(
    dtype: torch.dtype,
    zero_point: torch.Tensor | None,
) -> tuple[torch.dtype, torch.Tensor]:
    """Convert coreai-opt dtype to PyTorch quantized dtype and handle zero point.

    Args:
        dtype: The dtype from FakeQuantizeImplBase
        zero_point: The zero point tensor (may be None)

    Returns:
        Tuple of (converted_dtype, zero_point_tensor)

    Raises:
        ValueError: If dtype is not supported for MIL activation quantization

    """
    converted_zero_point = zero_point
    if dtype == torch.uint8:
        converted_dtype = torch.quint8
        if converted_zero_point is None:
            converted_zero_point = torch.tensor(_DEFAULT_UINT8_ZERO_POINT)
    elif dtype == torch.int8:
        converted_dtype = torch.qint8
        if converted_zero_point is None:
            converted_zero_point = torch.tensor(_DEFAULT_INT8_ZERO_POINT)
    else:
        msg = (
            f"Unsupported dtype {dtype} for MIL activation quantization. "
            f"Only torch.uint8 and torch.int8 are supported."
        )
        raise ValueError(msg)

    return converted_dtype, converted_zero_point


def is_module_fake_quant_target(
    module: nn.Module,
    target: CompressionTargetTensor,
) -> bool:
    """Check if a module is a fake quantization module targeting a specific tensor type.

    Args:
        module: The module to check
        target: The target tensor type (WEIGHT or ACTIVATION)

    Returns:
        True if module is a FakeQuantizeImplBase targeting the specified tensor type

    """
    return isinstance(module, FakeQuantizeImplBase) and module.quantization_target == target


def pack_fp4_to_float4tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Pack FP4-quantized values into Float4Tensor (uint8-backed, two values per byte).

    Args:
        tensor (torch.Tensor): Tensor whose values are representable in FP4.
            The last dimension must be even.

    Returns:
        Float4Tensor: Float4Tensor wrapping packed uint8 data.
    """
    from coreai_torch._compression._floatx import Float4Tensor  # noqa: PLC0415
    from torchao.prototype.mx_formats.kernels import f32_to_f4_unpacked  # noqa: PLC0415

    unpacked = f32_to_f4_unpacked(tensor.to(torch.float32))
    low = unpacked[..., 0::2]
    high = unpacked[..., 1::2]
    return Float4Tensor((high << 4) | low)


def validate_fp4_export(
    fake_quant_mod: FakeQuantizeImplBase,
    quantized_data: torch.Tensor | None = None,
) -> None:
    """Validate that FP4 quantization config is supported for MLIR export.

    FP4 export requires weight-only quantization with ``PerBlockGranularity``
    that resolves to blocks of size 32 along the weight's last axis and no
    blocking along any other axis (i.e. resolved block sizes
    ``(1, ..., 1, 32)``). This covers 2D Linear weights, which resolve to
    ``(1, 32)``, and higher-rank weights such as MoE experts that resolve to
    ``(1, ..., 1, 32)``.

    Args:
        fake_quant_mod (FakeQuantizeImplBase): The fake quantization module to validate.
        quantized_data (torch.Tensor | None): The quantized weight tensor used
            to resolve per-axis block sizes against the actual weight shape.

    Raises:
        ValueError: If the FP4 configuration is not supported for export.
    """
    if fake_quant_mod.quantization_target != CompressionTargetTensor.WEIGHT:
        raise ValueError(
            "FP4 quantization is only supported for weights during MLIR export. "
            f"Got quantization_target={fake_quant_mod.quantization_target}."
        )

    granularity = fake_quant_mod._granularity
    if not isinstance(granularity, PerBlockGranularity):
        raise ValueError(
            f"FP4 quantization requires PerBlockGranularity. Got {type(granularity).__name__}."
        )

    if quantized_data is None:
        return

    resolved_block_size = granularity.get_block_size(quantized_data.shape)
    expected_block_size = (1,) * (quantized_data.ndim - 1) + (_FP4_EXPORT_BLOCK_SIZE,)
    if resolved_block_size != expected_block_size:
        raise ValueError(
            f"FP4 export requires per-axis block sizes {expected_block_size} for a "
            f"{quantized_data.ndim}D weight (blocks of {_FP4_EXPORT_BLOCK_SIZE} along "
            f"the last axis, no blocking elsewhere). Got resolved block sizes "
            f"{resolved_block_size} from granularity={granularity!r}."
        )
