# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Fake quantization implementation base class and default implementation."""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any

import torch
from torch.autograd import Function
from torchao.quantization.pt2e import FakeQuantizeBase

from coreai_opt._utils.spec_utils import (
    PartialConstructor as _PartialConstructor,
    with_args as _with_args,
)
from coreai_opt._utils.torch_utils import (
    get_n_bits_from_dtype as _get_n_bits_from_dtype,
    is_float4_dtype as _is_float4_dtype,
    is_float8_dtype as _is_float8_dtype,
    is_float_quant_dtype as _is_float_quant_dtype,
)
from coreai_opt.config.spec import CompressionSimulatorBase, CompressionTargetTensor
from coreai_opt.quantization._utils import get_quantization_shapes as _get_quantization_shapes
from coreai_opt.quantization.spec.errors import _BlockSizeMismatchError
from coreai_opt.quantization.spec.qscheme import QuantizationScheme

from .granularity import QuantizationGranularity
from .qformulation import QuantizationFormulation
from .qparams_calculator import QParamsCalculatorBase, StatelessQParamsCalculatorBase

__all__ = ["FakeQuantizeImplBase"]

logger = logging.getLogger(__name__)


class FakeQuantizeImplBase(CompressionSimulatorBase, FakeQuantizeBase):
    """
    Base class for implementing fake quantization
    """

    def __init__(
        self,
        dtype: torch.dtype,
        qformulation: QuantizationFormulation,
        granularity: QuantizationGranularity,
        target_dtype: torch.dtype,
        quant_min: int | float,
        quant_max: int | float,
        qparams_calculator: QParamsCalculatorBase,
        quantization_target: CompressionTargetTensor,
        n_bits: int | None = None,
        **kwargs,
    ):
        super().__init__()
        self.dtype = dtype
        self.qformulation = qformulation
        self._granularity = granularity
        self.target_dtype = target_dtype
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.qparams_calculator = qparams_calculator
        self.quantization_target = quantization_target
        self.register_buffer("_disabled", torch.tensor(False))

        # Infer n_bits from dtype if not provided
        if n_bits is None:
            n_bits = _get_n_bits_from_dtype(dtype)
        self.n_bits = n_bits

    @property
    def qscheme(self) -> QuantizationScheme:
        """The quantization scheme, delegated to the qparams_calculator."""
        return self.qparams_calculator.qscheme

    @property
    def granularity(self) -> QuantizationGranularity:
        """Getter for granularity."""
        return self._granularity

    @granularity.setter
    def granularity(self, granularity: QuantizationGranularity) -> None:
        """Update granularity for the fake quantize class and its qparams calculator.

        Can only be performed before the first forward pass.
        """
        self.qparams_calculator.granularity = granularity
        self._granularity = granularity

    def extra_repr(self) -> str:
        obs = "on" if self.observer_enabled.item() else "off"
        fq = "on" if self.fake_quant_enabled.item() else "off"
        return f"qformulation={self.qformulation}, observer={obs}, fake_quant={fq}"

    def is_disabled(self) -> bool:
        """Return True if fake quantization has been disabled."""
        return self._disabled.item()

    def disable_observer(self) -> None:
        """Disable the observer, unless the qparams calculator is stateless.

        Applies to **any** caller (direct, ``apply(disable_observer)``,
        ``convert_pt2e``, QAT scheduling). Stateless calculators recompute per
        forward and need ``observer_enabled=1`` permanently — ``forward`` uses
        that flag to route between live recompute and the stateful
        ``get_qparams()`` cache (which stateless doesn't have).
        """
        if isinstance(self.qparams_calculator, StatelessQParamsCalculatorBase):
            return
        super().disable_observer()

    def enable_observer(self, enabled: bool = True) -> None:
        """Inverse of ``disable_observer``: ignore ``enabled=False`` when the
        qparams calculator is stateless. Covers callers that invoke
        ``enable_observer(False)`` directly (e.g. the QAT scheduler at
        ``quantizer.py:_maybe_apply_qat_schedule``); ``disable_observer()``
        itself routes through the override above.
        """
        if not enabled and isinstance(self.qparams_calculator, StatelessQParamsCalculatorBase):
            return
        super().enable_observer(enabled)

    def _warn_and_disable(self, error: _BlockSizeMismatchError) -> None:
        """Log a warning and permanently disable this module."""
        logger.warning(
            "Tensor (target: %s) incompatible with block size "
            "configuration: %s. Skipping quantization.",
            self.quantization_target,
            error,
        )
        self._disabled.fill_(True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs fake quantization of the given tensor using the qparams
        (scale, zero point, minval) computed by the QParamsCalculator.
        """
        if self._disabled.item():
            return tensor

        if self.observer_enabled[0] == 1:
            # Call the forward function of the qparams_calculator
            # to collect observer statistics when the observer is
            # enabled
            # Use no_grad to prevent gradients flowing through the scale/zp computation path.
            # Gradients should be computed through the actual QDQ path only.
            with torch.no_grad():
                try:
                    scale, zero_point, minval = self.qparams_calculator(tensor)
                except _BlockSizeMismatchError as e:
                    self._warn_and_disable(e)
                    return tensor
        else:
            # When the observer is not enabled, call the get_qparams
            # function to retrieved the stored statistics
            scale, zero_point, minval = self.qparams_calculator.get_qparams()

        if self.fake_quant_enabled[0] == 1:
            # Cast incoming tensor to fp32 to perform qdq operations in high precision.
            # Cast the tensor to return back to the original dtype.
            orig_dtype = tensor.dtype
            tensor = tensor.to(torch.float32)
            return self._fused_fake_quant_dequant(tensor, scale, zero_point, minval).to(orig_dtype)

        return tensor

    @abstractmethod
    def quantize(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor | None,
        minval: torch.Tensor | None,
        cast_to_target_dtype: bool = True,
    ) -> torch.Tensor:
        """
        Given a tensor, scale and zero point, perform quantization of the tensor based
        on the configuration in the ``QuantizationSpec``.

        Args:
            tensor: The tensor to quantize
            scale: The scale to use for quantization
            zero_point: The zero point computed by the qparams calculator
                (None for floating-point dtypes).
            minval: The minimum representable float value of the observed
                range, computed by the qparams calculator
                (None for floating-point dtypes).
            cast_to_target_dtype: If True, the quantized tensor is cast to the target_dtype.
                Otherwise, the values of the tensor are quantized to appropriate bins but the dtype
                used to represent the quantized tensor remains the same as the original tensor.
                This allows fake quantization to capture the quantization error while allowing
                gradients to backpropagate.
        """
        pass

    @abstractmethod
    def dequantize(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor | None,
        minval: torch.Tensor | None,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Given a quantized tensor, the scale and zero point used to perform quantization,
        perform de-quantization of the tensor based on the configuration in the
        ``QuantizationSpec`` and return it as a tensor with dtype as ``output_dtype``.

        Args:
            tensor: The tensor to dequantize
            scale: The scale to use for dequantization
            zero_point: The zero point computed by the qparams calculator
                (None for floating-point dtypes).
            minval: The minimum representable float value of the observed
                range, computed by the qparams calculator
                (None for floating-point dtypes).
            output_dtype: The dtype to use for the dequantized tensor
        """
        pass

    @abstractmethod
    def _fused_fake_quant_dequant(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor | None,
        minval: torch.Tensor | None,
    ) -> torch.Tensor:
        """Fused quantize → dequantize as a single autograd node with STE gradient.

        Expects the input tensor to already be in fp32. Returns an fp32 tensor;
        the caller is responsible for casting to the desired output dtype.
        """
        pass

    @classmethod
    def with_args(cls, **kwargs: dict) -> _PartialConstructor[FakeQuantizeImplBase]:
        # This is needed for compatibility with torch prepare_pt2e
        fake_quant_constructor = _with_args(cls, **kwargs)

        # need to assign the correct module to fake_quantize
        # constructors to satisfy public v private requirements
        fake_quant_constructor.__module__ = f"{cls.__module__}.{cls.__name__}"
        return fake_quant_constructor

    def calculate_qparams(self) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Returns the computed (scale, zero_point, minval).
        ``zero_point`` and ``minval`` are None for floating-point dtypes.
        """
        return self.qparams_calculator.get_qparams()

    def set_export_mode(self, enabled: bool = True) -> None:
        """
        Set or unset export mode.
        """
        self.qparams_calculator.set_export_mode(enabled=enabled)

    def convert(self, model: torch.fx.GraphModule, observer_node: torch.fx.Node) -> None:
        """No-op: keep fake quant nodes intact during convert_pt2e.

        If this method is not present, torchao's convert method will try to replace
        fake quant nodes with its standard quantize/dequantize ops and fails in the process
        """


@FakeQuantizeImplBase.register("default")
class _DefaultFakeQuantizeImpl(FakeQuantizeImplBase):
    def _select_int_offsets(
        self,
        tensor: torch.Tensor,
        zero_point: torch.Tensor,
        minval: torch.Tensor,
        reduced_shape: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pick ``(quant_offset, float_offset)`` for the active integer formulation.

        ``quant_offset`` lives in the integer (quantized) domain.
        ``float_offset`` lives in the float domain and is returned in
        ``tensor.dtype``. Works uniformly for signed and unsigned integers.

        - ZP:     ``(zero_point, 0)``
        - MINVAL: ``(quant_min, minval)``
        """
        if self.qformulation == QuantizationFormulation.ZP:
            return (
                zero_point.view(reduced_shape),
                tensor.new_zeros(()),
            )
        if self.qformulation == QuantizationFormulation.MINVAL:
            return (
                tensor.new_full((), self.quant_min, dtype=zero_point.dtype),
                minval.view(reduced_shape).to(tensor.dtype),
            )
        raise NotImplementedError(f"Unknown qformulation: {self.qformulation}")

    def quantize(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor | None,
        minval: torch.Tensor | None,
        cast_to_target_dtype: bool = True,
    ) -> torch.Tensor:
        # Cast incoming tensor to fp32 to perform quantize operations in high precision.
        # Track the original dtype of the incoming tensor in case we need to cast the returning
        # tensor back (if cast_to_target_dtype is False).
        orig_dtype = tensor.dtype
        tensor = tensor.to(torch.float32)
        if _is_float_quant_dtype(self.dtype):
            assert zero_point is None, "zero_point must be None for floating-point quantization"
            assert minval is None, "minval must be None for floating-point quantization"
            quantized_tensor = self._quantize_float(tensor, scale)

        else:
            assert zero_point is not None, "zero_point must not be None for integer quantization"
            assert minval is not None, "minval must not be None for integer quantization"
            quantized_tensor = self._quantize_int(tensor, scale, zero_point, minval)

        output_dtype = self.target_dtype if cast_to_target_dtype else orig_dtype
        return quantized_tensor.to(output_dtype)

    def dequantize(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor | None,
        minval: torch.Tensor | None,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        # Cast incoming tensor to fp32 to perform dequantize operations in high precision.
        tensor = tensor.to(torch.float32)
        if _is_float_quant_dtype(self.target_dtype):
            assert zero_point is None, "zero_point must be None for floating-point dequantization"
            assert minval is None, "minval must be None for floating-point dequantization"
            return self._dequantize_float(tensor, scale, output_dtype)

        # Integer dequantization
        assert zero_point is not None, "zero_point must not be None for integer dequantization"
        assert minval is not None, "minval must not be None for integer dequantization"
        return self._dequantize_int(tensor, scale, zero_point, minval, output_dtype)

    def _quantize_int(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        minval: torch.Tensor,
    ) -> torch.Tensor:
        """
        Integer quantization. See :func:`_quantize_int` for the math; offsets are
        selected from ``self.qformulation`` via :meth:`_select_int_offsets`.

        This function quantizes the values in tensor but keeps the quantized tensor dtype in FP.
        """
        block_size = self.granularity.get_block_size(tensor.shape)
        original_shape, blockwise_shape, reduced_shape = _get_quantization_shapes(
            tensor, block_size
        )

        tensor = tensor.view(blockwise_shape)
        scale = scale.view(reduced_shape)
        quant_offset, float_offset = self._select_int_offsets(
            tensor, zero_point, minval, reduced_shape
        )

        quant, _ = _quantize_int(
            tensor, scale, quant_offset, float_offset, self.quant_min, self.quant_max
        )
        return quant.view(original_shape)

    def _dequantize_int(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        minval: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Integer dequantization. See :func:`_dequantize_int` for the math."""
        block_size = self.granularity.get_block_size(tensor.shape)
        original_shape, blockwise_shape, reduced_shape = _get_quantization_shapes(
            tensor, block_size
        )

        tensor = tensor.view(blockwise_shape)
        scale = scale.view(reduced_shape)
        quant_offset, float_offset = self._select_int_offsets(
            tensor, zero_point, minval, reduced_shape
        )

        dequant = _dequantize_int(tensor, scale, quant_offset, float_offset)
        return dequant.view(original_shape).to(output_dtype)

    def _quantize_float(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Floating-point quantization: cast_to_low_precision(clamp(input / scale, min, max))
        """
        block_size = self.granularity.get_block_size(tensor.shape)
        original_shape, blockwise_shape, reduced_shape = _get_quantization_shapes(
            tensor, block_size
        )

        tensor = tensor.view(blockwise_shape)
        scale = scale.view(reduced_shape)

        quantized_tensor, _ = _quantize_float(
            tensor, scale, self.quant_min, self.quant_max, self.dtype
        )

        return quantized_tensor.view(original_shape)

    def _dequantize_float(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Floating-point dequantization: input * scale"""
        block_size = self.granularity.get_block_size(tensor.shape)
        original_shape, blockwise_shape, reduced_shape = _get_quantization_shapes(
            tensor, block_size
        )

        tensor = tensor.view(blockwise_shape)
        scale = scale.view(reduced_shape)

        dequant = _dequantize_float(tensor, scale)
        return dequant.view(original_shape).to(output_dtype)

    def _fused_fake_quant_dequant(
        self,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor | None,
        minval: torch.Tensor | None,
    ) -> torch.Tensor:
        """Fused quantize → dequantize as a single autograd node with STE gradient.

        Dispatches to the int or float fused STE class based on self.dtype.
        """
        block_size = self.granularity.get_block_size(tensor.shape)
        original_shape, blockwise_shape, reduced_shape = _get_quantization_shapes(
            tensor, block_size
        )
        if _is_float_quant_dtype(self.dtype):
            return _FusedFakeQuantizeFloatSTE.apply(
                tensor,
                scale,
                self.quant_min,
                self.quant_max,
                self.dtype,
                original_shape,
                blockwise_shape,
                reduced_shape,
            )

        quant_offset, float_offset = self._select_int_offsets(
            tensor, zero_point, minval, reduced_shape
        )

        return _FusedFakeQuantizeIntSTE.apply(
            tensor,
            scale,
            quant_offset,
            float_offset,
            self.quant_min,
            self.quant_max,
            original_shape,
            blockwise_shape,
            reduced_shape,
        )


def _qdq_int(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    quant_offset: torch.Tensor,
    float_offset: torch.Tensor,
    quant_min: int,
    quant_max: int,
    original_shape: torch.Size,
    blockwise_shape: list[int],
    reduced_shape: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused quantize → dequantize for integer types.

    The quantized-dequantized tensor as well as a mask tensor marking positions in the tensor which
    were clamped are both returned.
    """
    tensor = tensor.view(blockwise_shape)
    scale = scale.view(reduced_shape)

    quantized, mask = _quantize_int(tensor, scale, quant_offset, float_offset, quant_min, quant_max)
    dequantized = _dequantize_int(quantized, scale, quant_offset, float_offset)

    return dequantized.view(original_shape).clone(), mask


class _FusedFakeQuantizeIntSTE(Function):
    """
    Fused fake quantize + dequantize for integer types with STE gradient.

    Handles blockwise reshaping internally so the entire fake-quantize operation
    (reshape → quantize → dequantize → reshape back) is a single autograd node.

    Fusing into one node reduces QAT memory: intermediate tensors (scaled, rounded,
    clamped) are local to forward and freed immediately instead of being retained by
    the autograd graph. Only a boolean mask (1 byte/element) is saved for backward,
    replacing multiple float32 intermediates (4 bytes/element each).
    """

    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        quant_offset: torch.Tensor,
        float_offset: torch.Tensor,
        quant_min: int,
        quant_max: int,
        original_shape: torch.Size,
        blockwise_shape: list[int],
        reduced_shape: list[int],
    ) -> torch.Tensor:
        dequantized, mask = _qdq_int(
            tensor,
            scale,
            quant_offset,
            float_offset,
            quant_min,
            quant_max,
            original_shape,
            blockwise_shape,
            reduced_shape,
        )
        ctx.save_for_backward(mask)
        ctx.original_shape = original_shape
        return dequantized

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None, None, None, None, None]:
        (mask,) = ctx.saved_tensors
        # Reshape grad to blockwise shape to apply mask, then reshape back
        grad_blockwise = grad_output.view(mask.shape)
        return (
            (grad_blockwise * mask).view(ctx.original_shape),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _qdq_float(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    quant_min: float,
    quant_max: float,
    dtype: torch.dtype,
    original_shape: torch.Size,
    blockwise_shape: list[int],
    reduced_shape: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused quantize → dequantize for float types.

    The quantized-dequantized tensor as well as a mask tensor marking positions in the tensor which
    were clamped are both returned.
    """
    tensor = tensor.view(blockwise_shape)
    scale = scale.view(reduced_shape)

    quantized, mask = _quantize_float(tensor, scale, quant_min, quant_max, dtype)
    dequantized = _dequantize_float(quantized, scale)

    return dequantized.view(original_shape).clone(), mask


class _FusedFakeQuantizeFloatSTE(Function):
    """
    Fused fake quantize + dequantize for float types with STE gradient.

    Handles blockwise reshaping internally so the entire fake-quantize operation
    (reshape → quantize → dequantize → reshape back) is a single autograd node.

    Fusing into one node reduces QAT memory: intermediate tensors (scaled, rounded,
    clamped) are local to forward and freed immediately instead of being retained by
    the autograd graph. Only a boolean mask (1 byte/element) is saved for backward,
    replacing multiple float32 intermediates (4 bytes/element each).
    """

    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        scale: torch.Tensor,
        quant_min: float,
        quant_max: float,
        dtype: torch.dtype,
        original_shape: torch.Size,
        blockwise_shape: list[int],
        reduced_shape: list[int],
    ) -> torch.Tensor:
        dequantized, mask = _qdq_float(
            tensor,
            scale,
            quant_min,
            quant_max,
            dtype,
            original_shape,
            blockwise_shape,
            reduced_shape,
        )
        ctx.save_for_backward(mask)
        ctx.original_shape = original_shape
        return dequantized

    @staticmethod
    def backward(
        ctx: Any, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None, None, None, None, None, None, None]:
        (mask,) = ctx.saved_tensors
        # Reshape grad to blockwise shape to apply mask, then reshape back
        grad_blockwise = grad_output.view(mask.shape)
        return (
            (grad_blockwise * mask).view(ctx.original_shape),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _quantize_int(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    quant_offset: torch.Tensor,
    float_offset: torch.Tensor,
    quant_min: int,
    quant_max: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Integer quantization:
        clamp(round((tensor - float_offset) / scale) + quant_offset, quant_min, quant_max)

    Generic form parameterized by two offsets so the same kernel handles both
    formulations:

    - ZP:     ``quant_offset = zero_point``, ``float_offset = 0``
    - MINVAL: ``quant_offset = quant_min``,  ``float_offset = minval``

    The quantized tensor remains in FP dtype.
    The quantized tensor as well as a mask tensor marking positions in the tensor which were clamped
    are both returned.
    """
    result = (tensor - float_offset) / scale
    result.round_()
    result.add_(quant_offset)
    mask = result >= quant_min
    mask &= result <= quant_max
    result.clamp_(quant_min, quant_max)
    return result, mask


def _dequantize_int(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    quant_offset: torch.Tensor,
    float_offset: torch.Tensor,
) -> torch.Tensor:
    """
    Integer dequantization:
        (tensor - quant_offset) * scale + float_offset

    Inverse of :func:`_quantize_int`. See that function's docstring for the
    ZP / MINVAL offset conventions.
    """
    return (tensor - quant_offset) * scale + float_offset


def _quantize_float(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    quant_min: float,
    quant_max: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Float quantization: cast_decast(clamp(tensor / scale, min, max))

    The quantized tensor as well as a mask tensor marking positions in the tensor which were clamped
    are both returned.
    """
    result = tensor / scale
    mask = result >= quant_min
    mask &= result <= quant_max
    result.clamp_(min=quant_min, max=quant_max)
    if _is_float8_dtype(dtype):
        return _fp8_forward(result, dtype), mask
    elif _is_float4_dtype(dtype):
        return _fp4_forward(result), mask
    else:
        raise ValueError(f"Expected float4/float8 dtype, got {dtype}")


def _fp8_forward(tensor: torch.Tensor, dtype: torch.dtype):
    # Hardcoding return dtype to torch.float32 - all callers of this private method already cast
    # the incoming tensor to float32.
    return tensor.to(dtype).to(torch.float32)


def _fp4_forward(tensor: torch.Tensor):
    """Perform tensor quantization for fp4 dtype"""
    from torchao.prototype.mx_formats.kernels import (  # noqa: PLC0415
        f4_unpacked_to_f32,
        f32_to_f4_unpacked,
    )

    fp4_bits = f32_to_f4_unpacked(tensor)
    return f4_unpacked_to_f32(fp4_bits)


def _dequantize_float(
    tensor: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Float dequantization: tensor * scale"""
    return tensor * scale
