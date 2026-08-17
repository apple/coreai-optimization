# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Weight quantization pass for Core AI AIProgram."""

from __future__ import annotations

import logging
from typing import Any, cast

import ml_dtypes
import numpy as np

from coreai_opt.coreai_utils._coreai_imports import (
    AIProgram as _AIProgram,
    DenseElementsAttr as _DenseElementsAttr,
    DenseResourceElementsAttr as _DenseResourceElementsAttr,
    FloatAttr as _FloatAttr,
    InsertionPoint as _InsertionPoint,
    IntegerType as _IntegerType,
    RankedTensorType as _RankedTensorType,
    WalkResult as _WalkResult,
    _get_constant_value_as_np_array,
    compression_types as _compression_types,
    coreai as _coreai,
)
from coreai_opt.coreai_utils._utils.graph_utils import (
    _apply_compression_transform,
    _create_constant_value_from_np_array,
    _infer_quantization_block_sizes,
    _should_compress_op,
)
from coreai_opt.coreai_utils._utils.quantize_utils import _compute_qparams_by_dtype
from coreai_opt.coreai_utils._utils.type_utils import (
    _get_fp_mlir_and_ml_dtype,
    _get_scale_mlir_and_np_dtype,
    _get_string_to_mlir_type,
)
from coreai_opt.coreai_utils.common import CompressionGranularity, DType, QScheme
from coreai_opt.coreai_utils.passes import _OPS_WEIGHT_NEED_COMPRESSION

logger = logging.getLogger(__name__)

_FLOAT_DTYPE_TO_ML = {
    DType.FP4_E2M1FN: ml_dtypes.float4_e2m1fn,
    DType.FP8_E4M3FN: ml_dtypes.float8_e4m3fn,
    DType.FP8_E5M2: ml_dtypes.float8_e5m2,
}

# Valid dtypes for the weight being quantized.
_VALID_WEIGHT_DTYPES: frozenset[DType] = frozenset(
    {
        DType.FP4_E2M1FN,
        DType.FP8_E4M3FN,
        DType.FP8_E5M2,
        DType.INT2,
        DType.INT4,
        DType.INT8,
        DType.UINT2,
        DType.UINT4,
        DType.UINT8,
    }
)

# Valid non-None scale_dtype values for FP8 weight dtypes.
_VALID_FP8_SCALE_DTYPES: frozenset[DType] = frozenset({DType.FP8_E8M0FNU})


def _create_int_quantized_weight(
    quantized_data: Any,
    scale: Any,
    zero_point: Any,
    quantized_mlir_type: Any,
    weight_element_type: Any,
) -> Any:
    """Create the IR value for integer-quantized weights via blockwise_shift_scale."""
    quantized_data_val = _create_constant_value_from_np_array(quantized_data, quantized_mlir_type)
    scale_val = _create_constant_value_from_np_array(scale, weight_element_type)
    zero_point_val = _create_constant_value_from_np_array(zero_point, quantized_mlir_type)
    return _coreai.blockwise_shift_scale(
        data=quantized_data_val,
        scale=scale_val,
        offset1=zero_point_val,
        offset2=_create_constant_value_from_np_array(
            np.zeros_like(scale),
            weight_element_type,
        ),
    )


def _create_fp_quantized_weight(
    quantized_data: Any,
    scale: Any,
    fp_mlir_type: Any,
    weight_element_type: Any,
    scale_mlir_type: Any = None,
    scale_np_dtype: Any = None,
) -> Any:
    """Create the IR value for FP-quantized weights via blockwise_shift_scale.

    When ``scale_mlir_type`` and ``scale_np_dtype`` are provided (FP8_E8M0FNU scale),
    the scale constant is created in that dtype using DenseResourceElementsAttr.
    Otherwise the scale uses the uncompressed weight element type (default behavior).
    """
    tensor_type = _RankedTensorType.get(list(quantized_data.shape), fp_mlir_type)
    data_attr = _DenseResourceElementsAttr.get_from_buffer(
        quantized_data,
        "dense_resource",
        tensor_type,
    )
    quantized_data_val = cast("Any", _coreai.ConstantOp(value=data_attr).result)

    if scale_mlir_type is not None:
        scale_cast = np.ascontiguousarray(scale.astype(scale_np_dtype))
        scale_shape = list(scale_cast.shape)
        scale_tensor_type = _RankedTensorType.get(scale_shape, scale_mlir_type)
        scale_attr = _DenseResourceElementsAttr.get_from_buffer(
            scale_cast, "dense_resource", scale_tensor_type
        )
        scale_val = cast("Any", _coreai.ConstantOp(value=scale_attr).result)

        # offset2 must be in the output (weight) dtype so blockwise_shift_scale
        # returns a tensor in the original weight precision, not f8E8M0FNU.
        offset2_val = _create_constant_value_from_np_array(
            np.zeros(scale_cast.shape, dtype=np.float32),
            weight_element_type,
        )
    else:
        scale_val = _create_constant_value_from_np_array(scale, weight_element_type)
        offset2_val = _create_constant_value_from_np_array(
            np.zeros_like(scale),
            weight_element_type,
        )

    zero_point_tensor_type = _RankedTensorType.get(list(scale.shape), fp_mlir_type)
    zero_point_attr = _DenseElementsAttr.get_splat(
        zero_point_tensor_type, _FloatAttr.get(fp_mlir_type, 0.0)
    )
    zero_point_val = cast("Any", _coreai.ConstantOp(value=zero_point_attr).result)

    return _coreai.blockwise_shift_scale(
        data=quantized_data_val,
        scale=scale_val,
        offset1=zero_point_val,
        offset2=offset2_val,
    )


def quantize_weights(
    coreai_program: _AIProgram,
    dtype: DType,
    qscheme: QScheme = QScheme.SYMMETRIC,
    granularity: CompressionGranularity = CompressionGranularity.PER_CHANNEL,
    block_size: int = 32,
    weight_num_threshold: int = 1024,
    scale_dtype: DType | None = None,
    in_place: bool = False,
) -> _AIProgram:
    """Quantize weights in a Core AI AIProgram (MLIR<CoreAI> IR) by using Core AI ops.

    Walks through the IR and quantizes each coreai.constant op that needs to be
    compressed. Only constants consumed by ops in ``_OPS_WEIGHT_NEED_COMPRESSION``
    are candidates; ops that fail to be quantized are skipped with a warning.

    The ``granularity`` and ``block_size`` parameters determine the effective
    ``block_sizes`` per axis (``0`` means the full axis is one block):

    For a 2-D linear weight ``[C_out, C_in]``:

    .. code-block:: text

        |-------------------------------|--------------------------|
        | Granularity                   | block_sizes              |
        |-------------------------------|--------------------------|
        | PER_TENSOR                    | [0, 0]                   |
        | PER_CHANNEL                   | [1, 0]                   |
        | PER_BLOCK(bs=32)              | [1, 32]                  |
        |-------------------------------|--------------------------|

    For a 4-D Conv weight ``[C_out, C_in, KH, KW]``:

    .. code-block:: text

        |-------------------------------|--------------------------|
        | Granularity                   | block_sizes              |
        |-------------------------------|--------------------------|
        | PER_TENSOR                    | [0, 0, 0, 0]             |
        | PER_CHANNEL                   | [1, 0, 0, 0]             |
        | PER_BLOCK(bs=32)              | [1, 32, 0, 0]            |
        |-------------------------------|--------------------------|

    Args:
        coreai_program (AIProgram): The model to be quantized.
        dtype (DType): Target quantized data type (e.g. ``DType.INT8``,
            ``DType.INT4``, ``DType.FP8_E4M3FN``, ``DType.FP4_E2M1FN``).
        qscheme (QScheme): Quantization scheme. Use ``QScheme.SYMMETRIC`` or
            ``QScheme.ASYMMETRIC``. FP dtypes only support
            ``QScheme.SYMMETRIC``. Defaults to ``QScheme.SYMMETRIC``.
        granularity (CompressionGranularity): Quantization granularity. Supports
            ``CompressionGranularity.PER_TENSOR``,
            ``CompressionGranularity.PER_CHANNEL``, and
            ``CompressionGranularity.PER_BLOCK``.
            Defaults to ``CompressionGranularity.PER_CHANNEL``.
        block_size (int): Block size applied to the input channel axis. Only
            effective when ``granularity`` is
            ``CompressionGranularity.PER_BLOCK``. Defaults to ``32``.
        weight_num_threshold (int): Threshold of weight element count to determine
            whether to compress a weight. Defaults to ``1024``.
        scale_dtype (DType | None): Data type for the scale constants. Must be
            ``None`` for integer ``dtype`` values. Must be ``None`` for
            ``DType.FP4_E2M1FN`` (scale is always stored in ``DType.FP8_E8M0FNU``
            internally). For FP8 ``dtype`` values, ``None`` (default) uses the
            uncompressed weight dtype (e.g. ``f16`` or ``f32``) for the scale;
            ``DType.FP8_E8M0FNU`` stores the scale in the 8-bit E8M0FNU format
            (MXFP). Defaults to ``None``.
        in_place (bool): Whether to quantize the model in-place. Defaults to
            ``False``.

    Returns:
        AIProgram: A quantized Core AI program.

    Raises:
        ValueError: If ``dtype`` is not in the set of supported weight dtypes.
        ValueError: If ``dtype`` is an FP dtype and ``qscheme`` is
            ``QScheme.ASYMMETRIC``. FP quantization only supports symmetric mode.
        ValueError: If ``scale_dtype`` is not ``None`` for an integer ``dtype``.
        ValueError: If ``scale_dtype`` is not ``None`` for ``DType.FP4_E2M1FN``.
        ValueError: If ``dtype`` is ``DType.FP4_E2M1FN`` and ``granularity`` is not
            ``CompressionGranularity.PER_BLOCK`` or ``block_size`` is not ``32``.
            FP4 weights must use per-block quantization with a block size of 32
            to produce a valid MXFP4 encoding.
    """
    if dtype not in _VALID_WEIGHT_DTYPES:
        raise ValueError(
            f"Unsupported weight dtype {dtype!r}. "
            f"Valid weight dtypes: {sorted(d.value for d in _VALID_WEIGHT_DTYPES)}."
        )

    if dtype.is_int():
        if scale_dtype is not None:
            raise ValueError(
                f"scale_dtype must be None for integer dtype {dtype!r}, got {scale_dtype!r}."
            )
    else:
        if qscheme != QScheme.SYMMETRIC:
            raise ValueError(
                f"Asymmetric quantization is not supported for FP dtype {dtype}. "
                "Use QScheme.SYMMETRIC for FP dtypes."
            )
        if dtype == DType.FP4_E2M1FN:
            if scale_dtype is not None:
                raise ValueError(
                    f"scale_dtype must be None for FP4 dtype {dtype!r}, got {scale_dtype!r}."
                )
            if granularity != CompressionGranularity.PER_BLOCK or block_size != 32:
                raise ValueError(
                    f"DType.FP4_E2M1FN requires "
                    f"granularity=CompressionGranularity.PER_BLOCK "
                    f"and block_size=32 for a valid MXFP4 encoding, "
                    f"got granularity={granularity!r} and block_size={block_size}."
                )
        elif scale_dtype is not None and scale_dtype not in _VALID_FP8_SCALE_DTYPES:
            raise ValueError(
                f"Invalid scale_dtype {scale_dtype!r} for FP8 weight dtype {dtype!r}. "
                f"Valid scale dtypes: {sorted(d.value for d in _VALID_FP8_SCALE_DTYPES)}."
            )

    def replace_weight_with_compression_op(op: Any) -> Any:
        """Apply quantization to a single op.

        Op Chaining
        -----------
        .. code-block:: text

            coreai.constant (quantized_data) ──┐
                                               ├─> coreai.blockwise_shift_scale -> output
            coreai.constant (scale) ───────────┤
            coreai.constant (zero_point) ──────┘
        """
        if not _should_compress_op(op, weight_num_threshold, _OPS_WEIGHT_NEED_COMPRESSION):
            return _WalkResult.ADVANCE

        const_weight: Any = op
        weight = _get_constant_value_as_np_array(const_weight)
        block_sizes = _infer_quantization_block_sizes(
            const_weight,
            weight.shape,
            granularity,
            block_size,
        )

        try:
            if dtype.is_int():
                dtype_builtin = _compression_types.string_to_builtin(dtype)
                quant_params = _compute_qparams_by_dtype(
                    weight,
                    dtype_builtin,
                    qscheme,
                    list(block_sizes),
                )
            else:
                quant_params = _compute_qparams_by_dtype(
                    weight,
                    _FLOAT_DTYPE_TO_ML[dtype],
                    QScheme.SYMMETRIC,
                    list(block_sizes),
                )
            if quant_params is None:
                logger.warning(
                    "Failed to quantize op %s. Skipped this op.",
                    const_weight.name,
                )
                return _WalkResult.ADVANCE
        except ImportError:
            raise
        except Exception as e:
            logger.warning(
                "Failed to quantize op %s. Skipped this op. Detailed reason:\n%s",
                const_weight.name,
                e,
            )
            return _WalkResult.ADVANCE

        quantized_data, scale, zero_point = quant_params

        with const_weight.context, const_weight.location, _InsertionPoint(const_weight):
            weight_element_type = cast("Any", const_weight.result.type).element_type

            if dtype.is_int():
                ref_mlir_type = _get_string_to_mlir_type()[dtype]
                quantized_mlir_type = (
                    _IntegerType.get_signed(ref_mlir_type.width)
                    if ref_mlir_type.is_signed
                    else _IntegerType.get_unsigned(ref_mlir_type.width)
                )
                if zero_point is None:
                    zero_point = np.zeros_like(scale, dtype=quantized_data.dtype)

                quantized_weight = _create_int_quantized_weight(
                    quantized_data,
                    scale,
                    zero_point,
                    quantized_mlir_type,
                    weight_element_type,
                )
            else:
                fp_mlir_type, _ = _get_fp_mlir_and_ml_dtype(dtype)

                if dtype == DType.FP4_E2M1FN:
                    scale_mlir_type, scale_np_dtype = _get_scale_mlir_and_np_dtype(
                        DType.FP8_E8M0FNU
                    )
                elif scale_dtype is not None:
                    scale_mlir_type, scale_np_dtype = _get_scale_mlir_and_np_dtype(scale_dtype)
                else:
                    scale_mlir_type, scale_np_dtype = None, None

                quantized_weight = _create_fp_quantized_weight(
                    quantized_data,
                    scale,
                    fp_mlir_type,
                    weight_element_type,
                    scale_mlir_type,
                    scale_np_dtype,
                )

        const_weight.result.replace_all_uses_with(quantized_weight)

        return _WalkResult.ADVANCE

    return _apply_compression_transform(
        coreai_program,
        replace_weight_with_compression_op,
        in_place,
    )
