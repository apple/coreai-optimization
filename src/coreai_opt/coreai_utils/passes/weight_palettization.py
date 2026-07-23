# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Weight palettization pass for Core AI AIProgram."""

from __future__ import annotations

import logging
from typing import Any, cast

import ml_dtypes
import numpy as np

from coreai_opt.coreai_utils._coreai_imports import (
    AIProgram as _AIProgram,
    DenseResourceElementsAttr as _DenseResourceElementsAttr,
    F16Type as _F16Type,
    F32Type as _F32Type,
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
    _should_compress_op,
)
from coreai_opt.coreai_utils._utils.palettize_utils import (
    _blockwise_compress,
    _infer_palettization_block_sizes_and_channel_axis as _infer_block_sizes,
    _is_cluster_dim_valid,
)
from coreai_opt.coreai_utils._utils.quantize_utils import _compute_qparams_by_dtype
from coreai_opt.coreai_utils._utils.type_utils import (
    _get_fp_mlir_and_ml_dtype,
    _get_string_to_mlir_type,
)
from coreai_opt.coreai_utils.common import CompressionGranularity, DType, QScheme
from coreai_opt.coreai_utils.passes import _OPS_WEIGHT_NEED_COMPRESSION

logger = logging.getLogger(__name__)

_FP8_DTYPE_TO_ML = {
    DType.FP8_E4M3FN: ml_dtypes.float8_e4m3fn,
    DType.FP8_E5M2: ml_dtypes.float8_e5m2,
}

_VALID_LUT_DTYPES = {DType.INT8, DType.UINT8, DType.FP8_E4M3FN, DType.FP8_E5M2}
_VALID_N_BITS = {1, 2, 3, 4, 6, 8}
_VALID_PALETTIZATION_GRANULARITIES = {
    CompressionGranularity.PER_TENSOR,
    CompressionGranularity.PER_CHANNEL,
    CompressionGranularity.PER_GROUPED_CHANNEL,
}


def palettize_weights(
    coreai_program: _AIProgram,
    lut_dtype: DType | None,
    n_bits: int = 4,
    granularity: CompressionGranularity = CompressionGranularity.PER_TENSOR,
    group_size: int = 32,
    cluster_dim: int = 1,
    enable_per_channel_scale: bool = False,
    weight_num_threshold: int = 1024,
    num_kmeans_workers: int = 4,
    enable_fast_kmeans_mode: bool = True,
    rounding_precision: int = 4,
    in_place: bool = False,
) -> _AIProgram:
    """Palettize weights in a Core AI AIProgram (MLIR<CoreAI> IR) by using Core AI ops.

    Walks through the IR and palettizes each coreai.constant op that needs to be
    compressed. Only constants consumed by ops in ``_OPS_WEIGHT_NEED_COMPRESSION``
    are candidates; ops that fail to be palettized are skipped with a warning.

    Args:
        coreai_program (AIProgram): The model to be palettized.
        lut_dtype (DType | None): The datatype for values in the look-up table.
            Can be ``None`` (no LUT quantization), ``DType.INT8``,
            ``DType.UINT8``, ``DType.FP8_E4M3FN``, or ``DType.FP8_E5M2``.
            Symmetric quantization is used by default. Defaults to ``None``.
        n_bits (int): Number of bits for palettizing the weights. Defaults to ``4``.
            A LUT will have ``2**n_bits`` entries; n_bits must be in ``{1, 2, 3, 4, 6, 8}``.
        granularity (CompressionGranularity): Quantization granularity. Supports
            ``CompressionGranularity.PER_TENSOR``,
            ``CompressionGranularity.PER_CHANNEL``, and
            ``CompressionGranularity.PER_GROUPED_CHANNEL``.
            Defaults to ``CompressionGranularity.PER_TENSOR``.
        group_size (int): Number of channels in a group. Only effective when
            granularity is ``CompressionGranularity.PER_GROUPED_CHANNEL``.
        cluster_dim (int): Dimension of centroids for each lookup table. When
            ``cluster_dim > 1``, it indicates 2-D clustering. Defaults to 1
            (scalar palettization).
        enable_per_channel_scale (bool): When ``True``, weights are normalized along
            output channels using per-channel scales before palettization. Not
            supported with ``cluster_dim > 1``.
        weight_num_threshold (int): Threshold of weight element count to determine
            whether to compress a weight.
        num_kmeans_workers (int): Number of worker processes for k-means. Defaults
            to 4.
        enable_fast_kmeans_mode (bool): Whether to use weight rounding to speed up
            k-means. Weight rounding reduces precision but speeds up k-means clustering.
            Defaults to ``True``.
        rounding_precision (int): Number of decimal places to round weights to during
            fast K-means clustering. Only effective when ``enable_fast_kmeans_mode``
            is ``True``. Defaults to ``4``.
        in_place (bool): Whether to palettize the model in-place. Defaults to
            ``False``.

    Returns:
        AIProgram: A palettized Core AI program.
    """
    if enable_per_channel_scale and cluster_dim > 1:
        raise ValueError("`enable_per_channel_scale=True` is not supported with `cluster_dim > 1`.")
    if lut_dtype is not None and lut_dtype not in _VALID_LUT_DTYPES:
        raise ValueError(
            f"`lut_dtype` must be one of "
            f"{sorted(d.value for d in _VALID_LUT_DTYPES)} or None, got {lut_dtype!r}."
        )
    if n_bits not in _VALID_N_BITS:
        raise ValueError(f"`n_bits` must be one of {sorted(_VALID_N_BITS)}, got {n_bits}.")
    if granularity not in _VALID_PALETTIZATION_GRANULARITIES:
        raise ValueError(
            f"`granularity` must be one of "
            f"{sorted(g.value for g in _VALID_PALETTIZATION_GRANULARITIES)}, got {granularity!r}."
        )

    def replace_weight_with_compression_op(
        op: Any,
    ) -> Any:
        """Apply palettization to a single op, creating the appropriate op chain.

        Op Chaining Rules
        -----------------
        This function creates different op chains depending on the compression configuration:

        1. Palettization only (lut_dtype=None, enable_per_channel_scale=False):

           coreai.constant (indices) ───────┐
                                            ├─> coreai.lut_to_dense -> output
           coreai.constant (fp16/fp32 LUT) ─┘

        2. Palettization + LUT quantization (lut_dtype=DType.INT8, enable_per_channel_scale=False):

           coreai.constant (indices)  ──────┐
                                            ├─> coreai.lut_to_dense ──>
           coreai.constant (int8 LUT) ──────┘   coreai.blockwise_shift_scale -> output
           coreai.constant (scale) ───────────────────┘
           coreai.constant (zero_point) ──────────────┘

        3. Palettization + per-channel scaling (lut_dtype=None, enable_per_channel_scale=True):

           coreai.constant (indices)  ───────┐
                                             ├─> coreai.lut_to_dense ──>
           coreai.constant (fp16/fp32 LUT) ──┘   coreai.blockwise_shift_scale -> output
           coreai.constant (scale) ────────────────────┘
           coreai.constant (zero_point=0) ─────────────┘

        4. Palettization + LUT quantization + per-channel scaling
           (lut_dtype=DType.INT8, enable_per_channel_scale=True):

           coreai.constant (indices)  ──────┐
                                            ├─> coreai.lut_to_dense ──>
           coreai.constant (int8 LUT) ──────┘   coreai.blockwise_shift_scale -> output
           coreai.constant (fused_scale) ─────────────┘
           coreai.constant (fused_zero_point) ────────┘

        Note:
            In case 4, the two blockwise_shift_scale operations from cases 2 and 3 are
            fused into one. The fused scale = lut_scale * per_channel_scale, and
            fused zero_point = lut_zero_point * per_channel_scale.
        """
        if not _should_compress_op(op, weight_num_threshold, _OPS_WEIGHT_NEED_COMPRESSION):
            return _WalkResult.ADVANCE

        const_weight: Any = op
        weight = _get_constant_value_as_np_array(const_weight)
        block_sizes, channel_axis = _infer_block_sizes(
            const_weight,
            weight.shape,
            granularity,
            group_size,
        )

        if cluster_dim > 1 and not _is_cluster_dim_valid(op, cluster_dim, channel_axis):
            logger.warning(
                "The `cluster_dim` is invalid for %s. Skipped this op.",
                const_weight.name,
            )
            return _WalkResult.ADVANCE

        if enable_per_channel_scale:
            # Normalize by per channel scales before doing palettization.
            per_channel_scale = np.max(np.abs(weight), axis=channel_axis, keepdims=True)
            per_channel_scale[per_channel_scale == 0] = 1
            weight = weight / per_channel_scale

        original_weight_type = weight.dtype
        weight = weight.astype(np.float16)
        if enable_fast_kmeans_mode:
            weight = np.round(weight, decimals=rounding_precision)

        try:
            lut_params = _blockwise_compress(
                weight,
                mode="KMEANS",
                nbits=n_bits,
                block_sizes=block_sizes,
                cluster_dim=cluster_dim,
                channel_axis=channel_axis,
                num_kmeans_workers=num_kmeans_workers,
            )
            if lut_params is None:
                logger.warning(
                    "Cannot perform palettization on %s. Skipped this op.",
                    const_weight.name,
                )
                return _WalkResult.ADVANCE
        except ImportError:
            raise
        except Exception as e:
            logger.warning(
                "Failed to palettize op %s. Skipped this op. "
                "Detailed reason about palettization failure:\n%s ",
                const_weight.name,
                e,
            )
            return _WalkResult.ADVANCE

        with const_weight.context, const_weight.location, _InsertionPoint(const_weight):
            indices = _create_constant_value_from_np_array(
                lut_params.indices,  # same shape as weight tensor (for many cases)
                _IntegerType.get_unsigned(n_bits),
            )

            vector_axis = _create_constant_value_from_np_array(
                (
                    np.int16(lut_params.vector_axis)
                    if lut_params.vector_axis is not None
                    else np.int16(0)
                ),
                _IntegerType.get_signed(16),
            )

            weight_float_mlir_type = (
                _F32Type.get() if original_weight_type == np.float32 else _F16Type.get()
            )

            if lut_dtype is not None:
                quant_block_sizes = [0] * len(lut_params.lut.shape)
                target_shape = [1] * len(lut_params.indices.shape)
                weight_element_type = cast("Any", const_weight.result.type).element_type

                if lut_dtype.is_int():
                    lut_dtype_builtin = _compression_types.string_to_builtin(lut_dtype)
                    ref_mlir_type = _get_string_to_mlir_type()[lut_dtype]
                    quantized_mlir_type = (
                        _IntegerType.get_signed(ref_mlir_type.width)
                        if ref_mlir_type.is_signed
                        else _IntegerType.get_unsigned(ref_mlir_type.width)
                    )

                    quant_params = _compute_qparams_by_dtype(
                        lut_params.lut,
                        lut_dtype_builtin,
                        QScheme.SYMMETRIC,
                        quant_block_sizes,
                    )
                    if quant_params is None:
                        logger.warning(
                            "Failed to compute quantization parameters for %s. Skipped this op.",
                            const_weight.name,
                        )
                        return _WalkResult.ADVANCE

                    quantized_lut_data, lut_scale, lut_zero_point = quant_params
                    if lut_zero_point is None:
                        lut_zero_point = np.zeros_like(
                            lut_scale,
                            dtype=quantized_lut_data.dtype,
                        )

                    if enable_per_channel_scale:
                        lut_scale = lut_scale * per_channel_scale
                        lut_zero_point_dtype = lut_zero_point.dtype
                        lut_zero_point = np.round(
                            lut_zero_point * per_channel_scale,
                        ).astype(lut_zero_point_dtype)

                    # Reshape scale and zero_point from LUT rank to reconstructed weight
                    # rank. For per-tensor quantization, all values are the same, so we
                    # can just reshape.
                    lut_scale_reshaped = np.full(
                        target_shape,
                        lut_scale.flatten()[0],
                        dtype=lut_scale.dtype,
                    )
                    lut_zero_point_reshaped = np.full(
                        target_shape,
                        lut_zero_point.flatten()[0],
                        dtype=lut_zero_point.dtype,
                    )

                    lut_quantized = _create_constant_value_from_np_array(
                        quantized_lut_data,
                        quantized_mlir_type,
                    )
                    lut_scale_const = _create_constant_value_from_np_array(
                        lut_scale_reshaped,
                        weight_element_type,
                    )
                    lut_zero_point_const = _create_constant_value_from_np_array(
                        lut_zero_point_reshaped.astype(np.float32),
                        weight_element_type,
                    )

                else:
                    fp8_mlir_type, _ = _get_fp_mlir_and_ml_dtype(lut_dtype)

                    quant_params = _compute_qparams_by_dtype(
                        lut_params.lut,
                        _FP8_DTYPE_TO_ML[lut_dtype],
                        QScheme.SYMMETRIC,
                        quant_block_sizes,
                    )
                    if quant_params is None:
                        logger.warning(
                            "Failed to compute quantization parameters for %s. Skipped this op.",
                            const_weight.name,
                        )
                        return _WalkResult.ADVANCE

                    quantized_lut_data, lut_scale, _ = quant_params

                    if enable_per_channel_scale:
                        lut_scale = lut_scale * per_channel_scale

                    lut_scale_reshaped = np.full(
                        target_shape,
                        lut_scale.flatten()[0],
                        dtype=lut_scale.dtype,
                    )

                    tensor_type = _RankedTensorType.get(
                        list(quantized_lut_data.shape), fp8_mlir_type
                    )
                    lut_quantized_attr = _DenseResourceElementsAttr.get_from_buffer(
                        quantized_lut_data,
                        "dense_resource",
                        tensor_type,
                    )
                    lut_quantized = cast("Any", _coreai.ConstantOp(value=lut_quantized_attr).result)

                    lut_scale_const = _create_constant_value_from_np_array(
                        lut_scale_reshaped,
                        weight_element_type,
                    )

                    lut_zero_point_const = _create_constant_value_from_np_array(
                        np.zeros(target_shape, dtype=np.float32),
                        weight_element_type,
                    )

                compressed_weight_quantized = _coreai.lut_to_dense(
                    indices=indices,
                    lut=lut_quantized,
                    axis=vector_axis,
                )
                # Cast lut_to_dense output from quantized type to float so all
                # blockwise_shift_scale operands share the same element type.
                cast_type = _RankedTensorType.get(
                    cast("Any", compressed_weight_quantized.type).shape,
                    weight_float_mlir_type,
                )
                compressed_weight_float = cast(
                    "Any",
                    _coreai.CastOp(cast_type, compressed_weight_quantized).result,
                )
                lut_t = cast("Any", lut_scale_const.type)
                compressed_weight = _coreai.blockwise_shift_scale(
                    data=compressed_weight_float,
                    scale=lut_scale_const,
                    offset1=lut_zero_point_const,
                    offset2=_create_constant_value_from_np_array(
                        np.zeros(lut_t.shape, dtype=np.float32),
                        lut_t.element_type,
                    ),
                )
                # Cast back to the original weight IR type so downstream ops
                # (e.g. coreai.transpose) see the same type contract.
                compressed_weight = cast(
                    "Any",
                    _coreai.CastOp(
                        cast("Any", const_weight.result.type),
                        compressed_weight,
                    ).result,
                )

            else:
                lut = _create_constant_value_from_np_array(
                    lut_params.lut.astype(original_weight_type),
                    cast("Any", const_weight.result.type).element_type,
                )

                compressed_weight = _coreai.lut_to_dense(
                    indices=indices,
                    lut=lut,
                    axis=vector_axis,
                )

                if enable_per_channel_scale:
                    scale = _create_constant_value_from_np_array(
                        per_channel_scale,
                        weight_float_mlir_type,
                    )
                    zero_point = _create_constant_value_from_np_array(
                        np.zeros_like(per_channel_scale, dtype=original_weight_type),
                        weight_float_mlir_type,
                    )
                    compressed_weight_float = cast(
                        "Any",
                        _coreai.CastOp(
                            _RankedTensorType.get(
                                cast("Any", compressed_weight.type).shape,
                                weight_float_mlir_type,
                            ),
                            compressed_weight,
                        ).result,
                    )
                    compressed_weight = _coreai.blockwise_shift_scale(
                        data=compressed_weight_float,
                        scale=scale,
                        offset1=zero_point,
                        offset2=_create_constant_value_from_np_array(
                            np.zeros(per_channel_scale.shape, dtype=original_weight_type),
                            weight_float_mlir_type,
                        ),
                    )
                    compressed_weight = cast(
                        "Any",
                        _coreai.CastOp(
                            cast("Any", const_weight.result.type),
                            compressed_weight,
                        ).result,
                    )

        const_weight.result.replace_all_uses_with(compressed_weight)

        return _WalkResult.ADVANCE

    return _apply_compression_transform(
        coreai_program,
        replace_weight_with_compression_op,
        in_place,
    )
