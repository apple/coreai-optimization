# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Weight sparsification pass for Core AI AIProgram."""

from __future__ import annotations

import logging
from typing import Any, cast

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
    _select_input_output_channel_axis,
    _should_compress_op,
)
from coreai_opt.coreai_utils._utils.palettize_utils import _blockwise_compress
from coreai_opt.coreai_utils._utils.quantize_utils import _compute_qparams_by_dtype
from coreai_opt.coreai_utils._utils.sparse_utils import (
    _compress_by_magnitude,
    _compress_by_nm_sparsity,
)
from coreai_opt.coreai_utils._utils.type_utils import (
    _get_fp_mlir_and_ml_dtype,
    _get_string_to_mlir_type,
)
from coreai_opt.coreai_utils.common import DType, QScheme
from coreai_opt.coreai_utils.passes import _OPS_WEIGHT_NEED_COMPRESSION

logger = logging.getLogger(__name__)

_VALID_DTYPES: frozenset[DType] = frozenset(
    {DType.INT8, DType.UINT8, DType.FP8_E4M3FN, DType.FP8_E5M2}
)
_VALID_PALETTIZE_NBITS: frozenset[int] = frozenset({1, 2, 3, 4, 6, 8})


def sparsify_weights(
    coreai_program: _AIProgram,
    target_sparsity: float | None = 0.5,
    block_size: int | None = None,
    n_m_ratio: tuple[int, int] | None = None,
    quantize_dtype: DType | None = None,
    palettize_nbits: int | None = None,
    weight_num_threshold: int = 1024,
    in_place: bool = False,
) -> _AIProgram:
    """Sparsify weights in a Core AI AIProgram (MLIR<CoreAI> IR) by using Core AI ops.

    Walks through the IR and sparsifies each coreai.constant op that needs to be
    compressed. Only constants consumed by ops in ``_OPS_WEIGHT_NEED_COMPRESSION``
    are candidates; ops that fail to be sparsified are skipped with a warning.

    Args:
        coreai_program (AIProgram): The model to be sparsified.
        target_sparsity (float | None): Percentage of sparsity in ``[0, 1]``.
            ``n`` lowest absolute weight values are set to zero, where
            ``n = floor(size * target_sparsity)``. Mutually exclusive with
            ``n_m_ratio``. Defaults to ``0.5``.
        block_size (int | None): Block size for block sparsity along the output
            channel dimension. Only applied to ``linear`` and ``conv`` layers.
            If set, must be greater than ``1``. Defaults to ``None``.
        n_m_ratio (tuple[int, int] | None): ``(n, m)`` ratio for n:m structured
            pruning along the input channel axis. Out of every ``m`` elements,
            the ``n`` with lowest magnitude are set to zero. Only applied to
            ``linear`` and ``conv`` layers. Mutually exclusive with
            ``target_sparsity``. Defaults to ``None``.
        quantize_dtype (DType | None): Data type for storing non-zero values (joint
            compression). Must be ``None``, ``DType.INT8``, ``DType.UINT8``,
            ``DType.FP8_E4M3FN``, or ``DType.FP8_E5M2``. When set, non-zero values
            are quantized and a ``coreai.blockwise_shift_scale`` op dequantizes them
            back. Cannot be used with ``palettize_nbits``. Defaults to ``None``.
        palettize_nbits (int | None): Number of bits for palettizing non-zero values.
            When set, non-zero values are palettized using k-means with
            ``2**palettize_nbits`` clusters. Valid values: ``{1, 2, 3, 4, 6, 8}``.
            Cannot be used with ``quantize_dtype``. Defaults to ``None``.
        weight_num_threshold (int): Minimum weight element count required to
            compress a weight. Defaults to ``1024``.
        in_place (bool): Whether to sparsify the model in-place. Defaults to
            ``False``.

    Returns:
        AIProgram: A sparsified Core AI program.

    Raises:
        ValueError: If both ``target_sparsity`` and ``n_m_ratio`` are set.
        ValueError: If neither ``target_sparsity`` nor ``n_m_ratio`` is set.
        ValueError: If both ``quantize_dtype`` and ``palettize_nbits`` are set.
        ValueError: If ``quantize_dtype`` is not ``None``, ``DType.INT8``,
            ``DType.UINT8``, ``DType.FP8_E4M3FN``, or ``DType.FP8_E5M2``.
        ValueError: If ``palettize_nbits`` is not in ``{1, 2, 3, 4, 6, 8}``.
        ValueError: If ``block_size`` is set and not greater than ``1``.
        ValueError: If ``n_m_ratio`` does not have length 2, contains non-integers,
            has ``m <= 0``, or has ``n`` outside ``[0, m]``.
    """
    if target_sparsity is not None and n_m_ratio is not None:
        raise ValueError("`target_sparsity` and `n_m_ratio` cannot both be set.")

    if target_sparsity is None and n_m_ratio is None:
        raise ValueError("One of `target_sparsity` or `n_m_ratio` must be set.")

    if quantize_dtype is not None and palettize_nbits is not None:
        raise ValueError("`quantize_dtype` and `palettize_nbits` cannot both be set.")

    if quantize_dtype is not None and quantize_dtype not in _VALID_DTYPES:
        raise ValueError(
            f"Invalid quantize_dtype {quantize_dtype!r}. "
            f"Valid quantize dtypes: {sorted(d.value for d in _VALID_DTYPES)}."
        )

    if palettize_nbits is not None and palettize_nbits not in _VALID_PALETTIZE_NBITS:
        raise ValueError(
            f"Invalid palettize_nbits {palettize_nbits}. "
            f"Must be one of {sorted(_VALID_PALETTIZE_NBITS)}."
        )

    if block_size is not None and block_size <= 1:
        raise ValueError("`block_size` must be greater than 1.")

    if n_m_ratio is not None:
        if len(n_m_ratio) != 2:
            raise ValueError(
                f"`n_m_ratio` must be a tuple of length 2, got length {len(n_m_ratio)}."
            )
        n, m = n_m_ratio
        if not isinstance(n, int) or not isinstance(m, int):
            raise ValueError(
                f"`n_m_ratio` must contain integers, got ({type(n).__name__}, {type(m).__name__})."
            )
        if m <= 0:
            raise ValueError(f"`n_m_ratio` requires m > 0, got m={m}.")
        if not (0 <= n <= m):
            raise ValueError(f"`n_m_ratio` requires 0 <= n <= m, got n={n}, m={m}.")

    def replace_weight_with_compression_op(op: Any) -> Any:
        """Apply sparsification to a single op, creating the appropriate op chain.

        Op Chaining Rules
        -----------------
        This function creates different op chains depending on the compression configuration:

        1. Sparsification only (quantize_dtype=None, palettize_nbits=None):

           coreai.constant (nonzero_data) ─> coreai.build_sparse_with_bitmask
             -> coreai.sparse_with_bitmask_to_dense -> output

        2. Sparsification + quantization (quantize_dtype=DType.INT8 or DType.FP8_E4M3FN):

           coreai.constant (quantized_nonzero_data) ─> coreai.build_sparse_with_bitmask
             -> coreai.sparse_with_bitmask_to_dense ──> coreai.blockwise_shift_scale -> output

        3. Sparsification + palettization (palettize_nbits is set):

           coreai.constant (palettized_nonzero_data) ─> coreai.lut_to_dense
             ──> coreai.build_sparse_with_bitmask -> coreai.sparse_with_bitmask_to_dense -> output
        """
        if not _should_compress_op(op, weight_num_threshold, _OPS_WEIGHT_NEED_COMPRESSION):
            return _WalkResult.ADVANCE

        const_weight: Any = op
        weight = _get_constant_value_as_np_array(const_weight)
        input_channel_axis, output_channel_axis = _select_input_output_channel_axis(const_weight)
        if output_channel_axis is None:
            logger.warning(
                "Cannot determine output_channel_axis for %s, using 0 as default.",
                const_weight.name,
            )
            output_channel_axis = 0
        if input_channel_axis is None:
            logger.warning(
                "Cannot determine input_channel_axis for %s, using 1 as default.",
                const_weight.name,
            )
            input_channel_axis = 1

        try:
            if target_sparsity is not None:
                sparse_params = _compress_by_magnitude(
                    weight,
                    target_sparsity,
                    block_size=block_size,
                    dim=output_channel_axis,
                )
            else:
                sparse_params = _compress_by_nm_sparsity(
                    weight,
                    n_m_ratio,
                    dim=input_channel_axis,
                )
            if sparse_params is None:
                logger.warning(
                    "Cannot perform sparsification on %s. Skipped this op.",
                    const_weight.name,
                )
                return _WalkResult.ADVANCE
        except ImportError:
            raise
        except Exception as e:
            logger.warning(
                "Failed to sparsify op %s. Skipped this op. "
                "Detailed reason about sparsification failure:\n%s",
                const_weight.name,
                e,
            )
            return _WalkResult.ADVANCE

        with const_weight.context, const_weight.location, _InsertionPoint(const_weight):
            require_dequantization = False

            if palettize_nbits is not None:
                nonzero_data = sparse_params.nonzero_data
                original_shape = nonzero_data.shape

                if nonzero_data.size == 0:
                    logger.warning(
                        "No non-zero values in %s. Falling back to default sparsification "
                        "(palettize_nbits ignored).",
                        const_weight.name,
                    )
                    nonzero_data_const = _create_constant_value_from_np_array(
                        sparse_params.nonzero_data,
                        const_weight.result.type.element_type,
                    )
                else:
                    # nonzero_data is 1D; reshape to 2D for _blockwise_compress.
                    # LUT rank = indices.rank + 2, so for 1D indices: LUT is 3D [1, 2^nbits, 1].
                    lut_params = _blockwise_compress(
                        nonzero_data.reshape(-1, 1),
                        mode="KMEANS",
                        nbits=palettize_nbits,
                        block_sizes=[0, 0],
                        cluster_dim=1,
                        channel_axis=0,
                        num_kmeans_workers=4,
                    )

                    indices_reshaped = lut_params.indices.reshape(original_shape)
                    indices = _create_constant_value_from_np_array(
                        indices_reshaped,
                        _IntegerType.get_unsigned(palettize_nbits),
                    )

                    lut_data = lut_params.lut.astype(sparse_params.nonzero_data.dtype)
                    lut = _create_constant_value_from_np_array(
                        lut_data.reshape(1, 2**palettize_nbits, 1),
                        const_weight.result.type.element_type,
                    )

                    vector_axis = _create_constant_value_from_np_array(
                        np.int16(0),
                        _IntegerType.get_signed(16),
                    )

                    nonzero_data_const = _coreai.lut_to_dense(
                        indices=indices,
                        lut=lut,
                        axis=vector_axis,
                    )

            elif quantize_dtype is not None:
                if sparse_params.nonzero_data.size == 0:
                    logger.warning(
                        "No non-zero values in %s. Falling back to default sparsification "
                        "(quantize_dtype ignored).",
                        const_weight.name,
                    )
                    nonzero_data_const = _create_constant_value_from_np_array(
                        sparse_params.nonzero_data,
                        const_weight.result.type.element_type,
                    )
                else:
                    require_dequantization = True

                    quant_block_sizes = [0] * len(sparse_params.nonzero_data.shape)
                    if quantize_dtype.is_int():
                        quantize_dtype_builtin = _compression_types.string_to_builtin(
                            quantize_dtype
                        )
                        ref_mlir_type = _get_string_to_mlir_type()[quantize_dtype]
                        quantized_mlir_type = (
                            _IntegerType.get_signed(ref_mlir_type.width)
                            if ref_mlir_type.is_signed
                            else _IntegerType.get_unsigned(ref_mlir_type.width)
                        )
                        quant_params = _compute_qparams_by_dtype(
                            sparse_params.nonzero_data,
                            quantize_dtype_builtin,
                            QScheme.SYMMETRIC,
                            quant_block_sizes,
                        )
                    else:
                        fp8_mlir_type, fp8_ml_dtype = _get_fp_mlir_and_ml_dtype(quantize_dtype)
                        quant_params = _compute_qparams_by_dtype(
                            sparse_params.nonzero_data,
                            fp8_ml_dtype,
                            QScheme.SYMMETRIC,
                            quant_block_sizes,
                        )

                    if quant_params is None:
                        logger.warning(
                            "Failed to compute quantization parameters for %s. Skipped this op.",
                            const_weight.name,
                        )
                        return _WalkResult.ADVANCE

                    quantized_data, scale, zero_point = quant_params
                    if zero_point is None:
                        zero_point = np.zeros_like(scale, dtype=quantized_data.dtype)

                    target_shape = [1] * len(weight.shape)
                    scale_reshaped = np.full(target_shape, scale.flatten()[0], dtype=scale.dtype)
                    zero_point_reshaped = np.full(
                        target_shape, zero_point.flatten()[0], dtype=zero_point.dtype
                    )

                    if quantize_dtype.is_int():
                        nonzero_data_const = _create_constant_value_from_np_array(
                            quantized_data,
                            quantized_mlir_type,
                        )
                        scale_const = _create_constant_value_from_np_array(
                            scale_reshaped,
                            const_weight.result.type.element_type,
                        )
                        zero_point_const = _create_constant_value_from_np_array(
                            zero_point_reshaped,
                            quantized_mlir_type,
                        )
                    else:
                        quantized_tensor_type = _RankedTensorType.get(
                            list(quantized_data.shape), fp8_mlir_type
                        )
                        nonzero_data_const = cast(
                            "Any",
                            _coreai.ConstantOp(
                                value=_DenseResourceElementsAttr.get_from_buffer(
                                    quantized_data, "dense_resource", quantized_tensor_type
                                )
                            ).result,
                        )
                        scale_const = _create_constant_value_from_np_array(
                            scale_reshaped,
                            const_weight.result.type.element_type,
                        )
                        zero_point_const = cast(
                            "Any",
                            _coreai.ConstantOp(
                                value=_DenseElementsAttr.get_splat(
                                    _RankedTensorType.get(target_shape, fp8_mlir_type),
                                    _FloatAttr.get(fp8_mlir_type, 0.0),
                                )
                            ).result,
                        )

            else:
                nonzero_data_const = _create_constant_value_from_np_array(
                    sparse_params.nonzero_data,
                    const_weight.result.type.element_type,
                )

            mask = _create_constant_value_from_np_array(
                sparse_params.mask,
                _IntegerType.get_unsigned(1),
            )
            sparse_tensor = _coreai.build_sparse_with_bitmask(
                values=nonzero_data_const,
                bitmask=mask,
            )
            compressed_weight = _coreai.sparse_with_bitmask_to_dense(sparse_tensor)

            if require_dequantization:
                compressed_weight = _coreai.blockwise_shift_scale(
                    data=compressed_weight,
                    scale=scale_const,
                    offset1=zero_point_const,
                    offset2=_create_constant_value_from_np_array(
                        np.zeros(scale_reshaped.shape, dtype=np.float32),
                        const_weight.result.type.element_type,
                    ),
                )

        const_weight.result.replace_all_uses_with(compressed_weight)

        return _WalkResult.ADVANCE

    return _apply_compression_transform(
        coreai_program,
        replace_weight_with_compression_op,
        in_place,
    )
