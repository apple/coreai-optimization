# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Palettization utilities for Core AI compression passes."""

# TODO: add test enhancements for palettize utils.
from __future__ import annotations

import atexit
import logging
from collections import namedtuple
from collections.abc import Callable, Sequence
from itertools import repeat
from multiprocessing import Pool
from typing import cast

import numpy as np

from coreai_opt.coreai_utils._coreai_imports import (
    Operation,
    _get_constant_value_as_np_array,
)
from coreai_opt.coreai_utils._utils.graph_utils import _select_input_output_channel_axis
from coreai_opt.coreai_utils.common import CompressionGranularity as _CompressionGranularity
from coreai_opt.deps import _kmeans1d

logger = logging.getLogger(__name__)

_CONV2D_OP = "coreai.conv2d"

LutParams = namedtuple("LutParams", "indices lut vector_axis")

_SUPPORTED_NBITS: tuple[int, ...] = (1, 2, 3, 4, 6, 8)

_compress_pool: Pool | None = None


def _reshape_weight_for_vector_lut(
    weight: np.ndarray, vector_size: int, vector_axis: int
) -> np.ndarray:
    """Reshape weight so vectors of length ``vector_size`` are on the last axis."""
    weight = np.swapaxes(weight, -1, vector_axis)
    weight = weight.reshape((*weight.shape[:-1], weight.shape[-1] // vector_size, vector_size))
    return np.swapaxes(weight, -2, vector_axis)


def _get_nbits_for_unique_mode(
    val: np.ndarray,
    allowed_nbits: tuple[int, ...],
    cluster_dim: int = 1,
    vector_axis: int | None = None,
) -> int:
    if cluster_dim == 1:
        val = val.flatten()
        unique_vals_num = len(np.unique(val))
    else:
        if vector_axis is None:
            raise ValueError("The `vector_axis` must be specified when cluster_dim > 1")
        val = np.swapaxes(val, -1, vector_axis).reshape((-1, cluster_dim))
        unique_vals_num = len(np.unique(val, axis=0))

    for nbits in allowed_nbits:
        if unique_vals_num <= 1 << nbits:
            return nbits
    raise ValueError(
        f"Unique values in weight cannot be represented by {allowed_nbits[-1]} bits palettization."
    )


def _get_kmeans_lookup_table_and_weight(
    nbits: int,
    weight: np.ndarray,
    force_kmeans1d: bool = False,
    cluster_dim: int = 1,
    vector_axis: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if force_kmeans1d and cluster_dim > 1:
        raise ValueError("Cannot force kmeans1d for vector palettization (cluster_dim > 1).")

    num_weights = int(np.prod(weight.shape))
    lut_len = 1 << nbits

    if cluster_dim > 1:
        weight = _reshape_weight_for_vector_lut(weight, cluster_dim, vector_axis)

    weight = weight.reshape(-1, cluster_dim)
    lut = np.zeros((lut_len, cluster_dim))

    is_better_to_use_kmeans1d = (
        weight.shape[1] == 1 and num_weights >= 10_000 and weight.dtype == np.float16
    )

    if is_better_to_use_kmeans1d or force_kmeans1d:
        values, indices, counts = np.unique(weight, return_inverse=True, return_counts=True)
        indices = indices.flatten()
        n_clusters = min(len(values), lut_len)
        kmeans_results = _kmeans1d.cluster(values, n_clusters, weights=counts)
        lut = lut.squeeze(-1)
        lut[:n_clusters] = kmeans_results.centroids
        wq = np.array(kmeans_results.clusters)[indices]
    else:
        try:
            from sklearn.cluster import KMeans  # noqa: PLC0415
        except Exception as err:
            raise ModuleNotFoundError(
                "scikit-learn is required for k-means quantization."
                ' To install, run: "pip install scikit-learn".'
            ) from err
        if is_better_to_use_kmeans1d:
            logger.warning(
                "It would be better to use kmeans1d but that is not available. "
                "Using scikit-learn for K-means."
            )
        n_clusters = min(num_weights, lut_len)
        kmeans = KMeans(n_clusters, init="k-means++", tol=1e-2, n_init=1, random_state=0).fit(
            weight
        )
        wq = kmeans.labels_[:num_weights]
        lut[:n_clusters] = kmeans.cluster_centers_

    return lut, wq


def _get_lut_and_indices(
    val: np.ndarray,
    mode: str,
    nbits: int | None,
    lut_function: Callable | None,
    cluster_dim: int = 1,
    vector_axis: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    def compress_kmeans(
        val: np.ndarray, nbits: int, cluster_dim: int, vector_axis: int | None
    ) -> tuple[np.ndarray, np.ndarray]:
        lut, indices = _get_kmeans_lookup_table_and_weight(
            nbits, val, force_kmeans1d=False, cluster_dim=cluster_dim, vector_axis=vector_axis
        )
        return lut.astype(val.dtype), indices.astype(np.uint8)

    def compress_uniform(val: np.ndarray, nbits: int) -> tuple[np.ndarray, np.ndarray]:
        val = val.flatten()
        val_min = np.amin(val)
        val_max = np.amax(val)
        scale = (val_max - val_min) / ((1 << nbits) - 1)
        indices = np.round(((val - val_min) / (val_max - val_min)) * ((1 << nbits) - 1)).astype(
            np.uint8
        )
        lut = np.array(range(0, 1 << nbits)) * scale + val_min
        return lut.astype(val.dtype), indices

    def compress_unique(
        val: np.ndarray, nbits: int | None, cluster_dim: int, vector_axis: int | None
    ) -> tuple[np.ndarray, np.ndarray]:
        if nbits is None:
            nbits = _get_nbits_for_unique_mode(val, _SUPPORTED_NBITS, cluster_dim, vector_axis)
        if cluster_dim > 1:
            val = _reshape_weight_for_vector_lut(val, cluster_dim, vector_axis)
        val = val.reshape((-1, cluster_dim))
        unique_vals, unique_inverse = np.unique(val, axis=0, return_inverse=True)
        lut = np.zeros((1 << nbits, cluster_dim))
        lut[: len(unique_vals)] = unique_vals
        indices = unique_inverse.flatten()
        if cluster_dim == 1:
            lut = lut.squeeze(-1)
        return lut.astype(val.dtype), indices.astype(np.uint8)

    if mode == "KMEANS":
        lut, indices = compress_kmeans(val, nbits, cluster_dim, vector_axis)
    elif mode == "UNIFORM":
        if cluster_dim > 1:
            raise NotImplementedError(
                "Vector palettization (cluster_dim > 1) doesn't support UNIFORM mode."
            )
        lut, indices = compress_uniform(val, nbits)
    elif mode == "UNIQUE":
        lut, indices = compress_unique(val, nbits, cluster_dim, vector_axis)
    else:
        if mode != "CUSTOM":
            raise ValueError(
                f"Invalid mode {mode!r}. Must be one of 'KMEANS', 'UNIFORM', 'UNIQUE', 'CUSTOM'."
            )
        lut, indices = lut_function(val)

    return lut, indices


def _grouped_channelwise_compress(
    original_data: np.ndarray,
    mode: str,
    nbits: int | None,
    channel_axis: int,
    channel_group_size: int,
    lut_function: Callable | None = None,
    cluster_dim: int = 1,
    num_kmeans_workers: int = 1,
) -> LutParams | None:
    global _compress_pool

    if not isinstance(original_data, np.ndarray):
        raise ValueError(f"Only numpy arrays are supported, but got {type(original_data)}")
    if nbits is not None and nbits not in _SUPPORTED_NBITS:
        raise ValueError(f"Invalid nbits. Support {_SUPPORTED_NBITS}, but got {nbits}")

    data_rank = len(original_data.shape)
    if not (-data_rank <= channel_axis < data_rank):
        raise ValueError(
            f"Invalid channel_axis. Should be in range [{-data_rank}, {data_rank}), "
            f"but got {channel_axis}"
        )
    if channel_axis < 0:
        channel_axis += data_rank

    channel_num = original_data.shape[channel_axis]
    if channel_group_size == 0:
        channel_group_size = channel_num
    if channel_num % channel_group_size != 0:
        logger.warning(
            "Can't perform palettization: The number of channels at %dth axis (%d) "
            "is not divisible by channel_group_size (%d).",
            channel_axis,
            channel_num,
            channel_group_size,
        )
        return None
    channel_group_num = channel_num // channel_group_size

    if channel_group_size % cluster_dim != 0:
        logger.warning(
            "Can't perform palettization: The channel_group_size at %dth axis (%d) "
            "is not divisible by cluster_dim (%d).",
            channel_axis,
            channel_group_size,
            cluster_dim,
        )
        return None

    if channel_axis != 0:
        original_data = np.swapaxes(original_data, 0, channel_axis)
    grouped_channel_data = np.split(original_data, channel_group_num, axis=0)

    vector_axis = 0

    if mode == "UNIQUE":
        try:
            for per_group_data in grouped_channel_data:
                per_group_nbits = _get_nbits_for_unique_mode(
                    per_group_data, _SUPPORTED_NBITS, cluster_dim, vector_axis
                )
                if nbits is None or per_group_nbits > nbits:
                    nbits = per_group_nbits
        except ValueError as e:
            logger.warning("Can't perform palettization: %s", e)
            return None

    if mode == "KMEANS" and num_kmeans_workers > 1:
        if _compress_pool is None:
            _compress_pool = Pool(processes=num_kmeans_workers)
            atexit.register(lambda: _compress_pool.terminate())
        lut, indices = zip(
            *_compress_pool.starmap(
                _get_lut_and_indices,
                zip(
                    grouped_channel_data,
                    repeat(mode),
                    repeat(nbits),
                    repeat(lut_function),
                    repeat(cluster_dim),
                    repeat(vector_axis),
                    strict=False,
                ),
            ),
            strict=False,
        )
    else:
        lut, indices = zip(
            *[
                _get_lut_and_indices(
                    per_channel_group_data, mode, nbits, lut_function, cluster_dim, vector_axis
                )
                for per_channel_group_data in grouped_channel_data
            ],
            strict=False,
        )

    lut = np.stack(lut, axis=0)
    indices = np.stack(indices, axis=0)

    if mode == "CUSTOM":
        nbits = int(np.ceil(np.log2(lut.shape[1])))

    palette_num = 2**nbits
    indices_target_shape = list(original_data.shape)
    if cluster_dim > 1:
        indices_target_shape[vector_axis] //= cluster_dim
    indices = indices.reshape(indices_target_shape)

    lut_target_shape = [1] * (len(original_data.shape) + 2)
    lut_target_shape[0] = channel_group_num
    lut_target_shape[-1] = cluster_dim
    lut_target_shape[-2] = palette_num
    lut = lut.reshape(lut_target_shape)

    if channel_axis != 0:
        lut = np.swapaxes(lut, 0, channel_axis)
        indices = np.swapaxes(indices, 0, channel_axis)

    # For all supported nbits (1,2,3,4,6,8), numpy represents sub-byte ints as uint8.
    return LutParams(indices.astype(np.uint8), lut, None if cluster_dim == 1 else channel_axis)


def _blockwise_compress(
    original_data: np.ndarray,
    mode: str,
    nbits: int | None,
    block_sizes: list[int],
    lut_function: Callable | None = None,
    cluster_dim: int = 1,
    channel_axis: int | None = None,
    num_kmeans_workers: int = 1,
) -> LutParams | None:
    """Compress ``original_data`` into an n-bit palettized representation.

    Args:
        original_data (np.ndarray): Weight tensor to palettize.
        mode (str): Clustering mode — ``"KMEANS"``, ``"UNIFORM"``, ``"UNIQUE"``, or ``"CUSTOM"``.
        nbits (int | None): Number of bits; must be in ``{1, 2, 3, 4, 6, 8}``.
        block_sizes (list[int]): Block size per axis (0 means no blocking on that axis).
        lut_function (Callable | None): Custom LUT function for ``"CUSTOM"`` mode.
        cluster_dim (int): Centroid vector length; ``1`` for scalar palettization.
        channel_axis (int | None): Channel axis override; inferred from ``block_sizes`` if ``None``.
        num_kmeans_workers (int): Number of worker processes for k-means.

    Returns:
        LutParams | None: Palettization result, or ``None`` if compression is inapplicable.
    """
    mode = mode.upper()
    channel_group_size = 0
    for axis, block_size in enumerate(block_sizes):
        if block_size != 0 and block_size != original_data.shape[axis]:
            if channel_axis is not None and channel_axis != axis:
                raise NotImplementedError(
                    "General block-wise palettization is not supported. Please use "
                    "'per_grouped_channel' or 'per_tensor' for the 'granularity' in config."
                )
            channel_axis = axis
            channel_group_size = block_size

    if channel_axis is None:
        if cluster_dim > 1:
            raise ValueError(
                "Cannot infer channel axis, which is required for vector palettization."
            )
        channel_axis = 0

    return _grouped_channelwise_compress(
        original_data,
        mode,
        nbits,
        channel_axis,
        channel_group_size,
        lut_function,
        cluster_dim,
        num_kmeans_workers,
    )


def _infer_palettization_block_sizes_and_channel_axis(
    op: Operation,
    weight_shape: Sequence[int],
    granularity: _CompressionGranularity,
    group_size: int,
) -> tuple[Sequence[int], int]:
    """Infer per-axis block sizes and the output channel axis for palettization.

    The channel axis is auto-selected based on which downstream ops consume the
    constant, making the result hardware-friendly.

    Args:
        op (Operation): The constant operation whose downstream consumers determine
            the channel axis.
        weight_shape (Sequence[int]): Shape of the weight tensor.
        granularity (CompressionGranularity): Compression granularity; one of the
            :class:`CompressionGranularity` string values.
        group_size (int): Group size applied along the output channel axis for
            ``"per_grouped_channel"`` granularity.

    Returns:
        tuple[Sequence[int], int]: ``(block_sizes, output_channel_axis)`` where
            ``block_sizes[i]`` is the block size for axis ``i`` (0 means no blocking).
    """
    input_channel_axis, output_channel_axis = _select_input_output_channel_axis(op)

    if input_channel_axis is None:
        logger.warning("Cannot determine input_channel_axis for block_sizes, use 1 by default.")
        input_channel_axis = 1
    if output_channel_axis is None:
        logger.warning(
            "Cannot determine output_channel_axis for block_sizes, use 0 by default.",
        )
        output_channel_axis = 0

    block_sizes = [0] * len(weight_shape)
    if granularity == _CompressionGranularity.PER_TENSOR:
        input_channel_block_size = 0
        output_channel_block_size = 0
    elif granularity == _CompressionGranularity.PER_CHANNEL:
        input_channel_block_size = 0
        output_channel_block_size = 1
    else:
        assert granularity == _CompressionGranularity.PER_GROUPED_CHANNEL
        assert isinstance(group_size, int)
        input_channel_block_size = 0
        output_channel_block_size = group_size

    if input_channel_axis < len(block_sizes):
        block_sizes[input_channel_axis] = input_channel_block_size
    if output_channel_axis < len(block_sizes):
        block_sizes[output_channel_axis] = output_channel_block_size

    return block_sizes, output_channel_axis


def _is_cluster_dim_valid(op: Operation, cluster_dim: int, channel_axis: int) -> bool:
    """Check op-dependent restrictions for cluster_dim.

    For conv2d, the weight shape is ``[C_out, C_in/groups, ...]`` but the effective
    shape per group is ``[C_out/groups, C_in/groups, ...]``, so the effective dim on
    ``channel_axis`` must be divisible by ``cluster_dim``.

    Args:
        op (Operation): The constant operation to check.
        cluster_dim (int): The cluster dimension to validate.
        channel_axis (int): The output channel axis.

    Returns:
        bool: ``True`` if ``cluster_dim`` is valid for the given op, ``False`` otherwise.
    """
    result_type = op.result.type  # type: ignore[attr-defined]
    if channel_axis < 0:
        channel_axis += result_type.rank

    for child_op_use in op.result.uses:
        child_op: Operation = cast("Operation", child_op_use.owner)
        if child_op.name == _CONV2D_OP:
            effective_shape = list(result_type.shape)
            group_op = child_op.groups.owner  # type: ignore[attr-defined]
            if group_op is not None and group_op.name.endswith("constant"):
                group_val = int(_get_constant_value_as_np_array(group_op))
                if group_val > 1:
                    effective_shape[0] //= group_val
            if effective_shape[channel_axis] % cluster_dim != 0:
                return False
    return True
