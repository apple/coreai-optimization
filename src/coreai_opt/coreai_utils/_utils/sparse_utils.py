# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Sparsification utilities for Core AI Optimization passes."""

from __future__ import annotations

import logging
from collections import namedtuple

import numpy as np

logger = logging.getLogger(__name__)

SparseParams = namedtuple("SparseParams", "nonzero_data mask")


def _produce_sparse_param(val: np.ndarray) -> SparseParams:
    flattened_val = val.flatten()
    nonzero_mask = flattened_val != 0
    nonzero_data = flattened_val[nonzero_mask]
    mask = nonzero_mask.reshape(val.shape).astype(np.uint8)
    return SparseParams(nonzero_data=nonzero_data, mask=mask)


def _compress_by_magnitude(
    val: np.ndarray,
    target_sparsity: float,
    block_size: int | None = None,
    dim: int | None = None,
) -> SparseParams | None:
    """Prune weights by zeroing the lowest-magnitude elements.

    Args:
        val (np.ndarray): Weight tensor to sparsify.
        target_sparsity (float): Fraction of weights to zero out, in ``[0, 1]``.
        block_size (int | None): Block size for structured block sparsity along ``dim``.
        dim (int | None): Channel axis for block sparsity (0 or 1).

    Returns:
        SparseParams | None: Sparse parameters, or ``None`` if pruning is inapplicable.
    """

    def _apply_block_sparsity(val: np.ndarray, block_size: int, dim: int) -> np.ndarray:
        shape = val.shape
        rank = len(shape)
        if dim not in [0, 1]:
            raise ValueError("block sparsity pruning only supports dim [0, 1].")
        if rank not in [2, 3, 4, 5]:
            raise ValueError("block sparsity only supports weights of rank [2, 3, 4, 5].")
        if dim == 1:
            perm = [1, 0] + list(range(2, rank))
            val = np.transpose(val, axes=perm)

        channel = val.shape[0]
        if channel % block_size != 0:
            pad_size = block_size - channel % block_size
            pad_value = [(0, pad_size)] + [(0, 0)] * (rank - 1)
            val = np.pad(val, pad_value)
        shape_padded = val.shape
        assert shape_padded[0] % block_size == 0

        new_shape = list(shape_padded)
        new_shape.insert(1, block_size)
        new_shape[0] = new_shape[0] // block_size
        val = np.reshape(val, new_shape)

        val = val * val
        val = np.sum(val, axis=1, keepdims=True)
        val = np.sqrt(val)

        reps = [1] * (rank + 1)
        reps[1] = block_size
        val = np.tile(val, reps)
        val = np.reshape(val, shape_padded)
        val = val[:channel]

        if dim == 1:
            val = np.transpose(val, axes=perm)
        return val

    magnitude_map = np.abs(val)
    if block_size is not None:
        if dim is None:
            raise ValueError("`dim` must be provided when `block_size` is specified.")
        channel = magnitude_map.shape[dim]
        if block_size > channel / 2:
            logger.warning(
                "block_size > channel / 2 is not applicable for block sparsity. "
                "Got block_size = %d, channel = %d. Skipped.",
                block_size,
                channel,
            )
            return None
        magnitude_map = _apply_block_sparsity(magnitude_map, block_size, dim)

    q = target_sparsity * 100
    if q == 100:
        val = 0 * val
    elif q != 0:
        val = np.where(magnitude_map <= np.percentile(magnitude_map, q), 0, val)
    return _produce_sparse_param(val)


def _compress_by_nm_sparsity(
    val: np.ndarray,
    n_m_ratio: tuple[int, int],
    dim: int,
) -> SparseParams | None:
    """Prune weights using n:m structured sparsity.

    Args:
        val (np.ndarray): Weight tensor to sparsify.
        n_m_ratio (tuple[int, int]): ``(n, m)`` — zero the ``n`` lowest-magnitude
            elements out of every ``m`` consecutive elements along ``dim``.
        dim (int): Channel axis for n:m pruning (0 or 1).

    Returns:
        SparseParams | None: Sparse parameters, or ``None`` if pruning is inapplicable.
    """
    n, m = n_m_ratio
    assert n <= m
    shape = val.shape
    rank = len(shape)
    if dim not in [0, 1]:
        raise ValueError("n:m pruning only supports dim [0, 1].")
    if rank not in [2, 3, 4, 5]:
        raise ValueError("n:m pruning only supports weights of rank [2, 3, 4, 5].")

    perm = list(range(2, rank)) + [0, 1]
    if dim == 0:
        perm[-2], perm[-1] = 1, 0
    weight = np.copy(np.transpose(val, axes=perm))
    shape_begin = weight.shape

    weight = np.reshape(weight, (-1, weight.shape[-1]))
    channel = weight.shape[-1]
    if m > channel / 2:
        logger.warning(
            "m > channel / 2 is not applicable for n:m pruning. Got m = %d, channel = %d. Skipped.",
            m,
            channel,
        )
        return None
    if channel % m != 0:
        pad_size = m - channel % m
        weight = np.pad(weight, ((0, 0), (0, pad_size)))
    shape_padded = weight.shape
    assert shape_padded[-1] % m == 0

    weight = np.reshape(weight, (-1, m))
    magnitude = np.abs(weight)
    indices = np.argsort(magnitude, axis=-1)[:, :n]

    n_m_mask = np.zeros(weight.shape).astype(val.dtype)
    np.put_along_axis(n_m_mask, indices, 1.0, axis=-1)
    n_m_mask = np.reshape(n_m_mask, shape_padded)
    n_m_mask = n_m_mask[:, :channel]

    n_m_mask = np.reshape(n_m_mask, shape_begin)
    perm_back = [perm.index(i) for i in range(rank)]
    n_m_mask = np.transpose(n_m_mask, axes=perm_back)

    val = val * (1 - n_m_mask)
    return _produce_sparse_param(val)
