# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Quantization utilities and helper functions."""

import torch
from torchao.quantization.quant_primitives import _get_reduction_params


def get_quantization_shapes(
    tensor: torch.Tensor,
    block_size: tuple[int, ...] | list[int],
) -> tuple[torch.Size, list[int], list[int]]:
    """
    Calculate shapes for block-wise quantization.

    Returns shapes needed to reshape tensors for quantization parameter computation.
    The blockwise shape exposes the block structure, and the reduced shape indicates
    where quantization parameters are computed (with 1s in reduced dimensions).

    Args:
        tensor: Input tensor to quantize
        block_size: Block size for granularity

    Returns:
        Tuple of (original_shape, blockwise_shape, reduced_shape)

    Example:
        For tensor shape [4, 16] with per-block (axis=1, block_size=4):
        - original_shape: [4, 16]
        - blockwise_shape: [4, 4, 4] - reshaped to expose 4 blocks of size 4
        - reduced_shape: [4, 1, 4] - qparams computed per block (1 in middle dim)

        For tensor shape [8, 32] with per-channel (axis=0):
        - original_shape: [8, 32]
        - blockwise_shape: [8, 32] - no blocking needed
        - reduced_shape: [1, 32] - qparams computed per output channel
    """

    original_shape = tensor.shape
    blockwise_shape, reduction_dims = _get_reduction_params(block_size, tensor.size())
    reduced_shape = list(blockwise_shape)
    for i in reduction_dims:
        reduced_shape[i] = 1

    return original_shape, blockwise_shape, reduced_shape
