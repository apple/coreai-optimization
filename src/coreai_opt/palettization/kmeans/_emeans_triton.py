# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Fused nearest-centroid kernels for batched e-means on two-number vectors (Triton).

A step of e-means takes a batch of points per block, finds each point's nearest centroid, and
adds the point to that centroid's running ``(x, y)`` sums and count. The unfused torch step does
this with a gather, a batched matrix product whose inner size is 2, a ``(B, batch, K)`` distance
tile written to memory and read back by ``argmin``, and two ``scatter_add`` calls with their
zero-fills. This kernel keeps the distances in registers: it reads each point once, scans the
centroids in tiles keeping a running minimum, and adds the point to the packed statistics
buffer with atomics.

The scores are ``|c|^2 - 2 x.c`` (the point's own ``|x|^2`` does not change the argmin), rounded
the way the torch step rounds them: products and sums are separate operations with no fusing, the
dot product accumulating ``x0*c0`` first. Ties go to the lowest index, as ``torch.argmin`` does.

Two entry points share one scan of the centroids, so they cannot disagree on a point:
:func:`assign_accumulate` is the training step, and :func:`assign_labels` the final exact
assignment, which only stores each point's index.
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - no Triton: the caller keeps the torch step
    triton = None

_MAX_BLOCK_K = 64


def available(x: torch.Tensor) -> bool:
    """Whether the fused step can run on ``x``: CUDA, Triton, fp32 two-number vectors."""
    return triton is not None and x.is_cuda and x.shape[-1] == 2 and x.dtype == torch.float32


if triton is not None:

    @triton.jit
    def _nearest_centroid(
        x0,
        x1,
        centroids_ptr,
        block,
        k,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Index of the centroid of ``block`` nearest to each of the points ``(x0, x1)``."""
        best = tl.full([BLOCK_M], float("inf"), tl.float32)
        best_idx = tl.zeros([BLOCK_M], tl.int32)
        for start in range(0, k, BLOCK_K):
            offs_k = start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < k
            c_base = (block * k + offs_k) * 2
            c0 = tl.load(centroids_ptr + c_base, mask=mask_k, other=0.0)
            c1 = tl.load(centroids_ptr + c_base + 1, mask=mask_k, other=0.0)
            c_sq = c0 * c0 + c1 * c1
            # separate multiplies and adds, as torch's bmm and elementwise ops round them
            dot = x0[:, None] * c0[None, :]
            dot = tl.fma(x1[:, None], c1[None, :], dot)
            score = c_sq[None, :] - 2.0 * dot
            score = tl.where(mask_k[None, :], score, float("inf"))
            tile_best = tl.min(score, axis=1)
            tile_idx = tl.argmin(score, axis=1).to(tl.int32) + start
            better = tile_best < best
            best = tl.where(better, tile_best, best)
            best_idx = tl.where(better, tile_idx, best_idx)
        return best_idx

    @triton.jit
    def _assign_accumulate_kernel(
        x_ptr,
        picked_ptr,
        centroids_ptr,
        stats_ptr,
        n,
        m,
        k,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Add each picked point of block ``b`` to the stats row of its nearest centroid.

        ``stats[b, j] = (sum x, sum y, count)`` for cluster ``j``. Grid: (blocks, cdiv(m, BLOCK_M)).
        """
        block = tl.program_id(0)
        offs_m = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < m
        row = tl.load(picked_ptr + offs_m, mask=mask_m, other=0).to(tl.int64)
        x_base = (block * n + row) * 2
        x0 = tl.load(x_ptr + x_base, mask=mask_m, other=0.0)
        x1 = tl.load(x_ptr + x_base + 1, mask=mask_m, other=0.0)
        best_idx = _nearest_centroid(x0, x1, centroids_ptr, block, k, BLOCK_M, BLOCK_K)

        out = (block * k + best_idx.to(tl.int64)) * 3
        tl.atomic_add(stats_ptr + out, x0, mask=mask_m)
        tl.atomic_add(stats_ptr + out + 1, x1, mask=mask_m)
        tl.atomic_add(stats_ptr + out + 2, tl.full([BLOCK_M], 1.0, tl.float32), mask=mask_m)

    @triton.jit
    def _assign_labels_kernel(
        x_ptr,
        centroids_ptr,
        labels_ptr,
        n,
        k,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Store the index of each point's nearest centroid. Grid: (cdiv(n, BLOCK_M), blocks)."""
        block = tl.program_id(1)
        offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < n
        row = block.to(tl.int64) * n + offs_m
        x0 = tl.load(x_ptr + row * 2, mask=mask_m, other=0.0)
        x1 = tl.load(x_ptr + row * 2 + 1, mask=mask_m, other=0.0)
        best_idx = _nearest_centroid(x0, x1, centroids_ptr, block, k, BLOCK_M, BLOCK_K)
        tl.store(labels_ptr + row, best_idx.to(tl.int64), mask=mask_m)


def assign_accumulate(
    x: torch.Tensor, picked: torch.Tensor, centroids: torch.Tensor, stats: torch.Tensor
) -> None:
    """Add the rows ``picked`` of every block of ``x`` to ``stats`` (``(B, K, 3)``, zeroed).

    ``x`` is ``(B, N, 2)``, ``centroids`` ``(B, K, 2)``, ``picked`` the ``(m,)`` row indices
    shared by all blocks. All are contiguous fp32 on one CUDA device (``picked`` int64).
    """
    b, n, _ = x.shape
    k = centroids.shape[1]
    m = picked.shape[0]
    block_k = min(triton.next_power_of_2(k), _MAX_BLOCK_K)
    block_m = 128
    _assign_accumulate_kernel[(b, triton.cdiv(m, block_m))](
        x,
        picked,
        centroids,
        stats,
        n,
        m,
        k,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        enable_fp_fusion=False,
    )


def assign_labels(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """``(B, N)`` int64 index of the nearest centroid of every point of ``x``.

    ``x`` is ``(B, N, 2)`` and ``centroids`` ``(B, K, 2)``, both fp32 on one CUDA device.
    """
    x, centroids = x.contiguous(), centroids.contiguous()
    b, n, _ = x.shape
    k = centroids.shape[1]
    labels = torch.empty(b, n, dtype=torch.long, device=x.device)
    block_m = 128
    _assign_labels_kernel[(triton.cdiv(n, block_m), b)](
        x,
        centroids,
        labels,
        n,
        k,
        BLOCK_M=block_m,
        BLOCK_K=min(triton.next_power_of_2(k), _MAX_BLOCK_K),
        enable_fp_fusion=False,
    )
    return labels
