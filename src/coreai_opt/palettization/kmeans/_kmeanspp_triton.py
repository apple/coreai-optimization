# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Fused k-means++ seeding steps for two-number vectors on a CUDA GPU (Triton).

A step needs two things per block: the weighted potential each candidate would leave, and the
nearest-seed distances once one candidate is chosen. Two kernels compute exactly those, reading
the points once and writing nothing the size of ``(B, N, trials)``. The random draws stay
``torch.multinomial`` in the caller. Squared distances are computed directly, as
``(x0 - c0)^2 + (x1 - c1)^2``, instead of by the norm trick.
"""

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - no Triton: the caller keeps the torch loop
    triton = None

#: Most candidates per step the kernel handles (the recipe uses ``2 + int(log(k))``).
MAX_TRIALS = 16
_BLOCK = 1024


def available(x: torch.Tensor, n_trials: int) -> bool:
    """Whether the fused step can run on ``x``: CUDA, Triton, two-number vectors, few trials."""
    return (
        triton is not None
        and x.is_cuda
        and x.shape[-1] == 2
        and x.dtype == torch.float32
        and n_trials <= MAX_TRIALS
    )


if triton is not None:

    @triton.jit
    def _potentials_kernel(
        x_ptr, closest_ptr, cand_ptr, out_ptr, n, chunks, TRIALS: tl.constexpr, BLOCK: tl.constexpr
    ):
        """``out[b, chunk, t]``: this chunk's sum of ``min(closest, |x - candidate_t|^2)``."""
        block = tl.program_id(0)
        chunk = tl.program_id(1)
        rows = chunk * BLOCK + tl.arange(0, BLOCK)
        mask = rows < n
        base = block * n + rows
        x0 = tl.load(x_ptr + 2 * base, mask=mask, other=0.0)
        x1 = tl.load(x_ptr + 2 * base + 1, mask=mask, other=0.0)
        closest = tl.load(closest_ptr + base, mask=mask, other=0.0)
        for t in tl.static_range(TRIALS):
            c0 = tl.load(cand_ptr + (block * TRIALS + t) * 2)
            c1 = tl.load(cand_ptr + (block * TRIALS + t) * 2 + 1)
            d0 = x0 - c0
            d1 = x1 - c1
            nearest = tl.minimum(closest, d0 * d0 + d1 * d1)
            total = tl.sum(tl.where(mask, nearest, 0.0), axis=0)
            tl.store(out_ptr + (block * chunks + chunk) * TRIALS + t, total)

    @triton.jit
    def _update_kernel(x_ptr, closest_ptr, chosen_ptr, partial_ptr, n, chunks, BLOCK: tl.constexpr):
        """``closest = min(closest, |x - chosen|^2)`` in place; ``partial[b, chunk]`` = its sum."""
        block = tl.program_id(0)
        chunk = tl.program_id(1)
        rows = chunk * BLOCK + tl.arange(0, BLOCK)
        mask = rows < n
        base = block * n + rows
        x0 = tl.load(x_ptr + 2 * base, mask=mask, other=0.0)
        x1 = tl.load(x_ptr + 2 * base + 1, mask=mask, other=0.0)
        closest = tl.load(closest_ptr + base, mask=mask, other=0.0)
        c0 = tl.load(chosen_ptr + block * 2)
        c1 = tl.load(chosen_ptr + block * 2 + 1)
        d0 = x0 - c0
        d1 = x1 - c1
        nearest = tl.minimum(closest, d0 * d0 + d1 * d1)
        tl.store(closest_ptr + base, nearest, mask=mask)
        tl.store(partial_ptr + block * chunks + chunk, tl.sum(tl.where(mask, nearest, 0.0), axis=0))


def candidate_potentials(
    x: torch.Tensor, closest: torch.Tensor, cand_pts: torch.Tensor
) -> torch.Tensor:
    """``(B, trials)``: for each candidate, the summed nearest-seed distance if it became a seed.

    ``x`` is ``(B, N, 2)``, ``closest`` ``(B, N)`` the distance to the nearest seed so far, and
    ``cand_pts`` ``(B, trials, 2)``.
    """
    b, n, _ = x.shape
    trials = cand_pts.shape[1]
    chunks = triton.cdiv(n, _BLOCK)
    out = torch.empty(b, chunks, trials, device=x.device, dtype=torch.float32)
    _potentials_kernel[(b, chunks)](
        x, closest, cand_pts.contiguous(), out, n, chunks, TRIALS=trials, BLOCK=_BLOCK
    )
    return out.sum(dim=1)


def update_closest(x: torch.Tensor, closest: torch.Tensor, chosen: torch.Tensor) -> torch.Tensor:
    """Fold the chosen seed ``(B, 2)`` into ``closest`` (in place); return the ``(B,)`` sums."""
    b, n, _ = x.shape
    chunks = triton.cdiv(n, _BLOCK)
    partial = torch.empty(b, chunks, device=x.device, dtype=torch.float32)
    _update_kernel[(b, chunks)](x, closest, chosen.contiguous(), partial, n, chunks, BLOCK=_BLOCK)
    return partial.sum(dim=1)
