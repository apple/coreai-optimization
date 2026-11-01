# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""PyTorch implementation of DKM."""

from collections.abc import Callable

import torch
import torch.nn.functional as F


def _pin_zero_centroid(centroids: torch.Tensor) -> torch.Tensor:
    """Pin each partition's nearest-origin centroid to exactly zero, keeping grad on the rest."""
    centroids = centroids.clone()
    p_idx = torch.arange(centroids.shape[0], device=centroids.device)
    zero_idx = centroids.detach().norm(dim=-1).argmin(dim=1)  # (P,) nearest-origin code
    centroids[p_idx, zero_idx, :] = centroids[p_idx, zero_idx, :].detach().zero_()
    return centroids


def dkm_forward(
    weights: torch.Tensor,
    centroids: torch.Tensor,
    tau: float,
    kmeans_max_iter: int,
    zero_threshold: float = 1e-7,
    enforce_zero: bool = False,
    quantize_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run DKM soft k-means and return the reconstructed weights and updated centroids.

    Args:
        weights: Weight tensor of shape ``(P, N, D)``.
        centroids: Initial centroid tensor of shape ``(P, K, D)``.
        tau: Softmax temperature.
        kmeans_max_iter: Number of soft-assignment iterations.
        zero_threshold: Floor applied to the softmax assignment weights and the
            per-centroid mass (denominator) for numerical stability at small
            ``tau``.
        enforce_zero: If True, pins each partition's nearest-origin centroid to
            exactly zero after every update step, keeping an exact-zero codebook
            entry for joint pruning + palettization.
        quantize_fn: If provided, fake-quantizes centroids after every update
            step (after ``enforce_zero``'s pin, if both are set), keeping the
            EM loop quantization-aware at every iteration rather than only at
            the initial seed.

    Returns:
        ``(reconstructed_weights, centroids_new)`` — both of shape matching ``weights``
        and ``centroids`` respectively. ``reconstructed_weights`` is differentiable
        w.r.t. the input ``weights`` and ``centroids``.
    """
    # Work in float32 for numerical stability regardless of input dtype
    weights_f = weights.float()
    centroids_f = centroids.float()

    attn = None
    for _ in range(kmeans_max_iter):
        # (P, N, K) — squared L2 distance from each weight to each centroid
        dist = (weights_f.unsqueeze(-2) - centroids_f.unsqueeze(-3)).pow(2).sum(-1)

        # Soft assignment: (P, N, K)
        attn = F.softmax(-dist / tau, dim=-1).clamp(min=zero_threshold)

        # Weighted centroid update: (P, K, D)
        # attn.mT is (P, K, N), weights_f is (P, N, D) → matmul gives (P, K, D)
        attn_sum = attn.sum(dim=1, keepdim=True).mT.clamp(min=zero_threshold)
        centroids_new = attn.mT @ weights_f / attn_sum

        # Pin each partition's nearest-origin centroid to exactly zero so the
        # codebook keeps an exact-zero entry (block-sparse code) that never drifts.
        if enforce_zero:
            centroids_new = _pin_zero_centroid(centroids_new)

        if quantize_fn is not None:
            centroids_new = quantize_fn(centroids_new)

        centroids_f = centroids_new

    # Reconstructed weights: (P, N, D)
    reconstructed_weights = attn @ centroids_f

    return reconstructed_weights.to(weights.dtype), centroids_f.to(centroids.dtype)
