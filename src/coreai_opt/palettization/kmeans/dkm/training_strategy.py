# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""DKM (Differentiable K-Means) training strategy and its specification."""

from functools import partial

import torch
from pydantic import Field, PositiveInt

from coreai_opt.palettization.kmeans.kmeans_fake_palettize import _KMeansFakePalettize
from coreai_opt.palettization.spec import PalettizationSpec, PerTensorGranularity
from coreai_opt.palettization.spec.training_strategy import TrainingStrategy, TrainingStrategySpec

from ._dispatch import get_forward as _get_forward
from ._forward import dkm_forward as _dkm_forward

# Cluster dim values which are supported by DKM
SUPPORTED_CLUSTER_DIMS = {1, 2, 4}


def _quantize_centroids(module: _KMeansFakePalettize, centroids: torch.Tensor) -> torch.Tensor:
    """Bridge DKM's ``(P, K, D)`` centroid shape to ``module.quantize_lut()``'s
    ``(num_blocks, num_clusters[, cluster_dim])`` convention, which drops the
    trailing ``D`` axis when ``cluster_dim == 1`` (same as ``module._raw_lut``).
    """
    if module.cluster_dim == 1:
        # (P, K, 1) -> (P, K) for quantize_lut, then restore (P, K, 1)
        return module.quantize_lut(centroids.squeeze(-1)).unsqueeze(-1)
    return module.quantize_lut(centroids)


class _DKMTrainingStrategy(TrainingStrategy):
    """Differentiable soft k-means training step. Mutates ``module.centroids``
    in place every training step via the EM loop.

    For details on DKM, refer to:
    Cho et al., "DKM: Differentiable K-Means Clustering Layer for Neural Network Compression",
    ICLR 2022. arXiv:2108.12659
    """

    def __init__(
        self,
        tau: float = 1e-4,
        kmeans_max_iter: int = 3,
        enforce_zero: bool = False,
        zero_threshold: float = 1e-7,
    ):
        self.tau = tau
        self.kmeans_max_iter = kmeans_max_iter
        self.enforce_zero = enforce_zero
        self.zero_threshold = zero_threshold

    def train_forward(self, module: _KMeansFakePalettize, weight: torch.Tensor) -> torch.Tensor:
        """Run one DKM EM step and reconstruct the weight from the updated centroids."""
        if not isinstance(module, _KMeansFakePalettize):
            raise TypeError(
                "DKM training strategy only supports k-means palettization "
                f"(_KMeansFakePalettize), got {type(module).__name__}."
            )
        if module.cluster_dim not in SUPPORTED_CLUSTER_DIMS:
            raise ValueError(
                f"DKM supports cluster_dim in {SUPPORTED_CLUSTER_DIMS}, got {module.cluster_dim}."
            )

        # Shapes: P=#blocks, N=vectors/block, D=cluster_dim, K=#clusters.

        # Scale (if enabled), reshape, block-split, and vectorize -> (P, N, D).
        weights, vectorization = module.vectorize(weight)

        quantize_fn = partial(_quantize_centroids, module)

        # Clone is needed to ensure "centroids" and "module.centroids" do not share the same storage
        # (occurs in the case of no LUT quantization).
        # DKM requires gradients to flow through "centroids", however if they share the same
        # storage, module.centroids.copy_() further below will invalidate gradients.
        centroids = quantize_fn(module.centroids).clone()  # (P, K, D)

        assert centroids.shape[0] == weights.shape[0], (
            f"partition count mismatch: centroids P={centroids.shape[0]} "
            f"vs weight P={weights.shape[0]}"
        )

        forward = _get_forward(weight.device)

        if forward is not None:
            reconstructed_weights, centroids_new = forward(
                weights,
                centroids,
                self.tau,
                kmeans_max_iter=self.kmeans_max_iter,
                zero_threshold=self.zero_threshold,
                enforce_zero=self.enforce_zero,
                quantize_fn=quantize_fn,
            )
        else:
            reconstructed_weights, centroids_new = _dkm_forward(
                weights,
                centroids,
                self.tau,
                self.kmeans_max_iter,
                zero_threshold=self.zero_threshold,
                enforce_zero=self.enforce_zero,
                quantize_fn=quantize_fn,
            )

        module.centroids.copy_(centroids_new.detach())
        module._indices_stale = True

        # reconstructed_weights is (P, N, D): undo vectorize + block-split (+ unscale)
        # to rebuild the weight.
        return module.devectorize(reconstructed_weights, vectorization)


@TrainingStrategySpec.register("dkm")
class DKMTrainingSpec(TrainingStrategySpec):
    """DKM (Differentiable K-Means) training-strategy hyperparameters.

    Attributes:
        tau: Softmax temperature for DKM's soft centroid assignment. Smaller
            values sharpen toward hard k-means; larger values soften it.
            Typical range: 1e-5 to 1e-3.
        kmeans_max_iter: Number of soft-assignment EM iterations per DKM
            training forward pass.
        enforce_zero: If True, pins the centroid nearest the origin to
            exactly zero. Useful for joint pruning + palettization.
        zero_threshold: Floor applied to the softmax assignment weights and the
            per-centroid mass (denominator) for numerical stability at small
            ``tau``.
    """

    _strategy_cls = _DKMTrainingStrategy

    tau: float = Field(default=1e-4, gt=0)
    kmeans_max_iter: PositiveInt = 3
    enforce_zero: bool = False
    zero_threshold: float = Field(default=1e-7, gt=0)


def default_dkm_weight_palettization_spec() -> PalettizationSpec:
    """Return the default DKM weight-palettization spec (4-bit, per-tensor, no LUT quantization)."""
    return PalettizationSpec(
        n_bits=4,
        lut_qspec=None,
        granularity=PerTensorGranularity(),
        cluster_dim=1,
        enable_per_channel_scale=False,
        training_strategy_spec=DKMTrainingSpec(),
    )
