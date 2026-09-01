# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import logging
from collections.abc import Callable

import numpy as np
import torch

from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.deps import _kmeans1d
from coreai_opt.palettization.spec import (
    PalettizationGranularity,
    PerGroupedChannelGranularity,
    PerTensorGranularity,
)
from coreai_opt.palettization.spec.errors import (
    _IncompatibleClusterDimError,
    _IncompatibleGranularityError,
)
from coreai_opt.palettization.spec.fake_palettize import _FakePalettizeImplBase
from coreai_opt.palettization.spec.training_strategy import (
    DefaultTrainingSpec,
    TrainingStrategySpec,
)
from coreai_opt.quantization.spec import (
    PerChannelGranularity as _QuantPerChannelGranularity,
    QuantizationComponentFactory,
    QuantizationSpec,
)

from ._efficient_kmeans import _EfficientKMeans
from .kmeans_support_mixins import _LinearPalettizationMixin
from .supported_ops_registry import _KMeansPalettizerSupportedOpsRegistry

logger = logging.getLogger(__name__)


@_FakePalettizeImplBase.register("default")
class _KMeansFakePalettize(_FakePalettizeImplBase):
    """K-means based palettization implementation for neural network weights.

    This class implements weight palettization using k-means clustering to reduce the
    number of unique values in weight tensors. The palettization process creates a
    lookup table (LUT) of cluster centroids and maps each original weight to its
    nearest centroid index.

    Supports both per-tensor and per-grouped-channel granularities, fast k-means mode
    with optimizations for fp16 weights, and configurable bit precision (n_bits).

    The workflow proceeds in two steps:

    1. ``_initialize()``: Clusters weights using k-means and computes the
       resulting LUT and indices
    2. ``_palettize()``: Reconstructs palettized weights from LUT and indices

    Example:
        >>> from coreai_opt.palettization.spec import (
        ...     PalettizationSpec,
        ...     PerTensorGranularity,
        ... )
        >>> spec = PalettizationSpec(
        ...     n_bits=2, granularity=PerTensorGranularity(), cluster_dim=1
        ... )
        >>> palettizer = _KMeansFakePalettize(**spec.__dict__)
        >>> weight = torch.randn(4, 4)
        >>> palettizer._initialize(weight)
        >>> palettized_weight = palettizer._palettize(palettizer.lut, palettizer.indices, weight)
    """

    def __init__(
        self,
        n_bits: int,
        lut_qspec: QuantizationSpec | None,
        granularity: PalettizationGranularity,
        cluster_dim: int,
        enable_per_channel_scale: bool,
        sensitivities: torch.Tensor = None,
        enable_fast_kmeans_mode: bool = True,
        rounding_precision: int = 4,
        op_to_optimize: Callable | None = None,
        training_strategy_spec: TrainingStrategySpec | None = None,
        sparsity: float | None = None,
    ):
        super().__init__(
            n_bits=n_bits,
            lut_qspec=lut_qspec,
            granularity=granularity,
            cluster_dim=cluster_dim,
            enable_per_channel_scale=enable_per_channel_scale,
            sparsity=sparsity,
        )

        self.enable_fast_kmeans_mode = enable_fast_kmeans_mode
        self.rounding_precision = rounding_precision
        self._sensitivities = sensitivities

        # Create LUT fake quantizer if LUT quantization is enabled.
        # Use PerChannelGranularity(axis=0) so the stacked LUT tensor
        # (num_blocks, num_clusters[, cluster_dim]) gets independent
        # quantization parameters per palettization group.
        if self.lut_qspec is not None:
            batched_lut_qspec = self.lut_qspec.model_copy(
                update={"granularity": _QuantPerChannelGranularity(axis=0)}
            )
            self._lut_fake_quantizer = QuantizationComponentFactory.create_fake_quantizer(
                spec=batched_lut_qspec,
                quantization_target=CompressionTargetTensor.LUT,
            )
        else:
            self._lut_fake_quantizer = None

        # Instantiate op specific reshape strategy
        registry = _KMeansPalettizerSupportedOpsRegistry
        if op_to_optimize is not None and registry.supports_operation(op_to_optimize):
            palettization_mixin = registry.get_registry_entry_for_func(op_to_optimize)
            self.reshape_strategy = palettization_mixin()
        else:
            # Use _LinearPalettizationMixin as default (no-op for 2D tensors)
            logger.info(
                f"No reshape strategy found for {op_to_optimize}. "
                f"Using _LinearPalettizationMixin as default."
            )
            self.reshape_strategy = _LinearPalettizationMixin()

        # Resolve axis default for PerGroupedChannelGranularity using the op's mixin.
        if (
            isinstance(self.granularity, PerGroupedChannelGranularity)
            and self.granularity.axis is None
        ):
            self.granularity = self.granularity.model_copy(
                update={"axis": self.reshape_strategy.default_axis}
            )

        self.register_buffer("centroids", None)
        self._centroids_initialized: bool = False
        self._indices_stale: bool = True

        # Resolve and construct the training strategy from its paired spec.
        if training_strategy_spec is None:
            training_strategy_spec = DefaultTrainingSpec()
        self._training_strategy = training_strategy_spec.build_strategy()

    @property
    def sensitivities(self) -> torch.Tensor | None:
        """Get the sensitivity values used for weighted k-means clustering."""
        return self._sensitivities

    @sensitivities.setter
    def sensitivities(self, value: torch.Tensor | None) -> None:
        """Set sensitivity values. Centroids and indices are recomputed
        on the next forward call.
        """
        self._sensitivities = value
        self._centroids_initialized = False
        self._indices_stale = True

    def ensure_initialized(self, tensor: torch.Tensor) -> None:
        """Cluster centroids on first use; disable on an incompatible tensor.

        Re-clusters on every call while ``observer_enabled`` is set.
        """
        if self._centroids_initialized and self.observer_enabled[0] == 0:
            return
        try:
            self._initialize(tensor.detach())
        except (_IncompatibleClusterDimError, _IncompatibleGranularityError) as e:
            logger.warning(
                f"Tensor incompatible with configured spec: {e}. Skipping palettization."
            )
            self._disabled = True

    def forward_enabled(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self._training_strategy.train_forward(self, tensor)
        return self.hard_assign(tensor)

    def _initialize(self, weight: torch.Tensor) -> None:
        """(Re-)initialize centroids and indices from k-means.

        Also seeds the LUT quantizer's observed qparams from these initial
        centroids (see ``quantize_lut()``) — without this, a strategy that
        never otherwise calls ``quantize_lut()`` (e.g. the default one-shot
        strategy) would leave it permanently unobserved, and ``lut`` (which
        reads frozen qparams via ``get_qparams()``, never observes) would
        have nothing valid to read.

        Use ``_refresh_indices()`` instead to recompute indices from
        already-updated centroids, without re-clustering. ``lut`` is never
        stored — it's derived fresh from ``centroids`` on every access.
        """
        self.centroids, indices = self._cluster_to_centroids(weight, self._sensitivities)
        self.indices = indices.detach()

        # Run self.quantize_lut to seed lut quantizer qparams
        self.quantize_lut(self._raw_lut(self.centroids))

        self._centroids_initialized = True
        self._indices_stale = False

    def _maybe_refresh_indices(self, weight: torch.Tensor) -> None:
        """Recompute indices from the current centroids if self._indices_stale is true,
        without re-clustering.
        """
        if self._indices_stale:
            self.indices = self._assign_indices(weight, self.centroids).detach()
            self._indices_stale = False

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """Load centroids from a checkpoint, reconstructing them from a legacy
        ``lut`` buffer when present.
        """
        lut_key, centroids_key = prefix + "lut", prefix + "centroids"
        if centroids_key not in state_dict and lut_key in state_dict:
            state_dict[centroids_key] = self._centroids_from_lut(state_dict[lut_key])
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        if self.centroids is not None:
            self._centroids_initialized = True
            self._indices_stale = True

    def _centroids_from_lut(self, lut: torch.Tensor) -> torch.Tensor:
        """Invert ``_reshape_lut_tensor`` to recover ``(num_blocks, num_clusters,
        cluster_dim)`` centroids from a stored 4D LUT tensor.
        """
        if lut.ndim != 4:
            raise ValueError(
                "Legacy 'lut' buffer must be 4D (num_blocks_axis0, num_blocks_axis1, "
                f"num_clusters, cluster_dim); got shape {tuple(lut.shape)}."
            )
        ungrouped_dim = 0 if self.granularity.axis == 1 else 1
        centroids = lut.squeeze(-1) if self.cluster_dim == 1 else lut
        centroids = centroids.squeeze(ungrouped_dim)
        return centroids.unsqueeze(-1) if self.cluster_dim == 1 else centroids

    @property
    def lut(self) -> torch.Tensor | None:
        """Lookup table dequantized from ``centroids`` using the LUT
        quantizer's frozen qparams (read via ``get_qparams()``, never
        re-observed here — see ``quantize_lut()`` for the training-time
        observing path). Cheap (independent of weight size), so never cached.
        """
        if self.centroids is None:
            return None

        raw_lut = self._raw_lut(self.centroids)
        if self._lut_fake_quantizer is None:
            return self._reshape_lut_tensor(raw_lut)

        orig_dtype = raw_lut.dtype
        scale, zero_point, minval = self._lut_fake_quantizer.qparams_calculator.get_qparams()
        fq_lut = self._lut_fake_quantizer._fused_fake_quant_dequant(
            raw_lut.to(torch.float32), scale, zero_point, minval
        ).to(orig_dtype)
        return self._reshape_lut_tensor(fq_lut)

    @property
    def quantized_lut(self) -> torch.Tensor | None:
        """LUT quantized against ``lut_qspec``, derived fresh from
        ``centroids``. ``None`` if no LUT quantizer is configured.

        Reads qparams via ``get_qparams()`` (a pure buffer read) rather than
        calling the qparams calculator directly — some calculators (e.g.
        moving-average) mutate running statistics on every call, so calling
        them again here would drift the observer relative to ``lut``'s own
        computation and make repeated reads mutually inconsistent. This
        reflects the calculator's state as of the last access to ``lut``
        (which does call it, to legitimately observe the current centroids).
        """
        if self.centroids is None or self._lut_fake_quantizer is None:
            return None
        raw_lut = self._raw_lut(self.centroids)
        scale, zero_point, minval = self._lut_fake_quantizer.qparams_calculator.get_qparams()
        quantized = self._lut_fake_quantizer.quantize(raw_lut, scale, zero_point, minval)
        return self._reshape_lut_tensor(quantized.detach())

    @property
    def lut_quantization_scale(self) -> torch.Tensor | None:
        """Quantization scale for ``quantized_lut``. ``None`` if no LUT
        quantizer is configured. See ``quantized_lut`` for why this reads
        ``get_qparams()`` instead of invoking the calculator directly.
        """
        if self.centroids is None or self._lut_fake_quantizer is None:
            return None
        scale, _, _ = self._lut_fake_quantizer.qparams_calculator.get_qparams()
        return self._reshape_lut_tensor(scale.detach())

    @property
    def lut_quantization_zero_point(self) -> torch.Tensor | None:
        """Quantization zero point for ``quantized_lut``. ``None`` if no LUT
        quantizer is configured or the quantization scheme has no zero
        point. See ``quantized_lut`` for why this reads ``get_qparams()``
        instead of invoking the calculator directly.
        """
        if self.centroids is None or self._lut_fake_quantizer is None:
            return None
        _, zero_point, _ = self._lut_fake_quantizer.qparams_calculator.get_qparams()
        return self._reshape_lut_tensor(zero_point.detach()) if zero_point is not None else None

    def hard_assign(self, weight: torch.Tensor) -> torch.Tensor:
        """Nearest-centroid reconstruction against the current centroids,
        refreshing indices first if stale.
        """
        self._maybe_refresh_indices(weight)
        return self._palettize(self.lut, self.indices, weight)

    def _blocks_to_cluster(self, weight_2d: torch.Tensor, axis: int) -> list[torch.Tensor]:
        """Validate cluster_dim divisibility and split a 2D weight/sensitivity
        tensor into per-partition blocks.
        """
        if self.cluster_dim > 1:
            if isinstance(self.granularity, PerGroupedChannelGranularity) and axis == 0:
                weight_dim = self.granularity.group_size
            else:
                weight_dim = weight_2d.shape[0]
            if weight_dim % self.cluster_dim != 0:
                raise _IncompatibleClusterDimError(
                    f"Tensor dimension {weight_dim} along output channel axis "
                    f"is not divisible by cluster_dim {self.cluster_dim}."
                )
        return self.granularity.get_blocks_to_cluster(weight_2d)

    @torch.no_grad()
    def _cluster_to_centroids(
        self, original_weights: torch.Tensor, sensitivities: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cluster weight (+ optional sensitivities) into centroids via k-means.

        Returns ``(centroids, indices)``: ``centroids`` is a ``(num_blocks,
        num_clusters, cluster_dim)`` tensor, and ``indices`` is the per-element
        cluster assignment produced directly by the clustering algorithm.
        """
        weight = original_weights.cpu()
        if self.enable_per_channel_scale:
            weight = self._scale_by_per_channel_scale(weight)

        axis = self.granularity.axis if self.granularity.axis else 0
        weight = self.reshape_strategy.reshape_for_kmeans(weight, axis)
        block_weights_to_cluster = self._blocks_to_cluster(weight, axis)

        if sensitivities is not None:
            sensitivities = sensitivities.cpu()
            # numpy has no bfloat16 dtype, so cluster bf16 sensitivities as
            # float32, matching the block-weight handling in _cluster_weights_1d.
            if sensitivities.dtype == torch.bfloat16:
                sensitivities = sensitivities.float()
            sensitivities = self.reshape_strategy.reshape_for_kmeans(sensitivities, axis)
            block_sensitivities = self.granularity.get_blocks_to_cluster(sensitivities)
        else:
            block_sensitivities = [None] * len(block_weights_to_cluster)

        num_clusters = 2**self.n_bits
        centroids_per_block = []
        block_indices = []
        for block_weight, block_sensitivity in zip(
            block_weights_to_cluster, block_sensitivities, strict=True
        ):
            if self.cluster_dim == 1:
                centroids, clusters = self._cluster_weights_1d(block_weight, block_sensitivity)
            else:
                centroids, clusters = self._cluster_weights_2d(block_weight, block_sensitivity)
            centroids = self._pad_lut_to_num_clusters(centroids, num_clusters)
            centroids_per_block.append(centroids.to(weight.dtype))
            block_indices.append(self._build_block_indices(clusters, block_weight).to(torch.uint8))

        stacked = torch.stack(centroids_per_block)
        # Keep a trailing vector dimension so shape is (num_blocks,
        # num_clusters, cluster_dim) for both scalar and vector palettization.
        centroids_pnd = stacked if self.cluster_dim > 1 else stacked.unsqueeze(-1)
        centroids_pnd = centroids_pnd.detach().clone().to(original_weights.device)

        indices = self._combine_block_indices(block_indices, axis, original_weights)
        return centroids_pnd, indices

    @torch.no_grad()
    def _assign_indices(
        self, original_weights: torch.Tensor, centroids: torch.Tensor
    ) -> torch.Tensor:
        """Nearest-centroid hard assignment of ``original_weights`` against a
        given ``centroids`` (P, K, D) tensor.
        """
        weight = original_weights.detach().cpu()
        if self.enable_per_channel_scale:
            weight = self._scale_by_per_channel_scale(weight)

        axis = self.granularity.axis if self.granularity.axis else 0
        weight_2d = self.reshape_strategy.reshape_for_kmeans(weight, axis)
        blocks = self._blocks_to_cluster(weight_2d, axis)
        centroids_cpu = centroids.detach().cpu().float()

        block_indices = []
        for block_idx, block_weight in enumerate(blocks):
            vec = self._vectorize(block_weight)
            dist = torch.cdist(vec.float(), centroids_cpu[block_idx])
            clusters = dist.argmin(dim=-1)
            block_indices.append(self._build_block_indices(clusters, block_weight).to(torch.uint8))

        # self.indices is always CPU-resident, matching self.lut.
        return self._combine_block_indices(block_indices, axis, original_weights)

    def _combine_block_indices(
        self,
        block_indices: list[torch.Tensor],
        axis: int,
        original_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate per-block indices and reshape to the original weight
        shape (axis 0 reduced by ``cluster_dim`` for vector palettization).
        """
        indices = torch.cat(block_indices, dim=axis)
        indices_shape = list(original_weights.shape)
        indices_shape[0] = indices_shape[0] // self.cluster_dim
        return self.reshape_strategy.reshape_to_original(indices, axis, torch.Size(indices_shape))

    @torch.no_grad()
    def _raw_lut(self, centroids: torch.Tensor) -> torch.Tensor:
        """Reshape ``centroids`` (P, K, D) to the pre-quantization LUT shape
        ``(P, K[, D])``, detached.
        """
        centroids = centroids.detach()
        return centroids if self.cluster_dim > 1 else centroids.squeeze(-1)

    def _palettize(
        self, lut: torch.Tensor, indices: torch.Tensor, original_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Palettized weights from LUT and indices.

        Args:
            lut: Lookup table tensor from calculate_centroids
            indices: Index tensor from calculate_centroids
            original_weights: Original weight tensor

        Returns:
            Palettized weight tensor with the original shape

        Note:
            This method assumes that group_size is divisible by the weight shape
            along the grouped axis, so all blocks have the same size.
        """
        clustered_weight = None
        axis = self.granularity.axis if self.granularity.axis else 0

        lut = lut.to(indices.device)

        # Reshape indices back to 2D for block processing (reverse of
        # reshape_to_original in _assign_indices)
        indices = self.reshape_strategy.reshape_for_kmeans(indices, axis)
        # Cast to int for indexing since PyTorch treats uint8 as a boolean mask
        indices = indices.int()

        if isinstance(self.granularity, PerTensorGranularity):
            # Per-tensor granularity: single LUT for entire tensor
            # Scalar lut shape: (1, 1, num_clusters, 1) -> squeeze to (num_clusters,)
            # Vector lut shape: (1, 1, num_clusters, cluster_dim) -> squeeze to
            #   (num_clusters, cluster_dim)
            flat_lut = lut.squeeze()
            clustered_weight = flat_lut[indices]
            if self.cluster_dim > 1:
                clustered_weight = self._lookup_result_to_block(clustered_weight)
        elif isinstance(self.granularity, PerGroupedChannelGranularity):
            # Per-grouped-channel granularity: multiple LUTs for different blocks
            depalett_block_weights = []

            group_size = self.granularity.group_size
            num_blocks = self.granularity.num_blocks_to_cluster(original_weights)

            # Process each block with its corresponding LUT
            for block_idx in range(num_blocks):
                # Extract the LUT for this block
                # Scalar: lut shape (num_blocks, 1, num_clusters, 1) or
                #   (1, num_blocks, num_clusters, 1)
                # Vector: lut shape (num_blocks, 1, num_clusters, cluster_dim) or
                #   (1, num_blocks, num_clusters, cluster_dim)
                if axis == 0:
                    block_lut = lut[block_idx, 0]  # (num_clusters, cluster_dim)
                    # For vector palettization, indices are reduced along axis 0
                    # (output channel), so row slicing uses reduced group size.
                    reduced_group = group_size // self.cluster_dim
                    block_indices = indices[
                        block_idx * reduced_group : (block_idx + 1) * reduced_group, :
                    ]
                else:
                    block_lut = lut[0, block_idx]  # (num_clusters, cluster_dim)
                    # Column slicing is unaffected since vectorization is along axis 0
                    block_indices = indices[
                        :, block_idx * group_size : (block_idx + 1) * group_size
                    ]

                if self.cluster_dim == 1:
                    block_lut = block_lut.squeeze(-1)  # (num_clusters,)

                depalett_block_weight = block_lut[block_indices]
                if self.cluster_dim > 1:
                    depalett_block_weight = self._lookup_result_to_block(depalett_block_weight)

                depalett_block_weights.append(depalett_block_weight)

            clustered_weight = torch.cat(depalett_block_weights, dim=axis)
        else:
            # Unknown granularity
            raise ValueError(f"Unsupported granularity: {self.granularity}")

        # Reshape to original weight shape
        clustered_weight = self.reshape_strategy.reshape_to_original(
            clustered_weight, axis, original_weights.shape
        )

        if self.enable_per_channel_scale:
            clustered_weight = self._unscale_by_per_channel_scale(clustered_weight)

        return clustered_weight.to(original_weights.device, original_weights.dtype)

    def _pad_lut_to_num_clusters(
        self,
        centroids: torch.Tensor,
        num_clusters: int,
    ) -> torch.Tensor:
        """Pad centroids to ``num_clusters`` using the last centroid value.

        When k-means returns fewer centroids than ``2 ** n_bits`` (e.g. when
        the number of unique values is small), the LUT must still have
        ``num_clusters`` entries. Padding with the last centroid value, rather
        than zeros, avoids skewing the min/max range used for LUT quantization.
        Padded entries are never referenced by indices so their value is
        irrelevant for reconstruction.

        Args:
            centroids: Centroid tensor of shape ``(k,)`` for scalar or
                ``(k, cluster_dim)`` for vector palettization, where
                ``k < num_clusters``.
            num_clusters: Target number of clusters (``2 ** n_bits``).

        Returns:
            Padded centroid tensor of shape ``(num_clusters,)`` or
            ``(num_clusters, cluster_dim)``.
        """
        if len(centroids) >= num_clusters:
            return centroids

        if self.cluster_dim == 1:
            padded_lut = centroids[-1].expand(num_clusters).clone()
        else:
            padded_lut = centroids[-1:].expand(num_clusters, -1).clone()
        padded_lut[: len(centroids)] = centroids
        return padded_lut

    def quantize_lut(
        self,
        lut: torch.Tensor,
    ) -> torch.Tensor:
        """Quantize the stacked LUT tensor and dequantize it back via STE.

        Computes per-block quantization parameters on the stacked LUT of shape
        ``(num_blocks, num_clusters[, cluster_dim])``, quantizes it, then
        dequantizes back to the original dtype via a fused STE op.

        If no LUT fake quantizer is configured, returns the input unchanged.

        Args:
            lut: Stacked LUT tensor to quantize.

        Returns:
            Dequantized LUT tensor (same shape and dtype as input), or the
            original tensor if LUT quantization is not enabled.
        """
        if self._lut_fake_quantizer is None:
            return lut

        orig_dtype = lut.dtype
        scale, zero_point, minval = self._lut_fake_quantizer.qparams_calculator(lut)
        lut = self._lut_fake_quantizer._fused_fake_quant_dequant(
            lut.to(torch.float32), scale, zero_point, minval
        ).to(orig_dtype)
        return lut

    def _cluster_weights_1d(
        self,
        block_weight: torch.Tensor,
        block_sensitivity: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Cluster weights such that each centroid is a 1d scalar, i.e., cluster_dim == 1.
        """
        num_clusters = 2**self.n_bits

        # numpy has no bfloat16 dtype, so cluster bf16 weights as float32. The
        # centroids are cast back to the weight dtype by the caller.
        if block_weight.dtype == torch.bfloat16:
            block_weight = block_weight.float()

        block_weight_flatten = block_weight.flatten().numpy()
        if block_sensitivity is not None:
            block_sensitivity_flatten = block_sensitivity.flatten().numpy()
        else:
            block_sensitivity_flatten = None

        logger.debug(
            f"Clustering weights with kmeans 1d: "
            f"Weight dtype={block_weight_flatten.dtype}"
            f"enable_fast_kmeans_mode={self.enable_fast_kmeans_mode}"
            f"Range=({np.min(block_weight_flatten)},{np.max(block_weight_flatten)})"
        )
        if (block_weight_flatten.dtype == np.float16) or (
            self.enable_fast_kmeans_mode
            and (np.max(block_weight_flatten)) <= np.finfo(np.float16).max
            and np.min(block_weight_flatten) >= np.finfo(np.float16).min
        ):
            values, indices, counts = self._reduce_weights_to_cluster(block_weight_flatten)
            num_clusters = min(len(values), num_clusters)
            if block_sensitivity_flatten is not None:
                counts = np.bincount(indices, weights=block_sensitivity_flatten)

            kmeans_results: _kmeans1d.Clustered = _kmeans1d.cluster(
                values, num_clusters, weights=counts
            )

            # Expand clusters according to np.unique indices
            # kmeans_results is a namedtuple, which is why we use this constructor
            kmeans_results = type(kmeans_results)(
                clusters=np.array(kmeans_results.clusters)[indices].tolist(),
                centroids=kmeans_results.centroids,
            )
        else:
            kmeans_results: _kmeans1d.Clustered = _kmeans1d.cluster(
                block_weight_flatten, num_clusters, weights=block_sensitivity_flatten
            )

        # First create numpy array from list and then tensor from numpy array.
        # This is much faster than creating tensor from list.
        centroids = torch.from_numpy(np.array(kmeans_results.centroids))
        clusters = torch.from_numpy(np.array(kmeans_results.clusters))

        return centroids, clusters

    def _cluster_weights_2d(
        self,
        block_weight: torch.Tensor,
        block_sensitivity: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Cluster weights using vector k-means where each centroid is a vector
        of length cluster_dim, i.e., cluster_dim > 1.

        Vectorization is always along axis 0 (output channel axis).
        """
        num_clusters = 2**self.n_bits

        # Vectorize: reshape block_weight to (N, cluster_dim) along axis 0
        vectorized = self._vectorize(block_weight)
        num_clusters = min(len(vectorized), num_clusters)

        # Prepare sample weights from sensitivities
        sample_weight = None
        if block_sensitivity is not None:
            sens_vectorized = self._vectorize(block_sensitivity)
            # Sum sensitivities along cluster_dim for per-vector importance
            sample_weight = sens_vectorized.sum(dim=-1, keepdim=True)

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vectorized = vectorized.to(device)
        if sample_weight is not None:
            sample_weight = sample_weight.to(device)

        kmeans = _EfficientKMeans(
            n_clusters=num_clusters,
            init="kmeans++",
            n_init=5,
            max_iter=300,
        ).fit(vectorized.float(), sample_weight=sample_weight)

        centroids = kmeans.cluster_centers_.cpu()
        labels = kmeans.labels_.cpu()

        return centroids, labels

    def _vectorize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reshape a 2D tensor into (N, cluster_dim) vectors for k-means.

        Vectors are always formed along axis 0 (output channel axis). This transposes
        the tensor so consecutive elements along axis 0 are grouped into vectors. For
        ``cluster_dim == 1`` (scalar palettization), this is just a flatten.
        """
        if self.cluster_dim == 1:
            return tensor.reshape(-1, 1)
        return tensor.transpose(0, 1).reshape(-1, self.cluster_dim)

    def _lookup_result_to_block(self, looked_up: torch.Tensor) -> torch.Tensor:
        """Reshape a vector LUT lookup result back to 2D weight shape.

        Args:
            looked_up: Result of LUT[indices] of shape
                (rows // cluster_dim, cols, cluster_dim)

        Returns:
            Tensor of shape (rows, cols)
        """
        # (rows//cd, cols, cd) → (rows//cd, cd, cols) → (rows, cols)
        return looked_up.transpose(-2, -1).flatten(0, 1)

    def _build_block_indices(
        self, clusters: torch.Tensor, block_weight: torch.Tensor
    ) -> torch.Tensor:
        """Reshape raw cluster assignments into block-shaped indices.

        For scalar (cluster_dim==1): reshape to block_weight shape.
        For vector (cluster_dim>1): reshape to reduced shape where axis 0
        (output channel) is divided by cluster_dim.
        """
        if self.cluster_dim == 1:
            return clusters.reshape(block_weight.shape)

        rows, cols = block_weight.shape
        # Vectorized as: transpose(0,1) → (cols, rows) → reshape(-1, cd)
        # Labels shape: cols * rows / cd
        # Reshape to (cols, rows//cd) then transpose to (rows//cd, cols)
        return clusters.reshape(cols, rows // self.cluster_dim).transpose(0, 1)

    def _reduce_weights_to_cluster(self, block_weight_flatten: np.ndarray):
        # With fp16 values we often have a reduced amount of unique values
        # and performing weighted kmeans becomes much faster

        # Add rounding before computing unique values to further reduce
        # clustered weight size
        if self.enable_fast_kmeans_mode:
            # Cast fp32 -> fp16
            if block_weight_flatten.dtype != np.float16:
                block_weight_flatten = block_weight_flatten.astype(np.float16)

            # Rounding
            scale = 10**self.rounding_precision
            block_weight_flatten = np.round(block_weight_flatten.astype(np.float32) * scale) / scale

        # To speed up parallel kmeans, use numpy.unique instead of
        # torch.unique in multiprocessing setting.
        values, indices, counts = np.unique(
            block_weight_flatten,
            return_inverse=True,
            return_counts=True,
        )

        return values, indices, counts

    def _reshape_lut_tensor(self, lut: torch.Tensor) -> torch.Tensor:
        """Reshape a stacked LUT tensor into the 4D format expected by palettization.

        Transforms the input from shape ``(num_blocks, num_clusters[, cluster_dim])``
        to ``(num_blocks_axis0, num_blocks_axis1, num_clusters, cluster_dim)``:
          - Inserts an ungrouped dimension of size 1 along the axis not used for
            grouping (axis 0 if granularity axis is 1, and vice versa).
          - Appends a trailing vector dimension of size 1 when ``cluster_dim == 1``
            (scalar palettization).
        """
        # Add ungrouped dimension based on granularity axis
        ungrouped_dim = 0 if self.granularity.axis == 1 else 1
        lut = lut.unsqueeze(ungrouped_dim)

        # Add vector dimension for 1D clustering
        if self.cluster_dim == 1:
            lut = lut.unsqueeze(-1)

        return lut

    def _scale_by_per_channel_scale(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Compute per channel scales for scaling the parameter in the range ``[-1, 1]``.
        Also scale the parameter using the computed scales.
        """
        flattened_weight = weight.flatten(1)
        per_channel_scale = torch.max(torch.abs(flattened_weight), dim=1, keepdim=True).values
        # Handle zero scales
        per_channel_scale[per_channel_scale == 0] = 1
        scaled_weight = flattened_weight / per_channel_scale
        scaled_weight = scaled_weight.reshape(weight.shape)
        # Update scales
        self.per_channel_scale = per_channel_scale.detach()

        return scaled_weight

    def _unscale_by_per_channel_scale(self, scaled_weight: torch.Tensor) -> torch.Tensor:
        """
        Re-scale the parameter back to its original range by multiplying
        per channel scales.
        """
        flattened_scaled_weight = scaled_weight.flatten(1)
        flattened_unscaled_weight = flattened_scaled_weight * self.per_channel_scale
        unscaled_weight = flattened_unscaled_weight.reshape(scaled_weight.shape)
        return unscaled_weight
