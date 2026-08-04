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
from coreai_opt.palettization.spec.errors import _IncompatibleClusterDimError
from coreai_opt.palettization.spec.fake_palettize import _FakePalettizeImplBase
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

    1. ``_calculate_centroids()``: Clusters weights using k-means, returns LUT and indices
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
        >>> lut, indices = palettizer._calculate_centroids(weight)
        >>> palettized_weight = palettizer._palettize(lut, indices, weight)
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
        **kwargs,
    ):
        super().__init__(
            n_bits=n_bits,
            lut_qspec=lut_qspec,
            granularity=granularity,
            cluster_dim=cluster_dim,
            enable_per_channel_scale=enable_per_channel_scale,
            **kwargs,
        )

        self.enable_fast_kmeans_mode = enable_fast_kmeans_mode
        self.rounding_precision = rounding_precision
        self._sensitivities = sensitivities
        self._centroids_stale = False

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

    @property
    def sensitivities(self) -> torch.Tensor | None:
        """Get the sensitivity values used for weighted k-means clustering."""
        return self._sensitivities

    @sensitivities.setter
    def sensitivities(self, value: torch.Tensor | None) -> None:
        """Set sensitivity values and mark centroids as stale.

        When sensitivities are updated, the LUT and indices become stale and must
        be recomputed via _calculate_centroids before use. This is typically done
        by enabling the observer and running a forward pass through the model.
        """
        self._sensitivities = value
        self._centroids_stale = True

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply fake palettization to input tensor.

        Overrides base class to add stale centroids warning when sensitivities
        have been updated but centroids have not been recomputed.
        """
        # Check for stale centroids when observer is disabled and we have
        # initialized LUT/indices that would be used
        if (
            self._centroids_stale
            and self._initialized
            and self.observer_enabled[0] == 0
            and self.fake_palett_enabled[0] == 1
        ):
            logger.warning(
                "Sensitivities were updated but centroids have not been recomputed. "
                "The current LUT and indices do not reflect the new sensitivity values."
                "Enable observer and run a forward pass to recompute centroids, or use "
                "calibration_mode() which handles this automatically."
            )

        return super().forward(tensor)

    @torch.no_grad()
    def _calculate_centroids(
        self, original_weights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight = original_weights.cpu()

        if self.enable_per_channel_scale:
            weight = self._scale_by_per_channel_scale(weight)

        # Reshape weight into 2D matrix for clustering
        axis = self.granularity.axis if self.granularity.axis else 0
        weight = self.reshape_strategy.reshape_for_kmeans(weight, axis)

        # Validate cluster_dim divisibility along output channel axis (axis 0).
        if self.cluster_dim > 1:
            # For per-grouped-channel axis=0, each block has group_size rows,
            # so we must check group_size divisibility, not the full weight dim.
            if isinstance(self.granularity, PerGroupedChannelGranularity) and axis == 0:
                weight_dim = self.granularity.group_size
            else:
                weight_dim = weight.shape[0]
            if weight_dim % self.cluster_dim != 0:
                raise _IncompatibleClusterDimError(
                    f"Tensor dimension {weight_dim} along output channel axis "
                    f"is not divisible by cluster_dim {self.cluster_dim}."
                )

        block_weights_to_cluster = self.granularity.get_blocks_to_cluster(weight)

        # Reshape sensitivities if available
        if self.sensitivities is not None:
            sensitivities = self.sensitivities.cpu()
            # numpy has no bfloat16 dtype, so cluster bf16 sensitivities as
            # float32, matching the block-weight handling in _cluster_weights_1d.
            if sensitivities.dtype == torch.bfloat16:
                sensitivities = sensitivities.float()
            sensitivities = self.reshape_strategy.reshape_for_kmeans(sensitivities, axis)
            block_sensitivities = self.granularity.get_blocks_to_cluster(sensitivities)
        else:
            block_sensitivities = [None] * len(block_weights_to_cluster)

        lut = []
        indices = []
        num_clusters = 2**self.n_bits

        for block_weight, block_sensitivity in zip(
            block_weights_to_cluster, block_sensitivities, strict=True
        ):
            if self.cluster_dim == 1:
                centroids, clusters = self._cluster_weights_1d(block_weight, block_sensitivity)
            else:
                centroids, clusters = self._cluster_weights_2d(block_weight, block_sensitivity)

            centroids = self._pad_lut_to_num_clusters(centroids, num_clusters)

            lut.append(centroids.to(weight.dtype))
            block_indices = self._build_block_indices(clusters, block_weight)
            block_indices = block_indices.to(torch.uint8)
            indices.append(block_indices)

        # Handle concatenation based on granularity axis and convert to the shape
        # of original weight tensor (axis 0 reduced by cluster_dim for vector case)
        indices = torch.cat(indices, dim=axis)
        indices_shape = list(original_weights.shape)
        indices_shape[0] = indices_shape[0] // self.cluster_dim
        indices = self.reshape_strategy.reshape_to_original(
            indices, axis, torch.Size(indices_shape)
        )

        # Combine LUTs for all blocks into single tensor
        lut = torch.stack(lut)

        # Quantize the entire stacked LUT in one shot
        lut = self._quantize_lut(lut)

        lut = self._reshape_lut_tensor(lut)

        # Clear stale flag since centroids are now up-to-date with sensitivities
        self._centroids_stale = False

        return lut, indices

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

        # Reshape indices back to 2D for block processing (reverse of
        # reshape_to_original in _calculate_centroids)
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
                clustered_weight = self._devectorize(clustered_weight)
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
                    depalett_block_weight = self._devectorize(depalett_block_weight)

                depalett_block_weights.append(depalett_block_weight)

            clustered_weight = torch.cat(depalett_block_weights, dim=axis)
        else:
            # Unknown granularity
            raise ValueError(f"Unsupported granularity: {self.granularity}")

        clustered_weight.to(original_weights.dtype)

        # Reshape to original weight shape
        clustered_weight = self.reshape_strategy.reshape_to_original(
            clustered_weight, axis, original_weights.shape
        )

        if self.enable_per_channel_scale:
            clustered_weight = self._unscale_by_per_channel_scale(clustered_weight)

        return clustered_weight.to(original_weights.device)

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

    def _quantize_lut(
        self,
        lut: torch.Tensor,
    ) -> torch.Tensor:
        """Quantize the stacked LUT tensor and populate export buffers.

        Computes per-block quantization parameters on the stacked LUT of shape
        ``(num_blocks, num_clusters[, cluster_dim])``, quantizes it, then
        dequantizes back to the original dtype for STE-style training. Stores
        ``quantized_lut``, ``lut_quantization_scale``, and
        ``lut_quantization_zero_point`` as reshaped/detached buffers for export.

        If no LUT fake quantizer is configured, returns the input unchanged.

        Args:
            lut: Stacked LUT tensor to quantize.

        Returns:
            Dequantized LUT tensor (same shape and dtype as input), or the
            original tensor if LUT quantization is not enabled.
        """
        if self._lut_fake_quantizer is None:
            return lut

        scale, zero_point, minval = self._lut_fake_quantizer.qparams_calculator(lut)
        quantized_lut = self._lut_fake_quantizer.quantize(lut, scale, zero_point, minval)
        lut = self._lut_fake_quantizer.dequantize(
            quantized_lut, scale, zero_point, minval, output_dtype=lut.dtype
        )
        self.quantized_lut = self._reshape_lut_tensor(quantized_lut.detach())
        self.lut_quantization_scale = self._reshape_lut_tensor(scale.detach())
        self.lut_quantization_zero_point = (
            self._reshape_lut_tensor(zero_point.detach()) if zero_point is not None else None
        )
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
        """Reshape a 2D tensor into (N, cluster_dim) vectors for vector k-means.

        Vectors are always formed along axis 0 (output channel axis). This transposes
        the tensor so consecutive elements along axis 0 are grouped into vectors.
        """
        return tensor.transpose(0, 1).reshape(-1, self.cluster_dim)

    def _devectorize(self, looked_up: torch.Tensor) -> torch.Tensor:
        """Reshape vector LUT lookup result back to 2D weight shape.

        Reverses the vectorization along axis 0 (output channel axis).
        Since vectorization is always along axis 0, looked_up always has shape
        (rows // cluster_dim, cols, cluster_dim).

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
