# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import logging
from abc import abstractmethod

import torch
import torch.nn as nn

from coreai_opt._utils.spec_utils import (
    PartialConstructor as _PartialConstructor,
    with_args as _with_args,
)
from coreai_opt.config.spec import CompressionSimulatorBase
from coreai_opt.palettization.spec import (
    PalettizationGranularity,
)
from coreai_opt.palettization.spec.errors import (
    _IncompatibleClusterDimError,
    _IncompatibleGranularityError,
)
from coreai_opt.quantization.spec import QuantizationSpec

logger = logging.getLogger(__name__)


class _FakePalettizeImplBase(CompressionSimulatorBase, nn.Module):
    """Base class for fake palettization implementations with clustering and
    reconstruction methods.
    """

    lut: torch.Tensor
    indices: torch.Tensor
    per_channel_scale: torch.Tensor | None
    quantized_lut: torch.Tensor | None
    lut_quantization_scale: torch.Tensor | None
    lut_quantization_zero_point: torch.Tensor | None
    fake_palett_enabled: torch.Tensor
    observer_enabled: torch.Tensor

    def __init__(
        self,
        n_bits: int,
        lut_qspec: QuantizationSpec | None,
        granularity: PalettizationGranularity,
        cluster_dim: int,
        enable_per_channel_scale: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_bits = n_bits
        self.lut_qspec = lut_qspec
        self.granularity = granularity
        self.cluster_dim = cluster_dim
        self.enable_per_channel_scale = enable_per_channel_scale

        self.register_buffer("fake_palett_enabled", torch.tensor([1], dtype=torch.uint8))
        self.register_buffer("observer_enabled", torch.tensor([1], dtype=torch.uint8))
        self._disabled = False

        self.register_buffer("lut", None)
        self.register_buffer("indices", None)
        self.register_buffer("per_channel_scale", None)
        self.register_buffer("quantized_lut", None)
        self.register_buffer("lut_quantization_scale", None)
        self.register_buffer("lut_quantization_zero_point", None)

    @property
    def _initialized(self) -> bool:
        """Return True if lut and indices have been initialized (not None)."""
        return self.lut is not None and self.indices is not None

    def is_disabled(self) -> bool:
        """Return True if fake palettization has been disabled."""
        return self._disabled

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply fake palettization to input tensor"""
        # If permanently disabled due to incompatibility, return original tensor
        if self._disabled:
            return tensor

        if self.observer_enabled[0] == 1:
            # Cluster weights
            try:
                lut, indices = self._calculate_centroids(tensor)
            except _IncompatibleGranularityError as e:
                logger.warning(
                    f"Tensor incompatible with granularity: {e}. Skipping palettization."
                )
                self._disabled = True
                return tensor
            except _IncompatibleClusterDimError as e:
                logger.warning(
                    f"Tensor incompatible with cluster_dim: {e}. Skipping palettization."
                )
                self._disabled = True
                return tensor

            self.lut = lut.detach()
            self.indices = indices.detach()
        else:
            # Check that recomputed statistics exist
            if not self._initialized:
                # Not initialized yet, return original tensor
                return tensor

        if self.fake_palett_enabled[0] == 1:
            return self._palettize(lut=self.lut, indices=self.indices, original_weights=tensor)

        return tensor

    @abstractmethod
    def _palettize(
        self, lut: torch.Tensor, indices: torch.Tensor, original_weights: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct palettized weights from lookup table and indices."""
        raise NotImplementedError()

    @abstractmethod
    def _calculate_centroids(self, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Cluster weights and return lookup table (LUT) and corresponding indices.

        If tensor is incompatible with the specified granularity, this method
        should set self._disabled = True and return dummy values.

        Args:
            weight: The weight tensor to cluster

        Returns:
            Tuple of (lut, indices) where:
                - lut: Lookup table of cluster centroids
                - indices: Index tensor mapping each weight to its cluster

        LUT shape must be of the following form:
            [NUM_LUT_AXIS_0, NUM_LUT_AXIS_1, NUM_PALETTES, VECTOR_SIZE]
        where,
            NUM_LUT_* is the number of LUTs for the corresponding axis. The computation
            depends on the palettization granularity:
                - For per-tensor: NUM_LUT_* = 1 (single LUT for entire tensor)
                - For per-grouped channel: NUM_LUT_* = number of groups
                  (calculated as weight shape along axis // group size)
            NUM_PALETTES is lut.shape[-2] and needs to be 2^nbits
            VECTOR_SIZE is lut.shape[-1] and is added to support vector palettization.
                When VECTOR_SIZE is 1, it is scalar palettization.

        Indices shape much match the shape of the palettized weight.
        """
        raise NotImplementedError()

    @classmethod
    def with_args(cls, **kwargs: dict) -> _PartialConstructor[_FakePalettizeImplBase]:
        fake_palett_constructor = _with_args(cls, **kwargs)

        # need to assign the correct module to fake_palettize
        # constructors to satisfy public v private requirements
        fake_palett_constructor.__module__ = f"{cls.__module__}.{cls.__name__}"
        return fake_palett_constructor

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        """Custom state dict loading for palettization-specific buffers.

        This method handles the loading of palettization-specific buffers (lut, indices,
        per_channel_scale) that may be dynamically created during forward passes. By
        registering them here, we ensure they are properly loaded from saved checkpoints
         and don't generate unexpected key warnings.

        The method is called automatically by PyTorch during model loading (torch.load,
        load_state_dict, etc.) and should not be called directly.
        """
        buffer_names = {
            "lut",
            "indices",
            "per_channel_scale",
            "quantized_lut",
            "lut_quantization_scale",
            "lut_quantization_zero_point",
        }

        for buffer_name in buffer_names:
            prefixed_key = prefix + buffer_name
            if prefixed_key in state_dict:
                # Register the buffer with the correct name (without prefix)
                self.register_buffer(buffer_name, state_dict[prefixed_key])
                # Remove from unexpected keys if it was there to prevent warnings
                if prefixed_key in unexpected_keys:
                    unexpected_keys.remove(prefixed_key)

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def enable_fake_palett(self, enabled: bool = True) -> None:
        self.fake_palett_enabled[0] = 1 if enabled else 0

    def disable_fake_palett(self):
        self.enable_fake_palett(False)

    def enable_observer(self, enabled: bool = True) -> None:
        self.observer_enabled[0] = 1 if enabled else 0

    def disable_observer(self):
        self.enable_observer(False)


def _enable_fake_palett(mod):
    """Enable fake palettization for the module."""
    if isinstance(mod, _FakePalettizeImplBase):
        mod.enable_fake_palett()


def _disable_fake_palett(mod):
    """Disable fake palettization for the module."""
    if isinstance(mod, _FakePalettizeImplBase):
        mod.disable_fake_palett()


def _enable_observer(mod):
    """Enable observation for this module."""
    if isinstance(mod, _FakePalettizeImplBase):
        mod.enable_observer()


def _disable_observer(mod):
    """Disable observation for this module."""
    if isinstance(mod, _FakePalettizeImplBase):
        mod.disable_observer()
