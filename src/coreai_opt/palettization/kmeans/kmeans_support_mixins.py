# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Palettization support mixins for K-means clustering operations.

This module provides mixin classes that add palettization capabilities
to existing operation support classes in the registry.
"""

from abc import ABC, abstractmethod
from typing import ClassVar

import torch


class _PalettizationSupportMixin(ABC):
    """Abstract mixin for palettization support."""

    default_axis: ClassVar[int]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "default_axis"):
            raise TypeError(f"{cls.__name__} must define default_axis: ClassVar[int].")
        if cls.default_axis not in (0, 1):
            raise ValueError(f"{cls.__name__}.default_axis must be 0 or 1, got {cls.default_axis}.")

    @abstractmethod
    def reshape_for_kmeans(self, weight: torch.Tensor, axis: int) -> torch.Tensor:
        """Reshape weight tensor into 2D tensor for K-means clustering."""
        pass

    @abstractmethod
    def reshape_to_original(
        self,
        clustered_weight: torch.Tensor,
        axis: int,
        original_shape: torch.Size,
    ) -> torch.Tensor:
        """Reshape clustered weight back to original shape."""
        pass


class _LinearPalettizationMixin(_PalettizationSupportMixin):
    """Mixin providing palettization support for linear operations."""

    default_axis: ClassVar[int] = 0

    def reshape_for_kmeans(self, weight: torch.Tensor, axis: int) -> torch.Tensor:
        return weight

    def reshape_to_original(
        self,
        clustered_weight: torch.Tensor,
        axis: int,
        original_shape: torch.Size,
    ) -> torch.Tensor:
        return clustered_weight


class _ConvPalettizationMixin(_PalettizationSupportMixin):
    """Mixin providing palettization support for convolution operations."""

    default_axis: ClassVar[int] = 0

    def reshape_for_kmeans(self, weight: torch.Tensor, axis: int) -> torch.Tensor:
        if axis == 0:
            return weight.flatten(1)
        else:
            return weight.transpose(0, 1).flatten(1).transpose(0, 1)

    def reshape_to_original(
        self,
        clustered_weight: torch.Tensor,
        axis: int,
        original_shape: torch.Size,
    ) -> torch.Tensor:
        if axis == 0:
            return clustered_weight.reshape(original_shape)
        else:
            return (
                clustered_weight.transpose(0, 1)
                .reshape(
                    (
                        original_shape[1],
                        original_shape[0],
                        *[original_shape[i] for i in range(2, len(original_shape))],
                    )
                )
                .transpose(0, 1)
            )


class _ConvTransposePalettizationMixin(_ConvPalettizationMixin):
    """Mixin providing palettization support for transposed convolution operations.

    ``ConvTranspose`` weights are shaped ``[in_channels, out_channels, *kernel]``,
    so the output-channel axis is 1. The reshape logic from
    ``_ConvPalettizationMixin`` works unchanged.
    """

    default_axis: ClassVar[int] = 1
