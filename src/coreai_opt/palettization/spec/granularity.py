# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from abc import abstractmethod
from typing import Annotated, Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_serializer

from coreai_opt._utils.registry_utils import ConfigRegistryMixin

from .errors import _IncompatibleGranularityError


class PalettizationGranularity(BaseModel, ConfigRegistryMixin):
    """
    Base class for palettization granularity specifications.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    axis: int | None = Field(
        default=None,
        description="The axis along which palettization is applied. "
        "None for per-tensor granularity.",
    )

    @model_serializer
    def _serialize_model(self) -> dict[str, Any]:
        """Custom serializer that includes the registry type."""
        data = {}

        for field_name in type(self).model_fields:
            data[field_name] = getattr(self, field_name)

        # Find the registry key for this class type
        registry_key = None
        # Use the base class registry instead of instance registry
        for key, registered_class in PalettizationGranularity.REGISTRY.items():
            if registered_class is type(self):
                registry_key = key
                break

        if registry_key is not None:
            data["type"] = registry_key

        return data

    @abstractmethod
    def num_blocks_to_cluster(self, weight: torch.Tensor) -> int:
        """
        Return the number of weight blocks to cluster based on the
        specified granularity.

        Args:
            weight: The weight tensor to be palettized

        Returns:
            Number of LUTs for the weight tensor

        Raises:
            _IncompatibleGranularityError: If the tensor is incompatible with
                this granularity
        """
        pass

    @abstractmethod
    def get_blocks_to_cluster(self, weight: torch.Tensor) -> list[torch.Tensor]:
        """
        Extract weight blocks to cluster based on the specified granularity.

        Args:
            weight: The weight tensor to split into blocks

        Returns:
            A list of weight tensor blocks. Each block is a view or slice of the
            original weight tensor based on the granularity configuration.

        Raises:
            _IncompatibleGranularityError: If the tensor is incompatible with
                this granularity
        """
        pass


@PalettizationGranularity.register("per_tensor")
class PerTensorGranularity(PalettizationGranularity):
    """
    Per-tensor palettization granularity.

    This applies palettization to the tensor as a whole.
    """

    axis: Literal[None] = None

    def num_blocks_to_cluster(self, weight: torch.Tensor) -> int:
        return 1

    def get_blocks_to_cluster(self, weight: torch.Tensor) -> list[torch.Tensor]:
        """
        For per-tensor granularity, return the entire tensor as a single block.

        Args:
            weight: The weight tensor

        Returns:
            List containing the single weight tensor block
        """
        return [weight]


@PalettizationGranularity.register("per_grouped_channel")
class PerGroupedChannelGranularity(PalettizationGranularity):
    """
    Per-grouped-channel palettization granularity.

    This applies palettization to a specific channel which is selected through the
    ``axis`` argument. ``axis`` defaults to ``None``, in which case the default
    axis for the consuming op is used (e.g. 0 for ``Linear``/``Conv``, 1 for
    ``ConvTranspose``).
    """

    axis: Annotated[int | None, Field(default=None, ge=0, le=1)]
    group_size: int

    def num_blocks_to_cluster(self, weight: torch.Tensor) -> int:
        if self.axis is None:
            raise _IncompatibleGranularityError(
                "axis is None; it must be resolved against an op or set explicitly before use."
            )

        # Validate tensor has enough dimensions
        if len(weight.shape) <= self.axis:
            raise _IncompatibleGranularityError(
                f"Tensor shape {weight.shape} has insufficient dimensions for axis "
                f"{self.axis}. Parameter must have at least {self.axis + 1} dimensions."
            )

        # Validate divisibility
        shape_along_axis = weight.shape[self.axis]
        if shape_along_axis % self.group_size != 0:
            raise _IncompatibleGranularityError(
                f"Tensor size {weight.shape} along axis {self.axis} is not "
                f"divisible by group_size {self.group_size}. For per-grouped-channel "
                f"palettization, the tensor shape along the specified axis must be "
                f"divisible by group_size."
            )

        return shape_along_axis // self.group_size

    def get_blocks_to_cluster(self, weight: torch.Tensor) -> list[torch.Tensor]:
        """
        Split weight tensor into blocks along the specified axis with group_size.

        Args:
            weight: The weight tensor to split

        Returns:
            List of weight blocks, each of size group_size along the specified axis

        Raises:
            _IncompatibleGranularityError: If tensor is incompatible with
            this granularity
        """
        if self.axis is None:
            raise _IncompatibleGranularityError(
                "axis is None; it must be resolved against an op or set explicitly before use."
            )

        # Validate tensor has enough dimensions
        if len(weight.shape) <= self.axis:
            raise _IncompatibleGranularityError(
                f"Tensor shape {weight.shape} has insufficient dimensions for axis "
                f"{self.axis}. Parameter must have at least {self.axis + 1} dimensions."
            )

        # Validate divisibility
        shape_along_axis = weight.shape[self.axis]
        if shape_along_axis % self.group_size != 0:
            raise _IncompatibleGranularityError(
                f"Tensor size {shape_along_axis} along axis {self.axis} is not "
                f"divisible by group_size {self.group_size}. For per-grouped-channel "
                f"palettization, the tensor shape along the specified axis must be "
                f"divisible by group_size."
            )

        # Split tensor into blocks
        block_weights = []
        if self.axis == 0:
            for block_idx in range(0, weight.shape[0], self.group_size):
                block_weight = weight[block_idx : block_idx + self.group_size, :]
                block_weights.append(block_weight)
        else:
            for block_idx in range(0, weight.shape[1], self.group_size):
                block_weight = weight[:, block_idx : block_idx + self.group_size]
                block_weights.append(block_weight)

        return block_weights
