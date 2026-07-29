# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from abc import abstractmethod
from typing import Annotated, Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_serializer

from coreai_opt._utils.registry_utils import ConfigRegistryMixin as _ConfigRegistryMixin
from coreai_opt.quantization.spec.errors import _BlockSizeMismatchError


class QuantizationGranularity(BaseModel, _ConfigRegistryMixin):
    """
    Base class for quantization granularity specifications.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    axis: int | None = Field(
        default=None,
        description="The axis along which quantization is applied. "
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
        for key, registered_class in QuantizationGranularity.REGISTRY.items():
            if registered_class is type(self):
                registry_key = key
                break

        if registry_key is not None:
            data["type"] = registry_key

        return data

    @abstractmethod
    def _get_block_size(self, block_sizes_list: list[int]) -> list[int]:
        """
        Given an initial list of the tensor shape, return a list of block sizes
        corresponding to each axis:
        - if no structuring is being done for a certain axis, return the
          tensor's shape corresponding to that axis which is present in the initial
          ``block_sizes_list``
        - if per-channel structuring is being done for a certain axis, set the
          block size as ``1`` for that specific axis
        - if per-block structuring is being done for a certain axis, set the block
          size for that specific axis

        Example:
            - ``[10, 5, 2]`` with per-channel structuring on axis 1 results in
              ``[10, 1, 2]``
            - ``[10, 5, 2]`` with per-block structuring on axis 0 with block size 2
              results in ``[2, 5, 2]``
            - ``[10, 5, 2]`` with per-tensor structuring results in ``[10, 5, 2]``
        """
        pass

    def get_block_size(self, tensor_shape: torch.Size) -> tuple[int, ...]:
        """
        Get a list of block sizes based on the granularity.
        """
        return tuple(self._get_block_size(list(tensor_shape)))

    # The axis resolution logic lives here because it is granularity-specific.
    # Currently only PerChannelGranularity has a meaningful axis to resolve, but
    # this can be extended for other granularity types (e.g. negative axis support
    # for PerBlockGranularity) in the future. The resolved value is stored on
    # QParamsCalculator (not here) because granularity instances are shared across
    # nodes while the resolved axis is per-node.
    @staticmethod
    def _resolve_axis(granularity: QuantizationGranularity, tensor_ndim: int) -> int | None:
        """Resolve axis to a non-negative value based on granularity type.

        Converts negative Python-style axis indexing to a non-negative value
        using the tensor rank. Currently handles ``PerChannelGranularity``;
        can be extended for other granularity types as needed.

        Args:
            granularity: The granularity instance to resolve axis for.
            tensor_ndim: Rank of the tensor being quantized.

        Returns:
            Non-negative axis for granularity types that support it,
            None otherwise.

        """
        if not isinstance(granularity, PerChannelGranularity):
            return None
        axis = granularity.axis
        if axis is None:
            return None
        if axis < 0:
            axis += tensor_ndim
        return axis


@QuantizationGranularity.register("per_tensor")
class PerTensorGranularity(QuantizationGranularity):
    """
    Per-tensor quantization granularity.

    This applies quantization to the tensor as a whole.
    """

    axis: Literal[None] = None

    def _get_block_size(self, block_sizes_list: list[int]) -> list[int]:
        return block_sizes_list


@QuantizationGranularity.register("per_channel")
class PerChannelGranularity(QuantizationGranularity):
    """Per-channel quantization granularity.

    This applies quantization to a specific channel which is selected through the
    ``axis`` argument. When ``axis`` is ``None`` (the default), ``Quantizer.prepare()``
    automatically resolves it based on the module type for weight quantization.

    Note: axis can be negatively indexed as per standard Python style indexing.
    For example, with a block sizes list: [10, 20, 30], a valid set of axis include
    -3 <= axis < 3
    """

    axis: int | None = None

    def _get_block_size(self, block_sizes_list: list[int]) -> list[int]:

        if self.axis is None:
            raise ValueError(
                "PerChannelGranularity axis is None and was not resolved to a "
                "default. Please specify axis explicitly."
            )

        try:
            block_sizes_list[self.axis] = 1
        except IndexError:
            block_sizes_list_len = len(block_sizes_list)
            msg = (
                f"axis {self.axis} is out of bounds for tensor of "
                f"rank {block_sizes_list_len}. "
                f"Allowed axis range is "
                f"[{-block_sizes_list_len}, {block_sizes_list_len})"
            )
            raise ValueError(msg) from None

        return block_sizes_list


@QuantizationGranularity.register("per_block")
class PerBlockGranularity(QuantizationGranularity):
    """Per-block quantization granularity.

    This applies quantization to blocks of values within the tensor. Supports two modes:

    1. Single-axis mode: Quantize blocks along one specific axis
       (no blocking for axis>=2)

       - ``axis``: The axis to create blocks (0 or 1)
       - ``block_size``: Integer specifying block size for that axis

    2. Multi-axis mode: Create blocks across multiple axes simultaneously

       - ``axis``: Must be None
       - ``block_size``: Tuple specifying block size for each axis
         (-1 means no blocking)

    In single-axis mode, when ``axis`` is ``None`` and ``block_size`` is an integer,
    ``Quantizer.prepare()`` automatically resolves the axis based on the module type
    for weight quantization.

    .. list-table::
       :header-rows: 1

       * - Weight tensor shape (input)
         - axis
         - block_size
         - Weight shape of each block (output)
       * - [C_out, C_in]
         - 1
         - 32
         - [1, 32]
       * - [C_out, C_in]
         - None
         - (4, 8)
         - [4, 8]
       * - [C_out, C_in, KH, KW]
         - 0
         - 16
         - [16, 1, KH, KW]
       * - [C_out, C_in, KH, KW]
         - None
         - (4, 16, 3, -1)
         - [4, 16, 3, KW]
    """

    axis: Annotated[int, Field(ge=0, le=1)] | None = None
    block_size: Annotated[int, Field(gt=0)] | tuple[Annotated[int, Field(gt=0)] | Literal[-1], ...]

    def _get_block_size(self, block_sizes_list: list[int]) -> list[int]:
        if isinstance(self.block_size, tuple):
            return self._handle_multi_axis_block_size(block_sizes_list)
        else:
            return self._handle_single_axis_block_size(block_sizes_list)

    def _handle_multi_axis_block_size(self, block_sizes_list: list[int]) -> list[int]:
        """Handle blocking when self.block_size is a tuple"""
        if self.axis is not None:
            raise ValueError(
                "axis must be None when block_size is a tuple "
                "self.block_size tuple should have a block size "
                "for each of the tensor's dimensions"
            )

        if len(block_sizes_list) != len(self.block_size):
            raise ValueError(
                f"Rank of block_size ({len(self.block_size)}) must match "
                f"rank of weight tensor ({len(block_sizes_list)})"
            )

        for axis, block_sz in enumerate(self.block_size):
            if block_sz > 0:  # -1 means no quantization on this axis
                if block_sizes_list[axis] % block_sz != 0:
                    raise _BlockSizeMismatchError(
                        f"Tensor size {block_sizes_list[axis]} along axis {axis} "
                        f"is not divisible by block size {block_sz}. "
                        f"Full tensor size: {block_sizes_list}, "
                        f"block_size tuple: {self.block_size}"
                    )
                block_sizes_list[axis] = block_sz

        return block_sizes_list

    def _handle_single_axis_block_size(self, block_sizes_list: list[int]) -> list[int]:
        """Handle blocking when self.block_size is an integer"""
        # TODO: Logic to be added where if self.axis is None,
        #  we can figure out the optimal axis for the user
        if self.axis is None:
            raise ValueError("axis must be specified when block_size is an int")

        if self.axis >= len(block_sizes_list):
            raise ValueError(
                f"axis {self.axis} is out of bounds for tensor of rank {len(block_sizes_list)}"
            )

        if block_sizes_list[self.axis] % self.block_size != 0:
            raise _BlockSizeMismatchError(
                f"Tensor size {block_sizes_list[self.axis]} along axis {self.axis} "
                f"is not divisible by block size {self.block_size}"
            )

        # For integer block_size, only process the first two dimensions
        # (which would be input and output channel axis in no particular order)
        # Set the specified axis to block_size, set the other dimension (0 or 1) to 1
        # Leave all higher dimensions (index 2+) unchanged
        block_sizes_list[self.axis] = self.block_size
        for axis, _ in enumerate(block_sizes_list[:2]):
            if axis != self.axis:
                block_sizes_list[axis] = 1

        return block_sizes_list
