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
from coreai_opt.config.spec import CompressionTargetTensor as _CompressionTargetTensor
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
    def _get_block_size(
        self,
        block_sizes_list: list[int],
        quantization_target: _CompressionTargetTensor = _CompressionTargetTensor.WEIGHT,
    ) -> list[int]:
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

        ``quantization_target`` distinguishes weight from activation tensors.
        Only per-block granularity uses it (see
        :meth:`PerBlockGranularity._handle_single_axis_block_size`); the other
        granularities ignore it.

        Example:
            - ``[10, 5, 2]`` with per-channel structuring on axis 1 results in
              ``[10, 1, 2]``
            - ``[10, 5, 2]`` with per-block structuring on axis 0 with block size 2
              results in ``[2, 5, 2]``
            - ``[10, 5, 2]`` with per-tensor structuring results in ``[10, 5, 2]``
        """
        pass

    def get_block_size(
        self,
        tensor_shape: torch.Size,
        quantization_target: _CompressionTargetTensor = _CompressionTargetTensor.WEIGHT,
    ) -> tuple[int, ...]:
        """
        Get a list of block sizes based on the granularity.

        Args:
            tensor_shape: Shape of the tensor being quantized.
            quantization_target: Whether the tensor is a weight or an activation.
                Defaults to ``WEIGHT``, which preserves the historical behavior.
        """
        return tuple(self._get_block_size(list(tensor_shape), quantization_target))

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

    def _get_block_size(
        self,
        block_sizes_list: list[int],
        quantization_target: _CompressionTargetTensor = _CompressionTargetTensor.WEIGHT,
    ) -> list[int]:
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

    def _get_block_size(
        self,
        block_sizes_list: list[int],
        quantization_target: _CompressionTargetTensor = _CompressionTargetTensor.WEIGHT,
    ) -> list[int]:

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

       - ``axis``: The axis to create blocks. May be negative (Python-style
         indexing). For weight quantization this is typically ``0`` or ``1``
         (the channel axes); for activation quantization it is commonly the
         last / reduction axis (e.g. ``-1``).
       - ``block_size``: Integer specifying block size for that axis

    2. Multi-axis mode: Create blocks across multiple axes simultaneously

       - ``axis``: Must be None
       - ``block_size``: Tuple specifying block size for each axis
         (-1 means no blocking)

    In single-axis mode, when ``axis`` is ``None`` and ``block_size`` is an integer,
    ``Quantizer.prepare()`` automatically resolves the axis based on the module type
    for weight quantization.

    Single-axis mode treats weights and activations differently. For weights only
    the two leading channel axes participate in blocking, so trailing kernel
    dimensions span a whole block. For activations every axis other than the block
    axis gets its own scale.

    .. list-table::
       :header-rows: 1

       * - Tensor shape (input)
         - target
         - axis
         - block_size
         - Shape of each block (output)
       * - [C_out, C_in]
         - weight
         - 1
         - 32
         - [1, 32]
       * - [C_out, C_in]
         - weight
         - None
         - (4, 8)
         - [4, 8]
       * - [C_out, C_in, KH, KW]
         - weight
         - 0
         - 16
         - [16, 1, KH, KW]
       * - [C_out, C_in, KH, KW]
         - weight
         - None
         - (4, 16, 3, -1)
         - [4, 16, 3, KW]
       * - [B, S, D]
         - activation
         - -1
         - 16
         - [1, 1, 16]
       * - [B, C, H, W]
         - activation
         - 1
         - 16
         - [1, 16, 1, 1]
    """

    axis: int | None = None
    block_size: Annotated[int, Field(gt=0)] | tuple[Annotated[int, Field(gt=0)] | Literal[-1], ...]

    def _get_block_size(
        self,
        block_sizes_list: list[int],
        quantization_target: _CompressionTargetTensor = _CompressionTargetTensor.WEIGHT,
    ) -> list[int]:
        if isinstance(self.block_size, tuple):
            return self._handle_multi_axis_block_size(block_sizes_list)
        else:
            return self._handle_single_axis_block_size(block_sizes_list, quantization_target)

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

    def _handle_single_axis_block_size(
        self,
        block_sizes_list: list[int],
        quantization_target: _CompressionTargetTensor = _CompressionTargetTensor.WEIGHT,
    ) -> list[int]:
        """Handle blocking when self.block_size is an integer"""
        # TODO: Logic to be added where if self.axis is None,
        #  we can figure out the optimal axis for the user
        if self.axis is None:
            raise ValueError("axis must be specified when block_size is an int")

        # Resolve negative (Python-style) axis to a non-negative index using the
        # tensor rank. This allows activation quantization to target the last /
        # reduction axis via axis=-1 regardless of the tensor's rank.
        rank = len(block_sizes_list)
        axis = self.axis + rank if self.axis < 0 else self.axis

        if axis < 0 or axis >= rank:
            raise ValueError(
                f"axis {self.axis} is out of bounds for tensor of rank {rank}. "
                f"Allowed axis range is [{-rank}, {rank})"
            )

        if block_sizes_list[axis] % self.block_size != 0:
            raise _BlockSizeMismatchError(
                f"Tensor size {block_sizes_list[axis]} along axis {axis} "
                f"is not divisible by block size {self.block_size}"
            )

        # How the non-block axes are treated depends on the quantization target,
        #
        # WEIGHT: only the two leading channel axes participate. The other
        #   channel axis becomes 1 (one scale per slice) while any trailing
        #   dimensions (index 2+, e.g. conv kernel dims) keep their full size so
        #   each block spans the whole kernel.
        #     [C_out, C_in, KH, KW], axis=0, block=16 -> [16, 1, KH, KW]
        #
        # ACTIVATION: blocking runs along a single axis and every other
        #   dimension gets its own scale, so all non-block axes become 1.
        #     [B, S, D],    axis=-1, block=16 -> [1, 1, 16]
        #     [B, C, H, W], axis=1,  block=16 -> [1, 16, 1, 1]
        if quantization_target == _CompressionTargetTensor.ACTIVATION:
            collapse_upto = rank
        else:
            collapse_upto = 2

        block_sizes_list[axis] = self.block_size
        for i, _ in enumerate(block_sizes_list[:collapse_upto]):
            if i != axis:
                block_sizes_list[i] = 1

        return block_sizes_list
