# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Pruning parametrization modules."""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import torch

from coreai_opt._utils.spec_utils import (
    PartialConstructor as _PartialConstructor,
    with_args as _with_args,
)
from coreai_opt.config.spec import CompressionSimulatorBase

from .scheme import ChannelStructured, PruningScheme

if TYPE_CHECKING:
    # Imported only for type checking — runtime would be a circular import via
    # coreai_opt.pruning.config.
    from coreai_opt.pruning.config.sparsity_schedule import SparsityScheduleBase


class PruneImplBase(CompressionSimulatorBase):
    """Abstract base for pruning parametrizations that mask a layer's weight.

    Subclasses implement :meth:`compute_mask` — a pure static function from
    ``(weight, sparsity, pruning_scheme)`` to a binary mask. The base class
    handles the mask buffer and optional schedule-driven sparsity updates.
    """

    schedule: SparsityScheduleBase | None = None
    _sparsity: float
    _target_sparsity: float
    _pruning_scheme: PruningScheme
    _dirty: bool

    def __init__(
        self,
        target_sparsity: float,
        pruning_scheme: PruningScheme,
        **kwargs: Any,
    ):
        super().__init__()
        self._target_sparsity = target_sparsity
        self._sparsity = target_sparsity
        self._pruning_scheme = pruning_scheme
        self.schedule = None
        self._dirty = True
        self.register_buffer("mask", torch.empty(0))

    @property
    def sparsity(self) -> float:
        """Sparsity that the current mask reflects. Use ``update_sparsity`` to change."""
        return self._sparsity

    @staticmethod
    @abstractmethod
    def compute_mask(
        weight: torch.Tensor,
        sparsity: float,
        pruning_scheme: PruningScheme,
    ) -> torch.Tensor:
        """Compute a binary pruning mask for the given weight tensor.

        Args:
            weight (torch.Tensor): The weight tensor to compute a mask for.
            sparsity (float): Fraction of elements to prune, in [0, 1].
            pruning_scheme (PruningScheme): Structural pattern of sparsity.

        Returns:
            torch.Tensor: Binary mask with the same shape as *weight* (1 = keep,
            0 = prune).
        """
        ...

    def update_sparsity(self, step_count: int) -> None:
        """Update the sparsity based on the configured schedule and the provided step count.

        Raises:
            RuntimeError: If no schedule is attached. This method should be
                invoked only after setting the ``schedule`` property.
        """
        if self.schedule is None:
            raise RuntimeError(
                "update_sparsity called on a PruneImplBase with no schedule attached."
            )
        new = self.schedule.compute_sparsity(step_count, self._target_sparsity, self._sparsity)
        if new != self._sparsity:
            self._sparsity = new
            self._dirty = True

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """Compute / re-compute the mask if stale, and then apply it to the weight."""
        if self._dirty:
            new_mask = self.compute_mask(weight, self._sparsity, self._pruning_scheme)
            if self.mask.device != weight.device or self.mask.dtype != weight.dtype:
                self.mask = self.mask.to(device=weight.device, dtype=weight.dtype)
            if self.mask.shape != new_mask.shape:
                self.mask.resize_(new_mask.shape)
            self.mask.copy_(new_mask)
            self._dirty = False
        return weight * self.mask

    @classmethod
    def with_args(cls, **kwargs: Any) -> _PartialConstructor[PruneImplBase]:
        """Create a partial constructor with pre-filled arguments."""
        return _with_args(cls, **kwargs)


@PruneImplBase.register("default")
class _MagnitudePruneImpl(PruneImplBase):
    """Magnitude-based pruning supporting unstructured and channel-structured schemes.

    Prunes a given tensor to target sparsity by zero-ing out the smallest-magnitude
    elements until desired sparsity is achieved.
    """

    @staticmethod
    def compute_mask(
        weight: torch.Tensor,
        sparsity: float,
        pruning_scheme: PruningScheme,
    ) -> torch.Tensor:
        """Compute a magnitude-based mask respecting the pruning scheme.

        Args:
            weight (torch.Tensor): The weight tensor.
            sparsity (float): Fraction of elements to prune, in [0, 1].
            pruning_scheme (PruningScheme): Structural pattern of sparsity.

        Returns:
            torch.Tensor: Binary mask (1 = keep, 0 = prune).
        """
        if sparsity == 0.0:
            return torch.ones_like(weight)
        if sparsity >= 1.0:
            return torch.zeros_like(weight)

        # TODO: Replace this with generic abstractions
        if isinstance(pruning_scheme, ChannelStructured):
            return _MagnitudePruneImpl._compute_channel_mask(weight, sparsity, pruning_scheme.axis)
        return _MagnitudePruneImpl._compute_unstructured_mask(weight, sparsity)

    @staticmethod
    def _compute_unstructured_mask(weight: torch.Tensor, sparsity: float) -> torch.Tensor:
        """Element-wise magnitude pruning."""
        num_elements = weight.numel()
        num_keep = num_elements - math.floor(num_elements * sparsity)
        abs_weight = weight.abs()
        _, topk_indices = torch.topk(abs_weight.flatten(), num_keep)
        mask = torch.zeros(num_elements, dtype=weight.dtype, device=weight.device)
        mask[topk_indices] = 1.0
        return mask.reshape(weight.shape)

    @staticmethod
    def _compute_channel_mask(
        weight: torch.Tensor,
        sparsity: float,
        axis: int,
    ) -> torch.Tensor:
        """Channel-structured magnitude pruning along *axis*.

        Channel importance is measured by L1 norm. The least-important
        channels are pruned entirely.
        """
        num_channels = weight.shape[axis]
        num_prune = math.floor(num_channels * sparsity)

        if num_prune == 0:
            return torch.ones_like(weight)
        if num_prune >= num_channels:
            return torch.zeros_like(weight)

        reduce_dims = [d for d in range(weight.ndim) if d != axis]
        channel_norms = weight.abs().sum(dim=reduce_dims)

        num_keep = num_channels - num_prune
        _, keep_indices = torch.topk(channel_norms, num_keep, largest=True)
        channel_mask = torch.zeros(num_channels, dtype=weight.dtype, device=weight.device)
        channel_mask[keep_indices] = 1.0

        shape = [1] * weight.ndim
        shape[axis] = num_channels
        return channel_mask.view(shape).expand_as(weight)
