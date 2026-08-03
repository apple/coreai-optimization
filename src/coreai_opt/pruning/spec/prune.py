# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Pruning parametrization modules."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

import torch

from coreai_opt._utils.spec_utils import (
    PartialConstructor as _PartialConstructor,
    with_args as _with_args,
)
from coreai_opt.config.spec import CompressionSimulatorBase

from .scheme import PruningScheme

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
    """Magnitude-based pruning that delegates to the configured pruning scheme.

    Prunes a given tensor to target sparsity by zero-ing out the smallest-magnitude
    elements, per whatever structural pattern ``pruning_scheme`` defines (unstructured,
    channel-structured, block-structured, N:M-structured, or future schemes).
    """

    @staticmethod
    def compute_mask(
        weight: torch.Tensor,
        sparsity: float,
        pruning_scheme: PruningScheme,
    ) -> torch.Tensor:
        """Compute a magnitude-based mask by delegating to the pruning scheme.

        Args:
            weight (torch.Tensor): The weight tensor.
            sparsity (float): Fraction of elements to prune, in [0, 1].
            pruning_scheme (PruningScheme): Structural pattern of sparsity.

        Returns:
            torch.Tensor: Binary mask (1 = keep, 0 = prune).
        """
        return pruning_scheme.compute_mask(weight, sparsity)
