# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Pruning scheme specifications."""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from coreai_opt._utils.registry_utils import ConfigRegistryMixin

from .errors import _BlockSizeMismatchError


class PruningScheme(BaseModel, ConfigRegistryMixin):
    """Base class for pruning scheme specifications.

    A pruning scheme defines the structural pattern of sparsity applied
    to a tensor, and knows how to turn ``(weight, sparsity)`` into a binary
    mask. Call the public :meth:`compute_mask` to get a mask; subclasses
    implement the abstract :meth:`_compute_mask`, which handles sparsity
    strictly between 0 and 1 (the 0.0 / 1.0 edge cases are handled once, in
    the base class).

    The sole exception is :class:`NMStructured`, whose achieved sparsity is
    fixed by construction (``n / m``) rather than a free parameter — it
    overrides :meth:`compute_mask` directly and ignores the ``sparsity``
    argument entirely.

    Attributes:
        axis (int | None): The axis along which structured pruning is applied.
            ``None`` for unstructured (element-wise) pruning.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    axis: int | None = Field(
        default=None,
        description="Axis along which structured pruning is applied. None for unstructured.",
    )

    @model_serializer
    def _serialize_model(self) -> dict[str, Any]:
        """Custom serializer that includes the registry type."""
        data = {}

        for field_name in type(self).model_fields:
            data[field_name] = getattr(self, field_name)

        registry_key = None
        for key, registered_class in PruningScheme.REGISTRY.items():
            if registered_class is type(self):
                registry_key = key
                break

        if registry_key is not None:
            data["type"] = registry_key

        return data

    def compute_mask(self, weight: torch.Tensor, sparsity: float) -> torch.Tensor:
        """Compute a binary pruning mask for the given weight tensor.

        Args:
            weight (torch.Tensor): The weight tensor to compute a mask for.
            sparsity (float): Fraction of elements to prune, in [0, 1].

        Returns:
            torch.Tensor: Binary mask with the same shape as *weight* (1 = keep,
            0 = prune).
        """
        if sparsity == 0.0:
            return torch.ones_like(weight)
        if sparsity >= 1.0:
            return torch.zeros_like(weight)
        return self._compute_mask(weight, sparsity)

    @abstractmethod
    def _compute_mask(self, weight: torch.Tensor, sparsity: float) -> torch.Tensor:
        """Compute a mask for *sparsity* strictly between 0 and 1.

        Subclasses implement the scheme-specific masking logic here; the
        0.0 / 1.0 edge cases are already handled by :meth:`compute_mask`.
        """
        ...


@PruningScheme.register("unstructured")
class Unstructured(PruningScheme):
    """Unstructured pruning scheme.

    Individual elements are pruned independently — any element can be zeroed
    regardless of its position in the tensor.
    """

    axis: Literal[None] = None

    def _compute_mask(self, weight: torch.Tensor, sparsity: float) -> torch.Tensor:
        num_elements = weight.numel()
        num_keep = num_elements - math.floor(num_elements * sparsity)
        abs_weight = weight.abs()
        _, topk_indices = torch.topk(abs_weight.flatten(), num_keep)
        mask = torch.zeros(num_elements, dtype=weight.dtype, device=weight.device)
        mask[topk_indices] = 1.0
        return mask.reshape(weight.shape)


@PruningScheme.register("channel_structured")
class ChannelStructured(PruningScheme):
    """Channel-structured pruning scheme.

    Entire channels (slices along ``axis``) are pruned or kept together.
    Channel importance is determined by L1 norm of each channel.
    """

    axis: int = Field(default=0, description="Axis along which channels are pruned.")

    def _compute_mask(self, weight: torch.Tensor, sparsity: float) -> torch.Tensor:
        num_channels = weight.shape[self.axis]
        num_prune = math.floor(num_channels * sparsity)

        if num_prune == 0:
            return torch.ones_like(weight)
        if num_prune >= num_channels:
            return torch.zeros_like(weight)

        reduce_dims = [d for d in range(weight.ndim) if d != self.axis]
        channel_norms = weight.abs().sum(dim=reduce_dims)

        num_keep = num_channels - num_prune
        _, keep_indices = torch.topk(channel_norms, num_keep, largest=True)
        channel_mask = torch.zeros(num_channels, dtype=weight.dtype, device=weight.device)
        channel_mask[keep_indices] = 1.0

        shape = [1] * weight.ndim
        shape[self.axis] = num_channels
        return channel_mask.view(shape).expand_as(weight)


@PruningScheme.register("block_structured")
class BlockStructured(PruningScheme):
    """Block-structured pruning scheme.

    Generalizes :class:`ChannelStructured` to prune contiguous blocks of
    ``block_size`` slices along ``axis`` together, ranked by L2 norm
    (``ChannelStructured`` is equivalent to ``block_size=1``, ranked by L1
    norm — the two remain separate registered schemes).

    Unlike Phoenix's reference implementation, a tensor whose size along
    ``axis`` is not evenly divisible by ``block_size`` raises an error
    instead of being padded.
    """

    axis: int = Field(default=0, description="Axis along which blocks are formed.")
    block_size: int = Field(gt=0, description="Number of contiguous slices per block along axis.")

    def _compute_mask(self, weight: torch.Tensor, sparsity: float) -> torch.Tensor:
        num_along_axis = weight.shape[self.axis]
        if num_along_axis % self.block_size != 0:
            raise _BlockSizeMismatchError(
                f"Tensor size {num_along_axis} along axis {self.axis} is not "
                f"divisible by block_size {self.block_size}. Full tensor shape: "
                f"{tuple(weight.shape)}"
            )

        num_blocks = num_along_axis // self.block_size
        num_prune = math.floor(num_blocks * sparsity)

        if num_prune == 0:
            return torch.ones_like(weight)
        if num_prune >= num_blocks:
            return torch.zeros_like(weight)

        # axis is now at position 0; all other dims keep their relative order.
        moved = torch.movedim(weight, self.axis, 0)
        other_dims = moved.shape[1:]
        grouped = moved.view(num_blocks, self.block_size, *other_dims)
        block_norms = grouped.pow(2).sum(dim=tuple(range(1, grouped.ndim))).sqrt()

        num_keep = num_blocks - num_prune
        _, keep_indices = torch.topk(block_norms, num_keep, largest=True)
        block_mask = torch.zeros(num_blocks, dtype=weight.dtype, device=weight.device)
        block_mask[keep_indices] = 1.0

        mask = block_mask.repeat_interleave(self.block_size)
        mask = mask.view(num_along_axis, *([1] * len(other_dims))).expand(
            num_along_axis, *other_dims
        )
        # Restore axis to its original position.
        return torch.movedim(mask, 0, self.axis)


@PruningScheme.register("n_m_structured")
class NMStructured(PruningScheme):
    """N:M structured pruning scheme.

    Zeroes exactly ``n`` smallest-magnitude elements out of every contiguous
    group of ``m`` elements along ``axis`` — a hardware-friendly sparsity
    pattern with a fixed sparsity ratio of ``n / m``.

    Unlike other schemes, the achieved sparsity is fixed by construction and
    does not depend on ``PruningSpec.target_sparsity``: :meth:`compute_mask`
    overrides the base class directly and **ignores** its ``sparsity``
    argument.

    Unlike Phoenix's reference implementation, a tensor whose size along
    ``axis`` is not evenly divisible by ``m`` raises an error instead of
    being padded.
    """

    axis: int = Field(default=0, description="Axis along which N:M groups are formed.")
    n: int = Field(ge=0, description="Number of smallest-magnitude elements zeroed per group of m.")
    m: int = Field(gt=0, description="Group size along axis.")

    @model_validator(mode="after")
    def _validate_n_lt_m(self) -> NMStructured:
        if self.n >= self.m:
            raise ValueError(f"n ({self.n}) must be less than m ({self.m})")
        return self

    def _compute_mask(self, weight: torch.Tensor, sparsity: float) -> torch.Tensor:
        """Unreachable: :meth:`compute_mask` is overridden directly below and never
        delegates here. Defined only to satisfy the base class's abstract method.
        """
        raise NotImplementedError(
            "NMStructured overrides compute_mask directly; _compute_mask is unused."
        )

    def compute_mask(self, weight: torch.Tensor, sparsity: float) -> torch.Tensor:
        """Compute the N:M mask.

        ``sparsity`` is accepted for interface compatibility with
        :class:`PruneImplBase` but is ignored — achieved sparsity is fixed
        at ``n / m`` by construction.
        """
        num_along_axis = weight.shape[self.axis]
        if num_along_axis % self.m != 0:
            raise _BlockSizeMismatchError(
                f"Tensor size {num_along_axis} along axis {self.axis} is not "
                f"divisible by m {self.m}. Full tensor shape: {tuple(weight.shape)}"
            )

        if self.n == 0:
            return torch.ones_like(weight)

        # axis is now at position -1; all other dims keep their relative order.
        moved = torch.movedim(weight, self.axis, -1)
        original_shape = moved.shape
        grouped = moved.reshape(-1, self.m)

        prune_idx = torch.argsort(grouped.abs(), dim=1, stable=True)[:, : self.n]
        mask = torch.ones_like(grouped)
        mask.scatter_(1, prune_idx, 0.0)

        mask = mask.reshape(original_shape)
        # Restore axis to its original position.
        return torch.movedim(mask, -1, self.axis)
