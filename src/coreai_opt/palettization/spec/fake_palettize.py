# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

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
from coreai_opt.quantization.spec import QuantizationSpec


class _FakePalettizeImplBase(CompressionSimulatorBase, nn.Module):
    """Base class for fake palettization implementations with clustering and
    reconstruction methods.
    """

    indices: torch.Tensor
    per_channel_scale: torch.Tensor | None
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
        # Non-persistent (kept out of new checkpoints); when set to 1 (at runtime or
        # via a legacy checkpoint) the forward pass re-clusters centroids every call.
        self.register_buffer(
            "observer_enabled", torch.tensor([0], dtype=torch.uint8), persistent=False
        )
        self._disabled = False

        self.register_buffer("indices", None)
        self.register_buffer("per_channel_scale", None)

    def is_disabled(self) -> bool:
        """Return True if fake palettization has been disabled."""
        return self._disabled

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Fake-palettize ``tensor`` through the enable/disable lifecycle.

        Delegates representation-specific work to ``ensure_initialized`` and
        ``forward_enabled``.
        """
        if self._disabled:
            return tensor

        self.ensure_initialized(tensor)

        # Check for self._disabled again in case ensure_initialized disabled the palettizer.
        if self._disabled:
            return tensor
        if self.fake_palett_enabled[0] == 0:
            return tensor
        return self.forward_enabled(tensor)

    @abstractmethod
    def ensure_initialized(self, tensor: torch.Tensor) -> None:
        """Initialize compression parameters from ``tensor`` on first use.

        Set ``self._disabled = True`` if ``tensor`` is incompatible with the
        configured spec.
        """
        raise NotImplementedError()

    @abstractmethod
    def forward_enabled(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return the palettized output for ``tensor`` when enabled."""
        raise NotImplementedError()

    @abstractmethod
    def hard_assign(self, weight: torch.Tensor) -> torch.Tensor:
        """Return the hard-assigned (deployable) palettized reconstruction of ``weight``."""
        raise NotImplementedError()

    @abstractmethod
    def _palettize(
        self, lut: torch.Tensor, indices: torch.Tensor, original_weights: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct palettized weights from lookup table and indices.

        ``lut`` shape must be of the following form:
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

        ``indices`` shape must match the shape of the palettized weight.
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

        This method handles the loading of palettization-specific buffers (indices,
        per_channel_scale) that may be dynamically created during forward passes. By
        registering them here, we ensure they are properly loaded from saved checkpoints
         and don't generate unexpected key warnings.

        The method is called automatically by PyTorch during model loading (torch.load,
        load_state_dict, etc.) and should not be called directly.
        """
        buffer_names = {
            "indices",
            "per_channel_scale",
            "centroids",
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

        obs_key = prefix + "observer_enabled"
        if obs_key in state_dict:
            self.observer_enabled.copy_(state_dict[obs_key])
            if obs_key in unexpected_keys:
                unexpected_keys.remove(obs_key)

        # Accept (and ignore) buffers from legacy checkpoints that are now
        # derived properties, so old state dicts load without unexpected-key errors.
        # ``lut`` is consumed by the subclass to reconstruct ``centroids``.
        for legacy_name in (
            "lut",
            "quantized_lut",
            "lut_quantization_scale",
            "lut_quantization_zero_point",
        ):
            prefixed_key = prefix + legacy_name
            if prefixed_key in unexpected_keys:
                unexpected_keys.remove(prefixed_key)

    def enable_fake_palett(self, enabled: bool = True) -> None:
        self.fake_palett_enabled[0] = 1 if enabled else 0

    def disable_fake_palett(self):
        self.enable_fake_palett(False)


def _enable_fake_palett(mod):
    """Enable fake palettization for the module."""
    if isinstance(mod, _FakePalettizeImplBase):
        mod.enable_fake_palett()


def _disable_fake_palett(mod):
    """Disable fake palettization for the module."""
    if isinstance(mod, _FakePalettizeImplBase):
        mod.disable_fake_palett()
