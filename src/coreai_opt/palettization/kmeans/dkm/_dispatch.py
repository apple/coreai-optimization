# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Resolve the DKM forward function for a device.

Providers registered under the ``coreai_opt.dkm_providers`` entry-point group may supply
the forward.
"""

import logging
from collections.abc import Callable
from functools import cache
from importlib.metadata import entry_points
from typing import NamedTuple, Protocol

import torch

__all__ = ["Forward", "Provider", "get_forward"]

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "coreai_opt.dkm_providers"


class Forward(Protocol):
    """Signature of a DKM forward pass, as called by the DKM training strategy."""

    def __call__(
        self,
        weights: torch.Tensor,
        centroids: torch.Tensor,
        tau: float,
        kmeans_max_iter: int,
        zero_threshold: float,
        enforce_zero: bool,
        quantize_fn: Callable[[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(reconstructed_weights, centroids)`` for ``weights`` of shape ``(P, N, D)``."""
        ...


class Provider(NamedTuple):
    """A named provider; ``load(device)`` returns its forward, or ``None`` if it does not apply."""

    name: str
    load: Callable[[torch.device], Forward | None]


@cache
def _providers() -> tuple[Provider, ...]:
    providers: list[Provider] = []
    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        try:
            providers = entry_point.load()(providers)
        except Exception:
            logger.warning(
                "DKM: provider %s failed to load; skipping.", entry_point.name, exc_info=True
            )
    return tuple(providers)


@cache
def get_forward(device: torch.device) -> Forward | None:
    """Return the forward supplied by a provider for ``device``, or ``None``."""
    for provider in _providers():
        forward = provider.load(device)
        if forward is not None:
            logger.info("DKM: using the %s provider on %s", provider.name, device)
            return forward
    logger.info("DKM: using the PyTorch implementation on %s", device)
    return None
