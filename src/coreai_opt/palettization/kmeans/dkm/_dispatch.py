# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Resolve the DKM forward function for a device.

Backends registered with :class:`DKMBackend` may supply the forward.
"""

import logging
from collections.abc import Callable
from functools import cache
from typing import Protocol

import torch

from coreai_opt._utils.registry_utils import FunctionRegistryMixin

__all__ = ["DKMBackend", "Forward", "get_forward"]

logger = logging.getLogger(__name__)


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


class DKMBackend(FunctionRegistryMixin):
    """Registry of DKM backends, tried in registration order.

    Each entry is a loader ``(device) -> Forward | None`` that returns its forward, or ``None``
    if it does not apply to the device.
    """


@cache
def get_forward(device: torch.device) -> Forward | None:
    """Return the forward supplied by a registered backend for ``device``, or ``None``."""
    for name, load in DKMBackend.REGISTRY.items():
        forward = load(device)
        if forward is not None:
            logger.info("DKM: using the %s backend on %s", name, device)
            return forward
    logger.info("DKM: using the PyTorch implementation on %s", device)
    return None
