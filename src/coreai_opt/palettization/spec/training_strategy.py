# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Pluggable training-time behavior for fake-palettize modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from pydantic import BaseModel, ConfigDict, model_serializer

from coreai_opt._utils.registry_utils import ConfigRegistryMixin as _ConfigRegistryMixin

if TYPE_CHECKING:
    from coreai_opt.palettization.kmeans.kmeans_fake_palettize import _KMeansFakePalettize


class TrainingStrategy(ABC):
    """Contract for a fake-palettize module's training-time forward pass."""

    @abstractmethod
    def train_forward(self, module: _KMeansFakePalettize, weight: torch.Tensor) -> torch.Tensor:
        """Compute this training step's output for ``weight``.

        May mutate ``module.centroids`` in place (must set
        ``module._indices_stale = True`` if it does). Use ``module.quantize_lut()``
        to fake-quantize intermediate centroids against the module's
        configured LUT quantizer. Must leave ``module.centroids`` such that
        ``module.hard_assign(weight)`` produces a sensible result at eval time.
        """
        raise NotImplementedError


class _DefaultTrainingStrategy(TrainingStrategy):
    """Post-training, one-shot k-means — today's KMeansPalettizer behavior.

    This strategy does not train the palettized weights. Inside a
    ``training_mode()`` loop it reconstructs each forward from the frozen
    centroids/indices computed at ``prepare()`` time, so no gradient flows back
    to a palettized weight — palettized weights stay fixed at their one-shot
    k-means values. Once a module's ``pat_schedule`` has enabled fake
    palettization, the rest of the model still trains normally and its forward
    pass sees the palettized weights, so non-palettized parameters adapt around
    them (palettization-aware fine-tuning of the rest of the network). To learn
    the palettized weights/centroids themselves, register a custom
    ``TrainingStrategy``.
    """

    def train_forward(self, module: _KMeansFakePalettize, weight: torch.Tensor) -> torch.Tensor:
        return module.hard_assign(weight)


class TrainingStrategyConfig(BaseModel, _ConfigRegistryMixin):
    """Base class for a fake-palettize module's training-strategy settings.

    Each subclass points ``_strategy_cls`` at its paired ``TrainingStrategy``
    behavior class; ``build_strategy()`` constructs that strategy from this
    config's own fields.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Each subclass points this at its paired TrainingStrategy behavior class.
    _strategy_cls: ClassVar[type[TrainingStrategy]]

    @model_serializer
    def _serialize_model(self) -> dict[str, Any]:
        """Custom serializer that includes the registry type."""
        data = {}

        for field_name in type(self).model_fields:
            data[field_name] = getattr(self, field_name)

        # Find the registry key for this class type
        registry_key = None
        # Use the base class registry instead of instance registry
        for key, registered_class in TrainingStrategyConfig.REGISTRY.items():
            if registered_class is type(self):
                registry_key = key
                break

        if registry_key is not None:
            data["type"] = registry_key

        return data

    def build_strategy(self) -> TrainingStrategy:
        """Construct this config's paired ``TrainingStrategy`` behavior instance."""
        kwargs = {k: v for k, v in self.model_dump().items() if k != "type"}
        return self._strategy_cls(**kwargs)


@TrainingStrategyConfig.register("default")
class DefaultTrainingConfig(TrainingStrategyConfig):
    """Settings for the default, post-training one-shot k-means strategy. No fields."""

    _strategy_cls = _DefaultTrainingStrategy
