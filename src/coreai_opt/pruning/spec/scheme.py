# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Pruning scheme specifications."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer

from coreai_opt._utils.registry_utils import ConfigRegistryMixin as _ConfigRegistryMixin


class PruningScheme(BaseModel, _ConfigRegistryMixin):
    """Base class for pruning scheme specifications.

    A pruning scheme defines the structural pattern of sparsity applied
    to a tensor. Subclasses represent different ways of structuring the pruning.

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


@PruningScheme.register("unstructured")
class Unstructured(PruningScheme):
    """Unstructured pruning scheme.

    Individual elements are pruned independently — any element can be zeroed
    regardless of its position in the tensor.
    """

    axis: Literal[None] = None


@PruningScheme.register("channel_structured")
class ChannelStructured(PruningScheme):
    """Channel-structured pruning scheme.

    Entire channels (slices along ``axis``) are pruned or kept together.
    Channel importance is determined by the pruning algorithm (e.g. L1 norm
    of each channel for magnitude-based pruning).
    """

    axis: int = Field(default=0, description="Axis along which channels are pruned.")
