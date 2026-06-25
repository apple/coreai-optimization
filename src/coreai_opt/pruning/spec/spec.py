# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Pruning specification."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BeforeValidator, Field, PrivateAttr, field_validator, model_validator

from coreai_opt.common import CompressionType
from coreai_opt.config.spec import CompressionSpec

from .prune import PruneImplBase
from .scheme import PruningScheme, Unstructured


class PruningSpec(CompressionSpec):
    """Specification for pruning tensors.

    Attributes:
        target_sparsity (float): Fraction of elements to prune, in ``[0, 1]``.
            Default: 0.5.
        pruning_scheme (PruningScheme): Structural pattern of sparsity.
            Default: ``Unstructured()``.
        pruning_algo (type[PruneImplBase]): Pruning implementation class.
            Default: ``"default"`` (magnitude-based pruning).

    Example:
        >>> spec = PruningSpec()
        >>> spec.target_sparsity
        0.5
        >>> spec = PruningSpec(target_sparsity=0.75)
    """

    _compression_type: CompressionType = PrivateAttr(default=CompressionType.PRUNING)

    target_sparsity: float = Field(default=0.5, ge=0.0, le=1.0)
    pruning_scheme: Annotated[
        PruningScheme,
        BeforeValidator(PruningScheme.maybe_build_from_dict),
    ] = Field(default_factory=Unstructured)
    pruning_algo: type[PruneImplBase] = Field(default="default", validate_default=True)

    @field_validator("pruning_algo", mode="before")
    @classmethod
    def convert_pruning_algo(cls, data: Any) -> type[PruneImplBase]:
        """Resolve string keys to registered pruning implementation classes."""
        return PruneImplBase.resolve(data)

    @model_validator(mode="before")
    @classmethod
    def _strip_computed_fields(cls, data: Any) -> Any:
        """Strip computed fields when deserializing from dict for round-trip support."""
        if isinstance(data, dict):
            declared = set(cls.model_fields.keys())
            return {k: v for k, v in data.items() if k in declared}
        return data


def default_weight_pruning_spec() -> PruningSpec:
    """Return the default pruning spec for weight tensors."""
    return PruningSpec(
        target_sparsity=0.5,
        pruning_scheme=Unstructured(),
        pruning_algo="default",
    )
