# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from typing import Annotated, Any, Literal

import torch
from pydantic import (
    BaseModel,
    BeforeValidator,
    PositiveInt,
    PrivateAttr,
    model_validator,
)

from coreai_opt.common import CompressionType
from coreai_opt.config.spec import CompressionSpec
from coreai_opt.quantization.spec import (
    PerTensorGranularity as QuantPerTensorGranularity,
    QuantizationFormulation,
    QuantizationSpec,
)

from .granularity import PalettizationGranularity, PerTensorGranularity

_SUPPORTED_LUT_DTYPES = {torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2}


class PalettizationSpec(CompressionSpec):
    """
    Specification for palettization compression of neural network weights.

    Palettization is a compression technique that reduces memory usage by representing
    weights using a lookup table (LUT) instead of storing full precision values.
    Weights are clustered into a small number of representative values (the palette),
    and each weight is replaced with an index into this palette.

    This specification configures all aspects of the palettization process including
    the number of bits for indices, the quantization of the lookup table,
    and the granularity at which palettization is applied.

    Attributes:
        n_bits: Number of bits used for palette indices. Determines palette size
            (2^n_bits entries). Must be one of {1, 2, 3, 4, 6, 8}. Default: 4.
        lut_qspec: Quantization specification for the lookup table values.
            If None, no quantization is applied to the LUT. When specified,
            only ``torch.int8``, ``torch.uint8``, ``torch.float8_e4m3fn``, and
            ``torch.float8_e5m2`` dtypes are supported, and granularity must be
            ``PerTensorGranularity``. FP8 dtypes require symmetric quantization.
            Default: None.
        granularity: Defines how palettization is applied - per-tensor applies a
            single palette to the entire tensor, per-channel applies separate
            palettes to each channel. Default: PerTensorGranularity().
        cluster_dim: The dimension of centroids for each lookup table.
            The centroid is a scalar by default. When cluster_dim > 1, it indicates 2-D
            clustering, and each cluster_dim length of weight vectors along the output
            channel are palettized using the same 2-D centroid. The length of each entry
            in the lookup tables is equal to cluster_dim. Default: 1.
        enable_per_channel_scale: When set to True, weights are normalized along the
            output channels using per-channel scales before being palettized.
            Default: False.

    Example:
        >>> # Basic 4-bit palettization
        >>> spec = PalettizationSpec()

        >>> # 2-bit palettization with quantized int8 lookup table
        >>> from coreai_opt.quantization.spec import QuantizationSpec, QuantizationScheme
        >>> spec = PalettizationSpec(
        ...     n_bits=2,
        ...     lut_qspec=QuantizationSpec(
        ...         dtype=torch.int8,
        ...         qscheme=QuantizationScheme.SYMMETRIC,
        ...     ),
        ... )

        >>> # Per-channel palettization with scaling
        >>> from coreai_opt.palettization.spec import PerGroupedChannelGranularity
        >>> spec = PalettizationSpec(
        ...     granularity=PerGroupedChannelGranularity(axis=0, group_size=32),
        ...     enable_per_channel_scale=True
        ... )
    """

    n_bits: Literal[1, 2, 3, 4, 6, 8] = 4
    lut_qspec: QuantizationSpec | None = None
    granularity: Annotated[
        PalettizationGranularity,
        BeforeValidator(PalettizationGranularity.maybe_build_from_dict),
    ] = PerTensorGranularity()
    cluster_dim: PositiveInt = 1
    enable_per_channel_scale: bool = False

    # Private attribute for compression type
    _compression_type: CompressionType = PrivateAttr(default=CompressionType.PALETTIZATION)

    # Sparsity level, in [0, 1]. Set via the `_sparsity` constructor/dict key.
    _sparsity: float | None = PrivateAttr(default=None)

    def __init__(self, **data: Any) -> None:
        sparsity = data.pop("_sparsity", None)
        super().__init__(**data)
        if sparsity is not None:
            self._validate_sparsity(sparsity)
            self._sparsity = sparsity

    def _validate_sparsity(self, sparsity: float) -> None:
        """Reject sparsity combined with a position-dependent LUT/scale mapping."""
        if not (0.0 <= sparsity <= 1.0):
            raise ValueError(f"_sparsity must be in [0, 1], got {sparsity}")
        if self.lut_qspec is not None:
            raise ValueError("lut_qspec not supported for joint sparsity.")
        if not isinstance(self.granularity, PerTensorGranularity):
            raise ValueError(f"granularity={self.granularity} not supported for joint sparsity.")
        if self.enable_per_channel_scale:
            raise ValueError("enable_per_channel_scale not supported for joint sparsity.")

    @model_validator(mode="after")
    def validate_lut_qspec(self) -> "PalettizationSpec":
        """Validate that lut_qspec only uses supported configurations."""
        if self.lut_qspec is None:
            return self

        if self.lut_qspec.dtype not in _SUPPORTED_LUT_DTYPES:
            raise ValueError(
                f"lut_qspec.dtype must be one of {_SUPPORTED_LUT_DTYPES}, "
                f"got {self.lut_qspec.dtype}"
            )

        if not isinstance(self.lut_qspec.granularity, QuantPerTensorGranularity):
            raise ValueError(
                f"lut_qspec.granularity must be PerTensorGranularity, "
                f"got {type(self.lut_qspec.granularity).__name__}"
            )

        if self.lut_qspec.qformulation == QuantizationFormulation.MINVAL:
            raise ValueError(
                "lut_qspec.qformulation=MINVAL is not supported for palettization. "
                "Use lut_qspec.qformulation=ZP instead."
            )

        return self

    def model_dump_preserve_objects(self) -> dict[str, Any]:
        """
        Custom model dump that preserves Pydantic BaseModel instances as objects
        instead of serializing them.

        This method creates a dictionary representation of the spec while keeping
        all Pydantic BaseModel fields as the original Python objects rather than
        serializing them to dictionaries. Non-Pydantic model fields are serialized
        normally. This is useful when you want to work with actual object instances
        programmatically.

        Returns:
            Dictionary with serialized non-Pydantic fields and preserved Pydantic objs.
        """

        # Find all fields that contain Pydantic BaseModel instances
        exclude_set = set()
        pydantic_fields = {}

        for field_name in self.model_fields:
            field_value = getattr(self, field_name)
            if isinstance(field_value, BaseModel):
                exclude_set.add(field_name)
                pydantic_fields[field_name] = field_value

        # Get regular model dump but exclude Pydantic model fields
        data = self.model_dump(exclude=exclude_set)

        # Add back the Pydantic model fields as original objects
        data.update(pydantic_fields)

        return data


def default_weight_palettization_spec() -> PalettizationSpec:
    return PalettizationSpec(
        n_bits=4,
        lut_qspec=None,
        granularity=PerTensorGranularity(),
        cluster_dim=1,
        enable_per_channel_scale=False,
    )
