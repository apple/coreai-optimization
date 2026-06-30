# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Utilities for working with compression metadata and torch dtypes."""

from __future__ import annotations

import warnings
from typing import Any, Final

import torch
from pydantic import BaseModel, ConfigDict, field_validator

from coreai_opt.common import CompressionType

STATE_DICT_METADATA_BUFFER_PREFIX: Final = "_COREML_"
BUFFER_NAME_SEPARATOR: Final = "/"
METADATA_VERSION_BUFFER: Final = (
    STATE_DICT_METADATA_BUFFER_PREFIX + BUFFER_NAME_SEPARATOR + "metadata_version"
)
METADATA_VERSION_VALUE: Final = 1
EXPECTED_BUFFER_NAME_SPLITS: Final = 3


class MILCompressionMetadata(BaseModel):
    """Compression metadata for MIL export.

    Stores quantization parameters as buffers in state_dict with _COREML_/ prefix
    for CoreMLTools MIL converter to read during model conversion.

    Args:
        param_name (str): Name of parameter this metadata corresponds to
        compression_type (CompressionType | list[CompressionType]): Compression type(s)
            applied (e.g., CompressionType.QUANTIZATION, or a list for combined
            compression like [CompressionType.PALETTIZATION, CompressionType.QUANTIZATION])
        quantization_n_bits (int | None): Number of bits used for quantization
        quantization_scale (torch.Tensor | None): Scale tensor for quantization
        zero_point (torch.Tensor | None): Zero point tensor for affine/unsigned
            quantization
        lut (torch.Tensor | None): Lookup table for palettized weights.
        palettization_scale (torch.Tensor | None): Per channel scales used to
            normalize weights before being palettized.
        vector_axis (int | None): Axis along which vector palettization is performed.

    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allow torch.Tensor and enum types
        validate_assignment=True,
    )

    param_name: str
    compression_type: list[CompressionType]
    quantization_n_bits: int | None = None
    quantization_scale: torch.Tensor | None = None
    zero_point: torch.Tensor | None = None
    lut: torch.Tensor | None = None
    palettization_scale: torch.Tensor | None = None
    vector_axis: int | None = None

    @field_validator("compression_type", mode="before")
    @classmethod
    def _wrap_compression_type(
        cls,
        v: CompressionType | list[CompressionType],
    ) -> list[CompressionType]:
        if isinstance(v, CompressionType):
            return [v]
        return v

    def register(
        self,
        module: torch.nn.Module,
    ) -> None:
        """Register compression metadata as buffers in module's state_dict.

        Args:
            module: The module to register buffers in

        Raises:
            TypeError: If metadata value cannot be converted to tensor

        """
        # Validate that param_name corresponds to an actual parameter
        param_names: dict[str, torch.nn.Parameter] = dict(module.named_parameters())
        if self.param_name not in param_names:
            warnings.warn(
                f"Parameter '{self.param_name}' not found in module. "
                f"Metadata will be registered but may not correspond to any actual parameter.",  # noqa: E501
                UserWarning,
                stacklevel=2,
            )

        # Get dict representation, excluding None values
        metadata_dict: dict[str, Any] = self.model_dump(
            exclude_none=True,
            exclude={"param_name"},
        )

        for metadata_key, value in metadata_dict.items():
            buffer_name = self._get_metadata_buffer_name(metadata_key)
            buffer_value: torch.Tensor | int = value
            if metadata_key == "compression_type":
                # Handle compression type - convert enums to CoreML code tensor
                buffer_value = torch.tensor([v.to_coreml_code() for v in value])
            # For other metadata, convert to tensor if not already
            elif not torch.is_tensor(buffer_value):
                try:
                    buffer_value = torch.tensor(buffer_value)
                except (ValueError, TypeError) as e:
                    msg = f"Cannot convert metadata value for '{metadata_key}' to tensor: {e}"  # noqa: E501
                    raise TypeError(msg) from e

            module.register_buffer(buffer_name, buffer_value)

    def _get_metadata_buffer_name(self, metadata_key: str) -> str:
        """Get the buffer name for a metadata key."""
        return BUFFER_NAME_SEPARATOR.join(
            [STATE_DICT_METADATA_BUFFER_PREFIX, self.param_name, metadata_key],
        )

    @classmethod
    def register_version(cls, model: torch.nn.Module) -> None:
        """Register metadata version buffer in model.

        Args:
            model: The model to register the version buffer in

        """
        model.register_buffer(
            METADATA_VERSION_BUFFER,
            torch.tensor(METADATA_VERSION_VALUE),
        )
