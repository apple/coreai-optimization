# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from enum import StrEnum

import torch
from torchao.quantization import MappingType as TorchAOMappingType


class QuantizationScheme(StrEnum):
    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"
    SYMMETRIC_WITH_CLIPPING = "symmetric_with_clipping"

    @classmethod
    def _to_mapping_type(cls, qscheme: "QuantizationScheme") -> TorchAOMappingType:
        if qscheme == cls.SYMMETRIC or qscheme == cls.SYMMETRIC_WITH_CLIPPING:
            return TorchAOMappingType.SYMMETRIC
        elif qscheme == cls.ASYMMETRIC:
            return TorchAOMappingType.ASYMMETRIC
        else:
            raise ValueError(f"Unknown value for quantization scheme: {qscheme}")

    @classmethod
    def _maybe_clip_bounds(
        cls,
        qscheme: "QuantizationScheme",
        dtype: torch.dtype,
        min_val: int,
        max_val: int,
    ) -> tuple[int, int]:
        """
        Clip min_val for SYMMETRIC_WITH_CLIPPING to ensure equal bins on
        each side of zero.
        """
        if qscheme == cls.SYMMETRIC_WITH_CLIPPING and dtype.is_signed:
            min_val = -max_val
        return (min_val, max_val)
