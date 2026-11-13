# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Casting related utilities including FP32 -> FP16 and INT32 -> INT16 passes."""

from .casting import (
    ANE_UNSAFE_FP16_OPS,
    cast_fp32_to_fp16,
    cast_int32_to_int16,
    cast_to_16_bit_precision,
)

__all__ = [
    "ANE_UNSAFE_FP16_OPS",
    "cast_fp32_to_fp16",
    "cast_int32_to_int16",
    "cast_to_16_bit_precision",
]
