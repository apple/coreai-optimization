# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Quantization compressor, configuration, specs, and granularity classes."""

from .spec import (  # noqa: I001
    QuantizationSpec,
    fake_quantize,
    qparams_calculator,
    range_calculator,
)

# Import config after spec to avoid circular imports
from .config import (
    ExecutionMode,
    InvalidExecutionModeError,
    ModuleQuantizerConfig,
    QuantizerConfig,
)
from .quantizer import Quantizer

__all__ = [
    "ExecutionMode",
    "InvalidExecutionModeError",
    "ModuleQuantizerConfig",
    "QuantizationSpec",
    "Quantizer",
    "QuantizerConfig",
]
