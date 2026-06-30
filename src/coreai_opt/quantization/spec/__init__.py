# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Quantization specs, schemes, granularity classes, and parameter calculators."""

from .factory import QuantizationComponentFactory
from .granularity import (
    PerBlockGranularity,
    PerChannelGranularity,
    PerTensorGranularity,
    QuantizationGranularity,
)
from .qformulation import QuantizationFormulation
from .qparams_calculator import (
    DynamicQParamsCalculator,
    GlobalMinMaxQParamsCalculator,
    MovingAverageQParamsCalculator,
    QParamsCalculatorBase,
    RunningRangeMixin,
    StatefulQParamsCalculatorBase,
    StatelessQParamsCalculatorBase,
    StaticQParamsCalculator,
)
from .qscheme import QuantizationScheme
from .range_calculator import MinMaxRangeCalculator, RangeCalculatorBase
from .spec import (
    QuantizationSpec,
    default_activation_quantization_spec,
    default_weight_quantization_spec,
)

__all__ = [
    "DynamicQParamsCalculator",
    "GlobalMinMaxQParamsCalculator",
    "MinMaxRangeCalculator",
    "MovingAverageQParamsCalculator",
    "PerBlockGranularity",
    "PerChannelGranularity",
    "PerTensorGranularity",
    "QParamsCalculatorBase",
    "QuantizationComponentFactory",
    "QuantizationFormulation",
    "QuantizationGranularity",
    "QuantizationScheme",
    "QuantizationSpec",
    "RangeCalculatorBase",
    "RunningRangeMixin",
    "StatefulQParamsCalculatorBase",
    "StatelessQParamsCalculatorBase",
    "StaticQParamsCalculator",
    "default_activation_quantization_spec",
    "default_weight_quantization_spec",
]
