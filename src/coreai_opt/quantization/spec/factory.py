# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from coreai_opt._utils.spec_utils import PartialConstructor as _PartialConstructor
from coreai_opt.config.spec import (
    CompressionComponentFactoryBase,
    CompressionTargetTensor,
)

from .fake_quantize import FakeQuantizeImplBase
from .qparams_calculator import (
    DynamicQParamsCalculator,
    MovingAverageQParamsCalculator,
    QParamsCalculatorBase,
    StaticQParamsCalculator,
    _DefaultQParamsCalculator,
)
from .range_calculator import RangeCalculatorBase
from .spec import QuantizationSpec


class QuantizationComponentFactory(CompressionComponentFactoryBase):
    """
    Factory class for creating quantization components from QuantizationSpec.

    This factory eliminates circular dependencies between QuantizationSpec and
    component classes (FakeQuantizeImplBase, QParamsCalculatorBase, RangeCalculatorBase)
    by centralizing the creation logic.
    """

    @classmethod
    def create_range_calculator(cls, spec: QuantizationSpec) -> RangeCalculatorBase:
        """
        Create a RangeCalculatorBase instance from a QuantizationSpec.

        Args:
            spec: QuantizationSpec instance containing configuration

        Returns:
            RangeCalculatorBase instance configured from the spec
        """
        # Standard arguments for range calculator
        common_args = {
            "granularity": spec.granularity,
        }

        # Automatically detect and include any extra arguments
        extra_args = spec.get_extra_args()

        # Create instance with all arguments
        return spec.range_calculator_cls(**common_args, **extra_args)

    @classmethod
    def create_qparams_calculator(
        cls, spec: QuantizationSpec, quantization_target: CompressionTargetTensor
    ) -> QParamsCalculatorBase:
        """
        Create a QParamsCalculatorBase instance from a QuantizationSpec.

        Args:
            spec: QuantizationSpec instance containing configuration
            quantization_target: The target tensor for quantization (weight/activation)

        Returns:
            QParamsCalculatorBase instance configured from the spec
        """
        # Resolve "default" marker class based on quantization target
        qparam_calculator_cls = spec.qparam_calculator_cls
        if qparam_calculator_cls is _DefaultQParamsCalculator:
            if quantization_target in (
                CompressionTargetTensor.WEIGHT,
                CompressionTargetTensor.LUT,
            ):
                qparam_calculator_cls = StaticQParamsCalculator
            elif quantization_target == CompressionTargetTensor.ACTIVATION:
                qparam_calculator_cls = MovingAverageQParamsCalculator
            else:
                raise ValueError(
                    f"Unsupported quantization target: {quantization_target}. "
                    f"Expected WEIGHT, ACTIVATION, or LUT."
                )

        if (
            qparam_calculator_cls is DynamicQParamsCalculator
            and quantization_target != CompressionTargetTensor.ACTIVATION
        ):
            raise ValueError(
                f"DynamicQParamsCalculator is only supported for activation "
                f"quantization, got quantization_target={quantization_target}."
            )

        # Create range calculator first
        range_calculator = cls.create_range_calculator(spec)

        # Standard arguments for qparams calculator
        common_args = {
            "dtype": spec.dtype,
            "qscheme": spec.qscheme,
            "granularity": spec.granularity,
            "target_dtype": spec.target_dtype,
            "quant_min": spec.quant_min,
            "quant_max": spec.quant_max,
            "range_calculator": range_calculator,
            "float_range": spec.float_range,
            "scale_dtype": spec.scale_dtype,
        }

        # Automatically detect and include any extra arguments
        extra_args = spec.get_extra_args()

        # Create instance with all arguments
        return qparam_calculator_cls(**common_args, **extra_args)

    @classmethod
    def construct(
        cls, spec: QuantizationSpec | None, target: CompressionTargetTensor
    ) -> FakeQuantizeImplBase | None:
        """
        Create a fake quantizer instance from a QuantizationSpec.

        This method implements the base class interface and delegates to
        create_fake_quantizer.

        Args:
            spec: QuantizationSpec instance containing configuration
            target: The target tensor for compression (weight or activation)

        Returns:
            FakeQuantizeImplBase instance configured from the spec, or None if
            spec is None
        """
        if spec is None:
            return None
        return cls.create_fake_quantizer(spec, target)

    @classmethod
    def construct_partial(
        cls, spec: QuantizationSpec | None, target: CompressionTargetTensor
    ) -> _PartialConstructor[FakeQuantizeImplBase] | None:
        """
        Create a fake quantizer partial object for deferred construction.

        This method implements the base class interface and delegates to
        create_fake_quantizer_partial.

        Args:
            spec: QuantizationSpec instance containing configuration
            target: The target tensor for compression (weight or activation)

        Returns:
            PartialConstructor: A partial object for deferred construction, or None
            if spec is None
        """
        if spec is None:
            return None
        return cls.create_fake_quantizer_partial(spec, target)

    @classmethod
    def create_fake_quantizer(
        cls, spec: QuantizationSpec, quantization_target: CompressionTargetTensor
    ) -> FakeQuantizeImplBase:
        """
        Create a FakeQuantizeImplBase instance from a QuantizationSpec.

        This method automatically detects any extra arguments in the spec beyond
        the base QuantizationSpec fields and passes them to the fake quantizer
        constructor.

        Args:
            spec: QuantizationSpec instance containing configuration
            quantization_target: The target tensor for quantization

        Returns:
            FakeQuantizeImplBase instance configured from the spec

        Example:
            >>> spec = QuantizationSpec(...)
            >>> fake_quantize = QuantizationComponentFactory.create_fake_quantizer(
            ...     spec, quantization_target=CompressionTargetTensor.WEIGHT
            ... )
            >>> extended_spec = ExtraArgQuantizationSpec(eps=0.1, ...)
            >>> fake_quantize = QuantizationComponentFactory.create_fake_quantizer(
            ...     extended_spec,
            ...     quantization_target=CompressionTargetTensor.ACTIVATION
            ... )
        """
        # For direct instantiation, create qparams calculator immediately
        qparams_calculator = cls.create_qparams_calculator(spec, quantization_target)

        # Standard arguments that all fake quantizers need
        common_args = {
            "dtype": spec.dtype,
            "qscheme": spec.qscheme,
            "qformulation": spec.qformulation,
            "granularity": spec.granularity,
            "target_dtype": spec.target_dtype,
            "quant_min": spec.quant_min,
            "quant_max": spec.quant_max,
            "qparams_calculator": qparams_calculator,
            "quantization_target": quantization_target,
            "n_bits": spec.n_bits,
        }

        # Automatically detect and include any extra arguments
        extra_args = spec.get_extra_args()

        # Create instance with all arguments
        return spec.fake_quantize_cls(**common_args, **extra_args)

    @classmethod
    def create_fake_quantizer_partial(
        cls, spec: QuantizationSpec, quantization_target: CompressionTargetTensor
    ) -> _PartialConstructor[FakeQuantizeImplBase]:
        """
        Create a fake quantizer partial object for deferred construction
        by the graph-mode prepare API (torchao PT2E).

        Args:
            spec: QuantizationSpec instance containing configuration
            quantization_target: The target tensor for quantization

        Returns:
            PartialConstructor: A partial object that can be used by the graph-mode
                          prepare API to construct fake quantizer instances. Each call
                          to the partial will create a new instance with its own
                          qparams_calculator.
        """
        # For partial construction, we need to defer qparams_calculator creation
        # to ensure each instance gets its own calculator

        # Standard arguments that all fake quantizers need
        # (excluding qparams_calculator)
        common_args = {
            "dtype": spec.dtype,
            "qscheme": spec.qscheme,
            "qformulation": spec.qformulation,
            "granularity": spec.granularity,
            "target_dtype": spec.target_dtype,
            "quant_min": spec.quant_min,
            "quant_max": spec.quant_max,
            "quantization_target": quantization_target,
            "n_bits": spec.n_bits,
        }

        # Automatically detect and include any extra arguments
        extra_args = spec.get_extra_args()

        # Create a factory function that creates qparams_calculator on each call
        def qparams_calculator_factory():
            return cls.create_qparams_calculator(spec, quantization_target)

        # Create partially constructed class obj with callable args
        # for qparams_calculator
        return spec.fake_quantize_cls.with_args(**common_args, **extra_args).with_callable_args(
            qparams_calculator=qparams_calculator_factory
        )
