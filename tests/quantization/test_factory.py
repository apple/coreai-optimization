# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import pytest
import torch

from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.quantization.spec.factory import QuantizationComponentFactory
from coreai_opt.quantization.spec.fake_quantize import (
    FakeQuantizeImplBase,
    _DefaultFakeQuantizeImpl,
)
from coreai_opt.quantization.spec.granularity import (
    PerChannelGranularity,
    PerTensorGranularity,
)
from coreai_opt.quantization.spec.qparams_calculator import (
    DynamicQParamsCalculator,
    GlobalMinMaxQParamsCalculator,
    MovingAverageQParamsCalculator,
    QParamsCalculatorBase,
    StaticQParamsCalculator,
    _DefaultQParamsCalculator,
)
from coreai_opt.quantization.spec.range_calculator import (
    MinMaxRangeCalculator,
    RangeCalculatorBase,
)
from coreai_opt.quantization.spec.spec import (
    QuantizationSpec,
    default_weight_quantization_spec,
)


class TestQuantizationComponentFactory:
    """Test the QuantizationComponentFactory class"""

    def test_create_range_calculator(self):
        """Test creating range calculator from spec"""
        spec = QuantizationSpec(
            dtype=torch.int8,
            qscheme="symmetric",
            granularity=PerTensorGranularity(),
            fake_quantize_cls=_DefaultFakeQuantizeImpl,
            qparam_calculator_cls=StaticQParamsCalculator,
            range_calculator_cls=MinMaxRangeCalculator,
        )

        range_calc = QuantizationComponentFactory.create_range_calculator(spec)

        assert isinstance(range_calc, MinMaxRangeCalculator)
        assert range_calc.granularity == spec.granularity

    @pytest.mark.parametrize(
        "range",
        [
            (0, 10),
            (-10, 0),
            (-10, 10),
            (-20, -10),
            (10, 20),
        ],
    )
    def test_create_range_calculator_functional(self, range):
        spec = QuantizationSpec(
            dtype=torch.int8,
            qscheme="symmetric",
            granularity=PerTensorGranularity(),
            fake_quantize_cls=_DefaultFakeQuantizeImpl,
            qparam_calculator_cls=StaticQParamsCalculator,
            range_calculator_cls=MinMaxRangeCalculator,
        )
        range_calculator = QuantizationComponentFactory.create_range_calculator(spec)
        assert range_calculator is not None

        low, high = range
        # Generate tensor with random values in the range
        x = torch.randint(low=low, high=high + 1, size=(10, 10)).to(torch.float)

        # Ensure the tensor contains the exact boundary values we expect
        x[0, 0] = float(low)  # Set minimum value
        x[0, 1] = float(high)  # Set maximum value

        min_val, max_val = range_calculator(x)
        assert min_val == low and max_val == high

    def test_create_qparams_calculator(self):
        """Test creating qparams calculator from spec"""
        spec = QuantizationSpec(
            dtype=torch.int8,
            qscheme="symmetric",
            granularity=PerChannelGranularity(axis=1),
            fake_quantize_cls=_DefaultFakeQuantizeImpl,
            qparam_calculator_cls=StaticQParamsCalculator,
            range_calculator_cls=MinMaxRangeCalculator,
        )

        qparams_calc = QuantizationComponentFactory.create_qparams_calculator(
            spec, CompressionTargetTensor.WEIGHT
        )

        assert isinstance(qparams_calc, StaticQParamsCalculator)
        assert qparams_calc.dtype == spec.dtype
        assert qparams_calc.qscheme == spec.qscheme
        assert qparams_calc.granularity == spec.granularity
        assert qparams_calc.target_dtype == spec.target_dtype
        assert qparams_calc.quant_min == spec.quant_min
        assert qparams_calc.quant_max == spec.quant_max
        assert isinstance(qparams_calc.range_calculator, MinMaxRangeCalculator)

    def test_create_qparams_calculator_functional(self):
        """Test that qparams calculator created by factory works functionally"""
        spec = default_weight_quantization_spec()
        qparams_calc = QuantizationComponentFactory.create_qparams_calculator(
            spec, CompressionTargetTensor.WEIGHT
        )

        # Create test tensor and observe it
        x = torch.randn(10, 20)
        qparams_calc(x)  # This will update the qparams calculator

        # Get qparams
        scale, zero_point, minval = qparams_calc.get_qparams()

        # Verify qparams are tensors with expected properties
        assert isinstance(scale, torch.Tensor)
        assert isinstance(zero_point, torch.Tensor)
        assert isinstance(minval, torch.Tensor)
        assert scale.numel() > 0
        assert zero_point.numel() > 0
        assert minval.numel() > 0

    def test_create_fake_quantizer(self):
        """Test creating fake quantizer from spec"""
        spec = QuantizationSpec(
            dtype=torch.int8,
            qscheme="symmetric",
            granularity=PerTensorGranularity(),
            fake_quantize_cls=_DefaultFakeQuantizeImpl,
            qparam_calculator_cls=StaticQParamsCalculator,
            range_calculator_cls=MinMaxRangeCalculator,
        )

        fake_quantizer = QuantizationComponentFactory.create_fake_quantizer(
            spec,
            quantization_target="weight",
        )

        assert isinstance(fake_quantizer, _DefaultFakeQuantizeImpl)
        assert fake_quantizer.dtype == spec.dtype
        assert fake_quantizer.qscheme == spec.qscheme
        assert fake_quantizer.granularity == spec.granularity
        assert fake_quantizer.target_dtype == spec.target_dtype
        assert fake_quantizer.quant_min == spec.quant_min
        assert fake_quantizer.quant_max == spec.quant_max
        assert isinstance(fake_quantizer.qparams_calculator, StaticQParamsCalculator)

    def test_create_fake_quantizer_functional(self):
        """Test that fake quantizer created by factory works functionally"""
        spec = default_weight_quantization_spec()
        fake_quantizer = QuantizationComponentFactory.create_fake_quantizer(
            spec,
            quantization_target="weight",
        )

        # Create test tensor
        x = torch.randn(10, 20)

        # Test forward pass
        fq_x = fake_quantizer(x)

        # Verify output shape and type
        assert fq_x.shape == x.shape
        assert fq_x.dtype == x.dtype

        # Verify quantization actually happened (values should be different but close)
        torch.testing.assert_close(x, fq_x, atol=2e-2, rtol=1e-5)

    def test_create_fake_quantizer_export_mode(self):
        """Test export mode functionality on fake quantizer created from spec"""
        spec = default_weight_quantization_spec()
        fq = QuantizationComponentFactory.create_fake_quantizer(
            spec,
            quantization_target="weight",
        )

        # Initially not in export mode
        assert not fq.qparams_calculator._export_mode

        # Set export mode
        fq.set_export_mode(True)
        assert fq.qparams_calculator._export_mode

        # Unset export mode
        fq.set_export_mode(False)
        assert not fq.qparams_calculator._export_mode

    @pytest.mark.parametrize(
        "dtype",
        [torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2, torch.float4_e2m1fn_x2],
    )
    @pytest.mark.parametrize(
        "granularity",
        [
            PerTensorGranularity(),
            PerChannelGranularity(axis=0),
            PerChannelGranularity(axis=1),
        ],
    )
    @pytest.mark.parametrize(
        "qparam_calculator_cls",
        [StaticQParamsCalculator, MovingAverageQParamsCalculator, GlobalMinMaxQParamsCalculator],
    )
    def test_factory_with_different_configurations(self, dtype, granularity, qparam_calculator_cls):
        """Test factory with various configuration combinations"""
        spec = QuantizationSpec(
            dtype=dtype,
            qscheme="symmetric",
            granularity=granularity,
            fake_quantize_cls=_DefaultFakeQuantizeImpl,
            qparam_calculator_cls=qparam_calculator_cls,
            range_calculator_cls=MinMaxRangeCalculator,
        )

        fake_quantizer = QuantizationComponentFactory.create_fake_quantizer(
            spec,
            quantization_target="weight",
        )

        # Verify all attributes match the spec
        assert fake_quantizer.dtype == spec.dtype
        assert fake_quantizer.qscheme == spec.qscheme
        assert fake_quantizer.granularity == spec.granularity
        assert fake_quantizer.target_dtype == spec.target_dtype
        assert fake_quantizer.quant_min == spec.quant_min
        assert fake_quantizer.quant_max == spec.quant_max
        assert isinstance(fake_quantizer.qparams_calculator, qparam_calculator_cls)


class TestExtraArgsSupport:
    """Test factory support for extended specs with extra arguments"""

    def test_get_extra_args_base_spec(self):
        """Test get_extra_args returns empty dict for base spec"""
        spec = QuantizationSpec(
            dtype=torch.int8,
            qscheme="symmetric",
            granularity=PerTensorGranularity(),
            fake_quantize_cls=_DefaultFakeQuantizeImpl,
            qparam_calculator_cls=StaticQParamsCalculator,
            range_calculator_cls=MinMaxRangeCalculator,
        )

        extra_args = spec.get_extra_args()
        assert extra_args == {}

    def test_get_extra_args_extended_spec(self):
        """Test get_extra_args detects additional fields in extended specs"""

        class ExtraArgQuantizationSpec(QuantizationSpec):
            eps: float
            temperature: float = 1.0

        spec = ExtraArgQuantizationSpec(
            dtype=torch.int8,
            qscheme="symmetric",
            granularity=PerTensorGranularity(),
            fake_quantize_cls=_DefaultFakeQuantizeImpl,
            qparam_calculator_cls=StaticQParamsCalculator,
            range_calculator_cls=MinMaxRangeCalculator,
            eps=0.1,
            temperature=2.0,
        )

        extra_args = spec.get_extra_args()
        expected = {"eps": 0.1, "temperature": 2.0}
        assert extra_args == expected

    def test_factory_with_extra_args_fake_quantizer(self):
        """Test factory can handle fake quantizers that need extra arguments"""

        @FakeQuantizeImplBase.register("extra-arg-test")
        class ExtraArgFakeQuantizeImpl(_DefaultFakeQuantizeImpl):
            def __init__(
                self,
                dtype,
                qscheme,
                qformulation,
                granularity,
                target_dtype,
                quant_min,
                quant_max,
                qparams_calculator,
                eps,
                temperature=1.0,
                **kwargs,
            ):
                super().__init__(
                    dtype,
                    qscheme,
                    qformulation,
                    granularity,
                    target_dtype,
                    quant_min,
                    quant_max,
                    qparams_calculator,
                    **kwargs,
                )
                self.eps = eps
                self.temperature = temperature

        class ExtraArgQuantizationSpec(QuantizationSpec):
            eps: float
            temperature: float = 1.0

        spec = ExtraArgQuantizationSpec(
            dtype=torch.int8,
            qscheme="symmetric",
            granularity=PerTensorGranularity(),
            fake_quantize_cls=ExtraArgFakeQuantizeImpl,
            qparam_calculator_cls=StaticQParamsCalculator,
            range_calculator_cls=MinMaxRangeCalculator,
            eps=0.1,
            temperature=2.0,
        )

        fake_quantizer = QuantizationComponentFactory.create_fake_quantizer(
            spec,
            quantization_target="weight",
        )

        # Verify the fake quantizer was created with extra args
        assert isinstance(fake_quantizer, ExtraArgFakeQuantizeImpl)
        assert fake_quantizer.eps == 0.1
        assert fake_quantizer.temperature == 2.0

        # Verify it still works functionally
        x = torch.randn(5, 10)
        fq_x = fake_quantizer(x)
        assert fq_x.shape == x.shape
        assert fq_x.dtype == x.dtype

    def test_factory_with_extra_args_fake_quantizer_old_spec(self):
        """Test factory fails gracefully when extra args are missing"""

        @FakeQuantizeImplBase.register("extra-arg-required")
        class ExtraArgRequiredFakeQuantizeImpl(_DefaultFakeQuantizeImpl):
            def __init__(
                self,
                dtype,
                qscheme,
                granularity,
                target_dtype,
                quant_min,
                quant_max,
                qparams_calculator,
                eps,  # Required argument
                **kwargs,
            ):
                super().__init__(
                    dtype,
                    qscheme,
                    granularity,
                    target_dtype,
                    quant_min,
                    quant_max,
                    qparams_calculator,
                    **kwargs,
                )
                self.eps = eps

        # Try to create with base spec (missing eps argument)
        spec = QuantizationSpec(
            dtype=torch.int8,
            qscheme="symmetric",
            granularity=PerTensorGranularity(),
            fake_quantize_cls=ExtraArgRequiredFakeQuantizeImpl,
            qparam_calculator_cls=StaticQParamsCalculator,
            range_calculator_cls=MinMaxRangeCalculator,
        )

        # Should fail with TypeError due to missing required argument
        with pytest.raises(TypeError):
            QuantizationComponentFactory.create_fake_quantizer(
                spec,
                quantization_target="weight",
            )

    def test_factory_with_extra_args_qparams_calculator(self):
        """Test factory can handle qparams calculators that need extra arguments"""

        @QParamsCalculatorBase.register("extra-averaging_constant")
        class ExtraAveragingConstQParamsCalculator(MovingAverageQParamsCalculator):
            def __init__(self, averaging_constant=0.9, extra_param=1.0, **kwargs):
                super().__init__(averaging_constant=averaging_constant, **kwargs)
                self.extra_param = extra_param

        class ExtraArgQuantizationSpec(QuantizationSpec):
            averaging_constant: float = 0.95
            extra_param: float = 2.0

        spec = ExtraArgQuantizationSpec(
            dtype=torch.int8,
            qscheme="symmetric",
            granularity=PerTensorGranularity(),
            fake_quantize_cls=_DefaultFakeQuantizeImpl,
            qparam_calculator_cls=ExtraAveragingConstQParamsCalculator,
            range_calculator_cls=MinMaxRangeCalculator,
            averaging_constant=0.95,
            extra_param=2.0,
        )

        qparams_calc = QuantizationComponentFactory.create_qparams_calculator(
            spec, CompressionTargetTensor.ACTIVATION
        )

        # Verify the qparams calculator was created with extra args
        assert isinstance(qparams_calc, ExtraAveragingConstQParamsCalculator)
        assert qparams_calc.averaging_constant == 0.95
        assert qparams_calc.extra_param == 2.0

    def test_factory_with_extra_args_range_calculator(self):
        """Test factory can handle range calculators that need extra arguments"""

        @RangeCalculatorBase.register("extra-range")
        class ExtraRangeCalculator(MinMaxRangeCalculator):
            def __init__(self, granularity, scale_factor=1.0, **kwargs):
                super().__init__(granularity, **kwargs)
                self.scale_factor = scale_factor

        class ExtraArgQuantizationSpec(QuantizationSpec):
            scale_factor: float = 1.5

        spec = ExtraArgQuantizationSpec(
            dtype=torch.int8,
            qscheme="symmetric",
            granularity=PerTensorGranularity(),
            fake_quantize_cls=_DefaultFakeQuantizeImpl,
            qparam_calculator_cls=StaticQParamsCalculator,
            range_calculator_cls=ExtraRangeCalculator,
            scale_factor=1.5,
        )

        range_calc = QuantizationComponentFactory.create_range_calculator(spec)

        # Verify the range calculator was created with extra args
        assert isinstance(range_calc, ExtraRangeCalculator)
        assert range_calc.scale_factor == 1.5


class TestPartialQuantizerSharing:
    """Test that partial fake quantizers don't share qparams_calculator instances"""

    def test_single_partial_multiple_instantiations(self):
        """Test if a single partial creates different qparams_calculator instances"""
        spec = QuantizationSpec(
            dtype=torch.int8,
            qscheme="symmetric",
            granularity=PerTensorGranularity(),
            fake_quantize_cls=_DefaultFakeQuantizeImpl,
            qparam_calculator_cls=StaticQParamsCalculator,
            range_calculator_cls=MinMaxRangeCalculator,
        )

        partial = QuantizationComponentFactory.create_fake_quantizer_partial(
            spec,
            quantization_target="weight",
        )

        # Instantiate the same partial multiple times
        fq1 = partial()
        fq2 = partial()

        # Each instance should have its own qparams_calculator
        assert id(fq1.qparams_calculator) != id(fq2.qparams_calculator)

    def test_state_sharing_after_forward(self):
        """Test if qparams_calculators share state after forward passes"""
        spec = QuantizationSpec(
            dtype=torch.int8,
            qscheme="symmetric",
            granularity=PerTensorGranularity(),
            fake_quantize_cls=_DefaultFakeQuantizeImpl,
            qparam_calculator_cls=StaticQParamsCalculator,
            range_calculator_cls=MinMaxRangeCalculator,
        )

        partial1 = QuantizationComponentFactory.create_fake_quantizer_partial(
            spec,
            quantization_target="weight",
        )
        partial2 = QuantizationComponentFactory.create_fake_quantizer_partial(
            spec,
            quantization_target="weight",
        )

        fq1 = partial1()
        fq2 = partial2()

        # Create different tensors to see if state is shared
        x1 = torch.tensor([1.0, 2.0, 3.0])
        x2 = torch.tensor([10.0, 20.0, 30.0])

        # Forward pass on fq1
        fq1(x1)
        scale1_after_fq1, _, _ = fq1.calculate_qparams()

        # Forward pass on fq2
        fq2(x2)
        scale1_after_fq2, _, _ = fq1.calculate_qparams()

        # Check if fq1's state changed when fq2 was called
        # If they don't share state, fq1's scale should remain unchanged
        torch.testing.assert_close(scale1_after_fq1, scale1_after_fq2)

    def test_range_calculator_sharing(self):
        """Test if range_calculators are shared between qparams_calculators"""
        spec = QuantizationSpec(
            dtype=torch.int8,
            qscheme="symmetric",
            granularity=PerTensorGranularity(),
            fake_quantize_cls=_DefaultFakeQuantizeImpl,
            qparam_calculator_cls=StaticQParamsCalculator,
            range_calculator_cls=MinMaxRangeCalculator,
        )

        partial1 = QuantizationComponentFactory.create_fake_quantizer_partial(
            spec,
            quantization_target="weight",
        )
        partial2 = QuantizationComponentFactory.create_fake_quantizer_partial(
            spec,
            quantization_target="weight",
        )

        fq1 = partial1()
        fq2 = partial2()

        # Each qparams_calculator should have its own range_calculator
        assert id(fq1.qparams_calculator.range_calculator) != id(
            fq2.qparams_calculator.range_calculator
        )

    def test_different_partials_sharing(self):
        """Test that different partials create separate qparams_calculator instances"""
        spec = QuantizationSpec(
            dtype=torch.int8,
            qscheme="symmetric",
            granularity=PerTensorGranularity(),
            fake_quantize_cls=_DefaultFakeQuantizeImpl,
            qparam_calculator_cls=StaticQParamsCalculator,
            range_calculator_cls=MinMaxRangeCalculator,
        )

        # Create two partials from the same spec
        partial1 = QuantizationComponentFactory.create_fake_quantizer_partial(
            spec, quantization_target=CompressionTargetTensor.WEIGHT
        )
        partial2 = QuantizationComponentFactory.create_fake_quantizer_partial(
            spec, quantization_target=CompressionTargetTensor.WEIGHT
        )

        # Instantiate the partials
        fq1 = partial1()
        fq2 = partial2()

        # Each fake quantizer should have its own qparams_calculator
        assert id(fq1.qparams_calculator) != id(fq2.qparams_calculator)

    def test_partial_vs_direct_instantiation(self):
        """Test that partial and direct instantiation both work correctly"""
        spec = QuantizationSpec(
            dtype=torch.int8,
            qscheme="symmetric",
            granularity=PerTensorGranularity(),
            fake_quantize_cls=_DefaultFakeQuantizeImpl,
            qparam_calculator_cls=StaticQParamsCalculator,
            range_calculator_cls=MinMaxRangeCalculator,
        )

        # Create via partial
        partial = QuantizationComponentFactory.create_fake_quantizer_partial(
            spec, quantization_target=CompressionTargetTensor.WEIGHT
        )
        fq_partial = partial()

        # Create directly
        fq_direct = QuantizationComponentFactory.create_fake_quantizer(
            spec, quantization_target=CompressionTargetTensor.WEIGHT
        )

        # Both should be functional and have separate qparams_calculators
        assert id(fq_partial.qparams_calculator) != id(fq_direct.qparams_calculator)

        # Test functionality
        x = torch.randn(5, 10)

        fq_partial_out = fq_partial(x)
        fq_direct_out = fq_direct(x)

        assert fq_partial_out.shape == x.shape
        assert fq_direct_out.shape == x.shape
        assert fq_partial_out.dtype == x.dtype
        assert fq_direct_out.dtype == x.dtype


class TestCompressionTargetTensorAttribute:
    """Test the quantization_target attribute functionality"""

    @pytest.mark.parametrize(
        "quantization_target",
        [
            CompressionTargetTensor.WEIGHT,
            CompressionTargetTensor.ACTIVATION,
        ],
    )
    def test_set_quantization_target(self, quantization_target):
        """Test that quantization_target can be set correctly"""
        spec = default_weight_quantization_spec()

        fq = QuantizationComponentFactory.create_fake_quantizer(
            spec, quantization_target=quantization_target
        )

        assert hasattr(fq, "quantization_target")
        assert fq.quantization_target == quantization_target


class TestQParamCalculatorClassResolution:
    """Test that the factory correctly resolves the _DefaultQParamsCalculator class"""

    def test_resolution_for_weight(self):
        """
        Test that 'default' qparam calculator resolves
        to StaticQParamsCalculator for weights
        """
        spec = QuantizationSpec(qparam_calculator_cls="default")

        # Verify spec has the marker class
        assert spec.qparam_calculator_cls == _DefaultQParamsCalculator

        # Create qparams calculator via factory for weight
        qparams_calc = QuantizationComponentFactory.create_qparams_calculator(
            spec, CompressionTargetTensor.WEIGHT
        )

        # Should be resolved to StaticQParamsCalculator
        assert isinstance(qparams_calc, StaticQParamsCalculator)

    def test_resolution_for_activation(self):
        """
        Test that 'default' qparam calculator resolves
        to MovingAverageQParamsCalculator for activations
        """
        spec = QuantizationSpec(qparam_calculator_cls="default")

        # Verify spec has the marker class
        assert spec.qparam_calculator_cls == _DefaultQParamsCalculator

        # Create qparams calculator via factory for activation
        qparams_calc = QuantizationComponentFactory.create_qparams_calculator(
            spec, CompressionTargetTensor.ACTIVATION
        )

        # Should be resolved to MovingAverageQParamsCalculator
        assert isinstance(qparams_calc, MovingAverageQParamsCalculator)

    def test_resolution_in_fake_quantizer_weight(self):
        """Test marker resolution through full fake quantizer creation for weight"""
        spec = QuantizationSpec(qparam_calculator_cls="default")

        fake_quantizer = QuantizationComponentFactory.create_fake_quantizer(
            spec, CompressionTargetTensor.WEIGHT
        )

        # The qparams_calculator should be StaticQParamsCalculator
        assert isinstance(fake_quantizer.qparams_calculator, StaticQParamsCalculator)

    @pytest.mark.parametrize(
        "qparam_calculator_string,qparam_calculator_cls",
        [
            ("default", MovingAverageQParamsCalculator),
            ("dynamic", DynamicQParamsCalculator),
        ],
    )
    def test_resolution_in_fake_quantizer_activation(
        self, qparam_calculator_string, qparam_calculator_cls
    ):
        """Test marker resolution through full fake quantizer creation for activation"""
        spec = QuantizationSpec(qparam_calculator_cls=qparam_calculator_string)

        fake_quantizer = QuantizationComponentFactory.create_fake_quantizer(
            spec, CompressionTargetTensor.ACTIVATION
        )

        assert isinstance(fake_quantizer.qparams_calculator, qparam_calculator_cls)

    def test_default_class_not_callable(self):
        """Test that the marker class raises an error if forward() is called"""
        spec = QuantizationSpec(qparam_calculator_cls="default")

        # Verify spec has the marker class
        assert spec.qparam_calculator_cls == _DefaultQParamsCalculator

        # Instantiating the marker directly should raise an error
        with pytest.raises(RuntimeError, match="_DefaultQParamsCalculator is a marker class"):
            _marker_instance = _DefaultQParamsCalculator(
                dtype=torch.int8,
                qscheme=spec.qscheme,
                granularity=spec.granularity,
                target_dtype=torch.int8,
                quant_min=-128,
                quant_max=127,
                range_calculator=MinMaxRangeCalculator(granularity=spec.granularity),
                float_range=[None, None],
            )

    def test_explicit_class_not_resolved(self):
        """Test that explicit class specification is not resolved"""
        # When explicitly specifying StaticQParamsCalculator, it should stay that way
        spec = QuantizationSpec(qparam_calculator_cls="static")
        assert spec.qparam_calculator_cls == StaticQParamsCalculator

        # Should get StaticQParamsCalculator for both weight and activation
        qparams_weight = QuantizationComponentFactory.create_qparams_calculator(
            spec, CompressionTargetTensor.WEIGHT
        )
        qparams_activation = QuantizationComponentFactory.create_qparams_calculator(
            spec, CompressionTargetTensor.ACTIVATION
        )

        assert isinstance(qparams_weight, StaticQParamsCalculator)
        assert isinstance(qparams_activation, StaticQParamsCalculator)

    def test_global_minmax_string_resolves_to_class(self):
        """Test that 'global_minmax' string resolves to GlobalMinMaxQParamsCalculator"""
        spec = QuantizationSpec(qparam_calculator_cls="global_minmax")
        assert spec.qparam_calculator_cls == GlobalMinMaxQParamsCalculator

        # Should get GlobalMinMaxQParamsCalculator for both weight and activation
        qparams_weight = QuantizationComponentFactory.create_qparams_calculator(
            spec, CompressionTargetTensor.WEIGHT
        )
        qparams_activation = QuantizationComponentFactory.create_qparams_calculator(
            spec, CompressionTargetTensor.ACTIVATION
        )

        assert isinstance(qparams_weight, GlobalMinMaxQParamsCalculator)
        assert isinstance(qparams_activation, GlobalMinMaxQParamsCalculator)

    def test_resolution_for_dynamic_qparams(self):
        """Test that 'dynamic' string resolves to DynamicQParamsCalculator"""
        spec = QuantizationSpec(qparam_calculator_cls="dynamic")
        assert spec.qparam_calculator_cls == DynamicQParamsCalculator

        qparams_calc = QuantizationComponentFactory.create_qparams_calculator(
            spec, CompressionTargetTensor.ACTIVATION
        )
        assert isinstance(qparams_calc, DynamicQParamsCalculator)

    @pytest.mark.parametrize(
        "target",
        [CompressionTargetTensor.WEIGHT, CompressionTargetTensor.LUT],
    )
    def test_dynamic_rejected_for_non_activation(self, target):
        """Test that 'dynamic' raises ValueError when used for weight/LUT targets"""
        spec = QuantizationSpec(qparam_calculator_cls="dynamic")
        with pytest.raises(
            ValueError,
            match="DynamicQParamsCalculator is only supported for activation",
        ):
            QuantizationComponentFactory.create_qparams_calculator(spec, target)
