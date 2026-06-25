# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import pytest
import torch

from coreai_opt.config.spec.factory import CompressionTargetTensor
from coreai_opt.quantization import QuantizationSpec
from coreai_opt.quantization.spec import (
    PerBlockGranularity,
    PerChannelGranularity,
    PerTensorGranularity,
    QuantizationScheme,
)
from coreai_opt.quantization.spec.factory import QuantizationComponentFactory
from coreai_opt.quantization.spec.qparams_calculator import (
    DynamicQParamsCalculator,
    GlobalMinMaxQParamsCalculator,
    MovingAverageQParamsCalculator,
    StatelessQParamsCalculatorBase,
    StaticQParamsCalculator,
)
from coreai_opt.quantization.spec.range_calculator import MinMaxRangeCalculator


def test_uninitialized_warning():
    """Test that using get_qparams before forward pass issues a warning"""
    qparams_calc = StaticQParamsCalculator(
        dtype=torch.int8,
        qscheme=QuantizationScheme.ASYMMETRIC,
        granularity=PerTensorGranularity(),
        target_dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        range_calculator=MinMaxRangeCalculator(granularity=PerTensorGranularity()),
        float_range=[None, None],
    )

    # Check for warning
    with pytest.warns(UserWarning):
        scale, zero_point, minval = qparams_calc.get_qparams()

    # Default values (buffers are empty until first forward pass)
    assert scale.numel() == 0
    assert zero_point.numel() == 0
    assert minval.numel() == 0


def test_export_mode():
    x = torch.tensor([-3.0, 0, 3.0], dtype=torch.float32)
    qparams_calc = StaticQParamsCalculator(
        dtype=torch.int8,
        qscheme=QuantizationScheme.ASYMMETRIC,
        granularity=PerTensorGranularity(),
        target_dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        range_calculator=MinMaxRangeCalculator(granularity=PerTensorGranularity()),
        float_range=[None, None],
    )

    scale1, zero_point1, minval1 = qparams_calc(x)

    x_different = torch.tensor([-10.0, 0, 10.0], dtype=torch.float32)

    # Set export mode and see that the scale / zp *dont* change
    qparams_calc.set_export_mode(enabled=True)
    scale2, zero_point2, minval2 = qparams_calc(x_different)
    torch.testing.assert_close(scale1, scale2)
    torch.testing.assert_close(zero_point1, zero_point2)
    torch.testing.assert_close(minval1, minval2)

    # Unset export mode and see that the scale / zp *do* change
    qparams_calc.set_export_mode(enabled=False)
    scale3, zero_point3, minval3 = qparams_calc(x_different)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(scale1, scale3)
        torch.testing.assert_close(zero_point1, zero_point3)
        torch.testing.assert_close(minval1, minval3)


@pytest.mark.parametrize(
    "qscheme,dtype,x,expected_scale,expected_zero_point,expected_minval",
    [
        pytest.param(
            QuantizationScheme.ASYMMETRIC,
            torch.int8,
            [-1.0, 0, 7.0],
            torch.tensor(4.0 / 127.5),  # 8.0 / 255
            -96,
            -1.0,
            id="affine_signed",
        ),
        pytest.param(
            QuantizationScheme.ASYMMETRIC,
            torch.uint8,
            [-1.0, 0, 5.0],
            torch.tensor(3.0 / 127.5),  # 6.0 / 255
            42,
            -1.0,
            id="affine_unsigned",
        ),
        pytest.param(
            QuantizationScheme.SYMMETRIC,
            torch.int8,
            [-1.0, 0, 3.0],
            torch.tensor(3.0 / 127.5),  # 6.0 / 255
            0,
            -3.0,
            id="symmetric_signed",
        ),
        pytest.param(
            QuantizationScheme.SYMMETRIC,
            torch.uint8,
            [-3.0, 0, 1.0],
            torch.tensor(3.0 / 127.5),  # 6.0 / 255
            128,
            -3.0,
            id="symmetric_unsigned",
        ),
        pytest.param(
            QuantizationScheme.SYMMETRIC,
            torch.float8_e4m3fn,
            [-3.0, 0, 3.0],
            torch.tensor(3.0 / 448.0),
            None,
            None,
            id="fp8_e4m3fn_symmetric",
        ),
        pytest.param(
            QuantizationScheme.SYMMETRIC,
            torch.float8_e5m2,
            [-3.0, 0, 3.0],
            torch.tensor(3.0 / 57344.0),
            None,
            None,
            id="fp8_e5m2_symmetric",
        ),
        pytest.param(
            QuantizationScheme.SYMMETRIC_WITH_CLIPPING,
            torch.int8,
            [-3.0, 0, 3.0],
            torch.tensor(3.0 / 127),
            0,
            -3.0,
            id="symmetric_with_clipping_signed",
        ),
        pytest.param(
            QuantizationScheme.SYMMETRIC_WITH_CLIPPING,
            torch.int4,
            [-1.5, 0, 1.5],
            torch.tensor(1.5 / 7),
            0,
            -1.5,
            id="symmetric_with_clipping_int4",
        ),
        pytest.param(
            QuantizationScheme.SYMMETRIC_WITH_CLIPPING,
            torch.uint8,
            [-3.0, 0, 3.0],
            torch.tensor(3.0 / 127.5),
            128,
            -3.0,
            id="symmetric_with_clipping_unsigned",
        ),
        pytest.param(
            QuantizationScheme.SYMMETRIC,
            torch.float4_e2m1fn_x2,
            [-3.0, 0, 3.0],
            # e8m0 scale: 2^(floor(log2(3.0)) - target_max_pow2)
            # = 2^(1 - 2) = 0.5
            torch.tensor(0.5),
            None,
            None,
            id="fp4_e2m1_symmetric",
        ),
    ],
)
class TestStaticQParamsCalculator:
    @staticmethod
    def _validate_zp_minval(
        dtype,
        zero_point,
        minval,
        expected_zero_point,
        expected_minval,
        expected_scale_zp_minval_shape,
    ):
        # For FP4 and FP8, zero_point and minval should be None;
        # for INT8, they should be tensors
        if dtype.is_floating_point:
            assert zero_point is None, "zero_point should be None for FP4 and FP8"
            assert minval is None, "minval should be None for FP4 and FP8"
        else:
            assert zero_point is not None, "zero_point should not be None for INT8"
            assert zero_point.shape == torch.Size(expected_scale_zp_minval_shape)
            assert minval is not None, "minval should not be None for INT8"
            assert minval.shape == torch.Size(expected_scale_zp_minval_shape)

            # Determine expected dtype based on quantization dtype
            expected_zp_dtype = torch.int32
            expected_zp_tensor = torch.full(
                size=expected_scale_zp_minval_shape,
                fill_value=expected_zero_point,
                dtype=expected_zp_dtype,
            )
            assert torch.equal(zero_point, expected_zp_tensor)

            expected_minval_dtype = torch.float32
            expected_minval_tensor = torch.full(
                size=expected_scale_zp_minval_shape,
                fill_value=expected_minval,
                dtype=expected_minval_dtype,
            )
            assert torch.equal(minval, expected_minval_tensor)

    def test_per_tensor_scale_zero_point(
        self, qscheme, dtype, x, expected_scale, expected_zero_point, expected_minval
    ):
        x = torch.tensor(x, dtype=torch.float32)

        spec = QuantizationSpec(
            dtype=dtype,
            qscheme=qscheme,
            granularity=PerTensorGranularity(),
            fake_quantize_cls="default",
            qparam_calculator_cls="default",
            range_calculator_cls="minmax",
        )

        qparams_calc = QuantizationComponentFactory.create_qparams_calculator(
            spec, CompressionTargetTensor.WEIGHT
        )

        scale, zero_point, minval = qparams_calc(x)
        expected_scale_zp_minval_shape = [1] * x.ndim

        assert scale.shape == torch.Size(expected_scale_zp_minval_shape)
        torch.testing.assert_close(
            scale,
            torch.ones(size=expected_scale_zp_minval_shape) * expected_scale,
            atol=1e-10,
            rtol=0,
        )

        self._validate_zp_minval(
            dtype,
            zero_point,
            minval,
            expected_zero_point,
            expected_minval,
            expected_scale_zp_minval_shape,
        )

    @pytest.mark.parametrize("axis", [0, 1, 2, -1, -2, -3])
    def test_per_channel_scale_zero_point(
        self, qscheme, dtype, x, axis, expected_scale, expected_zero_point, expected_minval
    ):
        # tl;dr: to comprehensively (and conveniently) test all axes, we need to
        # construct a tensor such that picking any axis would (ideally) yeild
        # the same scale/zp so the test parametrization is easier.

        # so build a (3, 3, 3) tensor where every channel-slice along any axis
        # spans the full data range, so min/max are uniform across all axes.
        # assumes x is always [a, 0, b] where a = -b, so that rolling produces
        # rotations that keep {a, 0, b} in every slice.
        vals = torch.tensor(x, dtype=torch.float32)  # (3)
        row_a = vals  # (3) -> for x = [-3, 0, 3]
        row_b = vals.roll(1)  # (3) -> [3, -3, 0]
        row_c = vals.roll(2)  # (3) -> [0, 3, -3]
        layer = torch.stack([row_a, row_b, row_c])  # (3, 3)
        x = torch.stack([layer, layer, layer])  # (3, 3, 3)

        spec = QuantizationSpec(
            dtype=dtype,
            qscheme=qscheme,
            granularity=PerChannelGranularity(axis=axis),
            fake_quantize_cls="default",
            qparam_calculator_cls="default",
            range_calculator_cls="minmax",
        )
        qparams_calc = QuantizationComponentFactory.create_qparams_calculator(
            spec, CompressionTargetTensor.WEIGHT
        )
        scale, zero_point, minval = qparams_calc(x)

        # to handle negative axis
        normalized_axis = axis % x.ndim
        expected_scale_zp_minval_shape = [
            1 if i != normalized_axis else x.shape[i] for i in range(x.ndim)
        ]

        assert scale.shape == torch.Size(expected_scale_zp_minval_shape)
        expected_scale_tensor = torch.ones(size=expected_scale_zp_minval_shape) * expected_scale
        torch.testing.assert_close(scale, expected_scale_tensor, atol=1e-10, rtol=0)

        self._validate_zp_minval(
            dtype,
            zero_point,
            minval,
            expected_zero_point,
            expected_minval,
            expected_scale_zp_minval_shape,
        )

    @pytest.mark.parametrize("block_size", [3, 6])
    def test_per_block_scale_zero_point(
        self,
        qscheme,
        dtype,
        x,
        block_size,
        expected_scale,
        expected_zero_point,
        expected_minval,
    ):
        # Split values into two channels
        x = torch.tensor(x, dtype=torch.float32).repeat(1, 4, 6).permute(0, 2, 1)

        spec = QuantizationSpec(
            dtype=dtype,
            qscheme=qscheme,
            granularity=PerBlockGranularity(axis=1, block_size=block_size),
            fake_quantize_cls="default",
            qparam_calculator_cls="default",
            range_calculator_cls="minmax",
        )
        qparams_calc = QuantizationComponentFactory.create_qparams_calculator(
            spec, CompressionTargetTensor.WEIGHT
        )
        scale, zero_point, minval = qparams_calc(x)

        expected_scale_zp_minval_shape = [1, x.shape[1] // block_size, 1]

        assert scale.shape == torch.Size(expected_scale_zp_minval_shape)
        expected_scale_tensor = torch.ones(size=expected_scale_zp_minval_shape) * expected_scale
        torch.testing.assert_close(scale, expected_scale_tensor, atol=1e-10, rtol=0)

        self._validate_zp_minval(
            dtype,
            zero_point,
            minval,
            expected_zero_point,
            expected_minval,
            expected_scale_zp_minval_shape,
        )


def test_extra_repr():
    qparams_calc = StaticQParamsCalculator(
        dtype=torch.int8,
        qscheme=QuantizationScheme.ASYMMETRIC,
        granularity=PerTensorGranularity(),
        target_dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        range_calculator=MinMaxRangeCalculator(PerTensorGranularity()),
        float_range=[None, None],
    )

    repr_str = qparams_calc.extra_repr()
    assert "scale=tensor([])" in repr_str
    assert "zero_point=tensor([], dtype=torch.int32)" in repr_str
    assert "minval=tensor([])" in repr_str
    assert "export_mode=False" in repr_str

    qparams_calc.set_export_mode(True)
    repr_str = qparams_calc.extra_repr()
    assert "export_mode=True" in repr_str

    qparams_calc.set_export_mode(False)
    repr_str = qparams_calc.extra_repr()
    assert "export_mode=False" in repr_str


@pytest.mark.parametrize("averaging_constant", [0.9, 0.8, 0.75])
def test_moving_average_qparams_calculator(averaging_constant):
    qparams_calc = MovingAverageQParamsCalculator(
        dtype=torch.int8,
        qscheme=QuantizationScheme.SYMMETRIC,
        granularity=PerTensorGranularity(),
        target_dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        range_calculator=MinMaxRangeCalculator(PerTensorGranularity()),
        float_range=[None, None],
        averaging_constant=averaging_constant,
    )

    x = torch.tensor([-3.0, 0, 3.0], dtype=torch.float32)
    scale1, zp1, minval1 = qparams_calc(x)

    scale2, zp2, minval2 = qparams_calc(x * 2)

    assert torch.allclose((scale2 / scale1), torch.tensor(1 + averaging_constant))
    assert zp2 == zp1
    assert torch.allclose((minval2 / minval1), torch.tensor(1 + averaging_constant))


@pytest.mark.parametrize("float_range", [[None, None], [None, 10.0], [-10.0, None], [-10.0, 10.0]])
@pytest.mark.parametrize(
    "granularity",
    [
        PerTensorGranularity(),
        PerChannelGranularity(axis=0),
        PerBlockGranularity(axis=1, block_size=2),
    ],
)
def test_float_range(float_range, granularity):
    """
    Test float_range is honored for varying combinations of float ranges.
    """
    torch.manual_seed(0)
    qparams_calc = StaticQParamsCalculator(
        dtype=torch.int8,
        qscheme=QuantizationScheme.ASYMMETRIC,
        granularity=granularity,
        target_dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        range_calculator=MinMaxRangeCalculator(granularity),
        float_range=float_range,
    )

    x = torch.randn(4, 8) * 100.0
    scale, zp, minval = qparams_calc(x)

    if float_range[0] is not None:
        assert torch.all(minval == torch.min(torch.tensor(float_range[0]), torch.tensor(0)))

    qdq_min = scale * (-128 - zp)
    qdq_max = scale * (127 - zp)

    # Manually find min and max of the tensor taking granularity into account
    if isinstance(granularity, PerChannelGranularity):
        reduce_min = torch.clamp(torch.amin(x, dim=1, keepdim=True), max=0)
        reduce_max = torch.clamp(torch.amax(x, dim=1, keepdim=True), min=0)
    elif isinstance(granularity, PerBlockGranularity):
        reduce_min = torch.clamp(torch.amin(x.reshape(4, 4, 2), dim=2, keepdim=False), max=0)
        reduce_max = torch.clamp(torch.amax(x.reshape(4, 4, 2), dim=2, keepdim=False), min=0)
    else:
        reduce_min = torch.clamp(torch.min(x), max=0)
        reduce_max = torch.clamp(torch.max(x), min=0)

    # If float_range min and/or is not provided, qparams computed should line up with
    # manually computed min/maxes within a delta of scale. If float_range is provided,
    # qparams should instead line up with the provided min/max value within delta of
    # scale.
    if float_range[0] is None:
        assert torch.all(torch.abs(qdq_min - reduce_min) <= scale)
    else:
        assert torch.all(torch.abs(qdq_min - float_range[0]) <= scale)

    if float_range[1] is None:
        assert torch.all(torch.abs(qdq_max - reduce_max) <= scale)
    else:
        assert torch.all(torch.abs(qdq_max - float_range[1]) <= scale)


@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("float_range", [[None, 10.0], [-10.0, None]])
def test_float_range_one_sided_half_precision(input_dtype, float_range):
    """Test a one-sided float_range resolves on float16 and bfloat16 inputs."""
    qparams_calc = StaticQParamsCalculator(
        dtype=torch.int8,
        qscheme=QuantizationScheme.ASYMMETRIC,
        granularity=PerChannelGranularity(axis=0),
        target_dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        range_calculator=MinMaxRangeCalculator(PerChannelGranularity(axis=0)),
        float_range=float_range,
    )

    x = (torch.randn(4, 8) * 100.0).to(input_dtype)
    scale, _, _ = qparams_calc(x)

    # The resolved scale must carry the half-precision input dtype.
    assert scale.dtype == input_dtype
    assert torch.all(scale > 0)


@pytest.mark.parametrize("float_range", [[0.0, 5.0], [-5.0, 0.0]])
def test_float_range_with_symmetric(float_range):
    """
    When using symmetric quantization, it will not typically be possible to honor the
    float range unless it is already provided with symmetric min and max values, due to
    the way scale is computed for symmetric quantization.

    The expected behavior is that the provided float range will still be what is used as
    the starting point for computing scale, but ultimately the range should roughly
    encompass

    -max(-float_range_min, float_range_max), max(-float_range_min, float_range_max)
    """
    torch.manual_seed(0)
    qparams_calc = MovingAverageQParamsCalculator(
        dtype=torch.int8,
        qscheme=QuantizationScheme.SYMMETRIC,
        granularity=PerTensorGranularity(),
        target_dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        range_calculator=MinMaxRangeCalculator(PerTensorGranularity()),
        float_range=float_range,
    )

    x = torch.randn(1, 10) * 10.0
    scale, zp, minval = qparams_calc(x)
    qdq_min = scale * (-128 - zp)
    qdq_max = scale * (127 - zp)

    max_abs = max(abs(float_range[0]), abs(float_range[1]))
    assert torch.abs(qdq_min - (-max_abs)) <= scale
    assert torch.abs(qdq_max - max_abs) <= scale
    assert torch.all(minval == torch.tensor(-max_abs))


def test_float_range_with_moving_average_calculator():
    """
    Test that when float range min/max is fixed, moving average calculator provides the
    same scale/zp/minval across multiple tensors.
    """
    torch.manual_seed(0)
    float_range = [0.0, 5.0]
    qparams_calc = MovingAverageQParamsCalculator(
        dtype=torch.int8,
        qscheme=QuantizationScheme.ASYMMETRIC,
        granularity=PerTensorGranularity(),
        target_dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        range_calculator=MinMaxRangeCalculator(PerTensorGranularity()),
        float_range=float_range,
    )

    x = torch.randn(1, 10) * 10.0
    x2 = torch.randn(1, 10) * 10.0

    scale, zp, minval = qparams_calc(x)
    assert torch.all(minval == torch.tensor(0.0))
    qdq_min = scale * (-128 - zp)
    qdq_max = scale * (127 - zp)

    assert torch.abs(qdq_min - float_range[0]) <= scale
    assert torch.abs(qdq_max - float_range[1]) <= scale

    scale2, zp2, minval2 = qparams_calc(x2)
    assert torch.equal(scale, scale2)
    assert torch.equal(zp, zp2)
    assert torch.equal(minval, minval2)


@pytest.mark.parametrize(
    "calculator_cls,extra_kwargs",
    [
        pytest.param(
            StaticQParamsCalculator,
            {},
            id="default",
        ),
        pytest.param(
            MovingAverageQParamsCalculator,
            {"averaging_constant": 0.9},
            id="moving_average",
        ),
        pytest.param(
            GlobalMinMaxQParamsCalculator,
            {},
            id="global_minmax",
        ),
    ],
)
def test_gradient_preservation(calculator_cls, extra_kwargs):
    qparams_calc = calculator_cls(
        dtype=torch.int8,
        qscheme=QuantizationScheme.ASYMMETRIC,
        granularity=PerTensorGranularity(),
        target_dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        range_calculator=MinMaxRangeCalculator(PerTensorGranularity()),
        float_range=[None, None],
        **extra_kwargs,
    )

    x_init = torch.tensor([-3.0, 0, 3.0], dtype=torch.float32)
    qparams_calc(x_init)

    x = torch.tensor([-6.0, 0, 6.0], dtype=torch.float32, requires_grad=True)
    scale, _, _ = qparams_calc(x)

    assert scale.grad_fn is not None, "Scale should have grad_fn to preserve gradients"

    loss = torch.mean(x * scale)
    loss.backward()

    # Verify that gradients flow back to the input
    assert x.grad is not None, "Input tensor should have gradients"
    assert not torch.allclose(x.grad, torch.zeros_like(x.grad)), "Gradients should be non-zero"


@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "calculator_cls,extra_kwargs",
    [
        pytest.param(StaticQParamsCalculator, {}, id="static"),
        pytest.param(
            MovingAverageQParamsCalculator, {"averaging_constant": 0.9}, id="moving_average"
        ),
        pytest.param(GlobalMinMaxQParamsCalculator, {}, id="global_minmax"),
    ],
)
def test_compute_dtype_for_export(input_dtype, calculator_cls, extra_kwargs):

    qparams_calc = calculator_cls(
        dtype=torch.int8,
        qscheme=QuantizationScheme.ASYMMETRIC,
        granularity=PerTensorGranularity(),
        target_dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        range_calculator=MinMaxRangeCalculator(PerTensorGranularity()),
        float_range=[None, None],
        **extra_kwargs,
    )

    x = torch.tensor([-3.0, 0, 3.0], dtype=input_dtype)
    _, _, _ = qparams_calc(x)
    _compute_dtype_for_export = qparams_calc._compute_dtype_for_export

    # _compute_dtype_for_export should match input dtype
    assert _compute_dtype_for_export == input_dtype, (
        f"Expected _compute_dtype_for_export {input_dtype}, got {_compute_dtype_for_export}"
    )


def test_set_granularity():
    qparams_calc = MovingAverageQParamsCalculator(
        dtype=torch.int8,
        qscheme=QuantizationScheme.ASYMMETRIC,
        granularity=PerTensorGranularity(),
        target_dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        range_calculator=MinMaxRangeCalculator(PerTensorGranularity()),
        float_range=[None, None],
    )

    per_channel_ref = MovingAverageQParamsCalculator(
        dtype=torch.int8,
        qscheme=QuantizationScheme.ASYMMETRIC,
        granularity=PerChannelGranularity(axis=1),
        target_dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        range_calculator=MinMaxRangeCalculator(PerChannelGranularity(axis=1)),
        float_range=[None, None],
    )

    x = torch.randn(2, 5)
    per_channel_ref(x)

    qparams_calc.granularity = PerChannelGranularity(axis=1)
    assert qparams_calc.granularity == PerChannelGranularity(axis=1)
    assert qparams_calc.range_calculator.granularity == PerChannelGranularity(axis=1)
    qparams_calc(x)
    assert torch.equal(qparams_calc.scale, per_channel_ref.scale)
    assert torch.equal(qparams_calc.minval, per_channel_ref.minval)

    # Test switching the granularity after the first forward pass throws an error
    with pytest.raises(
        RuntimeError, match="Cannot change granularity after observer has been initialized."
    ):
        qparams_calc.granularity = PerTensorGranularity()


class TestGlobalMinMaxQParamsCalculator:
    """Tests specific to GlobalMinMaxQParamsCalculator accumulation behavior."""

    def _make_calculator(self, granularity=None, float_range=None):
        if granularity is None:
            granularity = PerTensorGranularity()
        if float_range is None:
            float_range = [None, None]
        return GlobalMinMaxQParamsCalculator(
            dtype=torch.int8,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=granularity,
            target_dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            range_calculator=MinMaxRangeCalculator(granularity),
            float_range=float_range,
        )

    def test_range_expands_on_wider_input(self):
        """Running min/max should expand when a wider-range tensor is observed."""
        calc = self._make_calculator()

        x1 = torch.tensor([-1.0, 0.0, 1.0])
        calc(x1)
        assert torch.equal(calc.running_min, torch.tensor([-1.0]))
        assert torch.equal(calc.running_max, torch.tensor([1.0]))

        x2 = torch.tensor([-3.0, 0.0, 3.0])
        calc(x2)
        assert torch.equal(calc.running_min, torch.tensor([-3.0]))
        assert torch.equal(calc.running_max, torch.tensor([3.0]))

    def test_range_does_not_contract(self):
        """Running min/max should never shrink when a narrower tensor is seen."""
        calc = self._make_calculator()

        x1 = torch.tensor([-5.0, 0.0, 5.0])
        calc(x1)

        x2 = torch.tensor([-1.0, 0.0, 1.0])
        calc(x2)
        assert torch.equal(calc.running_min, torch.tensor([-5.0]))
        assert torch.equal(calc.running_max, torch.tensor([5.0]))

    def test_range_expands_asymmetrically(self):
        """Min and max should track independently."""
        calc = self._make_calculator()

        x1 = torch.tensor([-1.0, 0.0, 2.0])
        calc(x1)

        # Only min expands
        x2 = torch.tensor([-5.0, 0.0, 1.0])
        calc(x2)
        assert torch.equal(calc.running_min, torch.tensor([-5.0]))
        assert torch.equal(calc.running_max, torch.tensor([2.0]))

        # Only max expands
        x3 = torch.tensor([-1.0, 0.0, 10.0])
        calc(x3)
        assert torch.equal(calc.running_min, torch.tensor([-5.0]))
        assert torch.equal(calc.running_max, torch.tensor([10.0]))

    def test_single_forward_matches_static(self):
        """With a single forward pass, GlobalMinMax should produce the same
        result as Static (both just see the current tensor)."""
        granularity = PerTensorGranularity()
        x = torch.tensor([-3.0, 0.0, 3.0])

        static_calc = StaticQParamsCalculator(
            dtype=torch.int8,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=granularity,
            target_dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            range_calculator=MinMaxRangeCalculator(granularity),
            float_range=[None, None],
        )
        minmax_calc = self._make_calculator(granularity)

        scale_s, zp_s, minval_s = static_calc(x)
        scale_m, zp_m, minval_m = minmax_calc(x)

        assert scale_s == scale_m
        assert zp_s == zp_m
        assert minval_s == minval_m

    @pytest.mark.parametrize(
        "granularity",
        [
            PerTensorGranularity(),
            PerChannelGranularity(axis=0),
            PerBlockGranularity(axis=1, block_size=2),
        ],
    )
    def test_accumulation_across_granularities(self, granularity):
        """Verify range expansion works for all granularity types."""
        calc = self._make_calculator(granularity)

        torch.manual_seed(0)
        x1 = torch.randn(4, 8)
        calc(x1)
        min_after_x1 = calc.running_min.clone()
        max_after_x1 = calc.running_max.clone()

        x2 = torch.randn(4, 8) * 3.0
        calc(x2)

        # Running range should be >= what it was after x1
        assert torch.all(calc.running_min <= min_after_x1)
        assert torch.all(calc.running_max >= max_after_x1)

    def test_export_mode_freezes_params(self):
        """In export mode, scale, zero_point and minval should not update."""
        calc = self._make_calculator()

        x1 = torch.tensor([-3.0, 0.0, 3.0])
        scale1, zp1, minval1 = calc(x1)

        calc.set_export_mode(True)
        x2 = torch.tensor([-7.0, 0.0, 10.0])
        scale2, zp2, minval2 = calc(x2)

        assert scale1 == scale2
        assert zp1 == zp2
        assert minval1 == minval2

    def test_float_range_full_freeze(self):
        """When both float_range bounds are set, scale/zp/minval should be constant."""
        calc = self._make_calculator(float_range=[-5.0, 5.0])

        x1 = torch.randn(10)
        scale1, zp1, minval1 = calc(x1)
        assert torch.all(minval1 == torch.tensor([-5.0]))

        x2 = torch.randn(10) * 100.0
        scale2, zp2, minval2 = calc(x2)

        assert scale1 == scale2
        assert zp1 == zp2
        assert minval1 == minval2

    def test_scale_monotonically_nondecreasing(self):
        """For symmetric quantization, scale should never decrease as range
        expands."""
        calc = self._make_calculator()

        torch.manual_seed(42)
        prev_scale = torch.tensor(0.0)
        for _ in range(5):
            x = torch.randn(20) * torch.rand(1).item() * 10
            scale, _, _ = calc(x)
            assert scale >= prev_scale
            prev_scale = scale.clone()


class TestDynamicQParamsCalculator:
    """Tests specific to DynamicQParamsCalculator runtime-recompute behavior."""

    def _make_calculator(self, granularity=None, float_range=None):
        if granularity is None:
            granularity = PerTensorGranularity()
        if float_range is None:
            float_range = [None, None]
        return DynamicQParamsCalculator(
            dtype=torch.int8,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=granularity,
            target_dtype=torch.int8,
            quant_min=-128,
            quant_max=127,
            range_calculator=MinMaxRangeCalculator(granularity),
            float_range=float_range,
        )

    def test_inherits_stateless_base(self):
        """DynamicQParamsCalculator must inherit from StatelessQParamsCalculatorBase"""
        assert issubclass(DynamicQParamsCalculator, StatelessQParamsCalculatorBase)

    def test_set_export_mode_true_raises(self):
        """Export mode is unsupported for stateless calculators — qparams are input-dependent."""
        calc = self._make_calculator()
        with pytest.raises(
            NotImplementedError,
            match="Stateless quantization .* does not support export mode",
        ):
            calc.set_export_mode(enabled=True)

    @pytest.mark.parametrize(
        "float_range",
        [[-1.0, 1.0], [-1.0, None], [None, 1.0]],
    )
    def test_rejects_non_none_float_range(self, float_range):
        """Dynamic requires float_range=[None, None]."""
        with pytest.raises(ValueError, match=r"requires float_range=\[None, None\]"):
            self._make_calculator(float_range=float_range)

    def test_get_qparams_not_defined(self):
        """get_qparams is structurally absent on stateless calculators — there
        is no cached value to return. Attribute access raises AttributeError."""
        calc = self._make_calculator()
        calc(torch.randn(4, 8))  # populate debug attributes
        with pytest.raises(AttributeError):
            calc.get_qparams()

    def test_qparams_recompute_per_forward(self):
        """Different inputs must yield recomputed qparams matching each input's range
        directly — locks both per-forward recompute (no caching) AND correct math on
        each forward."""
        calc = self._make_calculator()

        x1 = torch.tensor([-1.0, 0.0, 1.0])
        scale1, zp1, minval1 = calc(x1)
        torch.testing.assert_close(scale1, torch.tensor([1.0 / 127.5]))
        assert zp1 == 0
        torch.testing.assert_close(minval1, torch.tensor([-1.0]))

        x2 = torch.tensor([-10.0, 0.0, 10.0])
        scale2, zp2, minval2 = calc(x2)
        torch.testing.assert_close(scale2, torch.tensor([10.0 / 127.5]))
        assert zp2 == 0
        torch.testing.assert_close(minval2, torch.tensor([-10.0]))

    @pytest.mark.parametrize(
        "qscheme,dtype,x",
        [
            pytest.param(QuantizationScheme.SYMMETRIC, torch.int8, [-1.0, 0.0, 3.0], id="int8_sym"),
            pytest.param(
                QuantizationScheme.ASYMMETRIC, torch.int8, [-1.0, 0.0, 7.0], id="int8_asym"
            ),
            pytest.param(
                QuantizationScheme.SYMMETRIC_WITH_CLIPPING,
                torch.int8,
                [-3.0, 0.0, 3.0],
                id="int8_sym_clip",
            ),
            pytest.param(
                QuantizationScheme.SYMMETRIC, torch.uint8, [-3.0, 0.0, 1.0], id="uint8_sym"
            ),
            pytest.param(
                QuantizationScheme.ASYMMETRIC, torch.uint8, [-1.0, 0.0, 5.0], id="uint8_asym"
            ),
            pytest.param(
                QuantizationScheme.SYMMETRIC_WITH_CLIPPING,
                torch.int4,
                [-1.5, 0.0, 1.5],
                id="int4_sym_clip",
            ),
            pytest.param(
                QuantizationScheme.SYMMETRIC,
                torch.float8_e4m3fn,
                [-3.0, 0.0, 3.0],
                id="fp8_e4m3",
            ),
            pytest.param(
                QuantizationScheme.SYMMETRIC, torch.float8_e5m2, [-3.0, 0.0, 3.0], id="fp8_e5m2"
            ),
            pytest.param(
                QuantizationScheme.SYMMETRIC,
                torch.float4_e2m1fn_x2,
                [-3.0, 0.0, 3.0],
                id="fp4",
            ),
        ],
    )
    def test_matches_static_for_single_input(self, qscheme, dtype, x):
        """For a single forward, dynamic must match static across the
        dtype/qscheme matrix — equivalence with static's known-good math is
        the strongest correctness check for dynamic."""
        granularity = PerTensorGranularity()
        spec = QuantizationSpec(dtype=dtype, qscheme=qscheme, granularity=granularity)
        common_kwargs = dict(
            dtype=dtype,
            qscheme=qscheme,
            granularity=granularity,
            target_dtype=spec.target_dtype,
            quant_min=spec.quant_min,
            quant_max=spec.quant_max,
            range_calculator=MinMaxRangeCalculator(granularity),
            float_range=[None, None],
            scale_dtype=spec.scale_dtype,
        )
        static_calc = StaticQParamsCalculator(**common_kwargs)
        dynamic_calc = DynamicQParamsCalculator(**common_kwargs)
        x_tensor = torch.tensor(x, dtype=torch.float32)

        scale_s, zp_s, minval_s = static_calc(x_tensor)
        scale_d, zp_d, minval_d = dynamic_calc(x_tensor)

        assert torch.equal(scale_s, scale_d)
        # FP4/FP8 return None for zero_point and minval.
        if zp_s is None:
            assert zp_d is None
        else:
            assert torch.equal(zp_s, zp_d)
        if minval_s is None:
            assert minval_d is None
        else:
            assert torch.equal(minval_s, minval_d)

    def test_variable_shape_scale_supported(self):
        """LLM use case: per-token (per-channel along seq axis) dynamic
        quantization where seq_len varies across forwards. Scale shape must
        change correspondingly
        """
        granularity = PerChannelGranularity(axis=1)  # per-token on (B, seq, H)
        calc = self._make_calculator(granularity=granularity)

        x_short = torch.randn(2, 8, 16)  # seq_len = 8
        scale_short, _, _ = calc(x_short)
        assert scale_short.shape == (1, 8, 1)

        x_long = torch.randn(2, 32, 16)  # seq_len = 32
        scale_long, _, _ = calc(x_long)
        assert scale_long.shape == (1, 32, 1)


@pytest.mark.parametrize(
    "precision_dtype",
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ],
)
@pytest.mark.parametrize("dtype", ["int4", "int8"])
@pytest.mark.parametrize(
    "granularity",
    [
        PerTensorGranularity(),
        PerChannelGranularity(axis=0),
        PerBlockGranularity(axis=1, block_size=16),
    ],
)
@pytest.mark.parametrize(
    "compression_target", [CompressionTargetTensor.WEIGHT, CompressionTargetTensor.ACTIVATION]
)
def test_scale_clamping(precision_dtype, dtype, granularity, compression_target):
    """Verify quantization scales are not clamped to the input dtype's eps.

    torchao's choose_qparams_affine_with_min_max defaults to using
    torch.finfo(input_dtype).eps as the scale floor. For bf16 (eps=0.0078125)
    and fp16 (eps=9.77e-04), this destroys scales for small-magnitude weights.
    We pass float32 eps explicitly to avoid this.
    """
    spec = QuantizationSpec(
        dtype=dtype,
        qscheme=QuantizationScheme.SYMMETRIC,
        granularity=granularity,
    )
    qparams_calc = QuantizationComponentFactory.create_qparams_calculator(spec, compression_target)

    precision_eps = torch.finfo(precision_dtype).eps
    x = torch.randn(32, 32, dtype=precision_dtype) * precision_eps
    scale, _, _ = qparams_calc(x)
    assert torch.all(scale >= torch.finfo(torch.float32).eps), (
        f"Scale should be greater than float32 eps floor: min scale={scale.min().item()}"
    )
    if precision_dtype != torch.float32:
        assert torch.all(scale < precision_eps), (
            f"Scale should be less than {precision_dtype} eps floor {precision_eps}, "
            f"max scale={scale.max().item()}"
        )
