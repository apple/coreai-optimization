# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import copy

import pytest
import torch
from torchao.quantization.pt2e import (
    disable_observer,
    enable_observer,
)
from torchao.quantization.quant_primitives import (
    _dequantize_affine_float8,
    _quantize_affine_float8,
)

from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.quantization import QuantizationSpec
from coreai_opt.quantization.spec import (
    PerBlockGranularity,
    PerChannelGranularity,
    PerTensorGranularity,
    QuantizationFormulation,
    QuantizationGranularity,
    QuantizationScheme,
)
from coreai_opt.quantization.spec.factory import QuantizationComponentFactory
from coreai_opt.quantization.spec.fake_quantize import (
    _DefaultFakeQuantizeImpl,
    _FusedFakeQuantizeIntSTE,
)
from coreai_opt.quantization.spec.qparams_calculator import StaticQParamsCalculator
from coreai_opt.quantization.spec.range_calculator import MinMaxRangeCalculator


@pytest.mark.parametrize(
    "granularity",
    [PerChannelGranularity(axis=0), PerBlockGranularity(axis=None, block_size=(1, 1))],
)
@pytest.mark.parametrize(
    "qscheme",
    [
        QuantizationScheme.ASYMMETRIC,
        QuantizationScheme.SYMMETRIC,
        QuantizationScheme.SYMMETRIC_WITH_CLIPPING,
    ],
)
@pytest.mark.parametrize(
    "qformulation",
    [
        QuantizationFormulation.ZP,
        QuantizationFormulation.MINVAL,
    ],
)
def test_fake_quant_dequant_no_reduction(qscheme, granularity, qformulation):
    range_calculator = MinMaxRangeCalculator(granularity)
    kwargs = {
        "dtype": torch.int8,
        "qscheme": qscheme,
        "qformulation": qformulation,
        "granularity": granularity,
        "target_dtype": torch.int8,
        "quant_min": -128,
        "quant_max": 127,
        "float_range": [None, None],
    }
    qparam_calculator = StaticQParamsCalculator(range_calculator=range_calculator, **kwargs)
    fq = _DefaultFakeQuantizeImpl(
        qparams_calculator=qparam_calculator,
        **kwargs,
    )

    x = torch.randn(2, 1)
    fq_x = fq(x)
    if qscheme == QuantizationScheme.ASYMMETRIC:
        # With asymmetric, there should be very little change in the quantized tensor since
        # no min/max reduction was needed. We use assert_close instead of torch.equal
        # because the quantize→dequantize round-trip (x/scale → round → *scale) can
        # introduce tiny FP32 rounding errors.
        torch.testing.assert_close(fq_x, x)
    else:
        # With symmetric modes, even though no min/max reduction is done, the computed
        # scale with symmetric constraints still leads to a final min/max which is close
        # to but not equal to the tensor.
        # Instead, just check that the quantized tensor is close to the original within
        # a difference of scale.
        assert torch.all(torch.abs(fq_x - x) < torch.abs(qparam_calculator.scale))


def test_set_granularity():
    range_calculator = MinMaxRangeCalculator(PerTensorGranularity())
    kwargs = {
        "dtype": torch.int8,
        "qscheme": QuantizationScheme.ASYMMETRIC,
        "granularity": PerTensorGranularity(),
        "target_dtype": torch.int8,
        "quant_min": -128,
        "quant_max": 127,
        "qformulation": QuantizationFormulation.ZP,
        "float_range": [None, None],
    }
    qparam_calculator = StaticQParamsCalculator(range_calculator=range_calculator, **kwargs)
    fq = _DefaultFakeQuantizeImpl(
        qparams_calculator=qparam_calculator,
        **kwargs,
    )
    x = torch.randn(2, 5)

    fq.granularity = PerChannelGranularity(axis=1)
    assert fq.granularity == PerChannelGranularity(axis=1)
    assert fq.qparams_calculator.granularity == PerChannelGranularity(axis=1)

    fq(x)
    assert fq.qparams_calculator.scale.shape == (1, 5)

    # Test switching the granularity after the first forward pass throws an error
    with pytest.raises(
        RuntimeError, match="Cannot change granularity after observer has been initialized."
    ):
        fq.granularity = PerTensorGranularity()


@pytest.mark.parametrize("qscheme", ["asymmetric", "symmetric", "symmetric_with_clipping"])
@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "uint8",
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float4_e2m1fn_x2,
    ],
)
@pytest.mark.parametrize(
    "qformulation",
    [
        QuantizationFormulation.ZP,
        QuantizationFormulation.MINVAL,
    ],
)
@pytest.mark.parametrize(
    "granularity",
    [
        {"type": "per_tensor"},
        {"type": "per_channel", "axis": 1},
        {"type": "per_block", "axis": 1, "block_size": 5},
        {"type": "per_block", "axis": 1, "block_size": 10},
    ],
)
@pytest.mark.parametrize(
    "module",
    [
        torch.nn.Linear(100, 100),
        torch.nn.Conv2d(100, 100, 10),
    ],
)
class TestDefaultFakeQuantize:
    @pytest.fixture(autouse=True)
    def skip_invalid_fp_qscheme(self, dtype, qscheme, qformulation):
        fp_dtypes = {torch.float8_e4m3fn, torch.float8_e5m2, torch.float4_e2m1fn_x2}
        if dtype in fp_dtypes and (
            qscheme != "symmetric" or qformulation != QuantizationFormulation.ZP
        ):
            pytest.skip(f"{dtype} only supports symmetric qscheme")

    @pytest.fixture
    def fq(self, dtype, qscheme, qformulation, granularity):
        # Account for bug in pytest which fails if you run batched tests
        # with a dictionary value that is parametrized
        granularity = QuantizationGranularity.maybe_build_from_dict(copy.deepcopy(granularity))
        spec = QuantizationSpec(
            dtype=dtype, qscheme=qscheme, granularity=granularity, qformulation=qformulation
        )
        kwargs = {
            "dtype": spec.dtype,
            "qscheme": spec.qscheme,
            "granularity": granularity,
            "qformulation": qformulation,
            "target_dtype": spec.target_dtype,
            "quant_min": spec.quant_min,
            "quant_max": spec.quant_max,
            "float_range": [None, None],
        }
        range_calculator = MinMaxRangeCalculator(granularity)
        qparam_calculator = StaticQParamsCalculator(range_calculator=range_calculator, **kwargs)
        return _DefaultFakeQuantizeImpl(
            qparams_calculator=qparam_calculator,
            **kwargs,
        )

    def test_fake_quant_dequant(self, fq, module):
        x = module.weight
        fq_x = fq(x)

        assert not torch.equal(x, fq_x), "Quantization should introduce error"
        torch.testing.assert_close(x, fq_x, atol=2e-2, rtol=1e-5)

    def test_export_mode(self, fq, module):
        assert not fq.qparams_calculator._export_mode
        fq.set_export_mode(True)
        assert fq.qparams_calculator._export_mode


@pytest.mark.parametrize(
    "dtype,granularity,qformulation",
    [
        (torch.int8, PerTensorGranularity(), QuantizationFormulation.ZP),
        (torch.uint8, PerChannelGranularity(axis=1), QuantizationFormulation.MINVAL),
        (torch.float8_e4m3fn, PerTensorGranularity(), QuantizationFormulation.ZP),
        (torch.float8_e5m2, PerChannelGranularity(axis=0), QuantizationFormulation.ZP),
        (torch.float4_e2m1fn_x2, PerTensorGranularity(), QuantizationFormulation.ZP),
        (
            torch.float4_e2m1fn_x2,
            PerBlockGranularity(axis=1, block_size=4),
            QuantizationFormulation.ZP,
        ),
    ],
)
@pytest.mark.parametrize(
    "compression_target_tensor",
    [CompressionTargetTensor.WEIGHT, CompressionTargetTensor.ACTIVATION],
)
def test_qat_gradient_flow(dtype, granularity, qformulation, compression_target_tensor):
    """Test that gradients flow through custom INT8/FP8/FP4 STE implementations."""
    spec = QuantizationSpec(
        dtype=dtype,
        qscheme=QuantizationScheme.SYMMETRIC,
        granularity=granularity,
        qformulation=qformulation,
        fake_quantize_cls="default",
        qparam_calculator_cls="default",
        range_calculator_cls="minmax",
    )

    fq = QuantizationComponentFactory.create_fake_quantizer(spec, compression_target_tensor)
    tensor = torch.randn(8, 16, requires_grad=True)

    assert fq.observer_enabled.item() == 1
    assert fq.fake_quant_enabled.item() == 1
    fq(tensor)  # Forward pass to initialize scales

    # Test grad for disabled observer
    tensor.grad = None
    fq.apply(disable_observer)
    assert fq.observer_enabled.item() == 0

    output = fq(tensor)
    loss = output.sum()
    loss.backward()

    disabled_observer_grad = tensor.grad
    assert disabled_observer_grad is not None, "Gradient should flow through custom STE"
    assert disabled_observer_grad.shape == tensor.shape, "Gradient shape should match input"
    qparams_scale, _, _ = fq.qparams_calculator.get_qparams()
    assert qparams_scale.grad is None

    # Test grad for enabled observer
    tensor.grad = None
    fq.apply(enable_observer)
    assert fq.observer_enabled.item() == 1

    output = fq(tensor)
    loss = output.sum()
    loss.backward()

    enabled_observer_grad = tensor.grad

    assert enabled_observer_grad is not None, "Gradient should flow through custom STE"
    assert enabled_observer_grad.shape == tensor.shape, "Gradient shape should match input"
    qparams_scale, _, _ = fq.qparams_calculator.get_qparams()
    assert qparams_scale.grad is None

    # For weights (static calculator), grads with observer enabled vs disabled would
    # not differ. For activations (moving average), the scale updates between passes
    # so gradients can change, making this check flaky.
    if compression_target_tensor == CompressionTargetTensor.WEIGHT:
        assert torch.equal(disabled_observer_grad, enabled_observer_grad)


@pytest.mark.parametrize(
    "dtype, expected_quantized_values",
    [
        (
            torch.float8_e4m3fn,
            torch.tensor(
                [
                    [0.875, 1.75, 0, 3.5, 4.5],
                    [9, 18, -26, 36, 44],
                    [88, 176, 256, 352, 448],
                    [-0.0859375, 0.171875, 0.28125, -0.34375, 0.4375],
                ],
                dtype=torch.float8_e4m3fn,
            ),
        ),
        (
            torch.float8_e5m2,
            torch.tensor(
                [
                    [112, 224, 0, 448, 512],
                    [1024, 2048, -3584, 4096, 6144],
                    [12288, 24576, 32768, 49152, 57344],
                    [-0.109375, 0.21875, 0.375, -0.4375, 0.625],
                ],
                dtype=torch.float8_e5m2,
            ),
        ),
        (
            torch.float4_e2m1fn_x2,
            torch.tensor(
                [
                    [0, 0, 0, 0, 0],
                    [0.0, 0.5, -0.5, 0.5, 1.0],
                    [1.5, 3, 4, 6, 6],
                    [0, 0, 0, 0, 0],
                ],
                dtype=torch.float8_e4m3fn,
            ),
        ),
    ],
)
def test_float_quantized_values(dtype, expected_quantized_values):
    input_tensor = torch.tensor(
        [
            [1.0, 2.0, 0, 4.0, 5.0],
            [10.0, 20.0, -30.0, 40.0, 50.0],
            [100.0, 200.0, 300.0, 400.0, 500.0],
            [-0.1, 0.2, 0.3, -0.4, 0.5],
        ]
    )

    if dtype == torch.float8_e5m2:
        input_tensor[-1, :] = torch.tensor([-0.001, 0.002, 0.003, -0.004, 0.005])

    spec = QuantizationSpec(
        dtype=dtype,
        qscheme=QuantizationScheme.SYMMETRIC,
        granularity=PerTensorGranularity(),
        fake_quantize_cls="default",
        qparam_calculator_cls="default",
        range_calculator_cls="minmax",
    )

    fq = QuantizationComponentFactory.create_fake_quantizer(
        spec=spec,
        quantization_target=CompressionTargetTensor.WEIGHT,
    )

    scale, zero_point, minval = fq.qparams_calculator(input_tensor)
    quantized_tensor = fq.quantize(input_tensor, scale, zero_point, minval)

    assert torch.all(quantized_tensor == expected_quantized_values)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("granularity", [PerTensorGranularity(), PerChannelGranularity(axis=0)])
def test_float8_matches_torchao_primitives(dtype, granularity):
    """Verify custom FP8 quant/dequant matches torchao primitives exactly."""
    tensor = torch.randn(8, 16)
    spec = QuantizationSpec(
        dtype=dtype,
        qscheme=QuantizationScheme.SYMMETRIC,
        granularity=granularity,
        fake_quantize_cls="default",
        qparam_calculator_cls="default",
        range_calculator_cls="minmax",
    )

    fq = QuantizationComponentFactory.create_fake_quantizer(spec, CompressionTargetTensor.WEIGHT)
    scale, zero_point, minval = fq.qparams_calculator(tensor)

    # Test quantize
    custom_quant = fq.quantize(tensor, scale, zero_point, minval)
    torchao_quant = _quantize_affine_float8(tensor, scale, dtype)
    torch.testing.assert_close(custom_quant, torchao_quant)

    # Test dequantize
    custom_dequant = fq.dequantize(custom_quant, scale, zero_point, minval, torch.float32)
    torchao_dequant = _dequantize_affine_float8(custom_quant, scale, torch.float32)
    torch.testing.assert_close(custom_dequant, torchao_dequant)


class TestFusedFakeQuantizeIntSTEMatchesTorchBuiltin:
    """Verify FusedFakeQuantizeIntSTE produces identical results to
    torch.fake_quantize_per_{tensor,channel}_affine for forward and backward."""

    @pytest.mark.parametrize(
        "qscheme,dtype",
        [
            (QuantizationScheme.SYMMETRIC, torch.int8),
            (QuantizationScheme.ASYMMETRIC, torch.uint8),
        ],
    )
    @pytest.mark.parametrize(
        "shape",
        [(8, 16), (1, 1), (64, 128)],
    )
    def test_per_tensor_parity(self, shape, qscheme, dtype):
        """Verify parity of per tensor qdq"""
        quant_min, quant_max = QuantizationSpec.get_quant_range(dtype, qscheme)
        tensor_coreai_opt = torch.randn(*shape, requires_grad=True)
        tensor_torch = tensor_coreai_opt.detach().clone().requires_grad_(True)
        scale = torch.tensor([0.05])
        zero_point = (
            torch.zeros(1) if qscheme == QuantizationScheme.SYMMETRIC else torch.tensor([3.0])
        )

        original_shape = tensor_coreai_opt.shape
        blockwise_shape = list(tensor_coreai_opt.shape)
        reduced_shape = [1] * len(tensor_coreai_opt.shape)

        # ZP-form parity with torch.fake_quantize_per_tensor_affine:
        #   quant_offset = zero_point, float_offset = 0
        quant_offset = zero_point
        float_offset = torch.tensor(0.0)

        out_coreai_opt = _FusedFakeQuantizeIntSTE.apply(
            tensor_coreai_opt,
            scale,
            quant_offset,
            float_offset,
            quant_min,
            quant_max,
            original_shape,
            blockwise_shape,
            reduced_shape,
        )
        out_torch = torch.fake_quantize_per_tensor_affine(
            tensor_torch,
            scale,
            zero_point,
            quant_min,
            quant_max,
        )
        torch.testing.assert_close(out_coreai_opt, out_torch)

        grad = torch.randn_like(out_coreai_opt)
        out_coreai_opt.backward(grad)
        out_torch.backward(grad)
        torch.testing.assert_close(tensor_coreai_opt.grad, tensor_torch.grad)

    @pytest.mark.parametrize(
        "qscheme,dtype",
        [
            (QuantizationScheme.SYMMETRIC, torch.int8),
            (QuantizationScheme.ASYMMETRIC, torch.uint8),
        ],
    )
    @pytest.mark.parametrize(
        "shape,axis",
        [((8, 16), 0), ((8, 16), 1), ((4, 8, 3, 3), 0)],
    )
    def test_per_channel_parity(self, shape, axis, qscheme, dtype):
        """Verify parity of per channel qdq"""
        quant_min, quant_max = QuantizationSpec.get_quant_range(dtype, qscheme)
        tensor_coreai_opt = torch.randn(*shape, requires_grad=True)
        tensor_torch = tensor_coreai_opt.detach().clone().requires_grad_(True)
        n_channels = shape[axis]
        scale_1d = torch.rand(n_channels).abs() + 1e-6
        zero_point_1d = (
            torch.zeros(n_channels)
            if qscheme == QuantizationScheme.SYMMETRIC
            else torch.randint(0, 5, (n_channels,)).float()
        )

        # Build shapes for our implementation: scale/zp broadcast along axis
        reduced_shape = [1] * len(shape)
        reduced_shape[axis] = n_channels
        scale_nd = scale_1d.view(reduced_shape)
        zp_nd = zero_point_1d.view(reduced_shape)

        original_shape = torch.Size(shape)
        blockwise_shape = list(shape)

        # ZP-form parity with torch.fake_quantize_per_channel_affine:
        #   quant_offset = zero_point, float_offset = 0
        quant_offset = zp_nd
        float_offset = torch.tensor(0.0)

        out_coreai_opt = _FusedFakeQuantizeIntSTE.apply(
            tensor_coreai_opt,
            scale_nd,
            quant_offset,
            float_offset,
            quant_min,
            quant_max,
            original_shape,
            blockwise_shape,
            reduced_shape,
        )
        out_torch = torch.fake_quantize_per_channel_affine(
            tensor_torch,
            scale_1d,
            zero_point_1d,
            axis,
            quant_min,
            quant_max,
        )
        torch.testing.assert_close(out_coreai_opt, out_torch)

        grad = torch.randn_like(out_coreai_opt)
        out_coreai_opt.backward(grad)
        out_torch.backward(grad)
        torch.testing.assert_close(tensor_coreai_opt.grad, tensor_torch.grad)


@pytest.mark.parametrize("dtype", [torch.int8, torch.uint8, torch.int4, torch.uint4])
@pytest.mark.parametrize(
    "granularity",
    [
        PerTensorGranularity(),
        PerChannelGranularity(axis=0),
        PerBlockGranularity(axis=1, block_size=20),
    ],
)
@pytest.mark.parametrize(
    "qscheme",
    [
        QuantizationScheme.SYMMETRIC,
        QuantizationScheme.ASYMMETRIC,
    ],
)
def test_zp_minval_round_trip_delta_upper_bound(dtype, granularity, qscheme):
    """ZP and MINVAL formulations share the same scale.
    Per-element round-trip error is at most ``scale/2``:
        |dequant_zp     - f| <= scale/2
        |dequant_minval - f| <= scale/2
        |dequant_zp - dequant_minval| <= scale + eps.
    """

    x = torch.randn(100, 100) * 10.0

    def round_trip(qformulation):
        spec = QuantizationSpec(
            dtype=dtype,
            qscheme=qscheme,
            granularity=granularity,
            qformulation=qformulation,
        )
        fq = QuantizationComponentFactory.create_fake_quantizer(
            spec, CompressionTargetTensor.WEIGHT
        )
        return fq(x), fq.qparams_calculator.scale

    y_zp, scale_zp = round_trip(QuantizationFormulation.ZP)
    y_minval, scale_minval = round_trip(QuantizationFormulation.MINVAL)

    if isinstance(granularity, PerBlockGranularity):
        scale_zp = scale_zp.repeat_interleave(granularity.block_size, dim=granularity.axis)
        scale_minval = scale_minval.repeat_interleave(granularity.block_size, dim=granularity.axis)

    assert torch.equal(scale_zp, scale_minval)
    assert torch.all(torch.abs(y_minval - y_zp) <= scale_zp + 1e-5)
