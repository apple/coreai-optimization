# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for quantized qparams (``QuantizationSpec.qparam_qspec``), e.g. NVFP4.

The references below re-implement each format from its definition: quantize the per-block
qparams first, then choose the element codes against the decoded qparams.
"""

import logging
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchao.quantization.pt2e import disable_observer, enable_observer

from coreai_opt import ExportBackend
from coreai_opt._utils.torch_utils import nonzero_scale
from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.quantization import (
    ModuleQuantizerConfig,
    QuantizationSpec,
    Quantizer,
    QuantizerConfig,
)
from coreai_opt.quantization.config import ExecutionMode, QATSchedule
from coreai_opt.quantization.spec import (
    PerBlockGranularity,
    PerChannelGranularity,
    PerTensorGranularity,
    QuantizationFormulation,
    QuantizationScheme,
    fp4_forward,
)
from coreai_opt.quantization.spec.factory import QuantizationComponentFactory
from coreai_opt.quantization.spec.fake_quantize import FakeQuantizeImplBase
from coreai_opt.quantization.spec.qparams_calculator import MovingAverageQParamsCalculator
from tests.test_utils.general import (
    minval_spec,
    nvfp4_spec,
    twice_quantized_spec,
    weight_fake_quantizer,
)


def _nvfp4_reference(weight: torch.Tensor) -> torch.Tensor:
    """Fake-quantize a 2-D weight as NVFP4, with blocks of 16 along the last dim."""
    blocks = weight.float().reshape(weight.shape[0], -1, 16)
    ideal_scale = blocks.abs().amax(dim=-1, keepdim=True) / 6.0
    tensor_scale = ideal_scale.amax() / 448.0
    block_scale = (ideal_scale / tensor_scale).clamp(max=448.0).to(torch.float8_e4m3fn).float()
    scale = block_scale * tensor_scale
    elements = fp4_forward((blocks / nonzero_scale(scale)).clamp(-6.0, 6.0))
    return (elements * scale).reshape(weight.shape)


def _decode_uint8_qparams(values: torch.Tensor) -> torch.Tensor:
    """Round trip non-negative qparams, grouped 8 along the last dim, through uint8 codes."""
    parent = values.amax(dim=-1, keepdim=True) / 255.0
    return (values / parent).round().clamp(0, 255) * parent


def _minval_reference(weight: torch.Tensor) -> torch.Tensor:
    """Fake-quantize a 2-D weight as ``minval_spec``, with groups of 8x32 along the last dim."""
    blocks = weight.float().reshape(weight.shape[0], -1, 8, 32)
    low = blocks.amin(dim=-1).clamp(max=0)
    high = blocks.amax(dim=-1).clamp(min=0)
    scale = _decode_uint8_qparams((high - low) / 15.0).unsqueeze(-1)
    minval = -_decode_uint8_qparams(-low).unsqueeze(-1)
    codes = ((blocks - minval) / nonzero_scale(scale)).round().clamp(0, 15)
    return (codes * scale + minval).reshape(weight.shape)


_SPEC_FACTORIES = [nvfp4_spec, minval_spec]
_REFERENCE_CASES = [(nvfp4_spec, _nvfp4_reference), (minval_spec, _minval_reference)]


def _nvfp4_activation_spec(tensor_scale_calculator: str = "default") -> QuantizationSpec:
    """NVFP4 activations: block scales recomputed every forward, under a tensor scale."""
    return QuantizationSpec(
        dtype=torch.float4_e2m1fn_x2,
        granularity=PerBlockGranularity(axis=-1, block_size=16),
        qparam_calculator_cls="dynamic",
        qparam_qspec=QuantizationSpec(
            dtype=torch.float8_e4m3fn,
            granularity=PerTensorGranularity(),
            qparam_calculator_cls=tensor_scale_calculator,
        ),
    )


def _mlp() -> nn.Module:
    return nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 16)).eval()


def _config(
    weight_spec: QuantizationSpec, execution_mode: ExecutionMode, **module_config_kwargs: Any
) -> QuantizerConfig:
    """Return a config that quantizes weights with ``weight_spec``, and activations if given."""
    module_config_kwargs = {"op_input_spec": None, "op_output_spec": None} | module_config_kwargs
    return QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={"weight": weight_spec}, **module_config_kwargs
        ),
        execution_mode=execution_mode,
    )


def _worked_example_spec(qformulation: QuantizationFormulation) -> QuantizationSpec:
    """uint4 blocks of 4 whose qparams are uint4 codes under one parent per 2 blocks.

    A 1 x 8 weight is then 2 blocks forming 1 group, small enough to work out by hand, and the
    coarse nested codes make the effect of quantizing the qparams visible.
    """
    return QuantizationSpec(
        dtype=torch.uint4,
        qscheme=QuantizationScheme.ASYMMETRIC,
        qformulation=qformulation,
        granularity=PerBlockGranularity(axis=1, block_size=4),
        qparam_qspec=QuantizationSpec(
            dtype=torch.uint4,
            qscheme=QuantizationScheme.ASYMMETRIC,
            granularity=PerBlockGranularity(block_size=2),
        ),
    )


def _assert_values(actual: torch.Tensor, expected: list[float]) -> None:
    # Stricter than the fp32 defaults; still allows fp32 rounding of the decimal literals.
    torch.testing.assert_close(actual.flatten(), torch.tensor(expected), rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------
# NVFP4 numerics
# ---------------------------------------------------------------------------


def test_nvfp4_chooses_elements_against_the_quantized_block_scale():
    """Elements are chosen against the E4M3-quantized block scale.

    With a tensor scale of 1, a block maximum of 7 rounds the block scale 7/6 to the E4M3 value
    1.125, so 1.43 / 1.125 = 1.27 rounds up to the E2M1 value 1.5. Against the unrounded scale,
    1.43 / 1.167 = 1.23 would round down to 1.0 instead.
    """
    weight = torch.zeros(2, 16)
    weight[0, 0] = 2688.0  # 6 * 448, so the tensor scale is 1
    weight[1, 0] = 7.0
    weight[1, 1] = 1.43
    fake_quantize = weight_fake_quantizer(nvfp4_spec())

    out = fake_quantize(weight)

    calculator = fake_quantize.qparams_calculator
    assert calculator.scale_fake_quantize.qparams_calculator.scale.flatten().tolist() == [1.0]
    assert calculator.scale.flatten().tolist() == [448.0, 1.125]
    assert out[0, 0].item() == 2688.0
    # 7 / 1.125 = 6.2 saturates at 6.0.
    assert out[1, :3].tolist() == [6.75, 1.6875, 0.0]


def test_nvfp4_block_scale_underflow_decodes_to_zero():
    """A block far below the tensor maximum decodes to zeros rather than NaN.

    Its E4M3 scale code underflows to 0.
    """
    weight = torch.randn(2, 32)
    weight[0, :16] *= 1e6
    weight[1, 16:] *= 1e-6
    fake_quantize = weight_fake_quantizer(nvfp4_spec())

    out = fake_quantize(weight)

    assert fake_quantize.qparams_calculator.scale[1, 1].item() == 0.0
    assert torch.equal(out[1, 16:], torch.zeros(16))
    assert torch.isfinite(out).all()


def test_nvfp4_all_zero_weight_stays_zero():
    out = weight_fake_quantizer(nvfp4_spec())(torch.zeros(4, 64))
    assert torch.equal(out, torch.zeros(4, 64))


# ---------------------------------------------------------------------------
# Quantized minval numerics
# ---------------------------------------------------------------------------


def test_minval_elements_use_the_decoded_scale_and_minval():
    """Hand-computed values: elements are chosen against the decoded scale and minval.

    Block 0 is [-1.5, 0, 1.5, 3]: scale 4.5 / 15 = 0.3, minval -1.5. Block 1 is
    [-0.28, 0.1, 0.3, 0.555]: scale 0.835 / 15 = 0.0557, minval -0.28. The scales share one
    parent, 0.3 / 15 = 0.02, so block 1's scale becomes code 3 (2.78 rounded), 0.06. The
    minvals get their own parent, 1.5 / 15 = 0.1, so block 1's minval becomes code 3 (2.8
    rounded), -0.3. Block 1 then decodes to -0.3 + 0.06 * [0, 7, 10, 14]. Without the nested
    spec it would decode to [-0.28, 0.1097, 0.2767, 0.555].
    """
    fake_quantize = weight_fake_quantizer(_worked_example_spec(QuantizationFormulation.MINVAL))
    weight = torch.tensor([[-1.5, 0.0, 1.5, 3.0, -0.28, 0.1, 0.3, 0.555]])

    out = fake_quantize(weight)

    calculator = fake_quantize.qparams_calculator
    _assert_values(calculator.scale_fake_quantize.qparams_calculator.scale, [0.02])
    _assert_values(calculator.minval_fake_quantize.qparams_calculator.scale, [0.1])
    _assert_values(calculator.scale, [0.3, 0.06])
    _assert_values(calculator.minval, [-1.5, -0.3])
    _assert_values(out, [-1.5, 0.0, 1.5, 3.0, -0.3, 0.12, 0.3, 0.54])


def test_minval_qparams_decompose_into_codes_and_parents():
    """Scale and minval are uint8 codes times separate parents, one of each per 8 blocks.

    The nested specs see non-negative values, so their zero points are 0.
    """
    weight = torch.randn(4, 512) * 0.02
    fake_quantize = weight_fake_quantizer(minval_spec())
    fake_quantize(weight)
    calculator = fake_quantize.qparams_calculator

    for name, decoded, qparam_fq in (
        ("scale", calculator.scale, calculator.scale_fake_quantize),
        ("minval", -calculator.minval, calculator.minval_fake_quantize),
    ):
        parent = qparam_fq.qparams_calculator.scale
        assert parent.shape == (4, 2), name
        assert (qparam_fq.qparams_calculator.zero_point == 0).all(), name
        codes = decoded.reshape(4, 2, 8) / parent.unsqueeze(-1)
        torch.testing.assert_close(codes, codes.round())
        assert codes.round().min() >= 0 and codes.round().max() <= 255, name
    assert calculator.scale_fake_quantize is not calculator.minval_fake_quantize


def test_minval_constant_negative_block_decodes_to_its_minval():
    """A constant negative block decodes to the nearest minval its min parent can express."""
    weight = torch.zeros(1, 256)
    weight[0, :32] = torch.linspace(-1.0, 2.0, 32)
    weight[0, 32:64] = -0.65
    fake_quantize = weight_fake_quantizer(minval_spec())

    out = fake_quantize(weight)

    min_parent = fake_quantize.qparams_calculator.minval_fake_quantize.qparams_calculator.scale
    assert out[0, 32:64].unique().numel() == 1
    assert (out[0, 32] + 0.65).abs() <= min_parent.item() / 2


def test_minval_scale_code_zero_decodes_to_minval():
    """A block with scale code 0 decodes to its minval rather than NaN.

    Its range, which always includes zero, is below 1/510 of the largest range in its group of
    8 blocks.
    """
    weight = torch.zeros(1, 256)
    weight[0, :32] = torch.linspace(-1.0, 2.0, 32)
    weight[0, 32:64] = torch.linspace(0.0, 1e-4, 32)
    fake_quantize = weight_fake_quantizer(minval_spec())

    out = fake_quantize(weight)

    calculator = fake_quantize.qparams_calculator
    assert calculator.scale[0, 1].item() == 0.0
    assert torch.equal(out[0, 32:64], calculator.minval[0, 1].expand(32))


def test_minval_all_positive_block_has_zero_minval():
    weight = torch.zeros(1, 256)
    weight[0, :32] = torch.linspace(-1.0, 2.0, 32)
    weight[0, 32:64] = torch.linspace(0.1, 0.9, 32)
    fake_quantize = weight_fake_quantizer(minval_spec())

    out = fake_quantize(weight)

    assert fake_quantize.qparams_calculator.minval[0, 1].item() == 0.0
    assert (out[0, 32:64] >= 0).all()


# ---------------------------------------------------------------------------
# Shared behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("spec_factory", "reference"), _REFERENCE_CASES)
def test_matches_reference(spec_factory, reference):
    weight = torch.randn(8, 512) * 0.02
    out = weight_fake_quantizer(spec_factory())(weight)
    assert torch.equal(out, reference(weight))


@pytest.mark.parametrize("spec_factory", _SPEC_FACTORIES)
def test_half_precision_weight_matches_fp32_qparams(spec_factory):
    """The qparams are computed from an fp32 scale, so a bf16 weight is not rounded twice."""
    weight = (torch.randn(4, 512) * 0.02).bfloat16()
    bf16_fake_quantize = weight_fake_quantizer(spec_factory())
    fp32_fake_quantize = weight_fake_quantizer(spec_factory())

    bf16_fake_quantize(weight)
    fp32_fake_quantize(weight.float())

    torch.testing.assert_close(
        bf16_fake_quantize.calculate_qparams(),
        fp32_fake_quantize.calculate_qparams(),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("spec_factory", _SPEC_FACTORIES)
def test_gradient_flows_through_fake_quantize(spec_factory):
    weight = (torch.randn(4, 512) * 0.02).requires_grad_()
    weight_fake_quantizer(spec_factory())(weight).sum().backward()
    assert weight.grad is not None
    assert torch.isfinite(weight.grad).all()
    assert weight.grad.abs().sum() > 0


def test_nested_block_axis_follows_the_weight_block_axis():
    """Nested per-block granularities take the block axis once weight axis defaults set it."""
    fake_quantize = weight_fake_quantizer(minval_spec(axis=None))
    calculator = fake_quantize.qparams_calculator

    fake_quantize.granularity = fake_quantize.granularity.model_copy(update={"axis": 1})

    assert calculator.scale_fake_quantize.granularity.axis == 1
    assert calculator.minval_fake_quantize.granularity.axis == 1


def test_explicit_nested_axis_groups_along_that_axis():
    """A nested block axis other than the weight's groups the qparam grid along it."""
    spec = QuantizationSpec(
        dtype=torch.int4,
        granularity=PerBlockGranularity(axis=1, block_size=32),
        qparam_qspec=QuantizationSpec(
            dtype=torch.uint8,
            qscheme=QuantizationScheme.ASYMMETRIC,
            granularity=PerBlockGranularity(axis=0, block_size=2),
        ),
    )
    fake_quantize = weight_fake_quantizer(spec)

    fake_quantize(torch.randn(8, 128))

    # The 8 x 4 blocks share nested qparams in pairs of rows.
    nested_calculator = fake_quantize.qparams_calculator.scale_fake_quantize.qparams_calculator
    assert nested_calculator.scale.shape == (4, 4)


def test_ungroupable_blocks_disable_the_fake_quantizer(caplog):
    """288 input channels form 9 blocks of 32, which do not divide into groups of 8."""
    weight = torch.randn(4, 288)
    fake_quantize = weight_fake_quantizer(minval_spec())

    with caplog.at_level(logging.WARNING):
        out = fake_quantize(weight)

    assert torch.equal(out, weight)
    assert fake_quantize.is_disabled()
    assert "qparam_qspec cannot group the per-block scale grid of shape (4, 9)" in caplog.text


def test_per_tensor_parent_quantizes_its_scale_per_tensor():
    """A parent switched to per-tensor quantizes its single scale and minval per tensor.

    Graph mode forces some shared observers to per-tensor, which leaves nothing for a nested
    per-block granularity to group.
    """
    fake_quantize = weight_fake_quantizer(minval_spec())
    fake_quantize.granularity = PerTensorGranularity()
    weight = torch.randn(4, 512) * 0.02

    out = fake_quantize(weight)

    calculator = fake_quantize.qparams_calculator
    assert not fake_quantize.is_disabled()
    for qparam_fq in (calculator.scale_fake_quantize, calculator.minval_fake_quantize):
        assert qparam_fq.granularity == PerTensorGranularity()
    assert not torch.equal(out, weight)


@pytest.mark.parametrize(
    ("spec_factory", "quantizes_minval"), [(nvfp4_spec, False), (minval_spec, True)]
)
def test_qparam_fake_quantizers_per_formulation(spec_factory, quantizes_minval):
    """The scale is always quantized; minval only under the MINVAL formulation."""
    partial = QuantizationComponentFactory.create_fake_quantizer_partial(
        spec_factory(), CompressionTargetTensor.WEIGHT
    )
    first, second = partial(), partial()
    assert first.qparams_calculator.scale_fake_quantize is not None
    assert (first.qparams_calculator.minval_fake_quantize is not None) == quantizes_minval
    # Each constructed fake quantizer gets its own qparam fake quantizers.
    assert (
        first.qparams_calculator.scale_fake_quantize
        is not second.qparams_calculator.scale_fake_quantize
    )


# ---------------------------------------------------------------------------
# General qparam quantization: NVFP4 is a special case
# ---------------------------------------------------------------------------


def test_int8_scale_codes_per_channel_match_reference():
    """int4 blocks whose scales are int8 codes, with one tensor scale per output channel."""
    spec = QuantizationSpec(
        dtype=torch.int4,
        granularity=PerBlockGranularity(axis=1, block_size=32),
        qparam_qspec=QuantizationSpec(dtype=torch.int8, granularity=PerChannelGranularity(axis=0)),
    )
    weight = torch.randn(8, 256)

    out = weight_fake_quantizer(spec)(weight)

    blocks = weight.reshape(8, -1, 32)
    ideal_scale = blocks.abs().amax(dim=-1) / 7.5
    channel_scale = ideal_scale.amax(dim=-1, keepdim=True) / 127.5
    scale = ((ideal_scale / channel_scale).round().clamp(-128, 127) * channel_scale).unsqueeze(-1)
    expected = (blocks / nonzero_scale(scale)).round().clamp(-8, 7) * scale
    assert torch.equal(out, expected.reshape(weight.shape))


def test_asymmetric_zero_point_follows_the_quantized_scale():
    spec = QuantizationSpec(
        dtype=torch.uint8,
        qscheme=QuantizationScheme.ASYMMETRIC,
        granularity=PerBlockGranularity(axis=1, block_size=32),
        qparam_qspec=QuantizationSpec(
            dtype=torch.float8_e4m3fn, granularity=PerTensorGranularity()
        ),
    )
    weight = torch.randn(4, 128) + 0.3
    fake_quantize = weight_fake_quantizer(spec)

    fake_quantize(weight)

    scale, zero_point, _minval = fake_quantize.calculate_qparams()
    min_val = weight.reshape(4, -1, 32).amin(dim=-1).clamp(max=0)
    expected_zero_point = torch.clamp(-torch.round(min_val / scale), 0, 255).to(torch.int32)
    assert torch.equal(zero_point, expected_zero_point)


def test_zero_point_is_rederived_from_the_decoded_scale():
    """Hand-computed values: the zero point comes from the decoded scale, not the raw one.

    Block 0 is [-1.5, 0, 1.5, 3]: scale 0.3, zero point 5. Block 1 is [-0.3, 0, 0.3, 0.72]:
    scale 1.02 / 15 = 0.068, which the shared parent 0.02 rounds to code 3, so 0.06. The zero
    point of the raw scale, round(0.3 / 0.068) = 4, would clip -0.3 to -0.24. Re-derived from
    0.06 it is 5, so -0.3 stays exact and 0.72 clips to the top code, (15 - 5) * 0.06 = 0.6.
    """
    fake_quantize = weight_fake_quantizer(_worked_example_spec(QuantizationFormulation.ZP))
    weight = torch.tensor([[-1.5, 0.0, 1.5, 3.0, -0.3, 0.0, 0.3, 0.72]])

    out = fake_quantize(weight)

    calculator = fake_quantize.qparams_calculator
    _assert_values(calculator.scale_fake_quantize.qparams_calculator.scale, [0.02])
    _assert_values(calculator.scale, [0.3, 0.06])
    assert calculator.zero_point.flatten().tolist() == [5, 5]
    _assert_values(out, [-1.5, 0.0, 1.5, 3.0, -0.3, 0.0, 0.3, 0.6])


def test_qparam_qspec_nests_recursively():
    """A nested spec's own qparam_qspec quantizes the nested scales in turn."""
    fake_quantize = weight_fake_quantizer(twice_quantized_spec())

    fake_quantize(torch.randn(4, 256))

    nested_calculator = fake_quantize.qparams_calculator.scale_fake_quantize.qparams_calculator
    innermost_scale = nested_calculator.scale_fake_quantize.qparams_calculator.scale
    # One nested scale per 4 of the 4 x 8 blocks, each an E4M3 code times one tensor scale.
    assert nested_calculator.scale.shape == (4, 2)
    codes = nested_calculator.scale / innermost_scale
    torch.testing.assert_close(codes, codes.to(torch.float8_e4m3fn).float())


def test_activation_qparam_fake_quantizer_targets_activations():
    """The nested fake quantizer shares the target, so "default" resolves to a moving average."""
    fake_quantize = QuantizationComponentFactory.create_fake_quantizer(
        _nvfp4_activation_spec(), CompressionTargetTensor.ACTIVATION
    )
    qparam_fq = fake_quantize.qparams_calculator.scale_fake_quantize
    assert qparam_fq.quantization_target == CompressionTargetTensor.ACTIVATION
    assert isinstance(qparam_fq.qparams_calculator, MovingAverageQParamsCalculator)


def test_nested_scale_follows_the_observer_while_block_scales_stay_dynamic():
    """The nested tensor scale freezes with the observer; dynamic block scales keep updating.

    The tensor scale calibrates while observers are on and freezes once they are off, while the
    block scales follow the input on every forward.
    """
    fake_quantize = QuantizationComponentFactory.create_fake_quantizer(
        _nvfp4_activation_spec(tensor_scale_calculator="global_minmax"),
        CompressionTargetTensor.ACTIVATION,
    )
    tensor_scale = fake_quantize.qparams_calculator.scale_fake_quantize.qparams_calculator
    inputs = torch.randn(2, 64)
    fake_quantize(inputs)
    calibrated = tensor_scale.scale.clone()

    fake_quantize.apply(disable_observer)
    frozen_out = fake_quantize(inputs * 0.5)
    assert torch.equal(tensor_scale.scale, calibrated)
    assert torch.isfinite(frozen_out).all()

    fake_quantize.apply(enable_observer)
    fake_quantize(inputs * 10)
    assert tensor_scale.scale.item() > calibrated.item()


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("execution_mode", [ExecutionMode.EAGER, ExecutionMode.GRAPH])
def test_weight_and_activation_qparam_qspec_end_to_end(execution_mode):
    """NVFP4 weights and activations: calibrate, then run the prepared model in torch."""
    model = _mlp()
    inputs = torch.randn(4, 256)
    config = _config(
        nvfp4_spec(axis=None), execution_mode, op_input_spec={"*": _nvfp4_activation_spec()}
    )
    quantizer = Quantizer(model, config)
    prepared_model = quantizer.prepare((inputs,))
    with quantizer.calibration_mode():
        prepared_model(inputs)

    out = prepared_model(inputs)

    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, model(inputs), atol=0.3, rtol=0.3)
    assert torch.equal(quantizer.finalize(backend=ExportBackend._TORCH)(inputs), out)


@pytest.mark.parametrize("execution_mode", [ExecutionMode.EAGER, ExecutionMode.GRAPH])
@pytest.mark.parametrize(("spec_factory", "reference"), _REFERENCE_CASES)
def test_prepared_model_matches_reference(spec_factory, reference, execution_mode):
    model = _mlp()
    inputs = torch.randn(4, 256)
    first, second = model[0], model[2]
    hidden = F.relu(F.linear(inputs, reference(first.weight.detach()), first.bias))
    expected = F.linear(hidden, reference(second.weight.detach()), second.bias)

    quantizer = Quantizer(model, _config(spec_factory(axis=None), execution_mode))
    prepared_model = quantizer.prepare((inputs,))
    out = prepared_model(inputs)

    torch.testing.assert_close(out, expected)
    finalized_model = quantizer.finalize(backend=ExportBackend._TORCH)
    assert torch.equal(finalized_model(inputs), out)


@pytest.mark.parametrize("execution_mode", [ExecutionMode.EAGER, ExecutionMode.GRAPH])
@pytest.mark.parametrize(
    ("weight_spec", "backend"),
    [
        pytest.param(nvfp4_spec(axis=None), ExportBackend.CoreAI, id="nvfp4-coreai"),
        pytest.param(minval_spec(axis=None), ExportBackend.CoreML, id="minval-coreml"),
        pytest.param(
            QuantizationSpec(
                dtype=torch.int4,
                granularity=PerBlockGranularity(axis=1, block_size=32),
                qparam_qspec=QuantizationSpec(
                    dtype=torch.int8,
                    qscheme=QuantizationScheme.ASYMMETRIC,
                    granularity=PerChannelGranularity(axis=0),
                ),
            ),
            ExportBackend.CoreAI,
            id="scale_codes_with_an_offset-coreai",
        ),
        pytest.param(
            twice_quantized_spec(), ExportBackend.CoreML, id="scale_quantized_twice-coreml"
        ),
    ],
)
def test_export_is_not_supported_yet(weight_spec, backend, execution_mode):
    """Any qparam_qspec blocks CoreAI and CoreML export for now."""
    inputs = torch.randn(4, 256)
    quantizer = Quantizer(_mlp(), _config(weight_spec, execution_mode))
    quantizer.prepare((inputs,))

    with pytest.raises(NotImplementedError, match="quantized qparams") as excinfo:
        quantizer.finalize(backend=backend)

    # Only the weight fake quantizers are named, not the nested ones, which also quantize
    # qparams when a spec is quantized twice.
    assert "fake_quantize" not in str(excinfo.value).split("Affected FakeQuantize modules:")[1]


def test_qat_step_updates_weights():
    model = _mlp()
    inputs = torch.randn(4, 256)
    quantizer = Quantizer(model, _config(minval_spec(axis=None), ExecutionMode.EAGER))
    prepared_model = quantizer.prepare((inputs,))
    optimizer = torch.optim.SGD(prepared_model.parameters(), lr=0.1)
    before = [p.detach().clone() for p in prepared_model.parameters()]

    with quantizer.training_mode():
        prepared_model(inputs).square().mean().backward()
        optimizer.step()

    after = list(prepared_model.parameters())
    assert all(torch.isfinite(p).all() for p in after)
    assert any(not torch.equal(b, a) for b, a in zip(before, after, strict=True))


@pytest.mark.parametrize("execution_mode", [ExecutionMode.EAGER, ExecutionMode.GRAPH])
def test_qat_schedule_freezes_nested_scales(execution_mode):
    """A QAT schedule's disable_observer step also freezes the nested scales.

    The dynamic activation calculator quantizes its block scales on every forward, so the nested
    tensor scale only stops updating if the schedule reaches its fake quantizer.
    """
    inputs = torch.randn(4, 256)
    config = _config(
        nvfp4_spec(axis=None),
        execution_mode,
        op_input_spec={"*": _nvfp4_activation_spec(tensor_scale_calculator="global_minmax")},
        qat_schedule=QATSchedule(enable_observer=0, enable_fake_quant=0, disable_observer=1),
    )
    quantizer = Quantizer(_mlp(), config)
    prepared_model = quantizer.prepare((inputs,))
    nested_calculators = [
        mod.qparams_calculator.scale_fake_quantize.qparams_calculator
        for mod in prepared_model.modules()
        if isinstance(mod, FakeQuantizeImplBase)
        and mod.quantization_target == CompressionTargetTensor.ACTIVATION
        and mod.qparams_calculator.scale_fake_quantize is not None
    ]
    assert nested_calculators

    with quantizer.training_mode():
        prepared_model(inputs)
        quantizer.step()
        frozen = [calculator.scale.clone() for calculator in nested_calculators]
        prepared_model(inputs * 10)

    for calculator, scale in zip(nested_calculators, frozen, strict=True):
        assert torch.equal(calculator.scale, scale)
