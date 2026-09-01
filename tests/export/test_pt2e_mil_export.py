# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for PT2E quantizer export to CoreML backend."""

import pytest
import torch

from coreai_opt import ExportBackend
from coreai_opt.quantization import ModuleQuantizerConfig, Quantizer, QuantizerConfig
from coreai_opt.quantization.spec import (
    PerBlockGranularity,
    PerChannelGranularity,
    PerTensorGranularity,
    QuantizationGranularity,
    QuantizationScheme,
    QuantizationSpec,
)
from coreai_opt.quantization.spec.fake_quantize import _DefaultFakeQuantizeImpl
from coreai_opt.quantization.spec.qparams_calculator import (
    MovingAverageQParamsCalculator,
    StaticQParamsCalculator,
)
from coreai_opt.quantization.spec.range_calculator import MinMaxRangeCalculator
from tests.fixtures.quantization import (
    COREML_ACT_REJECT_DTYPES,
    COREML_WEIGHT_REJECT_DTYPES,
    make_quant_config,
)

from . import export_utils

# TODO: migrate to using conftest.py for fixtures.


def _build_simple_model_quant_config(
    weight_dtype: torch.dtype,
    weight_granularity: QuantizationGranularity,
    act_granularity: QuantizationGranularity,
    qscheme: QuantizationScheme,
) -> QuantizerConfig:
    """Build a QuantizerConfig for the simple conv/linear model's weight and activations."""
    weight_qspec = QuantizationSpec(
        dtype=weight_dtype,
        qscheme=qscheme,
        granularity=weight_granularity,
        fake_quantize_cls=_DefaultFakeQuantizeImpl,
        qparam_calculator_cls=StaticQParamsCalculator,
        range_calculator_cls=MinMaxRangeCalculator,
    )
    activation_qspec = QuantizationSpec(
        dtype=torch.uint8,
        qscheme=qscheme,
        granularity=act_granularity,
        fake_quantize_cls=_DefaultFakeQuantizeImpl,
        qparam_calculator_cls=MovingAverageQParamsCalculator,
        range_calculator_cls=MinMaxRangeCalculator,
    )
    return QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={"weight": weight_qspec},
            op_input_spec={"*": activation_qspec},
            op_output_spec={"*": activation_qspec},
        ),
    )


_test_params = (
    [
        # int8 weights with per-tensor granularity
        (torch.int8, PerTensorGranularity(), PerTensorGranularity(), qscheme)
        for qscheme in QuantizationScheme
    ]
    + [
        # uint8 weights with per-channel (axis=0) granularity
        (torch.uint8, PerChannelGranularity(axis=0), PerTensorGranularity(), qscheme)
        for qscheme in QuantizationScheme
    ]
    + [
        # int4 weights with per-tensor granularity (low-bit quantization)
        (torch.int4, PerTensorGranularity(), PerTensorGranularity(), qscheme)
        for qscheme in QuantizationScheme
    ]
    + [
        # int8 weights with per-channel (axis=1) - default config for Conv/Linear
        (torch.int8, PerChannelGranularity(axis=1), PerTensorGranularity(), qscheme)
        for qscheme in QuantizationScheme
    ]
    + [
        # int4 weights with per-channel (axis=1) - low-bit with per-channel
        (torch.int4, PerChannelGranularity(axis=1), PerTensorGranularity(), qscheme)
        for qscheme in QuantizationScheme
    ]
    + [
        # uint8 weights with per-tensor granularity, asymmetric qscheme
        (
            torch.uint8,
            PerTensorGranularity(),
            PerTensorGranularity(),
            QuantizationScheme.ASYMMETRIC,
        ),
    ]
)

_test_ids = [
    f"w:{str(dtype).split('.')[-1]}--"
    f"wg:{wg.__class__.__name__.replace('Granularity', '')}--"
    f"ag:{ag.__class__.__name__.replace('Granularity', '')}--"
    f"{f'axis{ag.axis}'}--"
    f"qscheme:{qscheme.value}"
    for dtype, wg, ag, qscheme in _test_params
]


@pytest.mark.parametrize(
    ("weight_dtype", "weight_granularity", "act_granularity", "qscheme"),
    _test_params,
    ids=_test_ids,
)
def test_simple_model_export(
    simple_conv_linear_model: torch.nn.Module,
    simple_model_input: torch.Tensor,
    weight_dtype: torch.dtype,
    weight_granularity: QuantizationGranularity,
    act_granularity: QuantizationGranularity,
    qscheme: QuantizationScheme,
) -> None:
    """Test PT2E CoreML export with various quantization configurations."""
    model = simple_conv_linear_model
    model.eval()

    config = _build_simple_model_quant_config(
        weight_dtype, weight_granularity, act_granularity, qscheme
    )
    quantizer = Quantizer(model, config)
    prepared_model = quantizer.prepare((simple_model_input,))

    with torch.no_grad():
        prepared_model_output = prepared_model(simple_model_input)
    expected_ops = {
        "constexpr_blockwise_shift_scale": 2,
        "quantize": 3,
        "dequantize": 3,
    }

    finalized_model = quantizer.finalize(backend=ExportBackend.CoreML)
    export_utils.convert_and_verify(
        finalized_model=finalized_model,
        input_data=simple_model_input,
        expected_ops=expected_ops,
        export_backend=ExportBackend.CoreML,
        prepared_model_output=prepared_model_output,
    )


# CoreML export rejects per-channel and per-block activation quantization outright,
# regardless of axis, dtype, or qscheme: a downstream coremltools MIL pass can
# silently corrupt it.
_perchannel_act_reject_params = [
    (
        torch.uint8,
        PerChannelGranularity(axis=0),
        PerChannelGranularity(axis=0),
        QuantizationScheme.SYMMETRIC,
    ),
    (
        torch.int8,
        PerChannelGranularity(axis=1),
        PerChannelGranularity(axis=-1),
        QuantizationScheme.ASYMMETRIC,
    ),
    (
        torch.int8,
        PerTensorGranularity(),
        PerBlockGranularity(axis=1, block_size=2),
        QuantizationScheme.SYMMETRIC,
    ),
]
_perchannel_act_reject_ids = ["axis0--symmetric", "axis-1--asymmetric", "per-block--symmetric"]


@pytest.mark.parametrize(
    ("weight_dtype", "weight_granularity", "act_granularity", "qscheme"),
    _perchannel_act_reject_params,
    ids=_perchannel_act_reject_ids,
)
def test_simple_model_export_rejects_perchannel_activation(
    simple_conv_linear_model: torch.nn.Module,
    simple_model_input: torch.Tensor,
    weight_dtype: torch.dtype,
    weight_granularity: QuantizationGranularity,
    act_granularity: QuantizationGranularity,
    qscheme: QuantizationScheme,
) -> None:
    """Per-channel activation quantization must be rejected on CoreML export."""
    model = simple_conv_linear_model
    model.eval()

    config = _build_simple_model_quant_config(
        weight_dtype, weight_granularity, act_granularity, qscheme
    )
    quantizer = Quantizer(model, config)
    quantizer.prepare((simple_model_input,))
    export_utils.assert_coreml_finalize_rejects(quantizer)


_mnist_test_params = [
    # Per-tensor weight + per-tensor activation
    (torch.int8, PerTensorGranularity(), PerTensorGranularity(), qscheme)
    for qscheme in QuantizationScheme
] + [
    # Per-channel weight (axis=0) + per-tensor activation
    (torch.int8, PerChannelGranularity(axis=0), PerTensorGranularity(), qscheme)
    for qscheme in QuantizationScheme
]

_mnist_test_ids = [
    f"w:{str(dtype).split('.')[-1]}--"
    f"wg:{wg.__class__.__name__.replace('Granularity', '')}--"
    f"ag:{ag.__class__.__name__.replace('Granularity', '')}--"
    f"{f'axis{ag.axis}'}--"
    f"qscheme:{qscheme.value}"
    for dtype, wg, ag, qscheme in _mnist_test_params
]


@pytest.mark.parametrize(
    ("weight_dtype", "weight_granularity", "act_granularity", "qscheme"),
    _mnist_test_params,
    ids=_mnist_test_ids,
)
def test_mnist_export(
    custom_test_mnist_model: torch.nn.Module,
    mnist_example_input: torch.Tensor,
    weight_dtype: torch.dtype,
    weight_granularity: QuantizationGranularity,
    act_granularity: QuantizationGranularity,
    qscheme: QuantizationScheme,
) -> None:
    """Test PT2E CoreML export with MNIST model."""
    model = custom_test_mnist_model
    model.eval()

    weight_qspec = QuantizationSpec(
        dtype=weight_dtype,
        qscheme=qscheme,
        granularity=weight_granularity,
        fake_quantize_cls=_DefaultFakeQuantizeImpl,
        qparam_calculator_cls=StaticQParamsCalculator,
        range_calculator_cls=MinMaxRangeCalculator,
    )
    activation_qspec = QuantizationSpec(
        dtype=torch.uint8,
        qscheme=qscheme,
        granularity=act_granularity,
        fake_quantize_cls=_DefaultFakeQuantizeImpl,
        qparam_calculator_cls=MovingAverageQParamsCalculator,
        range_calculator_cls=MinMaxRangeCalculator,
    )
    config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={"weight": weight_qspec},
            op_input_spec={"*": activation_qspec},
            op_output_spec={"*": activation_qspec},
        ),
    )
    quantizer = Quantizer(model, config)
    prepared_model = quantizer.prepare((mnist_example_input,))

    with torch.no_grad():
        prepared_model_output = prepared_model(mnist_example_input)

    expected_ops = {
        "constexpr_blockwise_shift_scale": 6,
        "quantize": 10,
        "dequantize": 10,
    }

    finalized_model = quantizer.finalize(backend=ExportBackend.CoreML)

    export_utils.convert_and_verify(
        finalized_model=finalized_model,
        input_data=mnist_example_input,
        expected_ops=expected_ops,
        export_backend=ExportBackend.CoreML,
        prepared_model_output=prepared_model_output,
    )


def test_resnet_export(
    resnet50_model: torch.nn.Module,
    resnet_example_input: torch.Tensor,
) -> None:
    """Test PT2E CoreML export with ResNet50 model."""
    model = resnet50_model
    model.eval()

    # Configure with uint8 activation quantization to match the default config
    weight_qspec = QuantizationSpec(
        dtype=torch.int8,
        qscheme=QuantizationScheme.SYMMETRIC,
        granularity=PerTensorGranularity(),
        fake_quantize_cls=_DefaultFakeQuantizeImpl,
        qparam_calculator_cls=StaticQParamsCalculator,
        range_calculator_cls=MinMaxRangeCalculator,
    )
    activation_qspec = QuantizationSpec(
        dtype=torch.uint8,
        qscheme=QuantizationScheme.SYMMETRIC,
        granularity=PerTensorGranularity(),
        fake_quantize_cls=_DefaultFakeQuantizeImpl,
        qparam_calculator_cls=MovingAverageQParamsCalculator,
        range_calculator_cls=MinMaxRangeCalculator,
    )
    config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={"weight": weight_qspec},
            op_input_spec={"*": activation_qspec},
            op_output_spec={"*": activation_qspec},
        ),
    )

    quantizer = Quantizer(model, config)
    prepared_model = quantizer.prepare((resnet_example_input,))

    with torch.no_grad():
        prepared_model_output = prepared_model(resnet_example_input)

    expected_ops = {
        "constexpr_blockwise_shift_scale": 54,
        "quantize": 73,
        "dequantize": 85,
    }

    finalized_model = quantizer.finalize(backend=ExportBackend.CoreML)

    export_utils.convert_and_verify(
        finalized_model=finalized_model,
        input_data=resnet_example_input,
        expected_ops=expected_ops,
        export_backend=ExportBackend.CoreML,
        prepared_model_output=prepared_model_output,
        snr_thresh=18.0,
        psnr_thresh=35.0,
    )


# Activation granularity test params for GatedMLPModel. CoreML export rejects
# PerChannelGranularity activations outright (see test_gated_mlp_export_rejects_
# perchannel_activation below), so only PerTensorGranularity actually exports here.
_gated_mlp_test_params = [(PerTensorGranularity(), qscheme) for qscheme in QuantizationScheme]

_gated_mlp_test_ids = [
    f"ag:{ag.__class__.__name__.replace('Granularity', '')}--"
    f"{f'axis{ag.axis}'}--"
    f"qscheme:{qscheme.value}"
    for ag, qscheme in _gated_mlp_test_params
]


def _build_gated_mlp_quant_config(
    act_granularity: QuantizationGranularity,
    qscheme: QuantizationScheme,
) -> QuantizerConfig:
    """Build a QuantizerConfig for GatedMLPModel's weight and activations."""
    weight_qspec = QuantizationSpec(
        dtype=torch.int8,
        qscheme=qscheme,
        granularity=PerTensorGranularity(),
        fake_quantize_cls=_DefaultFakeQuantizeImpl,
        qparam_calculator_cls=StaticQParamsCalculator,
        range_calculator_cls=MinMaxRangeCalculator,
    )
    activation_qspec = QuantizationSpec(
        dtype=torch.uint8,
        qscheme=qscheme,
        granularity=act_granularity,
        fake_quantize_cls=_DefaultFakeQuantizeImpl,
        qparam_calculator_cls=MovingAverageQParamsCalculator,
        range_calculator_cls=MinMaxRangeCalculator,
    )
    return QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={"weight": weight_qspec},
            op_input_spec={"*": activation_qspec},
            op_output_spec={"*": activation_qspec},
        ),
    )


@pytest.mark.parametrize(
    ("act_granularity", "qscheme"),
    _gated_mlp_test_params,
    ids=_gated_mlp_test_ids,
)
def test_gated_mlp_export(
    gated_mlp_model: torch.nn.Module,
    gated_mlp_model_input: torch.Tensor,
    act_granularity: QuantizationGranularity,
    qscheme: QuantizationScheme,
) -> None:
    """Test PT2E CoreML export with per-tensor activation quantization."""
    model = gated_mlp_model
    model.eval()

    config = _build_gated_mlp_quant_config(act_granularity, qscheme)
    quantizer = Quantizer(model, config)
    prepared_model = quantizer.prepare((gated_mlp_model_input,))

    with torch.no_grad():
        prepared_model_output = prepared_model(gated_mlp_model_input)
    expected_ops = {
        "constexpr_blockwise_shift_scale": 3,
        "quantize": 5,
        "dequantize": 5,
    }

    finalized_model = quantizer.finalize(backend=ExportBackend.CoreML)
    export_utils.convert_and_verify(
        finalized_model=finalized_model,
        input_data=gated_mlp_model_input,
        expected_ops=expected_ops,
        export_backend=ExportBackend.CoreML,
        prepared_model_output=prepared_model_output,
    )


# CoreML export rejects per-channel and per-block activation quantization outright,
# regardless of axis: a downstream coremltools MIL pass can silently corrupt it.
_gated_mlp_reject_params = [
    (PerChannelGranularity(axis=0), QuantizationScheme.SYMMETRIC),
    (PerChannelGranularity(axis=-1), QuantizationScheme.ASYMMETRIC),
    (PerBlockGranularity(axis=1, block_size=2), QuantizationScheme.SYMMETRIC),
]
_gated_mlp_reject_ids = ["axis0--symmetric", "axis-1--asymmetric", "per-block--symmetric"]


@pytest.mark.parametrize(
    ("act_granularity", "qscheme"),
    _gated_mlp_reject_params,
    ids=_gated_mlp_reject_ids,
)
def test_gated_mlp_export_rejects_perchannel_activation(
    gated_mlp_model: torch.nn.Module,
    gated_mlp_model_input: torch.Tensor,
    act_granularity: QuantizationGranularity,
    qscheme: QuantizationScheme,
) -> None:
    """Per-channel activation quantization must be rejected regardless of axis."""
    model = gated_mlp_model
    model.eval()

    config = _build_gated_mlp_quant_config(act_granularity, qscheme)
    quantizer = Quantizer(model, config)
    quantizer.prepare((gated_mlp_model_input,))
    export_utils.assert_coreml_finalize_rejects(quantizer)


# Unsupported dtypes (FP4, FP8, INT2, UINT2) must be rejected on CoreML export;
# finalize must reject them. Dtype lists and the config builder live in conftest
# (shared with the eager tests).
@pytest.mark.parametrize("weight_dtype", COREML_WEIGHT_REJECT_DTYPES)
def test_unsupported_weight_quant_coreml_export_rejected(
    simple_conv_linear_model: torch.nn.Module,
    simple_model_input: torch.Tensor,
    weight_dtype: torch.dtype | str,
) -> None:
    """Unsupported weight quantization dtypes must be rejected on graph-mode CoreML export."""
    config = make_quant_config(weight_dtype=weight_dtype, act_dtype=None, execution_mode="graph")
    model = simple_conv_linear_model
    model.eval()
    quantizer = Quantizer(model, config)
    quantizer.prepare((simple_model_input,))
    export_utils.assert_coreml_finalize_rejects(quantizer)


@pytest.mark.parametrize("act_dtype", COREML_ACT_REJECT_DTYPES)
def test_unsupported_activation_quant_coreml_export_rejected(
    simple_conv_linear_model: torch.nn.Module,
    simple_model_input: torch.Tensor,
    act_dtype: torch.dtype,
) -> None:
    """Unsupported activation quantization dtypes must be rejected."""
    config = make_quant_config(weight_dtype=torch.int8, act_dtype=act_dtype, execution_mode="graph")
    model = simple_conv_linear_model
    model.eval()
    quantizer = Quantizer(model, config)
    quantizer.prepare((simple_model_input,))
    export_utils.assert_coreml_finalize_rejects(quantizer)
