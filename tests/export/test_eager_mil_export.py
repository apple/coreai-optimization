# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from collections.abc import Mapping

import pytest
import torch

from coreai_opt import CoreMLExportError, ExportBackend
from coreai_opt.quantization import Quantizer, QuantizerConfig
from tests.fixtures.quantization import (
    COREML_ACT_REJECT_DTYPES,
    COREML_WEIGHT_REJECT_DTYPES,
    ParametrizedQuantConfigs,
    make_quant_config,
)

from . import export_utils


def _run_eager_mil_export_test_ex(
    model: torch.nn.Module,
    input_data: torch.Tensor,
    config: QuantizerConfig,
    expected_ops: Mapping[str, int],
    model_dtype: torch.dtype | None = None,
) -> None:
    """Run eager CoreML export test with expanded configuration parameters.

    Args:
        model: PyTorch model to quantize and export
        input_data: Input tensor for model
        config: Eager quantization configuration
        model_dtype: Model dtype (float16, float32, bfloat16, or None for no conversion)
        expected_ops: Expected operation counts in converted model

    """
    if model_dtype is not None:
        model = model.to(dtype=model_dtype)
        input_data = input_data.to(dtype=model_dtype)

    model.eval()
    quantizer = Quantizer(model, config)
    prepared_model = quantizer.prepare((input_data,))

    with torch.no_grad():
        prepared_model_output = prepared_model(input_data)

    finalized_model = quantizer.finalize(backend=ExportBackend.CoreML)

    export_utils.convert_and_verify(
        finalized_model=finalized_model,
        input_data=input_data,
        expected_ops=expected_ops,
        export_backend=ExportBackend.CoreML,
        prepared_model_output=prepared_model_output,
        snr_thresh=18.0,
        psnr_thresh=35.0,
    )


def _run_eager_mil_export_test(
    model: torch.nn.Module,
    input_data: torch.Tensor,
    parametrized_quant_config: ParametrizedQuantConfigs,
    expected_ops: Mapping[str, int],
) -> None:
    """Run eager CoreML export test with parametrized configuration.

    Wrapper around _run_eager_mil_export_test_ex that extracts model_dtype
    from the parametrized config object.

    Args:
        model: PyTorch model to quantize and export
        input_data: Input tensor for model
        parametrized_quant_config: Parametrized quantization configurations
        expected_ops: Expected operation counts in converted model

    """
    _run_eager_mil_export_test_ex(
        model=model,
        input_data=input_data,
        config=parametrized_quant_config.eager,
        expected_ops=expected_ops,
        model_dtype=parametrized_quant_config.model_dtype,
    )


def test_simple_model_export(
    simple_conv_linear_model: torch.nn.Module,
    simple_model_input: torch.Tensor,
    parametrized_quant_config_general: ParametrizedQuantConfigs,
) -> None:
    """Test eager CoreML export with various quantization configurations."""
    # CoreML rejects per-channel activation quantization outright.
    if parametrized_quant_config_general.has_per_channel_activation_granularity:
        pytest.skip("CoreML export rejects per-channel activation quantization")

    has_act_quant = parametrized_quant_config_general.has_activation_quantization

    _run_eager_mil_export_test(
        model=simple_conv_linear_model,
        input_data=simple_model_input,
        parametrized_quant_config=parametrized_quant_config_general,
        expected_ops={
            "constexpr_blockwise_shift_scale": 2,
            "quantize": 4 if has_act_quant else 0,
            "dequantize": 4 if has_act_quant else 0,
        },
    )


def test_mnist_export(
    custom_test_mnist_model: torch.nn.Module,
    mnist_example_input: torch.Tensor,
    parametrized_quant_config_general: ParametrizedQuantConfigs,
) -> None:
    """Test eager CoreML export on MNIST model with various quantization configurations."""
    # CoreML rejects per-channel activation quantization outright.
    if parametrized_quant_config_general.has_per_channel_activation_granularity:
        pytest.skip("CoreML export rejects per-channel activation quantization")

    has_act_quant = parametrized_quant_config_general.has_activation_quantization

    _run_eager_mil_export_test(
        model=custom_test_mnist_model,
        input_data=mnist_example_input,
        parametrized_quant_config=parametrized_quant_config_general,
        # Technically there are 18 quantize/dequantize pairs in the finalized torch model, but
        # in the exported CoreML model some of the quantize/dequantize pairs are optimized away,
        # resulting in 15 pairs. Specifically, the quantize/dequantize pairs around the 3 MaxPool
        # layers are optimized away.
        expected_ops={
            "constexpr_blockwise_shift_scale": 6,
            "quantize": 15 if has_act_quant else 0,
            "dequantize": 15 if has_act_quant else 0,
        },
    )


def test_resnet_export(
    resnet50_model: torch.nn.Module,
    resnet_example_input: torch.Tensor,
) -> None:
    """Test eager CoreML export on ResNet50 with default quantization configuration.

    Uses single default config instead of full parameter matrix to avoid excessive
    test execution time. Full parametrization coverage is provided by faster models
    (simple_conv_linear_model, custom_test_mnist_model).

    """
    config = QuantizerConfig(execution_mode="eager")
    _run_eager_mil_export_test_ex(
        model=resnet50_model,
        input_data=resnet_example_input,
        config=config,
        expected_ops={
            "constexpr_blockwise_shift_scale": 54,  # conv, linear
            # - 1 quantize/dequantize is optimized away at maxpool.
            # - dequantize count > quantize count because at each residual block,
            # the relu output feeds into both an add and a conv — the extra
            # quantize op between the two paths gets shared, but each path
            # has its own dequantize.
            "quantize": 142,  # conv, linear, maxpool, add
            "dequantize": 154,  # conv, linear, maxpool, add
        },
    )


def test_gated_mlp_perchannel_act_export(
    gated_mlp_model: torch.nn.Module,
    gated_mlp_model_input: torch.Tensor,
    parametrized_quant_config_perchannel_act_axis_coverage: ParametrizedQuantConfigs,
) -> None:
    """Test eager CoreML export with per-channel activation quantization axes.

    Uses GatedMLPModel (uniform rank-3 activations throughout the model). CoreML
    export rejects per-channel activation quantization outright; per-tensor cases
    from this fixture should still export and verify normally.
    """
    config = parametrized_quant_config_perchannel_act_axis_coverage

    if config.has_per_channel_activation_granularity:
        with pytest.raises(CoreMLExportError):
            _run_eager_mil_export_test(
                model=gated_mlp_model,
                input_data=gated_mlp_model_input,
                parametrized_quant_config=config,
                expected_ops={},
            )
        return

    _run_eager_mil_export_test(
        model=gated_mlp_model,
        input_data=gated_mlp_model_input,
        parametrized_quant_config=config,
        expected_ops={
            "constexpr_blockwise_shift_scale": 3,
            "quantize": 6,
            "dequantize": 6,
        },
    )


# Unsupported dtypes (FP4, FP8, INT2, UINT2) must be rejected on CoreML export;
# finalize must reject them. Dtype lists and the config builder live in conftest
# (shared with the pt2e tests).
@pytest.mark.parametrize("weight_dtype", COREML_WEIGHT_REJECT_DTYPES)
def test_unsupported_weight_quant_coreml_export_rejected(
    simple_conv_linear_model: torch.nn.Module,
    simple_model_input: torch.Tensor,
    weight_dtype: torch.dtype | str,
) -> None:
    """Unsupported weight quantization dtypes must be rejected on eager CoreML export."""
    config = make_quant_config(weight_dtype=weight_dtype, act_dtype=None, execution_mode="eager")
    with pytest.raises(CoreMLExportError):
        _run_eager_mil_export_test_ex(
            simple_conv_linear_model, simple_model_input, config, expected_ops={}
        )


@pytest.mark.parametrize("act_dtype", COREML_ACT_REJECT_DTYPES)
def test_unsupported_activation_quant_coreml_export_rejected(
    simple_conv_linear_model: torch.nn.Module,
    simple_model_input: torch.Tensor,
    act_dtype: torch.dtype,
) -> None:
    """Unsupported activation quantization dtypes must be rejected."""
    config = make_quant_config(weight_dtype=torch.int8, act_dtype=act_dtype, execution_mode="eager")
    with pytest.raises(CoreMLExportError):
        _run_eager_mil_export_test_ex(
            simple_conv_linear_model, simple_model_input, config, expected_ops={}
        )
