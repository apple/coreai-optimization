# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from collections.abc import Mapping

import pytest
import torch
from torch import nn

from coreai_opt import ExportBackend
from coreai_opt.quantization import (
    ModuleQuantizerConfig,
    QuantizationSpec,
    Quantizer,
    QuantizerConfig,
)
from coreai_opt.quantization.spec import (
    PerChannelGranularity,
    PerTensorGranularity,
    QuantizationFormulation,
    QuantizationScheme,
)
from tests.fixtures.fp4 import ParametrizedFP4Configs
from tests.fixtures.fp8 import ParametrizedFP8Configs
from tests.fixtures.quantization import ParametrizedQuantConfigs

from . import export_utils


def _run_eager_mlir_export_test_ex(
    model: torch.nn.Module,
    input_data: torch.Tensor,
    config: QuantizerConfig,
    expected_ops: Mapping[str, int],
    model_dtype: torch.dtype | None = None,
) -> None:
    """Run eager Core AI export test with expanded configuration parameters.

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

    finalized_model = quantizer.finalize(backend=ExportBackend.CoreAI)

    export_utils.convert_and_verify(
        finalized_model=finalized_model,
        input_data=input_data,
        expected_ops=expected_ops,
        export_backend=ExportBackend.CoreAI,
        prepared_model_output=prepared_model_output,
    )


def _run_eager_mlir_export_test(
    model: torch.nn.Module,
    input_data: torch.Tensor,
    parametrized_quant_config: ParametrizedQuantConfigs,
    expected_ops: Mapping[str, int],
) -> None:
    """Run eager Core AI export test with parametrized configuration.

    Wrapper around _run_eager_mlir_export_test_ex that extracts model_dtype
    from the parametrized config object.

    Args:
        model: PyTorch model to quantize and export
        input_data: Input tensor for model
        parametrized_quant_config: Parametrized quantization configurations
        expected_ops: Expected operation counts in converted model

    """
    _run_eager_mlir_export_test_ex(
        model=model,
        input_data=input_data,
        config=parametrized_quant_config.eager,
        expected_ops=expected_ops,
        model_dtype=parametrized_quant_config.model_dtype,
    )


def test_simple_model_export(
    simple_conv_linear_model: torch.nn.Module,
    simple_model_input: torch.Tensor,
    parametrized_quant_config_mlir: ParametrizedQuantConfigs,
    request: pytest.FixtureRequest,
) -> None:
    """Test eager Core AI export with various quantization configurations."""
    has_act_quant = parametrized_quant_config_mlir.has_activation_quantization

    # 4-bit-weight and int8 weight+activation per-tensor bfloat16 configs abort the
    # CoreAI interpreter (SIGABRT); xfail them without running so the native crash
    # cannot abort the session.
    parametrized_quant_config_mlir.xfail_if_unsupported(
        "eager",
        ExportBackend.CoreAI,
        unsupported_config=[
            {"model_dtype": torch.bfloat16, "weight_dtype": torch.int4},
            {"model_dtype": torch.bfloat16, "weight_dtype": torch.uint4},
            {
                "model_dtype": torch.bfloat16,
                "weight_dtype": torch.int8,
                "act_dtype": torch.int8,
                "granularity_type": "PerTensorGranularity",
            },
        ],
        reason="CoreAI interpreter aborts on this bfloat16 config.",
    )

    if parametrized_quant_config_mlir.model_dtype == torch.bfloat16:
        request.applymarker(
            pytest.mark.xfail(
                reason="bfloat16 CoreAI export not yet reliable (flaky SNR).",
                strict=False,
            )
        )

    _run_eager_mlir_export_test(
        model=simple_conv_linear_model,
        input_data=simple_model_input,
        parametrized_quant_config=parametrized_quant_config_mlir,
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
    """Test eager Core AI export on MNIST with various quantization configurations."""
    has_act_quant = parametrized_quant_config_general.has_activation_quantization

    # TODO: add model dtype support for mnist export test.
    _run_eager_mlir_export_test(
        model=custom_test_mnist_model,
        input_data=mnist_example_input,
        parametrized_quant_config=parametrized_quant_config_general,
        expected_ops={
            "constexpr_blockwise_shift_scale": 6,
            "quantize": 18 if has_act_quant else 0,
            "dequantize": 18 if has_act_quant else 0,
        },
    )


def test_resnet_export(
    resnet50_model: torch.nn.Module,
    resnet_example_input: torch.Tensor,
) -> None:
    """Test eager Core AI export on ResNet50 with default quantization configuration.

    Uses single default config instead of full parameter matrix to avoid excessive
    test execution time. Full parametrization coverage is provided by faster models
    (simple_conv_linear_model, custom_test_mnist_model).

    """
    config = QuantizerConfig(execution_mode="eager")
    _run_eager_mlir_export_test_ex(
        model=resnet50_model,
        input_data=resnet_example_input,
        config=config,
        expected_ops={
            "constexpr_blockwise_shift_scale": 54,
            "quantize": 160,
            "dequantize": 160,
        },
    )


def test_shared_param_model_export(shared_params_model, shared_params_model_input):
    model = shared_params_model
    model.eval()

    config = QuantizerConfig(execution_mode="eager")
    quantizer = Quantizer(model, config)
    prepared_model = quantizer.prepare((shared_params_model_input,))

    # Verify there is a shared instance of FakeQuant module for shared parameter layers
    assert prepared_model.layer1.parametrizations["weight"][0]
    assert (
        prepared_model.layer1.parametrizations["weight"][0]
        == prepared_model.layer2.parametrizations["weight"][0]
    )

    with torch.no_grad():
        prepared_model_output = prepared_model(shared_params_model_input)

    finalized_model = quantizer.finalize(backend=ExportBackend.CoreAI)

    # Verify that post finalize also the MLIR paremetrization module is shared
    assert (
        finalized_model.layer1.parametrizations["weight"][0]
        == finalized_model.layer2.parametrizations["weight"][0]
    )

    export_utils.convert_and_verify(
        finalized_model=finalized_model,
        input_data=shared_params_model_input,
        expected_ops={
            "constexpr_blockwise_shift_scale": 4,
            "quantize": 8,
            "dequantize": 8,
        },
        export_backend=ExportBackend.CoreAI,
        prepared_model_output=prepared_model_output,
    )


def test_duplicate_module_export():
    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.p1 = nn.Linear(20, 10)
            self.p2 = nn.Linear(20, 10)
            self.l = nn.Linear(10, 10)
            self.l1 = self.l

        def forward(self, x):
            x1 = self.l(self.p1(x))
            x2 = self.l1(self.p2(x))
            return torch.add(x1, x2)

    model = MyModule()
    model.eval()

    config = QuantizerConfig(execution_mode="eager")
    quantizer = Quantizer(model, config)
    input_data = torch.rand(1, 20)
    prepared_model = quantizer.prepare((input_data,))

    # Verify that post finalize same MLIR module is used for duplicated modules
    assert prepared_model.l.linear_quantize_input == prepared_model.l1.linear_quantize_input

    with torch.no_grad():
        prepared_model_output = prepared_model(input_data)

    finalized_model = quantizer.finalize(backend=ExportBackend.CoreAI)

    # Verify same FakeQuant module used for activation quantization
    # of duplicated modules
    assert finalized_model.l.linear_quantize_input == finalized_model.l1.linear_quantize_input

    export_utils.convert_and_verify(
        finalized_model=finalized_model,
        input_data=input_data,
        expected_ops={
            "constexpr_blockwise_shift_scale": 4,
            "quantize": 11,
            "dequantize": 11,
        },
        export_backend=ExportBackend.CoreAI,
        prepared_model_output=prepared_model_output,
    )


def test_fp8_simple_model_export(
    simple_conv_linear_model: torch.nn.Module,
    simple_model_input: torch.Tensor,
    parametrized_fp8_config: ParametrizedFP8Configs,
) -> None:
    """Test eager Core AI export with FP8 quantization.

    FP8 quantization requires symmetric scheme and per-tensor granularity.
    Tests both weight-only and weight+activation FP8 quantization.

    """
    _run_eager_mlir_export_test_ex(
        model=simple_conv_linear_model,
        input_data=simple_model_input,
        config=parametrized_fp8_config.eager,
        model_dtype=parametrized_fp8_config.model_dtype,
        expected_ops={
            "constexpr_blockwise_shift_scale": 2,
            "quantize": 4 if parametrized_fp8_config.with_activation_quant else 0,
            "dequantize": 4 if parametrized_fp8_config.with_activation_quant else 0,
        },
    )


def test_fp4_simple_model_export(
    simple_linear_model: torch.nn.Module,
    simple_linear_model_input: torch.Tensor,
    parametrized_fp4_config: ParametrizedFP4Configs,
) -> None:
    """Test eager MLIR export with FP4 quantization.

    FP4 quantization requires symmetric scheme and per-block granularity.
    Tests both weight-only and weight+activation FP4 quantization.
    """
    _run_eager_mlir_export_test_ex(
        model=simple_linear_model,
        input_data=simple_linear_model_input,
        config=parametrized_fp4_config.eager,
        model_dtype=parametrized_fp4_config.model_dtype,
        expected_ops={
            "constexpr_blockwise_shift_scale": 2,
            "quantize": 4 if parametrized_fp4_config.with_activation_quant else 0,
            "dequantize": 4 if parametrized_fp4_config.with_activation_quant else 0,
        },
    )


def test_gated_mlp_perchannel_act_export(
    gated_mlp_model: torch.nn.Module,
    gated_mlp_model_input: torch.Tensor,
    parametrized_quant_config_perchannel_act_axis_coverage: ParametrizedQuantConfigs,
) -> None:
    """Test eager Core AI export with per-channel activation quantization axes.
    Uses GatedMLPModel (uniform rank-3 activations throughout the model) to
    test per-channel activation quantization across all valid axis values without
    out-of-bounds errors.
    """
    has_act_quant = (
        parametrized_quant_config_perchannel_act_axis_coverage.has_activation_quantization
    )

    _run_eager_mlir_export_test(
        model=gated_mlp_model,
        input_data=gated_mlp_model_input,
        parametrized_quant_config=parametrized_quant_config_perchannel_act_axis_coverage,
        expected_ops={
            "constexpr_blockwise_shift_scale": 3,
            "quantize": 9 if has_act_quant else 0,
            "dequantize": 9 if has_act_quant else 0,
        },
    )


@pytest.mark.parametrize(
    "qscheme",
    [QuantizationScheme.SYMMETRIC, QuantizationScheme.ASYMMETRIC],
    ids=["symmetric", "asymmetric"],
)
@pytest.mark.parametrize(
    "has_activation_quant",
    [False, True],
    ids=["weight_only", "weight_and_activation"],
)
@pytest.mark.parametrize(
    "weight_dtype",
    [torch.int4, torch.int8],
    ids=["4bit_weight", "8bit_weight"],
)
def test_integer_quant_minval_export(
    simple_conv_linear_model: torch.nn.Module,
    simple_model_input: torch.Tensor,
    qscheme: QuantizationScheme,
    has_activation_quant: bool,
    weight_dtype: torch.dtype,
) -> None:
    """Eager MLIR export with MINVAL integer quantization, end-to-end."""
    weight_spec = QuantizationSpec(
        dtype=weight_dtype,
        qscheme=qscheme,
        qformulation=QuantizationFormulation.MINVAL,
        granularity=PerChannelGranularity(axis=0),
    )
    if has_activation_quant:
        activation_spec = QuantizationSpec(
            dtype=torch.int8,
            qscheme=qscheme,
            qformulation=QuantizationFormulation.MINVAL,
            granularity=PerTensorGranularity(),
        )
        op_input_spec = {"*": activation_spec}
        op_output_spec = {"*": activation_spec}
    else:
        op_input_spec = None
        op_output_spec = None

    config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={"weight": weight_spec},
            op_input_spec=op_input_spec,
            op_output_spec=op_output_spec,
        ),
        execution_mode="eager",
    )

    _run_eager_mlir_export_test_ex(
        model=simple_conv_linear_model,
        input_data=simple_model_input,
        config=config,
        expected_ops={
            "constexpr_blockwise_shift_scale": 2,
            "quantize": 4 if has_activation_quant else 0,
            "dequantize": 4 if has_activation_quant else 0,
        },
    )
