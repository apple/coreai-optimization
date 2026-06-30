# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for graph-mode quantizer export to Core AI backend."""

import sys
from collections.abc import Mapping

import pytest
import torch

from coreai_opt import ExportBackend
from coreai_opt.palettization.kmeans import KMeansPalettizer
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
from tests.fixtures.compression import ParametrizedP4A8CompressionConfigs
from tests.fixtures.fp4 import ParametrizedFP4Configs
from tests.fixtures.fp8 import ParametrizedFP8Configs
from tests.fixtures.quantization import ParametrizedQuantConfigs

from . import export_utils


def _run_graph_mode_mlir_export_test_ex(
    model: torch.nn.Module,
    input_data: torch.Tensor,
    config: QuantizerConfig,
    expected_ops: Mapping[str, int],
    model_dtype: torch.dtype | None = None,
) -> None:
    """Run graph-mode Core AI export test with expanded configuration parameters.

    Args:
        model: PyTorch model to quantize and export
        input_data: Input tensor for model
        config: graph-mode quantization configuration
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


def _run_graph_mode_mlir_export_test(
    model: torch.nn.Module,
    input_data: torch.Tensor,
    parametrized_quant_config: ParametrizedQuantConfigs,
    expected_ops: Mapping[str, int],
) -> None:
    """Run graph-mode Core AI export test with parametrized configuration.

    Wrapper around _run_graph_mode_mlir_export_test_ex that extracts model_dtype
    from the parametrized config object.

    Args:
        model: PyTorch model to quantize and export
        input_data: Input tensor for model
        parametrized_quant_config: Parametrized quantization configurations
        expected_ops: Expected operation counts in converted model

    """
    _run_graph_mode_mlir_export_test_ex(
        model=model,
        input_data=input_data,
        config=parametrized_quant_config.pt2e,
        model_dtype=parametrized_quant_config.model_dtype,
        expected_ops=expected_ops,
    )


def _skip_heavy_mnist_configs(config: ParametrizedQuantConfigs) -> None:
    """Skip configs redundant with simple-model coverage to bound mnist memory.

    test_simple_model_export already exercises the full 756-config matrix on a
    smaller model. On mnist (Conv2d + ConvTranspose2d + BatchNorm + MaxPool),
    keep only the axes that interact with model topology -- qscheme,
    w_granularity, act_granularity -- and pin the model-agnostic numerical /
    dtype axes to a single representative value. Reduces 756 -> 36 effective
    configs and ~37 GB of accumulation-bucket memory at -n 8 baseline
    .
    """
    if config.model_dtype != torch.float32:
        pytest.skip(f"MNIST pins model_dtype=float32, got {config.model_dtype}")

    weight_qspec = config.pt2e.global_config.op_state_spec["weight"]
    if weight_qspec.dtype != torch.int8:
        pytest.skip(f"MNIST pins weight_dtype=int8, got {weight_qspec.dtype}")

    act_qspec = config.pt2e.global_config.op_input_spec["*"]
    if act_qspec is not None and act_qspec.dtype != torch.int8:
        pytest.skip(f"MNIST tests act_dtype in {{int8, None}}, got {act_qspec.dtype}")


@pytest.mark.slow
def test_simple_model_export(
    simple_conv_linear_model: torch.nn.Module,
    simple_model_input: torch.Tensor,
    parametrized_quant_config_mlir: ParametrizedQuantConfigs,
    request: pytest.FixtureRequest,
) -> None:
    """Test graph-mode Core AI export with various quantization configurations."""
    has_act_quant = parametrized_quant_config_mlir.has_activation_quantization

    # 4-bit-weight and int8 weight+activation per-tensor bfloat16 configs abort the
    # CoreAI interpreter (SIGABRT); xfail them without running so the native crash
    # cannot abort the session.
    parametrized_quant_config_mlir.xfail_if_unsupported(
        "graph",
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

    _run_graph_mode_mlir_export_test(
        model=simple_conv_linear_model,
        input_data=simple_model_input,
        parametrized_quant_config=parametrized_quant_config_mlir,
        expected_ops={
            "constexpr_blockwise_shift_scale": 2,
            "quantize": 4 if has_act_quant else 0,
            "dequantize": 4 if has_act_quant else 0,
        },
    )


@pytest.mark.slow
def test_mnist_export(
    custom_test_mnist_model: torch.nn.Module,
    mnist_example_input: torch.Tensor,
    parametrized_quant_config_mlir: ParametrizedQuantConfigs,
) -> None:
    """Test graph-mode Core AI export on MNIST with various quantization configurations."""
    _skip_heavy_mnist_configs(parametrized_quant_config_mlir)

    has_act_quant = parametrized_quant_config_mlir.has_activation_quantization

    # Per-channel activation axis=-1 causes shape mismatch during quantization
    parametrized_quant_config_mlir.skip_if_unsupported(
        "graph",
        ExportBackend.CoreAI,
        unsupported_configs={"act_granularity_axis": -1},
        reason="RuntimeError: tensor size mismatch at "
        "non-singleton dimension during per-channel activation quantization "
        "with pooling layers",
    )

    _run_graph_mode_mlir_export_test(
        model=custom_test_mnist_model,
        input_data=mnist_example_input,
        parametrized_quant_config=parametrized_quant_config_mlir,
        expected_ops={
            "constexpr_blockwise_shift_scale": 6,
            "quantize": 12 if has_act_quant else 0,
            "dequantize": 12 if has_act_quant else 0,
        },
    )


@pytest.mark.slow
@pytest.mark.parametrize("config", [QuantizerConfig()], ids=["default-config"])
@pytest.mark.xfail(
    sys.platform == "linux",
    reason=(
        "[Export Tests] ResNet50 graph-mode int8 quantization"
        " SNR falls below threshold on Linux (19.44 < 20.0)"
    ),
    strict=False,
)
def test_resnet_export(
    resnet50_model: torch.nn.Module,
    resnet_example_input: torch.Tensor,
    config: QuantizerConfig,
) -> None:
    """Test graph-mode Core AI export on ResNet50 with default quantization configuration.

    Uses single default config instead of full parameter matrix to avoid excessive
    test execution time. Full parametrization coverage is provided by faster models
    (simple_conv_linear_model, custom_test_mnist_model).

    After conv+bn folding, all 53 conv+bn pairs are folded and quantized as weights.
    ResNet50 has 53 conv layers with BN + 1 fc layer = 54 total weight quantizations.

    """
    _run_graph_mode_mlir_export_test_ex(
        model=resnet50_model,
        input_data=resnet_example_input,
        config=config,
        model_dtype=torch.float32,
        expected_ops={
            "constexpr_blockwise_shift_scale": 54,
            "quantize": 74,
            "dequantize": 74,
        },
    )


def test_fp8_simple_model_export(
    simple_conv_linear_model: torch.nn.Module,
    simple_model_input: torch.Tensor,
    parametrized_fp8_config: ParametrizedFP8Configs,
) -> None:
    """Test graph-mode Core AI export with FP8 quantization.

    FP8 quantization requires symmetric scheme and per-tensor granularity.
    Tests both weight-only and weight+activation FP8 quantization.

    """
    _run_graph_mode_mlir_export_test_ex(
        model=simple_conv_linear_model,
        input_data=simple_model_input,
        config=parametrized_fp8_config.pt2e,
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
    """Test graph-mode MLIR export with FP4 quantization.

    FP4 quantization requires symmetric scheme and per-block granularity.
    Tests both weight-only and weight+activation FP4 quantization.
    """
    _run_graph_mode_mlir_export_test_ex(
        model=simple_linear_model,
        input_data=simple_linear_model_input,
        config=parametrized_fp4_config.pt2e,
        model_dtype=parametrized_fp4_config.model_dtype,
        expected_ops={
            "constexpr_blockwise_shift_scale": 2,
            "quantize": 3 if parametrized_fp4_config.with_activation_quant else 0,
            "dequantize": 3 if parametrized_fp4_config.with_activation_quant else 0,
        },
    )


@pytest.mark.slow
def test_gated_mlp_perchannel_act_export(
    gated_mlp_model: torch.nn.Module,
    gated_mlp_model_input: torch.Tensor,
    parametrized_quant_config_perchannel_act_axis_coverage: ParametrizedQuantConfigs,
) -> None:
    """Test graph-mode Core AI export with per-channel activation quantization axes.
    Uses GatedMLPModel (uniform rank-3 activations throughout the model)
    to test per-channel activation quantization across all valid axis
    values without out-of-bounds errors.
    """
    has_act_quant = (
        parametrized_quant_config_perchannel_act_axis_coverage.has_activation_quantization
    )

    _run_graph_mode_mlir_export_test(
        model=gated_mlp_model,
        input_data=gated_mlp_model_input,
        parametrized_quant_config=parametrized_quant_config_perchannel_act_axis_coverage,
        expected_ops={
            "constexpr_blockwise_shift_scale": 3,
            "quantize": 5 if has_act_quant else 0,
            "dequantize": 5 if has_act_quant else 0,
        },
    )


def _run_p4a8_compression_export_test(
    model: torch.nn.Module,
    input_data: torch.Tensor,
    config: ParametrizedP4A8CompressionConfigs,
    expected_ops: Mapping[str, int],
) -> None:
    """Run P4-A8 compression (palettize then quantize) MLIR export test.

    Steps:
    1. Palettize model with the provided palettization config.
    2. Apply activation-only quantization on the palettized model.
    3. Export to MLIR and verify op counts and SNR/PSNR.

    Args:
        model (torch.nn.Module): PyTorch model to compress and export.
        input_data (torch.Tensor): Input tensor for model.
        config (ParametrizedP4A8CompressionConfigs): P4-A8 compression config.
        expected_ops (Mapping[str, int]): Expected operation counts.

    """
    model.eval()

    # Palettize
    palettizer = KMeansPalettizer(model, config.palett_config)
    palettizer.prepare((input_data,))
    palettized = palettizer.finalize(backend=ExportBackend.MLIR)

    # Quantize activations
    quantizer = Quantizer(palettized, config.quant_config)
    prepared = quantizer.prepare((input_data,))

    with torch.no_grad():
        prepared_model_output = prepared(input_data)

    joint_model = quantizer.finalize(backend=ExportBackend.MLIR)

    # Export and verify
    export_utils.convert_and_verify(
        finalized_model=joint_model,
        input_data=input_data,
        expected_ops=expected_ops,
        export_backend=ExportBackend.MLIR,
        prepared_model_output=prepared_model_output,
    )


def test_mnist_p4a8_compression_export(
    custom_test_mnist_model: torch.nn.Module,
    mnist_example_input: torch.Tensor,
    parametrized_p4a8_compression_config: ParametrizedP4A8CompressionConfigs,
) -> None:
    """Test P4-A8 compression export on MNIST model.

    Verifies that 4-bit palettized weights with int8 activation quantization
    export correctly to MLIR. Checks lut_to_dense ops (weight decompression),
    constexpr_blockwise_shift_scale ops (LUT dequantization, only when LUT is
    quantized), and quantize/dequantize ops (activation quantization).
    """
    has_lut = parametrized_p4a8_compression_config.has_lut_quantization
    _run_p4a8_compression_export_test(
        model=custom_test_mnist_model,
        input_data=mnist_example_input,
        config=parametrized_p4a8_compression_config,
        expected_ops={
            "lut_to_dense": 6,
            "constexpr_blockwise_shift_scale": 6 if has_lut else 0,
            "quantize": 12,
            "dequantize": 12,
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
    """Graph-mode MLIR export with MINVAL integer quantization, end-to-end."""
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
        execution_mode="graph",
    )

    _run_graph_mode_mlir_export_test_ex(
        model=simple_conv_linear_model,
        input_data=simple_model_input,
        config=config,
        expected_ops={
            "constexpr_blockwise_shift_scale": 2,
            "quantize": 4 if has_activation_quant else 0,
            "dequantize": 4 if has_activation_quant else 0,
        },
    )
