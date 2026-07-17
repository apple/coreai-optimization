# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import pytest
import torch

from coreai_opt import ExportBackend
from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig
from tests.fixtures.palettization import ParametrizedPalettConfigs

from . import export_utils


def _run_kmeans_export_test(
    model: torch.nn.Module,
    input_data: torch.Tensor,
    config: KMeansPalettizerConfig,
    backend: ExportBackend,
    expected_count: int,
    skip_finalized_model_verify: bool = False,
) -> None:
    """Run KMeans palettization export test.

    Args:
        model: PyTorch model to palettize and export
        input_data: Input tensor for model
        config: KMeans palettization configuration
        backend: Export backend (CoreML or CoreAI)
        expected_count: Expected number of lut_to_dense ops in converted model
        skip_finalized_model_verify: If True, skip forward pass verification on
            the finalized model

    """
    model.eval()
    palettizer = KMeansPalettizer(model, config)
    prepared_model = palettizer.prepare((input_data,))

    with torch.no_grad():
        prepared_model_output = prepared_model(input_data)

    finalized_model = palettizer.finalize(backend=backend)

    expected_op_name = (
        "constexpr_lut_to_dense" if backend == ExportBackend.CoreML else "lut_to_dense"
    )

    export_utils.convert_and_verify(
        finalized_model=finalized_model,
        input_data=input_data,
        expected_ops={
            expected_op_name: expected_count,
        },
        export_backend=backend,
        prepared_model_output=prepared_model_output,
        skip_finalized_model_verify=skip_finalized_model_verify,
    )


def _has_float_lut(config: ParametrizedPalettConfigs) -> bool:
    """Whether the config uses a floating-point LUT dtype."""
    return config.lut_qspec is not None and config.lut_qspec.dtype.is_floating_point


def _assert_coreml_rejects_unsupported_lut(
    model: torch.nn.Module,
    input_data: torch.Tensor,
    config: KMeansPalettizerConfig,
) -> None:
    """Assert finalize(CoreML) rejects an unsupported LUT dtype.

    CoreML/MIL does not support FP or INT2 LUT quantization, so finalize must raise
    rather than emit an invalid model.
    """
    model.eval()
    palettizer = KMeansPalettizer(model, config)
    palettizer.prepare((input_data,))
    export_utils.assert_coreml_finalize_rejects(palettizer)


def _skip_heavy_mnist_configs(config: ParametrizedPalettConfigs) -> None:
    """Skip configs not needed for MNIST to reduce test matrix.

    MNIST tests run a subset of the full config space: n_bits=4 only, and at most
    one integer and one floating-point LUT dtype (plus None).
    """
    if config.n_bits != 4:
        pytest.skip(f"MNIST only tests n_bits=4, got {config.n_bits}")

    if config.lut_qspec is not None:
        dtype = config.lut_qspec.dtype
        # Keep one representative int dtype (int8) and one float dtype (float8_e4m3fn)
        if dtype not in (torch.int8, torch.float8_e4m3fn):
            pytest.skip(f"MNIST only tests lut_qspec with int8 and float8_e4m3fn, got {dtype}")


def _skip_unsupported_mil_configs(
    backend: ExportBackend,
    config: ParametrizedPalettConfigs,
) -> None:
    """Skip CoreML configs with unsupported feature combinations."""
    if backend != ExportBackend.CoreML:
        return

    is_vector = config.cluster_dim > 1
    has_lut_quant = config.lut_qspec is not None
    has_pcs = config.enable_per_channel_scale

    # Vector palettization + LUT quantization
    if is_vector and has_lut_quant:
        # TODO: add CoreML export support for palettization combos.
        pytest.skip("CoreML export not supported for vector palettization + LUT quantization.")

    # Vector palettization + per-channel scale
    if is_vector and has_pcs:
        # TODO: add CoreML export support for palettization combos.
        pytest.skip("CoreML export not supported for vector palettization + per-channel scale.")

    # LUT quantization + per-channel scale
    if has_lut_quant and has_pcs:
        # TODO: add CoreML export support for palettization combos.
        pytest.skip("CoreML export not supported for LUT quantization + per-channel scale.")


@pytest.mark.parametrize("backend", [ExportBackend.CoreML, ExportBackend.CoreAI])
def test_simple_model_export(
    simple_conv_linear_model: torch.nn.Module,
    simple_model_input: torch.Tensor,
    parametrized_palett_config: ParametrizedPalettConfigs,
    backend: ExportBackend,
) -> None:
    """Test KMeans palettization export with various configurations."""
    config = parametrized_palett_config.config
    granularity = parametrized_palett_config.granularity

    if backend == ExportBackend.CoreML and _has_float_lut(parametrized_palett_config):
        _assert_coreml_rejects_unsupported_lut(simple_conv_linear_model, simple_model_input, config)
        return

    _skip_unsupported_mil_configs(backend, parametrized_palett_config)

    if (
        backend == ExportBackend.CoreML
        and parametrized_palett_config.cluster_dim > 1
        and parametrized_palett_config.n_bits == 2
        and hasattr(granularity, "axis")
        and granularity.axis == 1
    ):
        # TODO: fix low SNR for 2-bit vector palettization with axis=1 grouped-channel granularity.
        pytest.skip(reason="CoreML: 2-bit vector palettization + axis=1 produces SNR mismatch.")

    # For axis = 1, group_size is not divisible for conv layer
    expected_count = 1 if granularity.axis == 1 else 2

    _run_kmeans_export_test(
        model=simple_conv_linear_model,
        input_data=simple_model_input,
        config=config,
        backend=backend,
        expected_count=expected_count,
    )


@pytest.mark.parametrize("backend", [ExportBackend.CoreML, ExportBackend.CoreAI])
def test_mnist_export(
    custom_test_mnist_model: torch.nn.Module,
    mnist_example_input: torch.Tensor,
    parametrized_palett_config: ParametrizedPalettConfigs,
    backend: ExportBackend,
) -> None:
    """Test KMeans palettization export on MNIST with various configurations."""
    _skip_heavy_mnist_configs(parametrized_palett_config)

    config = parametrized_palett_config.config
    granularity = parametrized_palett_config.granularity

    if backend == ExportBackend.CoreML and _has_float_lut(parametrized_palett_config):
        _assert_coreml_rejects_unsupported_lut(custom_test_mnist_model, mnist_example_input, config)
        return

    _skip_unsupported_mil_configs(backend, parametrized_palett_config)

    # The MNIST model has 6 weight-bearing layers (conv1, conv2, conv_transpose1,
    # conv_transpose2, dense1, dense2). For axis=1 with group_size=2, conv1's
    # axis-1 (in_channels=1) is not divisible, so palettization is skipped there.
    expected_count = 5 if granularity.axis == 1 else 6

    _run_kmeans_export_test(
        model=custom_test_mnist_model,
        input_data=mnist_example_input,
        config=config,
        backend=backend,
        expected_count=expected_count,
    )


@pytest.mark.parametrize("backend", [ExportBackend.CoreML, ExportBackend.CoreAI])
def test_resnet_export(
    resnet50_model: torch.nn.Module,
    resnet_example_input: torch.Tensor,
    backend: ExportBackend,
) -> None:
    """Test KMeans palettization export on ResNet50 with default configuration.

    Uses default config instead of full parameter matrix to avoid excessive
    test execution time. Full parametrization coverage is provided by faster models
    (simple_conv_linear_model, custom_test_mnist_model).

    """
    _run_kmeans_export_test(
        model=resnet50_model,
        input_data=resnet_example_input,
        config=KMeansPalettizerConfig(),  # default config
        backend=backend,
        expected_count=54,  # conv, linear
    )
