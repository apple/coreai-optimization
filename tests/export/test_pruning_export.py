# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for pruning export (CoreAI/MLIR and CoreML/MIL)."""

import pytest
import torch
import torch.nn as nn

from coreai_opt import ExportBackend
from coreai_opt.pruning import MagnitudePruner, MagnitudePrunerConfig
from coreai_opt.pruning.spec import ChannelStructured
from tests.fixtures.pruning import ParametrizedPruneConfigs

from . import export_utils


def _run_pruning_export_test(
    model: nn.Module,
    input_data: torch.Tensor,
    config: MagnitudePrunerConfig,
    backend: ExportBackend,
    expected_count: int,
    parametrized_prune_config: ParametrizedPruneConfigs,
) -> None:
    """Run pruning export test for a given backend.

    Args:
        model (nn.Module): PyTorch model to prune and export.
        input_data (torch.Tensor): Input tensor for the model.
        config (MagnitudePrunerConfig): Pruning configuration.
        backend (ExportBackend): Export backend (CoreAI or CoreML).
        expected_count (int): Expected number of sparse ops in converted model.
        parametrized_prune_config (ParametrizedPruneConfigs): Parametrized config.
    """
    if isinstance(parametrized_prune_config.pruning_scheme, ChannelStructured):
        # TODO: enable channel-structured pruning export
        pytest.skip("Channel structured pruning export not yet supported")

    model.eval()
    pruner = MagnitudePruner(model, config)
    prepared_model = pruner.prepare((input_data,))

    with torch.no_grad():
        prepared_model_output = prepared_model(input_data)

    finalized_model = pruner.finalize(backend=backend)

    expected_op_name = (
        "constexpr_sparse_to_dense" if backend == ExportBackend.CoreML else "sparse_to_dense"
    )

    export_utils.convert_and_verify(
        finalized_model=finalized_model,
        input_data=input_data,
        expected_ops={expected_op_name: expected_count},
        export_backend=backend,
        prepared_model_output=prepared_model_output,
    )


def test_simple_model_export(
    simple_conv_linear_model: nn.Module,
    simple_model_input: torch.Tensor,
    parametrized_prune_config: ParametrizedPruneConfigs,
) -> None:
    """Test pruning export on simple conv+linear model with various configs."""
    _run_pruning_export_test(
        model=simple_conv_linear_model,
        input_data=simple_model_input,
        config=parametrized_prune_config.config,
        backend=parametrized_prune_config.backend,
        expected_count=2,
        parametrized_prune_config=parametrized_prune_config,
    )


def test_mnist_export(
    custom_test_mnist_model: nn.Module,
    mnist_example_input: torch.Tensor,
    parametrized_prune_config: ParametrizedPruneConfigs,
) -> None:
    """Test pruning export on MNIST model with various configs."""
    _run_pruning_export_test(
        model=custom_test_mnist_model,
        input_data=mnist_example_input,
        config=parametrized_prune_config.config,
        backend=parametrized_prune_config.backend,
        expected_count=6,
        parametrized_prune_config=parametrized_prune_config,
    )


def test_resnet_export(
    resnet50_model: nn.Module,
    resnet_example_input: torch.Tensor,
    parametrized_prune_config: ParametrizedPruneConfigs,
) -> None:
    """Test pruning export on ResNet50 with default config."""
    _run_pruning_export_test(
        model=resnet50_model,
        input_data=resnet_example_input,
        config=parametrized_prune_config.config,
        backend=parametrized_prune_config.backend,
        expected_count=54,
        parametrized_prune_config=parametrized_prune_config,
    )
