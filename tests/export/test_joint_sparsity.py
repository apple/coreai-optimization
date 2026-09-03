# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""End-to-end export tests for joint post-training quantization/palettization + sparsity."""

import pytest
import torch
import torch.nn as nn

from coreai_opt import ExportBackend
from coreai_opt.palettization import (
    KMeansPalettizer,
    KMeansPalettizerConfig,
    ModuleKMeansPalettizerConfig,
    PalettizationSpec,
)
from coreai_opt.quantization import ModuleQuantizerConfig, Quantizer, QuantizerConfig
from coreai_opt.quantization.config import ExecutionMode
from coreai_opt.quantization.spec import PerTensorGranularity, QuantizationScheme, QuantizationSpec

from . import export_utils


class TestJointSparsityExport:
    """PTQ/PTP + PTS (post-training quantization/palettization + sparsity), end to end."""

    @staticmethod
    def _run_quant_sparsity_export(
        model: nn.Module, input_data: torch.Tensor, expected_count: int
    ) -> None:
        model.eval()
        config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec={
                    "weight": QuantizationSpec(
                        dtype=torch.int8,
                        qscheme=QuantizationScheme.SYMMETRIC,
                        granularity=PerTensorGranularity(),
                        _sparsity=0.5,
                    )
                },
                op_input_spec=None,
                op_output_spec=None,
            ),
            execution_mode=ExecutionMode.GRAPH,
        )

        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare((input_data,))

        with torch.no_grad():
            prepared_model_output = prepared_model(input_data)

        finalized_model = quantizer.finalize(backend=ExportBackend.CoreAI)

        export_utils.convert_and_verify(
            finalized_model=finalized_model,
            input_data=input_data,
            expected_ops={
                "sparse_to_dense": expected_count,
                "constexpr_blockwise_shift_scale": expected_count,
            },
            export_backend=ExportBackend.CoreAI,
            prepared_model_output=prepared_model_output,
        )

    @staticmethod
    def _run_palettization_sparsity_export(
        model: nn.Module, input_data: torch.Tensor, expected_count: int
    ) -> None:
        model.eval()
        config = KMeansPalettizerConfig(
            global_config=ModuleKMeansPalettizerConfig(
                op_state_spec={"weight": PalettizationSpec(n_bits=8, _sparsity=0.5)}
            )
        )

        palettizer = KMeansPalettizer(model, config)
        prepared_model = palettizer.prepare((input_data,))

        with torch.no_grad():
            prepared_model_output = prepared_model(input_data)

        finalized_model = palettizer.finalize(backend=ExportBackend.CoreAI)

        export_utils.convert_and_verify(
            finalized_model=finalized_model,
            input_data=input_data,
            expected_ops={
                "lut_to_dense": expected_count,
                "sparse_to_dense": expected_count,
            },
            export_backend=ExportBackend.CoreAI,
            prepared_model_output=prepared_model_output,
        )

    def test_quant_sparsity_mnist_export(self, custom_test_mnist_model, mnist_example_input):
        self._run_quant_sparsity_export(
            custom_test_mnist_model, mnist_example_input, expected_count=6
        )

    @pytest.mark.slow
    def test_quant_sparsity_resnet_export(self, resnet50_model, resnet_example_input):
        self._run_quant_sparsity_export(resnet50_model, resnet_example_input, expected_count=54)

    def test_palettization_sparsity_mnist_export(
        self, custom_test_mnist_model, mnist_example_input
    ):
        self._run_palettization_sparsity_export(
            custom_test_mnist_model, mnist_example_input, expected_count=6
        )

    @pytest.mark.slow
    def test_palettization_sparsity_resnet_export(self, resnet50_model, resnet_example_input):
        self._run_palettization_sparsity_export(
            resnet50_model, resnet_example_input, expected_count=54
        )
