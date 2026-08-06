# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for PT2E weight-only quantization export to CoreML backend."""

# TODO: enable activation quantization for CoreML export and delete this file.
#       These tests exist only to verify weight-only quantization works with
#       CoreML; once activation quantization is supported, remove this file and
#       use the full quantization tests in test_pt2e_export.py instead.

import pytest
import torch

from coreai_opt import ExportBackend
from coreai_opt.quantization import ModuleQuantizerConfig, Quantizer, QuantizerConfig
from coreai_opt.quantization.spec import (
    PerBlockGranularity,
    PerChannelGranularity,
    PerTensorGranularity,
    QuantizationScheme,
    QuantizationSpec,
)

from . import export_utils


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
@pytest.mark.parametrize("qscheme", [QuantizationScheme.SYMMETRIC])
@pytest.mark.parametrize(
    "granularity",
    [
        PerTensorGranularity(),
        PerChannelGranularity(axis=0),
        PerBlockGranularity(axis=0, block_size=2),
    ],
)
@pytest.mark.parametrize("backend", [ExportBackend.CoreML])
def test_simple_model_weight_only_mil_export(
    simple_conv_linear_model: torch.nn.Module,
    simple_model_input: torch.Tensor,
    dtype: str,
    qscheme: QuantizationScheme,
    granularity: PerTensorGranularity | PerChannelGranularity,
    backend: ExportBackend,
) -> None:
    """Test weight-only quantization export to CoreML backend."""
    model = simple_conv_linear_model
    model.eval()

    config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={
                "weight": QuantizationSpec(
                    dtype=dtype,
                    qscheme=qscheme,
                    granularity=granularity,
                    fake_quantize_cls="default",
                    qparam_calculator_cls="default",
                    range_calculator_cls="minmax",
                ),
            },
            op_input_spec=None,
            op_output_spec=None,
        ),
    )

    quantizer = Quantizer(model, config)
    prepared_model = quantizer.prepare((simple_model_input,))

    with torch.no_grad():
        prepared_model_output = prepared_model(simple_model_input)

    finalized_model = quantizer.finalize(backend=backend)

    export_utils.convert_and_verify(
        finalized_model=finalized_model,
        input_data=simple_model_input,
        expected_ops={
            "constexpr_blockwise_shift_scale": 2,
        },
        export_backend=backend,
        prepared_model_output=prepared_model_output,
    )


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
@pytest.mark.parametrize(
    "granularity",
    [
        PerTensorGranularity(),
        PerChannelGranularity(axis=0),
        PerBlockGranularity(axis=0, block_size=2),
    ],
)
def test_root_module_weight_only_mil_export(
    dtype: str,
    granularity: PerTensorGranularity | PerChannelGranularity | PerBlockGranularity,
) -> None:
    """Test weight-only quantization export to CoreML for a root-module weight.

    A bare ``nn.Linear`` owns its weight directly, so the fake-quant input target is the
    dot-less ``"weight"`` rather than ``"<submodule>.weight"``. The graph CoreML export
    path rejected that with ``Invalid weight target path: weight`` instead of resolving it
    to the root module.
    """
    model = torch.nn.Linear(8, 4)
    model.eval()
    model_input = torch.randn(2, 8)

    config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={
                "weight": QuantizationSpec(
                    dtype=dtype,
                    qscheme=QuantizationScheme.SYMMETRIC,
                    granularity=granularity,
                    fake_quantize_cls="default",
                    qparam_calculator_cls="default",
                    range_calculator_cls="minmax",
                ),
            },
            op_input_spec=None,
            op_output_spec=None,
        ),
    )

    quantizer = Quantizer(model, config)
    prepared_model = quantizer.prepare((model_input,))

    with torch.no_grad():
        prepared_model_output = prepared_model(model_input)

    finalized_model = quantizer.finalize(backend=ExportBackend.CoreML)

    export_utils.convert_and_verify(
        finalized_model=finalized_model,
        input_data=model_input,
        expected_ops={
            "constexpr_blockwise_shift_scale": 1,
        },
        export_backend=ExportBackend.CoreML,
        prepared_model_output=prepared_model_output,
    )
