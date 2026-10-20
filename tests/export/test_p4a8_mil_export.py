# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from collections.abc import Callable

import pytest
import torch
from torch import nn

from coreai_opt import ExportBackend
from coreai_opt.palettization import KMeansPalettizer
from coreai_opt.quantization import ExecutionMode, Quantizer
from coreai_opt.quantization.spec import QuantizationScheme, QuantizationSpec
from tests.fixtures.compression import ParametrizedP4A8CompressionConfigs

from . import export_utils

_INT8_LUT_QSPEC = QuantizationSpec(dtype=torch.int8, qscheme=QuantizationScheme.SYMMETRIC)

_LAYERS = pytest.mark.parametrize(
    ("make_layer", "input_shape"),
    [
        (lambda: nn.Linear(64, 32), (4, 64)),
        (lambda: nn.Conv2d(16, 32, kernel_size=3), (1, 16, 8, 8)),
    ],
    ids=["linear", "conv"],
)
_EXECUTION_MODES = pytest.mark.parametrize(
    "execution_mode", [ExecutionMode.GRAPH, ExecutionMode.EAGER]
)


def _palettize_then_quantize(
    make_layer: Callable[[], nn.Module],
    input_shape: tuple[int, ...],
    config: ParametrizedP4A8CompressionConfigs,
    execution_mode: ExecutionMode,
) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    """Palettize a single layer, then quantize its activations, finalizing both for CoreML.

    Returns:
        tuple[nn.Module, torch.Tensor, torch.Tensor]: The finalized model, the input data,
            and the prepared (fake-quantized) model's output on that input.
    """
    torch.manual_seed(0)
    model = nn.Sequential(make_layer()).eval()
    input_data = torch.randn(input_shape)

    palettizer = KMeansPalettizer(model, config.palett_config)
    palettizer.prepare((input_data,))
    palettized = palettizer.finalize(backend=ExportBackend.CoreML)

    quantizer = Quantizer(palettized, config.quant_config.set_execution_mode(execution_mode))
    prepared = quantizer.prepare((input_data,))

    with torch.no_grad():
        prepared_model_output = prepared(input_data)

    return quantizer.finalize(backend=ExportBackend.CoreML), input_data, prepared_model_output


@_EXECUTION_MODES
@pytest.mark.parametrize("lut_qspec", [_INT8_LUT_QSPEC, None], ids=["int8lut", "nolut"])
@pytest.mark.parametrize("n_bits", [4, 8], ids=["P4", "P8"])
@_LAYERS
def test_palettization_then_act_quant_mil_export(
    make_layer: Callable[[], nn.Module],
    input_shape: tuple[int, ...],
    n_bits: int,
    lut_qspec: QuantizationSpec | None,
    execution_mode: ExecutionMode,
) -> None:
    """Palettize a single layer, then quantize its activations, then export to CoreML.

    The palettization compression metadata registered by the first finalize must
    survive the activation quantization pass so the exported model still contains
    the compressed weight. 8-bit palettization with an int8 LUT is exported as int8 weight
    quantization instead of a LUT, in both execution modes.
    """
    config = ParametrizedP4A8CompressionConfigs.from_params(lut_qspec=lut_qspec, n_bits=n_bits)
    joint_model, input_data, prepared_model_output = _palettize_then_quantize(
        make_layer, input_shape, config, execution_mode
    )

    folds_to_weight_quantization = n_bits == 8 and lut_qspec is not None
    export_utils.convert_and_verify(
        finalized_model=joint_model,
        input_data=input_data,
        expected_ops={
            "constexpr_lut_to_dense": 0 if folds_to_weight_quantization else 1,
            "constexpr_blockwise_shift_scale": 1 if lut_qspec is not None else 0,
            "quantize": 2,
            "dequantize": 2,
        },
        export_backend=ExportBackend.CoreML,
        prepared_model_output=prepared_model_output,
    )
