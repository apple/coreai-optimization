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


@pytest.mark.parametrize("execution_mode", [ExecutionMode.GRAPH, ExecutionMode.EAGER])
@pytest.mark.parametrize(
    "lut_qspec",
    [QuantizationSpec(dtype=torch.int8, qscheme=QuantizationScheme.SYMMETRIC), None],
    ids=["int8lut", "nolut"],
)
@pytest.mark.parametrize("n_bits", [4, 8], ids=["P4", "P8"])
@pytest.mark.parametrize(
    ("make_layer", "input_shape"),
    [
        (lambda: nn.Linear(64, 32), (4, 64)),
        (lambda: nn.Conv2d(16, 32, kernel_size=3), (1, 16, 8, 8)),
    ],
    ids=["linear", "conv"],
)
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
    the palettized weight.
    """
    torch.manual_seed(0)
    model = nn.Sequential(make_layer()).eval()
    input_data = torch.randn(input_shape)
    config = ParametrizedP4A8CompressionConfigs.from_params(lut_qspec=lut_qspec, n_bits=n_bits)

    palettizer = KMeansPalettizer(model, config.palett_config)
    palettizer.prepare((input_data,))
    palettized = palettizer.finalize(backend=ExportBackend.CoreML)

    quantizer = Quantizer(palettized, config.quant_config.set_execution_mode(execution_mode))
    prepared = quantizer.prepare((input_data,))

    with torch.no_grad():
        prepared_model_output = prepared(input_data)

    joint_model = quantizer.finalize(backend=ExportBackend.CoreML)

    export_utils.convert_and_verify(
        finalized_model=joint_model,
        input_data=input_data,
        expected_ops={
            "constexpr_lut_to_dense": 1,
            "quantize": 2,
            "dequantize": 2,
        },
        export_backend=ExportBackend.CoreML,
        prepared_model_output=prepared_model_output,
    )
