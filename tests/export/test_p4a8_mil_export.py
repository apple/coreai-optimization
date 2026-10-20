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
from coreai_opt.quantization import ExecutionMode, ModuleQuantizerConfig, Quantizer, QuantizerConfig
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


def _uint8_act_quant_config() -> QuantizerConfig:
    """Return a uint8 asymmetric per-tensor input/output activation quantization config."""
    act_spec = QuantizationSpec(dtype=torch.uint8, qscheme=QuantizationScheme.ASYMMETRIC)
    return QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec=None,
            op_input_spec={"*": act_spec},
            op_output_spec={"*": act_spec},
        ),
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
    the compressed weight. In graph mode, 8-bit palettization with an int8 LUT and int8
    activations is exported as int8 weight quantization instead of a LUT.
    """
    config = ParametrizedP4A8CompressionConfigs.from_params(lut_qspec=lut_qspec, n_bits=n_bits)
    joint_model, input_data, prepared_model_output = _palettize_then_quantize(
        make_layer, input_shape, config, execution_mode
    )

    folds_to_weight_quantization = (
        execution_mode == ExecutionMode.GRAPH and n_bits == 8 and lut_qspec is not None
    )
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


@pytest.mark.parametrize("act_dtype", [torch.int8, torch.uint8], ids=["A8int", "A8uint"])
@pytest.mark.parametrize(
    "lut_qscheme",
    [QuantizationScheme.SYMMETRIC, QuantizationScheme.ASYMMETRIC],
    ids=["symmetric", "asymmetric"],
)
@_LAYERS
def test_p8_int8_lut_with_8bit_act_exports_as_int8_weight_quantization_exactly(
    make_layer: Callable[[], nn.Module],
    input_shape: tuple[int, ...],
    lut_qscheme: QuantizationScheme,
    act_dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In graph mode, the int8 weight quantization export must match the LUT export bit for bit.

    The reference is the same model exported with the rewrite disabled, i.e. as
    ``int8 LUT -> constexpr_blockwise_shift_scale -> constexpr_lut_to_dense``.
    """
    config = ParametrizedP4A8CompressionConfigs.from_params(
        lut_qspec=QuantizationSpec(dtype=torch.int8, qscheme=lut_qscheme), n_bits=8
    )
    if act_dtype == torch.uint8:
        config.quant_config = _uint8_act_quant_config()
    converter = export_utils.MILConverter()

    def convert(model: nn.Module, input_data: torch.Tensor):
        return converter.convert(converter.trace(model, input_data, {}), input_data)

    model, input_data, _ = _palettize_then_quantize(
        make_layer, input_shape, config, ExecutionMode.GRAPH
    )
    folded = convert(model, input_data)

    with monkeypatch.context() as m:
        m.setattr(
            "coreai_opt.quantization._graph._prepare_for_export."
            "fold_int8_lut_palettization_into_quantization",
            lambda *_: None,
        )
        model, input_data, _ = _palettize_then_quantize(
            make_layer, input_shape, config, ExecutionMode.GRAPH
        )
        reference = convert(model, input_data)

    ops = ["constexpr_lut_to_dense", "constexpr_blockwise_shift_scale"]
    assert converter._get_op_counts(reference, ops) == {  # noqa: SLF001
        "constexpr_lut_to_dense": 1,
        "constexpr_blockwise_shift_scale": 1,
    }
    assert converter._get_op_counts(folded, ops) == {  # noqa: SLF001
        "constexpr_lut_to_dense": 0,
        "constexpr_blockwise_shift_scale": 1,
    }
    assert torch.equal(
        converter._run_inference(folded, input_data)[0],  # noqa: SLF001
        converter._run_inference(reference, input_data)[0],  # noqa: SLF001
    )


@_LAYERS
def test_p8_int8_lut_with_uint8_act_exports_as_int8_weight_quantization(
    make_layer: Callable[[], nn.Module],
    input_shape: tuple[int, ...],
) -> None:
    """In graph mode, with uint8 input activations, the int8-LUT palettized weight is
    exported as int8 weight quantization, the same as with int8 input activations.
    """
    config = ParametrizedP4A8CompressionConfigs.from_params(lut_qspec=_INT8_LUT_QSPEC, n_bits=8)
    config.quant_config = _uint8_act_quant_config()
    joint_model, input_data, prepared_model_output = _palettize_then_quantize(
        make_layer, input_shape, config, ExecutionMode.GRAPH
    )

    export_utils.convert_and_verify(
        finalized_model=joint_model,
        input_data=input_data,
        expected_ops={
            "constexpr_lut_to_dense": 0,
            "constexpr_blockwise_shift_scale": 1,
            "quantize": 2,
            "dequantize": 2,
        },
        export_backend=ExportBackend.CoreML,
        prepared_model_output=prepared_model_output,
    )


@_LAYERS
def test_p8_int8_lut_without_input_act_quant_keeps_lut(
    make_layer: Callable[[], nn.Module],
    input_shape: tuple[int, ...],
) -> None:
    """In graph mode, without input activation quantization, the int8-LUT weight stays a LUT.

    Only the output activation is quantized, so the layer's input activation is not
    8-bit and the int8 weight quantization rewrite must not apply.
    """
    config = ParametrizedP4A8CompressionConfigs.from_params(lut_qspec=_INT8_LUT_QSPEC, n_bits=8)
    act_spec = QuantizationSpec(dtype=torch.int8, qscheme=QuantizationScheme.SYMMETRIC)
    config.quant_config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec=None,
            op_input_spec=None,
            op_output_spec={"*": act_spec},
        ),
    )
    joint_model, input_data, prepared_model_output = _palettize_then_quantize(
        make_layer, input_shape, config, ExecutionMode.GRAPH
    )

    export_utils.convert_and_verify(
        finalized_model=joint_model,
        input_data=input_data,
        expected_ops={
            "constexpr_lut_to_dense": 1,
            "constexpr_blockwise_shift_scale": 1,
            "quantize": 1,
            "dequantize": 1,
        },
        export_backend=ExportBackend.CoreML,
        prepared_model_output=prepared_model_output,
    )
