# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import pytest
import torch
import torch.nn as nn
import torch.nn.init as init

from coreai_opt import ExportBackend
from coreai_opt._utils.torch_utils import get_n_bits_from_dtype
from coreai_opt.quantization import (
    ModuleQuantizerConfig,
    Quantizer,
    QuantizerConfig,
)
from coreai_opt.quantization.config import ExecutionMode
from coreai_opt.quantization.spec import (
    PerTensorGranularity,
    QuantizationScheme,
    QuantizationSpec,
)
from tests.fixtures.quantization import make_quant_config
from tests.models.simple import SharedParamsModel, SimpleLinearModel, WeightReadOutsideOp


def weight_activation_quant_config(
    weight_dtype, activation_dtype, weight_qscheme, activation_qscheme, qformulation
):
    weight_qspec_dict = {
        "dtype": weight_dtype,
        "qscheme": weight_qscheme,
        "qformulation": qformulation,
        "granularity": {"type": "per_channel", "axis": 0},
        "fake_quantize_cls": "default",
        "qparam_calculator_cls": "default",
        "range_calculator_cls": "minmax",
    }
    weight_qspec = QuantizationSpec(**weight_qspec_dict)

    activation_qspec_dict = {
        "dtype": activation_dtype,
        "qscheme": activation_qscheme,
        "qformulation": qformulation,
        "granularity": {"type": "per_tensor"},
        "fake_quantize_cls": "default",
        "qparam_calculator_cls": "default",
        "range_calculator_cls": "minmax",
    }
    activation_qspec = QuantizationSpec(**activation_qspec_dict)

    return QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={"weight": weight_qspec},
            op_input_spec={"*": activation_qspec},
            op_output_spec={"*": activation_qspec},
        )
    )


def quantize_model(model, input_tensor, quant_config, backend):
    quantizer = Quantizer(model, quant_config)
    _ = quantizer.prepare((input_tensor,))
    finalized_model = quantizer.finalize(backend=backend)
    return finalized_model


@pytest.mark.parametrize("qformulation", ["zp", "minval"])
@pytest.mark.parametrize("weight_dtype", [torch.uint4, torch.int8])
@pytest.mark.parametrize(
    "activation_dtype",
    [
        pytest.param(
            torch.int4,
            # TODO: handle sub-byte dtypes in coreai quantize/dequantize torch custom ops.
            marks=pytest.mark.xfail(
                reason=(
                    "Sub-byte activation dtypes are not cast correctly during "
                    "graph-mode quantization buffer setup."
                ),
            ),
        ),
        torch.uint8,
    ],
)
@pytest.mark.parametrize("weight_qscheme", ["symmetric", "symmetric_with_clipping", "asymmetric"])
@pytest.mark.parametrize(
    "activation_qscheme", ["symmetric", "symmetric_with_clipping", "asymmetric"]
)
def test_activation_quantization_buffer(
    simple_conv_linear_model,
    simple_model_input,
    weight_dtype,
    activation_dtype,
    weight_qscheme,
    activation_qscheme,
    qformulation,
):
    quant_config = weight_activation_quant_config(
        weight_dtype, activation_dtype, weight_qscheme, activation_qscheme, qformulation
    )

    quantized_model = quantize_model(
        simple_conv_linear_model,
        simple_model_input,
        quant_config,
        backend=ExportBackend.CoreAI,
    )

    activation_scale_buffers = {
        key: value
        for key, value in quantized_model.named_buffers()
        if "activation_post_process_" in key and "_scale" in key
    }
    assert len(activation_scale_buffers) > 1, (
        f"Expected multiple scale buffers, got {len(activation_scale_buffers)}"
    )
    for scale_name, scale_tensor in activation_scale_buffers.items():
        assert scale_tensor.numel() > 0, f"Empty scale tensor: {scale_name}"
        assert torch.all(scale_tensor > 0), f"Invalid scale values in {scale_name}"

    activation_zp_buffers = {
        key: value
        for key, value in quantized_model.named_buffers()
        if "activation_post_process_" in key and "_zero_point" in key
    }
    activation_minval_buffers = {
        key: value
        for key, value in quantized_model.named_buffers()
        if "activation_post_process_" in key and "_minval" in key
    }

    if qformulation == "zp":
        # ZP formulation exports a zero_point buffer per activation FQ;
        # no minval buffer is registered.
        assert len(activation_zp_buffers) == len(activation_scale_buffers), (
            "Every activation buffer needs to have both scale and zero-point"
        )
        assert len(activation_minval_buffers) == 0, (
            "ZP formulation should not register minval buffers"
        )

        for zp_name, zp_tensor in activation_zp_buffers.items():
            assert zp_tensor.numel() > 0, f"Empty zero_point tensor: {zp_name}"
            if (
                activation_qscheme in ["symmetric", "symmetric_with_clipping"]
                and activation_dtype.is_signed
            ):
                assert torch.all(zp_tensor == 0), f"Invalid zero_point values in {zp_name}"
            else:
                n_bits = get_n_bits_from_dtype(activation_dtype)
                zp_min = 0
                zp_max = 2**n_bits - 1
                if activation_dtype.is_signed:
                    zp_min = zp_min - 2 ** (n_bits - 1)
                    zp_max = zp_max - 2 ** (n_bits - 1)

                assert torch.all(zp_tensor >= zp_min), f"Invalid zero_point values in {zp_name}"
                assert torch.all(zp_tensor <= zp_max), f"Invalid zero_point values in {zp_name}"
    else:  # qformulation == "minval"
        # MINVAL formulation exports a minval buffer per activation FQ;
        # no zero_point buffer is registered.
        assert len(activation_zp_buffers) == 0, (
            "MINVAL formulation should not register zero_point buffers"
        )
        assert len(activation_minval_buffers) == len(activation_scale_buffers), (
            "Every activation buffer needs to have both scale and minval"
        )

        for mv_name, mv_tensor in activation_minval_buffers.items():
            assert mv_tensor.numel() > 0, f"Empty minval tensor: {mv_name}"
            # minval is the floor of the float range:
            #   symmetric: -max_abs (<= 0)
            #   asymmetric: min(observed_min, 0) (<= 0)
            assert torch.all(mv_tensor <= 0), f"Invalid minval values in {mv_name}"

    output = quantized_model(simple_model_input)
    assert output is not None, "Quantized model failed to run with CoreAI backend"
    assert output.shape == (1, 10), f"Unexpected output shape: {output.shape}"


@pytest.mark.parametrize("qformulation", ["zp", "minval"])
@pytest.mark.parametrize("param_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_weight_quantization_buffer(param_dtype, qformulation):
    """
    Test that after prepare+finalize, the weight quantization buffers stored
    in the finalized model are correct:

    - Scale matches a manually computed scale using the export dtype
      (verifies scale is cast to export dtype before quantization, so that
      quantization and dequantization are self-consistent).
    - Quantized data matches the manually computed quantized weights.
    - The formulation-specific offset buffer is registered: ZP exports
      ``<weight>_zero_point`` only; MINVAL exports ``<weight>_minval`` only.
    """

    model = nn.Conv2d(64, 64, kernel_size=3).to(param_dtype)
    init.kaiming_normal_(model.weight)

    example_input = torch.randn(1, 64, 3, 3).to(param_dtype)

    weight_dtype = torch.int4
    weight_qscheme = QuantizationScheme.SYMMETRIC
    quant_config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={
                "weight": QuantizationSpec(
                    dtype=weight_dtype,
                    qscheme=weight_qscheme,
                    qformulation=qformulation,
                    granularity=PerTensorGranularity(),
                )
            },
            op_input_spec=None,
            op_output_spec=None,
        ),
        execution_mode=ExecutionMode.GRAPH,
    )
    original_weight = model.weight.data.clone()

    quantizer = Quantizer(model, quant_config)
    quantizer.prepare((example_input,))
    finalized_model = quantizer.finalize(backend=ExportBackend.CoreAI)

    weight_scale_buffers = {
        key: value
        for key, value in finalized_model.named_buffers()
        if "scale" in key and "activation" not in key
    }
    assert len(weight_scale_buffers) == 1, (
        f"Expected 1 weight scale buffer, got {len(weight_scale_buffers)}: "
        f"{list(weight_scale_buffers.keys())}"
    )
    stored_scale = next(iter(weight_scale_buffers.values()))

    quant_min, quant_max = QuantizationSpec.get_quant_range(weight_dtype, weight_qscheme)
    expected_scale = (2 * original_weight.abs().max() / (quant_max - quant_min)).to(param_dtype)

    assert stored_scale.dtype == param_dtype, (
        f"Expected scale dtype {param_dtype}, got {stored_scale.dtype}"
    )
    assert stored_scale == expected_scale, f"Scale mismatch for param_dtype={param_dtype}"

    weight_quantized_data_buffers = {
        key: value for key, value in finalized_model.named_buffers() if "weight_quantized" in key
    }
    assert len(weight_quantized_data_buffers) == 1, (
        f"Expected 1 quantized_data buffer, got {len(weight_quantized_data_buffers)}: "
        f"{list(weight_quantized_data_buffers.keys())}"
    )

    # Verify the formulation-specific offset buffer is registered:
    # ZP → only <weight>_zero_point; MINVAL → only <weight>_minval.
    weight_zp_buffers = {
        key: value for key, value in finalized_model.named_buffers() if "weight_zero_point" in key
    }
    weight_minval_buffers = {
        key: value for key, value in finalized_model.named_buffers() if "weight_minval" in key
    }

    if qformulation == "zp":
        assert len(weight_zp_buffers) == 1, (
            f"Expected 1 weight zero_point buffer for ZP, got {len(weight_zp_buffers)}"
        )
        assert len(weight_minval_buffers) == 0, (
            "ZP formulation should not register a weight minval buffer"
        )
        # SYMMETRIC int4 zero_point = 0
        stored_zp = next(iter(weight_zp_buffers.values()))
        assert torch.all(stored_zp == 0), f"Expected zero_point == 0, got {stored_zp}"
    else:  # qformulation == "minval"
        assert len(weight_zp_buffers) == 0, (
            "MINVAL formulation should not register a weight zero_point buffer"
        )
        assert len(weight_minval_buffers) == 1, (
            f"Expected 1 weight minval buffer for MINVAL, got {len(weight_minval_buffers)}"
        )

        stored_minval = next(iter(weight_minval_buffers.values()))
        assert stored_minval.dtype == param_dtype, (
            f"Expected minval dtype {param_dtype}, got {stored_minval.dtype}"
        )
        # SYMMETRIC: minval = -max_abs(weight)
        expected_minval = (-original_weight.abs().max()).to(param_dtype)
        assert stored_minval == expected_minval, (
            f"Minval mismatch: expected {expected_minval}, got {stored_minval}"
        )


def test_finalize_releases_superseded_dense_weights(simple_linear_model, simple_linear_model_input):
    """Graph-mode ``finalize(CoreAI)`` releases the dense weights,
    while leaving the model's structure intact."""
    model = simple_linear_model.eval()
    original_bytes = {k: v.nbytes for k, v in model.state_dict().items()}

    config = make_quant_config(weight_dtype=torch.int8, act_dtype=None, execution_mode="graph")
    finalized = quantize_model(model, simple_linear_model_input, config, ExportBackend.CoreAI)

    state = finalized.state_dict()

    for weight_key in ("l1.weight", "l2.weight"):
        assert weight_key in state, f"{weight_key} was dropped from state_dict, not released"
        assert state[weight_key].nbytes == 0, (
            f"{weight_key} should have been released, still holds {state[weight_key].nbytes} bytes"
        )

    # The buffers that superseded them are present.
    quantized_buffers = [
        name for name, _ in finalized.named_buffers() if "weight_quantized" in name
    ]
    assert len(quantized_buffers) == 2, (
        f"Expected 2 weight_quantized buffers, got {quantized_buffers}"
    )

    # Biases are still read by the graph, so their storage is untouched.
    for bias_key in ("l1.bias", "l2.bias"):
        assert state[bias_key].nbytes == original_bytes[bias_key], (
            f"{bias_key} storage changed; it is still read and must be untouched"
        )


def _graph_int8_config(exclude: list[str] | None = None) -> QuantizerConfig:
    """Weight-only int8 graph config (reuses ``make_quant_config``); ``exclude`` names
    modules to leave unquantized via ``module_name_configs``."""
    base = make_quant_config(weight_dtype=torch.int8, act_dtype=None, execution_mode="graph")
    if not exclude:
        return base
    return QuantizerConfig(
        global_config=base.global_config,
        execution_mode=ExecutionMode.GRAPH,
        module_name_configs={name: None for name in exclude},
    )


@pytest.mark.parametrize(
    ("model_cls", "example_input", "config", "kept_weight_keys", "n_quantized_buffers"),
    [
        # A layer excluded from quantization keeps its weight
        # intact while the quantized layer's weight is freed.
        pytest.param(
            SimpleLinearModel,
            torch.randn(4, 64),
            _graph_int8_config(exclude=["l2"]),
            ["l2.weight"],
            1,
            id="unquantized-layer-kept",
        ),
        # A quantized weight that is also read outside the op keeps a live get_attr after DCE,
        # so it is not freed.
        pytest.param(
            WeightReadOutsideOp,
            torch.randn(4, 64),
            _graph_int8_config(),
            ["fc.weight"],
            1,
            id="direct-read-kept",
        ),
        # in case of shared weights being quantized, the dense weight
        # should be freed
        pytest.param(
            SharedParamsModel,
            torch.randn(1, 784),
            _graph_int8_config(),
            [],
            3,
            id="shared-weight-freed",
        ),
    ],
)
def test_finalize_dense_weight_release_scenarios(
    model_cls, example_input, config, kept_weight_keys, n_quantized_buffers
):
    """A dense weight is released only when it was quantized and nothing still reads it.

    - Unquantized layer: this dense weight should not be released.
    - A weight that also accessed directly/explicitly outside the op still has
      a live ``get_attr`` and is kept even though it was quantized.
    - In case of a shared quantized weight, the dense weight is still freed.
    """
    model = model_cls()
    original_bytes = {k: v.nbytes for k, v in model.state_dict().items()}

    finalized = quantize_model(model, example_input, config, ExportBackend.CoreAI)
    state = finalized.state_dict()

    # Every dense weight is freed except the ones expected to stay live.
    for key in (k for k in state if k.endswith(".weight")):
        if key in kept_weight_keys:
            assert state[key].nbytes == original_bytes[key], (
                f"{key} is still read and must not be released"
            )
        else:
            assert state[key].nbytes == 0, f"{key} should have been released"

    quantized_buffers = [
        name for name, _ in finalized.named_buffers() if "weight_quantized" in name
    ]
    assert len(quantized_buffers) == n_quantized_buffers, (
        f"expected {n_quantized_buffers} weight_quantized buffers, got {quantized_buffers}"
    )

    # Biases are never release candidates, so any that survive keep their storage.
    for key, nbytes in original_bytes.items():
        if key.endswith(".bias") and key in state:
            assert state[key].nbytes == nbytes, f"{key} must be untouched"
