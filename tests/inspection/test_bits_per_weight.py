# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the bits_per_weight utility."""

import pytest
import torch
import torch.nn as nn

from coreai_opt.inspection import bits_per_weight
from coreai_opt.palettization import (
    KMeansPalettizer,
    KMeansPalettizerConfig,
    ModuleKMeansPalettizerConfig,
)
from coreai_opt.palettization.spec import (
    PalettizationSpec,
    PerGroupedChannelGranularity,
    PerTensorGranularity as PalettPerTensorGranularity,
)
from coreai_opt.pruning import MagnitudePruner
from coreai_opt.quantization import ModuleQuantizerConfig, Quantizer, QuantizerConfig
from coreai_opt.quantization.spec import (
    PerBlockGranularity,
    PerChannelGranularity,
    PerTensorGranularity,
    QuantizationSpec,
)
from tests.models.simple import LinearBatchNormModel, SharedParamsModel, SimpleLinearModel

# SimpleLinearModel and LinearBatchNormModel are both Linear(64, 128) -> Linear(128, 64),
# so their weights are 128 x 64 and 64 x 128.
# Expected bits and num_weights are derived by
# hand from these shapes and used as the golden values for testing the utility.
_IN_FEATURES = 64
_HIDDEN_FEATURES = 128
_OUT_FEATURES = 64
_WEIGHT_ELEMS = _HIDDEN_FEATURES * _IN_FEATURES + _OUT_FEATURES * _HIDDEN_FEATURES

_BIAS_ELEMS = _HIDDEN_FEATURES + _OUT_FEATURES
# One per-channel qparam (quantization scale, or palettization per-channel scale) per
# output channel of each layer.
_OUT_CHANNELS = _HIDDEN_FEATURES + _OUT_FEATURES
_FP32_BITS = 32
_INT64_BITS = 64

# Sanity envelope for the qparam / LUT / bias overhead a sane config adds on top of the
# nominal bit width, in bpw. See _assert_bpw_is_plausible for why 2 and not 1.
_MAX_EXPECTED_OVERHEAD_BPW = 2.0

_EXAMPLE_INPUT = torch.rand(4, _IN_FEATURES)


def _expected_elems(bias: bool) -> int:
    """Logical parameter elements: the dense weights, plus biases when present."""
    return _WEIGHT_ELEMS + (_BIAS_ELEMS if bias else 0)


def _expected_quant_bits(
    n_bits: int,
    num_qparam_blocks: int,
    qscheme: str,
    bias: bool,
    qformulation: str = "zp",
) -> int:
    """Hand-derived cost of the weight-quantized model, in bits.

    Payload is ``n_bits`` per weight. Overhead is one fp32 scale per qparam block, the
    per-block dequantization offset, and the fp32 biases, which quantization does not
    target.
    """
    scale_bits = num_qparam_blocks * _FP32_BITS
    if qformulation == "minval":
        offset_bits = num_qparam_blocks * _FP32_BITS
    elif qscheme == "asymmetric":
        offset_bits = num_qparam_blocks * n_bits
    else:
        offset_bits = 0
    bias_bits = _BIAS_ELEMS * _FP32_BITS if bias else 0
    return _WEIGHT_ELEMS * n_bits + scale_bits + offset_bits + bias_bits


def _expected_palettized_bits(
    n_bits: int, num_lut_blocks: int, bias: bool, per_channel_scale: bool
) -> int:
    """Hand-derived cost of the palettized model, in bits.

    Payload is one ``n_bits`` index per weight (``cluster_dim=1``). Overhead is one LUT
    of ``2**n_bits`` fp32 centroids per LUT block, one fp32 per-channel scale per output
    channel when that is enabled, and the fp32 biases, which palettization does not
    target.
    """
    lut_bits = num_lut_blocks * (2**n_bits) * _FP32_BITS
    per_channel_scale_bits = _OUT_CHANNELS * _FP32_BITS if per_channel_scale else 0
    bias_bits = _BIAS_ELEMS * _FP32_BITS if bias else 0
    return _WEIGHT_ELEMS * n_bits + lut_bits + per_channel_scale_bits + bias_bits


def _assert_bpw_is_plausible(bpw: float, n_bits: int) -> None:
    """Structural sanity bounds on bpw, independent of how the cost is modelled."""
    assert n_bits <= bpw < n_bits + _MAX_EXPECTED_OVERHEAD_BPW


def _prepare_eager_quant(
    model: nn.Module,
    dtype: torch.dtype,
    qscheme: str = "symmetric",
    granularity: object | None = None,
    qformulation: str = "zp",
) -> nn.Module:
    """Prepare an eager-mode weight-quantized model."""
    spec = QuantizationSpec(
        dtype=dtype,
        qscheme=qscheme,
        granularity=granularity or PerChannelGranularity(axis=0),
        qformulation=qformulation,
    )
    config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(op_state_spec={"weight": spec}, op_input_spec=None),
        execution_mode="eager",
    )
    return Quantizer(model, config).prepare(example_inputs=(_EXAMPLE_INPUT,))


def _prepare_palettized(model: nn.Module, spec: PalettizationSpec) -> nn.Module:
    """Prepare a palettized model."""
    config = KMeansPalettizerConfig(
        global_config=ModuleKMeansPalettizerConfig(op_state_spec={"weight": spec})
    )
    return KMeansPalettizer(model, config).prepare((_EXAMPLE_INPUT,))


def test_full_precision_bits_per_weight():
    assert bits_per_weight(SimpleLinearModel()).bpw == 32.0
    assert bits_per_weight(SimpleLinearModel().half()).bpw == 16.0
    assert bits_per_weight(SimpleLinearModel().bfloat16()).bpw == 16.0


@pytest.mark.parametrize("bias", [False, True], ids=["no_bias", "bias"])
@pytest.mark.parametrize("qscheme", ["symmetric", "asymmetric"])
@pytest.mark.parametrize(
    ("granularity", "num_qparam_blocks", "qformulation"),
    [
        pytest.param(PerTensorGranularity(axis=None), 2, "zp", id="per_tensor_zp"),
        pytest.param(PerTensorGranularity(axis=None), 2, "minval", id="per_tensor_minval"),
        pytest.param(PerChannelGranularity(axis=0), _OUT_CHANNELS, "zp", id="per_channel_zp"),
        pytest.param(
            PerChannelGranularity(axis=0), _OUT_CHANNELS, "minval", id="per_channel_minval"
        ),
        # One qparam block per 32 weights along the input-feature axis of each layer.
        pytest.param(
            PerBlockGranularity(axis=1, block_size=32),
            _HIDDEN_FEATURES * (_IN_FEATURES // 32) + _OUT_FEATURES * (_HIDDEN_FEATURES // 32),
            "zp",
            id="per_block_zp",
        ),
    ],
)
@pytest.mark.parametrize(
    ("dtype", "n_bits"),
    [
        (torch.int8, 8),
        (torch.int4, 4),
        (torch.int2, 2),
        (torch.uint8, 8),
        (torch.uint4, 4),
        (torch.uint2, 2),
    ],
    ids=["int8", "int4", "int2", "uint8", "uint4", "uint2"],
)
def test_eager_quant_matches_analytical(
    dtype: torch.dtype,
    n_bits: int,
    granularity: object,
    num_qparam_blocks: int,
    qformulation: str,
    qscheme: str,
    bias: bool,
):
    prepared = _prepare_eager_quant(
        SimpleLinearModel(bias=bias), dtype, qscheme, granularity, qformulation
    )
    result = bits_per_weight(prepared)

    expected_bits = _expected_quant_bits(n_bits, num_qparam_blocks, qscheme, bias, qformulation)
    expected_elems = _expected_elems(bias)
    assert result.total_bits == expected_bits
    # The inserted scale / zero-point buffers must not inflate the denominator: only the
    # original dense parameters are counted.
    assert result.total_weights == expected_elems
    assert result.bpw == expected_bits / expected_elems
    _assert_bpw_is_plausible(result.bpw, n_bits)


@pytest.mark.parametrize("bias", [False, True], ids=["no_bias", "bias"])
@pytest.mark.parametrize("per_channel_scale", [False, True], ids=["no_pcs", "pcs"])
@pytest.mark.parametrize(
    ("n_bits", "granularity", "num_lut_blocks"),
    [
        pytest.param(n_bits, granularity, num_lut_blocks, id=f"n{n_bits}_{granularity_id}")
        for n_bits in (1, 2, 4)
        for granularity, num_lut_blocks, granularity_id in (
            (PalettPerTensorGranularity(axis=None), 2, "per_tensor"),
            (
                PerGroupedChannelGranularity(axis=0, group_size=8),
                _HIDDEN_FEATURES // 8 + _OUT_FEATURES // 8,
                "group8",
            ),
            (
                PerGroupedChannelGranularity(axis=0, group_size=32),
                _HIDDEN_FEATURES // 32 + _OUT_FEATURES // 32,
                "group32",
            ),
        )
    ]
    # n_bits=8 only at per-tensor granularity. Its grouped variants carry more LUT than
    # payload, so they break the sanity envelope and live in
    # test_overhead_heavy_configs_exceed_bitwidth instead.
    + [pytest.param(8, PalettPerTensorGranularity(axis=None), 2, id="n8_per_tensor")],
)
def test_palettized_matches_analytical(
    n_bits: int,
    granularity: object,
    num_lut_blocks: int,
    per_channel_scale: bool,
    bias: bool,
):
    prepared = _prepare_palettized(
        SimpleLinearModel(bias=bias),
        PalettizationSpec(
            n_bits=n_bits,
            granularity=granularity,
            enable_per_channel_scale=per_channel_scale,
        ),
    )
    result = bits_per_weight(prepared)

    expected_bits = _expected_palettized_bits(n_bits, num_lut_blocks, bias, per_channel_scale)
    expected_elems = _expected_elems(bias)
    assert result.total_bits == expected_bits
    # The inserted LUT buffers must not inflate the denominator.
    assert result.total_weights == expected_elems
    assert result.bpw == expected_bits / expected_elems
    _assert_bpw_is_plausible(result.bpw, n_bits)


@pytest.mark.parametrize(
    ("compression", "dtype", "n_bits", "granularity", "num_blocks"),
    [
        # One fp32 scale and zero-point per 8 weights: 4.25 bpw of qparams over a 2 bpw
        # payload.
        pytest.param(
            "quantization",
            torch.int2,
            2,
            PerBlockGranularity(axis=1, block_size=8),
            _HIDDEN_FEATURES * (_IN_FEATURES // 8) + _OUT_FEATURES * (_HIDDEN_FEATURES // 8),
            id="quant_int2_block8",
        ),
        pytest.param(
            "quantization",
            torch.int4,
            4,
            PerBlockGranularity(axis=1, block_size=4),
            _HIDDEN_FEATURES * (_IN_FEATURES // 4) + _OUT_FEATURES * (_HIDDEN_FEATURES // 4),
            id="quant_int4_block4",
        ),
        # One 2**8-entry fp32 LUT per 8 output channels: 12 bpw of LUT over an 8 bpw
        # payload. Palettization takes n_bits directly, so it needs no dtype.
        pytest.param(
            "palettization",
            None,
            8,
            PerGroupedChannelGranularity(axis=0, group_size=8),
            _HIDDEN_FEATURES // 8 + _OUT_FEATURES // 8,
            id="palett_n8_group8",
        ),
        pytest.param(
            "palettization",
            None,
            8,
            PerGroupedChannelGranularity(axis=0, group_size=32),
            _HIDDEN_FEATURES // 32 + _OUT_FEATURES // 32,
            id="palett_n8_group32",
        ),
    ],
)
def test_overhead_heavy_configs_exceed_bitwidth(
    compression: str,
    dtype: torch.dtype | None,
    n_bits: int,
    granularity: object,
    num_blocks: int,
):
    """Configs whose qparam or LUT overhead outweighs the payload it serves.

    So the bpw will exceed n_bits by a non-trivial amount. These are in-efficient
    compression configs which the utility should be agnostic to. And their bpw values
    will not be close to n_bits, so they will not adhere to the
    ``_assert_bpw_is_plausible`` check that the above tests perform.
    """
    model = SimpleLinearModel(bias=False)
    if compression == "quantization":
        prepared = _prepare_eager_quant(model, dtype, "asymmetric", granularity)
        expected_bits = _expected_quant_bits(n_bits, num_blocks, "asymmetric", bias=False)
    else:
        prepared = _prepare_palettized(
            model, PalettizationSpec(n_bits=n_bits, granularity=granularity)
        )
        expected_bits = _expected_palettized_bits(
            n_bits, num_blocks, bias=False, per_channel_scale=False
        )
    result = bits_per_weight(prepared)

    assert result.total_bits == expected_bits
    assert result.total_weights == _expected_elems(bias=False)
    assert n_bits + _MAX_EXPECTED_OVERHEAD_BPW < result.bpw < _FP32_BITS


def test_persistent_buffers_counted():
    """BatchNorm running stats ship in ``state_dict()``, so they are amortized in."""
    result = bits_per_weight(LinearBatchNormModel())

    # Params: the two fp32 weights (the model is bias-free), plus BatchNorm's fp32
    # weight and bias.
    param_elems = _WEIGHT_ELEMS + 2 * _HIDDEN_FEATURES
    # Buffers: fp32 running_mean and running_var, one of each per hidden feature, plus
    # a scalar int64 num_batches_tracked.
    buffer_elems = 2 * _HIDDEN_FEATURES + 1

    assert result.total_weights == param_elems + buffer_elems
    assert result.total_bits == (
        param_elems * _FP32_BITS + 2 * _HIDDEN_FEATURES * _FP32_BITS + _INT64_BITS
    )


def test_multi_original_parametrization_is_unsupported():
    """``weight_norm`` stores original0 / original1 rather than a single original.

    There is no one dense tensor to attribute a storage cost to, so this must raise
    rather than fail with ``AttributeError``.
    """
    model = nn.utils.parametrizations.weight_norm(nn.Linear(_IN_FEATURES, _HIDDEN_FEATURES))

    with pytest.raises(NotImplementedError, match="multiple original tensors"):
        bits_per_weight(model)


def test_pruned_weight_is_unsupported():
    """A pruned weight raises rather than being priced as a dense tensor."""
    model = SimpleLinearModel(bias=False)
    prepared = MagnitudePruner(model).prepare((_EXAMPLE_INPUT,))

    with pytest.raises(NotImplementedError, match="pruned"):
        bits_per_weight(prepared)


def test_non_persistent_buffer_counted():
    """Buffers count regardless of ``persistent=``: every tensor the model carries."""

    class _ModuleWithScratch(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.l = nn.Linear(8, 8, bias=False)
            self.register_buffer("scratch", torch.zeros(1000), persistent=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.l(x) + self.scratch[: x.shape[-1]]

    result = bits_per_weight(_ModuleWithScratch())
    assert result.total_weights == 8 * 8 + 1000
    assert result.total_bits == (8 * 8 + 1000) * 32


@pytest.mark.parametrize(
    ("dtype", "granularity"),
    [
        (torch.float8_e4m3fn, PerChannelGranularity(axis=0)),
        (torch.float4_e2m1fn_x2, PerBlockGranularity(axis=1, block_size=32)),
    ],
    ids=["fp8_per_channel", "fp4_per_block"],
)
def test_float_quant_is_unsupported(dtype, granularity):
    spec = QuantizationSpec(dtype=dtype, qscheme="symmetric", granularity=granularity)
    config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(op_state_spec={"weight": spec}, op_input_spec=None),
        execution_mode="eager",
    )
    prepared = Quantizer(SimpleLinearModel(), config).prepare(example_inputs=(_EXAMPLE_INPUT,))

    with pytest.raises(NotImplementedError, match="floating-point weight quantization"):
        bits_per_weight(prepared)


def test_per_module_attributes_cost_to_the_owning_module():
    """Under mixed precision, each module's bpw lands on that module."""

    spec = QuantizationSpec(
        dtype=torch.int8, qscheme="symmetric", granularity=PerChannelGranularity(axis=0)
    )
    config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(op_state_spec={"weight": spec}, op_input_spec=None),
        # l2 stays fp32 while l1 is int8
        module_name_configs={"l2": None},
        execution_mode="eager",
    )
    prepared = Quantizer(SimpleLinearModel(bias=False), config).prepare(
        example_inputs=(_EXAMPLE_INPUT,)
    )
    prepared(_EXAMPLE_INPUT)
    result = bits_per_weight(prepared)

    # l1 carries 8 bits per weight plus one fp32 scale per output channel.
    l1_elems = _HIDDEN_FEATURES * _IN_FEATURES
    expected_l1_bits = l1_elems * 8 + _HIDDEN_FEATURES * _FP32_BITS
    l2_elems = _OUT_FEATURES * _HIDDEN_FEATURES

    assert result.per_module_map == {
        "l1": pytest.approx(expected_l1_bits / l1_elems),
        "l2": _FP32_BITS,
    }

    assert result.total_bits == expected_l1_bits + l2_elems * _FP32_BITS
    assert result.bpw == result.total_bits / (l1_elems + l2_elems)

    assert result.per_module_map["l1"] < result.bpw < result.per_module_map["l2"]


def test_per_module_counts_a_tied_weight_once():
    """A weight shared by several modules is charged only to the first one."""
    per_module_map = bits_per_weight(SharedParamsModel()).per_module_map

    assert per_module_map["shared_linear"] == _FP32_BITS
    assert "layer1" not in per_module_map
    assert "layer2" not in per_module_map
    # Modules that own tensors of their own are unaffected.
    assert per_module_map["input_layer"] == _FP32_BITS
    assert per_module_map["output"] == _FP32_BITS
