# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Validate the bits-per-weight prediction against the actual Core AI exported
asset size."""

import pytest
import torch
import torch.nn as nn

from coreai_opt import ExportBackend
from coreai_opt.inspection import BitsPerWeightResult, bits_per_weight
from coreai_opt.palettization import (
    KMeansPalettizer,
    KMeansPalettizerConfig,
    ModuleKMeansPalettizerConfig,
)
from coreai_opt.palettization.spec import (
    PalettizationSpec,
    PerGroupedChannelGranularity,
    default_weight_palettization_spec,
)
from coreai_opt.quantization import ModuleQuantizerConfig, Quantizer, QuantizerConfig
from coreai_opt.quantization.spec import (
    PerBlockGranularity,
    PerChannelGranularity,
    QuantizationSpec,
)
from tests.models.simple import GatedMLPModel

from . import export_utils

# Fractional budget for what a serialized artifact carries beyond the weight payload.
_MAX_COREAI_EXPORT_OVERHEAD = 0.025

# GatedMLPModel at this width holds ~3.1M params (12 MiB fp32, 1.5 MiB at int4).
_MLP_DIM = 1024


def _gated_mlp(bias: bool = False) -> nn.Module:
    return GatedMLPModel(dim=_MLP_DIM, hidden_dim=_MLP_DIM, bias=bias)


def _mlp_input() -> torch.Tensor:
    return torch.rand(1, 4, _MLP_DIM)


def _eager_quant_config(
    dtype: torch.dtype, qscheme: str, granularity: object | None = None
) -> QuantizerConfig:
    """Build an eager-mode weight-quantization config, per-channel by default."""
    spec = QuantizationSpec(
        dtype=dtype,
        qscheme=qscheme,
        granularity=granularity or PerChannelGranularity(axis=0),
    )
    return QuantizerConfig(
        global_config=ModuleQuantizerConfig(op_state_spec={"weight": spec}, op_input_spec=None),
        execution_mode="eager",
    )


def _assert_prediction_matches_export(
    finalized_model: nn.Module,
    input_data: torch.Tensor,
    result: BitsPerWeightResult,
) -> None:
    """Assert the bits-per-weight prediction matches the exported asset size.

    The lower bound holds only for fixtures whose tensors are all reachable from
    ``forward`` and randomly initialized. The export ships what the graph needs,
    and may shrink tensors it can represent more compactly, so a fixture with a
    dead buffer or uniform-valued weights would export below the prediction.
    """
    predicted_bytes = result.total_bits / 8
    actual_bytes = export_utils.coreai_export_size_bytes(finalized_model, input_data)
    actual_bpw = actual_bytes * 8 / result.total_weights
    overhead_bytes = actual_bytes - predicted_bytes
    context = f"predicted bpw={result.bpw:.4f}, actual bpw={actual_bpw:.4f}"

    assert overhead_bytes >= 0, (
        f"exported asset {actual_bytes:,} bytes fell below predicted "
        f"{predicted_bytes:,.0f} by {-overhead_bytes:,.0f} bytes ({context})"
    )
    # The only excess should be structural metadata, which stays a small fraction
    # of the payload.
    assert overhead_bytes <= predicted_bytes * _MAX_COREAI_EXPORT_OVERHEAD, (
        f"exported asset {actual_bytes:,} bytes exceeded predicted "
        f"{predicted_bytes:,.0f} by {overhead_bytes:,.0f} bytes "
        f"({overhead_bytes / predicted_bytes:.2%}, budget {_MAX_COREAI_EXPORT_OVERHEAD:.1%}) "
        f"({context})"
    )


def _assert_quantized_matches_export(
    model: nn.Module,
    input_data: torch.Tensor,
    dtype: torch.dtype,
    qscheme: str = "symmetric",
    granularity: object | None = None,
) -> None:
    """Quantize and check the prediction against the export."""
    quantizer = Quantizer(model, _eager_quant_config(dtype, qscheme, granularity))
    prepared_model = quantizer.prepare((input_data,))

    result = bits_per_weight(prepared_model)
    finalized_model = quantizer.finalize(backend=ExportBackend.CoreAI)
    _assert_prediction_matches_export(finalized_model, input_data, result)


def _assert_palettized_matches_export(
    model: nn.Module, input_data: torch.Tensor, spec: PalettizationSpec
) -> None:
    """Palettize and check the prediction against the export."""
    config = KMeansPalettizerConfig(
        global_config=ModuleKMeansPalettizerConfig(op_state_spec={"weight": spec})
    )
    palettizer = KMeansPalettizer(model, config)
    prepared_model = palettizer.prepare((input_data,))

    result = bits_per_weight(prepared_model)
    finalized_model = palettizer.finalize(backend=ExportBackend.CoreAI)
    _assert_prediction_matches_export(finalized_model, input_data, result)


@pytest.mark.parametrize(
    ("dtype", "qscheme"),
    [
        (torch.int8, "symmetric"),
        (torch.int8, "asymmetric"),
        (torch.int4, "symmetric"),
        (torch.int4, "asymmetric"),
    ],
    ids=["int8_symmetric", "int8_asymmetric", "int4_symmetric", "int4_asymmetric"],
)
def test_eager_quant_prediction_matches_export(dtype: torch.dtype, qscheme: str) -> None:
    """Per-channel weight quantization: predicted deploy size matches the export."""
    # bias=True so the fp32 biases are amortized into the prediction too.
    _assert_quantized_matches_export(_gated_mlp(bias=True), _mlp_input(), dtype, qscheme)


@pytest.mark.parametrize(
    "spec",
    [
        PalettizationSpec(n_bits=2),
        PalettizationSpec(n_bits=4),
        PalettizationSpec(n_bits=8),
        PalettizationSpec(n_bits=4, granularity=PerGroupedChannelGranularity(axis=0, group_size=8)),
        PalettizationSpec(
            n_bits=4, granularity=PerGroupedChannelGranularity(axis=0, group_size=32)
        ),
    ],
    ids=[
        "per_tensor_n2",
        "per_tensor_n4",
        "per_tensor_n8",
        "per_grouped_channel_group8",
        "per_grouped_channel_group32",
    ],
)
def test_palettized_prediction_matches_export(spec: PalettizationSpec) -> None:
    """Weight palettization across bit widths and granularities.

    Per-grouped-channel multiplies the LUT count by the number of channel groups, so
    it is what exercises the ``num_blocks_to_cluster`` term in the LUT cost.
    bias=False isolates the palettized weights so the deploy size is dominated by the
    indices and LUTs, not amortized fp32 biases.
    """
    _assert_palettized_matches_export(_gated_mlp(bias=False), _mlp_input(), spec)


@pytest.mark.parametrize(
    "weight_dtype", [torch.float32, torch.float16], ids=["fp32_weights", "fp16_weights"]
)
@pytest.mark.parametrize("dtype", [torch.int4, torch.int2], ids=["int4", "int2"])
def test_perblock_asymmetric_subbyte_prediction_matches_export(
    dtype: torch.dtype, weight_dtype: torch.dtype
) -> None:
    """Per-block asymmetric sub-byte weights: zero-points are packed at ``n_bits``.

    Per-channel granularity keeps the zero-point term small enough to hide its
    width; per-block ``block_size=32`` makes it ~3% of the payload, so this is what
    catches a zero-point charged at ``target_dtype``'s byte width (``element_size()``
    is 1 for int4 and int2 alike) instead of at ``n_bits``.

    The ``fp16_weights`` case additionally pins the scale width to the weight dtype.
    """
    _assert_quantized_matches_export(
        _gated_mlp(bias=False).to(weight_dtype),
        _mlp_input().to(weight_dtype),
        dtype,
        "asymmetric",
        granularity=PerBlockGranularity(axis=1, block_size=32),
    )


def test_quantized_lut_prediction_matches_export() -> None:
    """Palettization with a quantized LUT: the LUT's own qparams are amortized too.

    One group per channel at 1 bit maximizes the qparam-to-payload ratio (one LUT
    scale and zero-point per palettization block against a 1-bit index payload), so
    omitting the LUT-quantization qparams shows up as a ~4% shortfall here while
    staying under 0.1% for the default per-tensor 4-bit config.
    """
    _assert_palettized_matches_export(
        _gated_mlp(bias=False),
        _mlp_input(),
        PalettizationSpec(
            n_bits=1,
            granularity=PerGroupedChannelGranularity(axis=0, group_size=1),
            lut_qspec=QuantizationSpec(dtype=torch.uint8, qscheme="asymmetric"),
        ),
    )


@pytest.mark.parametrize("dtype", [torch.int8, torch.int4], ids=["int8", "int4"])
def test_resnet18_quant_prediction_matches_export(
    resnet18_model: nn.Module,
    resnet_example_input: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    """Pretrained ResNet-18 quantized: a deep real model with ~50 leaf modules."""
    _assert_quantized_matches_export(resnet18_model, resnet_example_input, dtype)


def test_resnet18_palettized_prediction_matches_export(
    resnet18_model: nn.Module,
    resnet_example_input: torch.Tensor,
) -> None:
    """Pretrained ResNet-18 palettized at the default 4 bits.

    Only the default config, matching ``test_kmeans_export.test_resnet_export``: the
    per-``n_bits`` matrix runs on the faster synthetic model instead.
    """
    _assert_palettized_matches_export(
        resnet18_model, resnet_example_input, default_weight_palettization_spec()
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
def test_mnist_dense_prediction_matches_export(
    custom_test_mnist_model: nn.Module,
    mnist_example_input: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    """Real conv/BN/linear model, dense fp32 and fp16: the export matches prediction.

    Exercises persistent BatchNorm buffers (running_mean / running_var), which are
    counted by bits_per_weight and survive export as constants rather than being
    folded away.
    """
    model = custom_test_mnist_model.to(dtype)
    model.eval()
    input_data = mnist_example_input.to(dtype)

    result = bits_per_weight(model)
    _assert_prediction_matches_export(model, input_data, result)


@pytest.mark.parametrize("dtype", [torch.int8, torch.int4], ids=["int8", "int4"])
def test_mnist_quant_prediction_matches_export(
    custom_test_mnist_model: nn.Module,
    mnist_example_input: torch.Tensor,
    dtype: torch.dtype,
) -> None:
    """Real conv/BN/linear model, int8 and int4 per-channel weights."""
    _assert_quantized_matches_export(custom_test_mnist_model, mnist_example_input, dtype)
