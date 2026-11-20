# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import logging
from collections.abc import Callable

import pytest
import torch
from torch import nn

from coreai_opt import CoreMLExportError, ExportBackend
from coreai_opt.palettization import KMeansPalettizer
from coreai_opt.palettization.kmeans._prepare_for_export import (
    LUTQuantizationInfo,
    PalettizationInfo,
    _is_8bit_scalar_quantized_lut_palettization,
)
from coreai_opt.palettization.spec import PerGroupedChannelGranularity, PerTensorGranularity
from coreai_opt.quantization.spec import QuantizationScheme, QuantizationSpec
from tests.config.env import get_compute_unit_kind
from tests.fixtures.palettization import ParametrizedPalettConfigs
from tests.test_utils.joint_sparsity import (
    SPARSITY,
    SYMMETRIC_LUT_QSPEC,
    build_palettizer,
    presparsify,
)

from . import export_utils

_LAYERS = pytest.mark.parametrize(
    ("make_layer", "input_shape"),
    [
        (lambda: nn.Linear(64, 32), (4, 64)),
        (lambda: nn.Conv2d(16, 32, kernel_size=3), (1, 16, 8, 8)),
    ],
    ids=["linear", "conv"],
)
_FP8_DTYPES = pytest.mark.parametrize(
    "lut_dtype", [torch.float8_e4m3fn, torch.float8_e5m2], ids=["e4m3fn", "e5m2"]
)
_8BIT_LUT_DTYPES = pytest.mark.parametrize(
    "lut_dtype",
    [torch.int8, torch.uint8, torch.float8_e4m3fn, torch.float8_e5m2],
    ids=["int8", "uint8", "e4m3fn", "e5m2"],
)
_SYM = QuantizationScheme.SYMMETRIC
_ASYM = QuantizationScheme.ASYMMETRIC
_8BIT_LUT_DTYPES_AND_QSCHEMES = pytest.mark.parametrize(
    ("lut_dtype", "lut_qscheme"),
    [
        (torch.int8, _ASYM),
        (torch.int8, _SYM),
        (torch.uint8, _ASYM),
        (torch.uint8, _SYM),
        (torch.float8_e4m3fn, _SYM),
        (torch.float8_e5m2, _SYM),
    ],
    ids=["int8-asym", "int8-sym", "uint8-asym", "uint8-sym", "e4m3fn", "e5m2"],
)
_INT8_LUT_DTYPES_AND_QSCHEMES = pytest.mark.parametrize(
    ("lut_dtype", "lut_qscheme"),
    [(torch.int8, _ASYM), (torch.int8, _SYM), (torch.uint8, _ASYM), (torch.uint8, _SYM)],
    ids=["int8-asym", "int8-sym", "uint8-asym", "uint8-sym"],
)
_GRANULARITIES = pytest.mark.parametrize(
    "granularity",
    [PerTensorGranularity(), PerGroupedChannelGranularity(axis=0, group_size=8)],
    ids=["pertensor", "grouped"],
)
_PER_CHANNEL_SCALE = pytest.mark.parametrize(
    "enable_per_channel_scale", [False, True], ids=["nopcs", "pcs"]
)
_HALF_PRECISION_MODEL_DTYPES = pytest.mark.parametrize(
    "model_dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"]
)


def _palettize(
    make_layer: Callable[[], nn.Module],
    input_shape: tuple[int, ...],
    n_bits: int,
    lut_dtype: torch.dtype,
    enable_per_channel_scale: bool = False,
    backend: ExportBackend = ExportBackend.CoreAI,
    cluster_dim: int = 1,
    granularity: PerTensorGranularity | PerGroupedChannelGranularity = PerTensorGranularity(),  # noqa: B008
    lut_qscheme: QuantizationScheme | None = None,
    model_dtype: torch.dtype = torch.float32,
    lut_scale_dtype: torch.dtype | None = None,
    lut_float_range: list[float] | None = None,
) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    """Palettize a single layer and finalize it for ``backend``.

    Returns:
        tuple[nn.Module, torch.Tensor, torch.Tensor]: The finalized model, the input data,
            and the prepared (fake-palettized) model's output on that input.
    """
    torch.manual_seed(0)
    model = nn.Sequential(make_layer()).to(model_dtype).eval()
    input_data = torch.randn(input_shape, dtype=model_dtype)
    qscheme = lut_qscheme or (
        QuantizationScheme.ASYMMETRIC
        if lut_dtype in (torch.int8, torch.uint8)
        else QuantizationScheme.SYMMETRIC
    )
    config = ParametrizedPalettConfigs.from_palett_params(
        n_bits=n_bits,
        granularity=granularity,
        cluster_dim=cluster_dim,
        enable_per_channel_scale=enable_per_channel_scale,
        lut_qspec=QuantizationSpec(
            dtype=lut_dtype,
            qscheme=qscheme,
            scale_dtype=lut_scale_dtype,
            float_range=lut_float_range or [None, None],
        ),
    ).config

    palettizer = KMeansPalettizer(model, config)
    prepared = palettizer.prepare((input_data,))
    with torch.no_grad():
        prepared_model_output = prepared(input_data)
    return palettizer.finalize(backend=backend), input_data, prepared_model_output


@_PER_CHANNEL_SCALE
@_8BIT_LUT_DTYPES_AND_QSCHEMES
@_GRANULARITIES
@_LAYERS
def test_p8_lut_exports_as_weight_quantization(
    make_layer: Callable[[], nn.Module],
    input_shape: tuple[int, ...],
    lut_dtype: torch.dtype,
    lut_qscheme: QuantizationScheme,
    enable_per_channel_scale: bool,
    granularity: PerTensorGranularity | PerGroupedChannelGranularity,
) -> None:
    """8-bit scalar palettization with an int8, uint8 or float8 LUT is exported without a LUT op."""
    model, input_data, prepared_model_output = _palettize(
        make_layer,
        input_shape,
        8,
        lut_dtype,
        enable_per_channel_scale,
        granularity=granularity,
        lut_qscheme=lut_qscheme,
    )

    export_utils.convert_and_verify(
        finalized_model=model,
        input_data=input_data,
        expected_ops={"lut_to_dense": 0, "constexpr_blockwise_shift_scale": 1},
        export_backend=ExportBackend.CoreAI,
        prepared_model_output=prepared_model_output,
    )


@_PER_CHANNEL_SCALE
@_8BIT_LUT_DTYPES_AND_QSCHEMES
@_GRANULARITIES
@_LAYERS
def test_p8_weight_quantization_matches_lut_export_exactly(
    make_layer: Callable[[], nn.Module],
    input_shape: tuple[int, ...],
    lut_dtype: torch.dtype,
    lut_qscheme: QuantizationScheme,
    enable_per_channel_scale: bool,
    monkeypatch: pytest.MonkeyPatch,
    granularity: PerTensorGranularity | PerGroupedChannelGranularity,
) -> None:
    """The decompressed weight must match bit for bit that of the LUT export it replaces."""
    model, input_data, _ = _palettize(
        make_layer,
        input_shape,
        8,
        lut_dtype,
        enable_per_channel_scale,
        granularity=granularity,
        lut_qscheme=lut_qscheme,
    )
    with monkeypatch.context() as m:
        m.setattr(
            "coreai_opt.palettization.kmeans._prepare_for_export."
            "_is_8bit_scalar_quantized_lut_palettization",
            lambda _: False,
        )
        reference, _, _ = _palettize(
            make_layer,
            input_shape,
            8,
            lut_dtype,
            enable_per_channel_scale,
            granularity=granularity,
            lut_qscheme=lut_qscheme,
        )

    with torch.no_grad():
        assert torch.equal(model(input_data), reference(input_data))
    assert torch.equal(model[0].weight, reference[0].weight)


@_FP8_DTYPES
@_LAYERS
def test_p4_fp8_lut_keeps_lut(
    make_layer: Callable[[], nn.Module],
    input_shape: tuple[int, ...],
    lut_dtype: torch.dtype,
) -> None:
    """Palettization with fewer than 8 bits keeps the LUT."""
    model, input_data, prepared_model_output = _palettize(make_layer, input_shape, 4, lut_dtype)

    export_utils.convert_and_verify(
        finalized_model=model,
        input_data=input_data,
        expected_ops={"lut_to_dense": 1, "constexpr_blockwise_shift_scale": 1},
        export_backend=ExportBackend.CoreAI,
        prepared_model_output=prepared_model_output,
    )


@_HALF_PRECISION_MODEL_DTYPES
@_PER_CHANNEL_SCALE
@_LAYERS
@pytest.mark.parametrize(("n_bits", "lut_ops"), [(4, 1), (8, 0)], ids=["p4", "p8"])
def test_quantized_lut_export_half_precision_model(
    n_bits: int,
    lut_ops: int,
    make_layer: Callable[[], nn.Module],
    input_shape: tuple[int, ...],
    enable_per_channel_scale: bool,
    model_dtype: torch.dtype,
) -> None:
    """The LUT scale follows the model dtype, so a half-precision model can be exported."""
    is_conv = len(input_shape) == 4
    if get_compute_unit_kind() == "cpu" and model_dtype == torch.bfloat16 and is_conv:
        pytest.skip(reason="The CPU unit fails inference on a bf16 conv, palettized or not.")

    model, input_data, prepared_model_output = _palettize(
        make_layer,
        input_shape,
        n_bits,
        torch.int8,
        enable_per_channel_scale,
        model_dtype=model_dtype,
    )

    export_utils.convert_and_verify(
        finalized_model=model,
        input_data=input_data,
        expected_ops={
            "lut_to_dense": 1 if enable_per_channel_scale else lut_ops,
            "constexpr_blockwise_shift_scale": 2 if enable_per_channel_scale else 1,
        },
        export_backend=ExportBackend.CoreAI,
        prepared_model_output=prepared_model_output,
    )


def _small_channel_layer() -> nn.Module:
    """Return a Linear layer whose first output channel is 1e5 times smaller than the rest."""
    layer = nn.Linear(64, 32, bias=False)
    with torch.no_grad():
        layer.weight[0] *= 1e-5
    return layer


@_HALF_PRECISION_MODEL_DTYPES
@pytest.mark.parametrize("n_bits", [4, 8], ids=["p4", "p8"])
def test_small_channel_export_half_precision_model(n_bits: int, model_dtype: torch.dtype) -> None:
    """The exported scales reproduce the prepared weights, down to the smallest channel."""
    model, input_data, prepared_model_output = _palettize(
        _small_channel_layer,
        (4, 64),
        n_bits,
        torch.int8,
        enable_per_channel_scale=True,
        lut_qscheme=QuantizationScheme.SYMMETRIC,
        model_dtype=model_dtype,
    )

    assert prepared_model_output.any(dim=0).all()
    converted_model = export_utils.convert_and_verify(
        finalized_model=model,
        input_data=input_data,
        expected_ops={},
        export_backend=ExportBackend.CoreAI,
        prepared_model_output=prepared_model_output,
    )

    # Per output channel, so a zeroed small channel fails. fp16 outputs near 1e-5 are subnormal.
    (exported_output,) = export_utils.MLIRConverter()._run_inference(converted_model, input_data)
    error = (exported_output.float() - prepared_model_output.float()).abs().amax(dim=0)
    assert (error <= 0.1 * prepared_model_output.float().abs().amax(dim=0)).all()


def test_sparse_small_channel_export_fp16_model() -> None:
    """A sparse weight also keeps its per-channel scale apart from the LUT scale."""
    torch.manual_seed(0)
    # Pruned per output channel, so the small channel keeps half of its weights.
    model = presparsify(nn.Sequential(_small_channel_layer()), SPARSITY, group_size=1)
    model = model.to(torch.float16)
    input_data = torch.randn((4, 64), dtype=torch.float16)
    palettizer = build_palettizer(
        model, n_bits=4, enable_per_channel_scale=True, lut_qspec=SYMMETRIC_LUT_QSPEC
    )
    prepared = palettizer.prepare((input_data,))
    with torch.no_grad():
        prepared_model_output = prepared(input_data)

    assert prepared_model_output.any(dim=0).all()
    converted_model = export_utils.convert_and_verify(
        finalized_model=palettizer.finalize(),
        input_data=input_data,
        expected_ops={"sparse_to_dense": 1, "constexpr_blockwise_shift_scale": 2},
        export_backend=ExportBackend.CoreAI,
        prepared_model_output=prepared_model_output,
    )

    (exported_output,) = export_utils.MLIRConverter()._run_inference(converted_model, input_data)
    error = (exported_output.float() - prepared_model_output.float()).abs().amax(dim=0)
    assert (error <= 0.1 * prepared_model_output.float().abs().amax(dim=0)).all()


def _small_layer(bound: float) -> nn.Module:
    """Return a Linear layer with weights within ``bound`` of 0 and its first 4 rows at 0."""
    layer = nn.Linear(64, 32, bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.linspace(-bound, bound, 32 * 64).reshape(32, 64))
        layer.weight[:4] = 0
    return layer


@pytest.mark.parametrize(
    ("weight_bound", "lut_dtype", "lut_kwargs", "fallback", "max_error"),
    [
        # The e8m0 scale is 0 in float16
        (1e-6, torch.float8_e4m3fn, {"lut_scale_dtype": torch.float8_e8m0fnu}, True, 0.1),
        # The float_range scale is an inexact float16 subnormal
        (1.5e-4, torch.int8, {"lut_float_range": [-1.7e-4, 1.7e-4]}, True, 0.005),
        # The float_range scale overflows float16 and the prepared output is 0
        (0.3, torch.int8, {"lut_float_range": [-1e7, 1e7]}, True, 0.0),
        # The e8m0 scale is an exact float16 subnormal, or 0 for the all-zero group
        (1e-3, torch.float8_e4m3fn, {"lut_scale_dtype": torch.float8_e8m0fnu}, False, 0.01),
        # The float_range scale is an inexact normal float16 value
        (0.3, torch.int8, {"lut_float_range": [-0.37, 0.37]}, False, 0.01),
    ],
    ids=[
        "e8m0-zero",
        "float-range-subnormal",
        "float-range-overflow",
        "e8m0-subnormal",
        "float-range-normal",
    ],
)
def test_lut_scale_out_of_fp16_range_exports_unquantized_lut(
    caplog: pytest.LogCaptureFixture,
    weight_bound: float,
    lut_dtype: torch.dtype,
    lut_kwargs: dict[str, torch.dtype | list[float]],
    fallback: bool,
    max_error: float,
) -> None:
    """Only a LUT scale that overflows or loses precision in float16 exports the LUT unquantized."""
    if get_compute_unit_kind() == "neural_engine" and weight_bound < 1e-3:
        pytest.skip(reason="The Neural Engine is inaccurate on fp16 weights this small.")

    with caplog.at_level(
        logging.WARNING, logger="coreai_opt.palettization.kmeans._prepare_for_export"
    ):
        model, input_data, prepared_model_output = _palettize(
            lambda: _small_layer(weight_bound),
            (4, 64),
            8,
            lut_dtype,
            granularity=PerGroupedChannelGranularity(group_size=4),
            lut_qscheme=QuantizationScheme.SYMMETRIC,
            model_dtype=torch.float16,
            **lut_kwargs,
        )
    assert ("exporting the LUT unquantized" in caplog.text) == fallback

    converted_model = export_utils.convert_and_verify(
        finalized_model=model,
        input_data=input_data,
        expected_ops=(
            {"lut_to_dense": 1}
            if fallback
            else {"lut_to_dense": 0, "constexpr_blockwise_shift_scale": 1}
        ),
        export_backend=ExportBackend.CoreAI,
        prepared_model_output=prepared_model_output,
    )

    # The SNR check can pass an all-zero output, so compare values directly.
    (exported_output,) = export_utils.MLIRConverter()._run_inference(converted_model, input_data)
    error = (exported_output.float() - prepared_model_output.float()).abs().max()
    assert error <= max_error * prepared_model_output.float().abs().max()


def test_p8_lut_fold_small_values_stay_8bit() -> None:
    """Folded 8-bit values that fit in fewer bits keep their 8-bit zero point."""

    def make_layer() -> nn.Module:
        layer = nn.Linear(64, 32)
        nn.init.constant_(layer.weight, -0.05)
        return layer

    model, input_data, prepared_model_output = _palettize(make_layer, (4, 64), 8, torch.uint8)

    export_utils.convert_and_verify(
        finalized_model=model,
        input_data=input_data,
        expected_ops={"lut_to_dense": 0, "constexpr_blockwise_shift_scale": 1},
        export_backend=ExportBackend.CoreAI,
        prepared_model_output=prepared_model_output,
    )


@_INT8_LUT_DTYPES_AND_QSCHEMES
@_GRANULARITIES
@_LAYERS
def test_p8_int8_lut_exports_as_weight_quantization_coreml(
    make_layer: Callable[[], nn.Module],
    input_shape: tuple[int, ...],
    lut_dtype: torch.dtype,
    lut_qscheme: QuantizationScheme,
    granularity: PerTensorGranularity | PerGroupedChannelGranularity,
) -> None:
    """CoreML export of 8-bit scalar palettization with an int8 or uint8 LUT has no LUT op."""
    model, input_data, prepared_model_output = _palettize(
        make_layer,
        input_shape,
        8,
        lut_dtype,
        backend=ExportBackend.CoreML,
        granularity=granularity,
        lut_qscheme=lut_qscheme,
    )

    export_utils.convert_and_verify(
        finalized_model=model,
        input_data=input_data,
        expected_ops={"constexpr_lut_to_dense": 0, "constexpr_blockwise_shift_scale": 1},
        export_backend=ExportBackend.CoreML,
        prepared_model_output=prepared_model_output,
    )


@_FP8_DTYPES
def test_p8_fp8_lut_coreml_export_stays_unsupported(lut_dtype: torch.dtype) -> None:
    """CoreML does not support float8, so the fold must not make fp8 palettization exportable."""
    with pytest.raises(CoreMLExportError, match=export_utils.COREML_REJECTION_MATCH):
        _palettize(lambda: nn.Linear(64, 32), (4, 64), 8, lut_dtype, backend=ExportBackend.CoreML)


@_8BIT_LUT_DTYPES
@_LAYERS
def test_p8_vector_palettization_keeps_lut(
    make_layer: Callable[[], nn.Module],
    input_shape: tuple[int, ...],
    lut_dtype: torch.dtype,
) -> None:
    """Folding vector palettization would store 8 bits per weight instead of 8/cluster_dim."""
    model, input_data, prepared_model_output = _palettize(
        make_layer, input_shape, 8, lut_dtype, cluster_dim=2
    )

    export_utils.convert_and_verify(
        finalized_model=model,
        input_data=input_data,
        expected_ops={"lut_to_dense": 1, "constexpr_blockwise_shift_scale": 1},
        export_backend=ExportBackend.CoreAI,
        prepared_model_output=prepared_model_output,
    )


@_8BIT_LUT_DTYPES
@pytest.mark.parametrize(
    ("cluster_dim", "expected"), [(1, True), (2, False)], ids=["scalar", "vector"]
)
def test_only_8bit_scalar_lut_palettization_is_rewritten(
    lut_dtype: torch.dtype, cluster_dim: int, expected: bool
) -> None:
    """Vector palettization is excluded: folding it would store 8 bits per weight instead of
    8 / cluster_dim, undoing its compression.
    """
    n_clusters = 2**8
    palett_info = PalettizationInfo(
        lut=torch.zeros(1, 1, n_clusters, cluster_dim),
        indices=torch.zeros(32, 64, dtype=torch.uint8),
        cluster_dim=cluster_dim,
        lut_quantization=LUTQuantizationInfo(
            n_bits=8,
            quantized_lut=torch.zeros(1, 1, n_clusters, cluster_dim).to(lut_dtype),
            scale=torch.ones(1, 1, 1, 1),
        ),
    )

    assert _is_8bit_scalar_quantized_lut_palettization(palett_info) is expected
