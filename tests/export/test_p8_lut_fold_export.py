# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

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
from tests.fixtures.palettization import ParametrizedPalettConfigs

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
) -> tuple[nn.Module, torch.Tensor, torch.Tensor]:
    """Palettize a single layer and finalize it for ``backend``.

    Returns:
        tuple[nn.Module, torch.Tensor, torch.Tensor]: The finalized model, the input data,
            and the prepared (fake-palettized) model's output on that input.
    """
    torch.manual_seed(0)
    model = nn.Sequential(make_layer()).eval()
    input_data = torch.randn(input_shape)
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
        lut_qspec=QuantizationSpec(dtype=lut_dtype, qscheme=qscheme),
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
