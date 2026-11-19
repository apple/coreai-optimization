# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for coreai_opt.coreai_utils._utils.quantize_utils.

Outputs are compared against the equivalent coremltools functions:
  - coremltools.optimize._utils.get_quant_range_by_dtype
  - coremltools.optimize._utils.quantize_weight_by_dtype
"""

import ml_dtypes
import numpy as np
import pytest

from coreai_opt.coreai_utils._coreai_imports import compression_types
from coreai_opt.coreai_utils._utils.quantize_utils import (
    _compute_qparams_by_dtype,
    _get_quantize_range_by_dtype,
    _quantize_data_by_dtype,
)
from coreai_opt.coreai_utils.common import QScheme
from tests.test_utils.markers import requires_coremltools

try:
    from coremltools.converters.mil.mil import types as cto_types
    from coremltools.optimize._utils import (
        get_quant_range_by_dtype as cto_get_quantize_range_by_dtype,
        quantize_weight_by_dtype as cto_quantize_data_by_dtype,
    )
except ImportError:  # coremltools is optional (the `coreml` extra).
    cto_types = None

# Int dtypes supported by both coreai and coremltools — used for direct comparison.
_SHARED_INT_DTYPES = ["int4", "uint4", "int8", "uint8"]
# Additional int dtypes supported by coreai only.
_COREAI_ONLY_INT_DTYPES = ["int2", "uint2"]
_MODES = [QScheme.SYMMETRIC, QScheme.ASYMMETRIC]

# Mapping from QScheme to the string expected by coremltools functions.
_CTO_MODE = {
    QScheme.SYMMETRIC: "LINEAR_SYMMETRIC",
    QScheme.ASYMMETRIC: "LINEAR",
}


class TestGetQuantRangeByDtype:
    @requires_coremltools
    @pytest.mark.parametrize("dtype_str", _SHARED_INT_DTYPES)
    @pytest.mark.parametrize("mode", _MODES)
    def test_matches_coremltools(self, dtype_str: str, mode: QScheme) -> None:
        """Output matches coremltools get_quant_range_by_dtype for shared dtypes."""
        coreai_dtype = compression_types.string_to_builtin(dtype_str)
        cto_dtype = cto_types.string_to_builtin(dtype_str)
        assert _get_quantize_range_by_dtype(coreai_dtype, mode) == cto_get_quantize_range_by_dtype(
            cto_dtype, _CTO_MODE[mode]
        )

    @pytest.mark.parametrize(
        ("dtype_str", "mode", "expected"),
        [
            ("int2", QScheme.ASYMMETRIC, (-2, 1)),
            ("int2", QScheme.SYMMETRIC, (-1, 1)),
            ("uint2", QScheme.ASYMMETRIC, (0, 3)),
            ("uint2", QScheme.SYMMETRIC, (0, 2)),
        ],
    )
    def test_coreai_only_dtypes(
        self, dtype_str: str, mode: QScheme, expected: tuple[int, int]
    ) -> None:
        """coreai-only sub-byte dtypes produce the correct range."""
        coreai_dtype = compression_types.string_to_builtin(dtype_str)
        assert _get_quantize_range_by_dtype(coreai_dtype, mode) == expected

    def test_raises_for_non_int_dtype(self) -> None:
        with pytest.raises(
            NotImplementedError, match="Only support getting quant range for int dtype"
        ):
            _get_quantize_range_by_dtype(compression_types.types_fp32, QScheme.ASYMMETRIC)

    @pytest.mark.parametrize(
        ("dtype", "expected"),
        [
            (ml_dtypes.float8_e4m3fn, (-448.0, 448.0)),
            (ml_dtypes.float8_e5m2, (-57344.0, 57344.0)),
        ],
    )
    def test_fp8_dtypes(self, dtype: type, expected: tuple[float, float]) -> None:
        """FP8 dtypes return the correct symmetric range regardless of mode."""
        assert _get_quantize_range_by_dtype(dtype, QScheme.SYMMETRIC) == expected
        assert _get_quantize_range_by_dtype(dtype, QScheme.ASYMMETRIC) == expected


class TestQuantizeDataByDtype:
    @requires_coremltools
    @pytest.mark.parametrize("dtype_str", _SHARED_INT_DTYPES)
    @pytest.mark.parametrize("mode", _MODES)
    def test_scale_close_to_coremltools(self, dtype_str: str, mode: QScheme) -> None:
        """Scale is approximately equal to coremltools (difference bounded by epsilon term)."""
        weight = np.array([[-2.0, 0.5, 1.0, -0.5], [0.2, 1.8, -1.5, 0.0]], dtype=np.float32)
        coreai_dtype = compression_types.string_to_builtin(dtype_str)
        cto_dtype = cto_types.string_to_builtin(dtype_str)
        axes = (0, 1)

        _, our_scale, our_zp = _quantize_data_by_dtype(weight, axes, coreai_dtype, mode)
        _, cto_scale, cto_zp = cto_quantize_data_by_dtype(weight, axes, cto_dtype, _CTO_MODE[mode])

        np.testing.assert_allclose(our_scale, cto_scale, rtol=1e-4)
        if our_zp is None:
            assert cto_zp is None
        else:
            assert cto_zp is not None
            np.testing.assert_allclose(our_zp, cto_zp, atol=1)

    @pytest.mark.parametrize("dtype_str", _SHARED_INT_DTYPES)
    def test_per_channel_output_shapes(self, dtype_str: str) -> None:
        """Per-channel quantization produces one scale value per output channel."""
        weight = np.random.randn(8, 4).astype(np.float32)
        coreai_dtype = compression_types.string_to_builtin(dtype_str)
        qdata, scale, _ = _quantize_data_by_dtype(weight, (1,), coreai_dtype, QScheme.SYMMETRIC)
        assert qdata.shape == weight.shape
        assert scale.shape == (8,)

    def test_raises_for_non_float_input(self) -> None:
        int_weight = np.array([[1, 2], [3, 4]], dtype=np.int32)
        coreai_dtype = compression_types.string_to_builtin("int8")
        with pytest.raises(ValueError, match="Only floating numpy arrays are supported"):
            _quantize_data_by_dtype(int_weight, (0, 1), coreai_dtype, QScheme.ASYMMETRIC)

    @pytest.mark.parametrize(
        ("fp8_dtype", "rtol"),
        [
            (ml_dtypes.float8_e4m3fn, 0.1),  # eps/2 = 0.0625
            (ml_dtypes.float8_e5m2, 0.15),  # eps/2 = 0.125
        ],
    )
    def test_fp8_per_tensor_dequantize(self, fp8_dtype: type, rtol: float) -> None:
        """Per-tensor FP8 quantization: dequantized values are close to input."""
        data = np.array([[-2.0, 0.5, 1.0, -0.5], [0.2, 1.8, -1.5, 0.0]], dtype=np.float32)
        qdata, scale, zp = _quantize_data_by_dtype(data, (0, 1), fp8_dtype, QScheme.SYMMETRIC)

        assert qdata.dtype == fp8_dtype
        assert qdata.shape == data.shape
        assert zp is not None
        assert np.all(zp.astype(np.float32) == 0.0)

        reconstructed = qdata.astype(np.float32) * scale
        np.testing.assert_allclose(reconstructed, data, rtol=rtol)

    @pytest.mark.parametrize(
        ("fp8_dtype", "rtol"),
        [
            (ml_dtypes.float8_e4m3fn, 0.1),  # eps/2 = 0.0625
            (ml_dtypes.float8_e5m2, 0.15),  # eps/2 = 0.125
        ],
    )
    def test_fp8_per_channel_dequantize(self, fp8_dtype: type, rtol: float) -> None:
        """Per-channel FP8 quantization: dequantized values are close to input."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((8, 16)).astype(np.float32)
        qdata, scale, zp = _quantize_data_by_dtype(data, (1,), fp8_dtype, QScheme.SYMMETRIC)

        assert qdata.dtype == fp8_dtype
        assert qdata.shape == data.shape
        assert scale.shape == (8,)
        assert zp is not None
        assert np.all(zp.astype(np.float32) == 0.0)

        reconstructed = qdata.astype(np.float32) * scale[:, np.newaxis]
        np.testing.assert_allclose(reconstructed, data, rtol=rtol)


class TestComputeQparamsByDtype:
    def test_raises_for_non_ndarray(self) -> None:
        with pytest.raises(ValueError, match="Only numpy arrays are supported"):
            _compute_qparams_by_dtype(
                [[1.0, 2.0]],
                compression_types.string_to_builtin("int8"),
                QScheme.ASYMMETRIC,
                [0, 0],
            )

    def test_raises_for_wrong_block_sizes_rank(self) -> None:
        weight = np.ones((4, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="must be equal"):
            _compute_qparams_by_dtype(
                weight, compression_types.string_to_builtin("int8"), QScheme.ASYMMETRIC, [0]
            )

    def test_returns_none_for_non_divisible_block_size(self) -> None:
        weight = np.ones((4, 8), dtype=np.float32)
        result = _compute_qparams_by_dtype(
            weight, compression_types.string_to_builtin("int8"), QScheme.ASYMMETRIC, [0, 3]
        )
        assert result is None

    @requires_coremltools
    @pytest.mark.parametrize("dtype_str", _SHARED_INT_DTYPES)
    @pytest.mark.parametrize("mode", _MODES)
    def test_per_tensor_close_to_coremltools(self, dtype_str: str, mode: QScheme) -> None:
        """Per-tensor scale and zero-point are approximately equal to coremltools."""
        weight = np.array([[-2.0, 0.5, 1.0, -0.5], [0.2, 1.8, -1.5, 0.0]], dtype=np.float32)
        coreai_dtype = compression_types.string_to_builtin(dtype_str)
        cto_dtype = cto_types.string_to_builtin(dtype_str)

        result = _compute_qparams_by_dtype(weight, coreai_dtype, mode, [0, 0])
        assert result is not None
        _, our_scale, our_zp = result

        _, cto_scale, cto_zp = cto_quantize_data_by_dtype(
            weight, (0, 1), cto_dtype, _CTO_MODE[mode]
        )

        np.testing.assert_allclose(our_scale, cto_scale, rtol=1e-4)
        if our_zp is None:
            assert cto_zp is None
        else:
            assert cto_zp is not None
            np.testing.assert_allclose(our_zp, cto_zp, atol=1)

    @pytest.mark.parametrize("dtype_str", _SHARED_INT_DTYPES)
    def test_per_channel_output_shapes(self, dtype_str: str) -> None:
        """Per-channel quantization produces scale shape (C_out, 1)."""
        weight = np.random.randn(8, 16).astype(np.float32)
        coreai_dtype = compression_types.string_to_builtin(dtype_str)
        result = _compute_qparams_by_dtype(weight, coreai_dtype, QScheme.SYMMETRIC, [1, 0])
        assert result is not None
        qdata, scale, _ = result
        assert qdata.shape == weight.shape
        assert scale.shape == (8, 1)

    @pytest.mark.parametrize("dtype_str", _SHARED_INT_DTYPES)
    def test_per_block_output_shapes(self, dtype_str: str) -> None:
        """Per-block quantization produces scale shape (C_out, n_blocks)."""
        weight = np.random.randn(8, 16).astype(np.float32)
        coreai_dtype = compression_types.string_to_builtin(dtype_str)
        result = _compute_qparams_by_dtype(weight, coreai_dtype, QScheme.SYMMETRIC, [1, 4])
        assert result is not None
        qdata, scale, _ = result
        assert qdata.shape == weight.shape
        assert scale.shape == (8, 4)
