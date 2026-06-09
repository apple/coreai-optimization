# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import os
import tempfile
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

from coreai_opt.palettization.kmeans.kmeans_fake_palettize import _KMeansFakePalettize
from coreai_opt.palettization.kmeans.kmeans_support_mixins import _PalettizationSupportMixin
from coreai_opt.palettization.kmeans.supported_ops_registry import (
    _KMeansPalettizerSupportedOpsRegistry,
)
from coreai_opt.palettization.spec import (
    PalettizationSpec,
    PerGroupedChannelGranularity,
    PerTensorGranularity,
)
from coreai_opt.palettization.spec.errors import (
    _IncompatibleClusterDimError,
    _IncompatibleGranularityError,
)
from coreai_opt.palettization.spec.spec import _SUPPORTED_LUT_DTYPES
from coreai_opt.quantization.spec import QuantizationScheme, QuantizationSpec


def _make_lut_qspec(
    lut_dtype: torch.dtype | None,
    lut_qscheme: QuantizationScheme | None = None,
) -> QuantizationSpec | None:
    """Helper to build a QuantizationSpec for LUT quantization, or None."""
    if lut_dtype is None:
        return None
    if lut_qscheme is None:
        lut_qscheme = QuantizationScheme.SYMMETRIC
    return QuantizationSpec(dtype=lut_dtype, qscheme=lut_qscheme)


def _valid_lut_dtype_qscheme_combinations():
    """Return valid (dtype, qscheme) pairs for LUT quantization.

    FP8 dtypes only support symmetric quantization, while integer dtypes
    support both symmetric and asymmetric.
    """
    combos = []
    for dtype in sorted(_SUPPORTED_LUT_DTYPES, key=str):
        dtype_name = str(dtype).removeprefix("torch.")
        combos.append(
            pytest.param(dtype, QuantizationScheme.SYMMETRIC, id=f"{dtype_name}-symmetric")
        )
        if not dtype.is_floating_point:
            combos.append(
                pytest.param(dtype, QuantizationScheme.ASYMMETRIC, id=f"{dtype_name}-asymmetric")
            )
    return combos


class Test_KMeansFakePalettize:
    """Test cases for _KMeansFakePalettize class."""

    @pytest.mark.parametrize(
        "weight_dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["fp32", "fp16", "bf16"]
    )
    @pytest.mark.parametrize("lut_dtype", [None, *_SUPPORTED_LUT_DTYPES])
    def test__calculate_centroids_simple_per_tensor(self, lut_dtype, weight_dtype):
        """Test _calculate_centroids with simple random weight matrix
        using per-tensor granularity.
        """
        # Create a simple weight matrix with known values for easy verification
        weight = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=weight_dtype)

        # Create palettization spec with 2-bit (4 centroids)
        spec = PalettizationSpec(
            n_bits=2,  # 4 centroids max
            granularity=PerTensorGranularity(),
            cluster_dim=1,
            lut_qspec=_make_lut_qspec(lut_dtype),
        )

        # Create _KMeansFakePalettize instance
        palettizer = _KMeansFakePalettize(
            n_bits=spec.n_bits,
            lut_qspec=spec.lut_qspec,
            granularity=spec.granularity,
            cluster_dim=spec.cluster_dim,
            enable_per_channel_scale=spec.enable_per_channel_scale,
        )

        # Verify centroid properties
        lut, indices = palettizer._calculate_centroids(weight)
        assert lut.shape == (1, 1, 2**spec.n_bits, 1)
        assert lut.dtype == weight.dtype

        # Test weight reconstruction
        palettized_weight = palettizer._palettize(lut, indices, weight)

        # Verify the reconstructed weight has the same shape and dtype
        assert palettized_weight.shape == weight.shape
        assert palettized_weight.dtype == weight.dtype

        # Verify that reconstruction gives a reasonable approximation
        mse = torch.mean((weight - palettized_weight) ** 2)
        print(f"Reconstruction MSE: {mse}")
        assert mse < 0.5

    def test_cluster_weights_1d_with_few_unique_values(self):
        """Test _cluster_weights_1d when there are fewer unique values than clusters."""
        # Create weight with only 2 unique values but request 4 clusters
        weight = torch.tensor([[1.0, 1.0, 2.0], [2.0, 1.0, 2.0]], dtype=torch.float32)

        spec = PalettizationSpec(n_bits=2, granularity=PerTensorGranularity(), cluster_dim=1)
        palettizer = _KMeansFakePalettize(
            n_bits=spec.n_bits,
            lut_qspec=spec.lut_qspec,
            granularity=spec.granularity,
            cluster_dim=spec.cluster_dim,
            enable_per_channel_scale=spec.enable_per_channel_scale,
        )

        centroids, indices = palettizer._calculate_centroids(weight)

        assert centroids.shape == (1, 1, 2**spec.n_bits, 1)

        centroids = centroids.flatten()

        # Should have exactly 2 unique centroid values
        assert len(torch.unique(centroids)) == 2

        # Padded entries should be filled with the last real centroid (not zeros)
        assert torch.all(centroids[2:] == centroids[1])

        # All cluster indices should be valid and have the same shape as weight
        assert indices.shape == weight.shape
        assert torch.all(indices >= 0) and torch.all(indices < 2)

    @pytest.mark.parametrize("lut_dtype", [None, *_SUPPORTED_LUT_DTYPES])
    def test__calculate_centroids_per_grouped_channel_axis_0(self, lut_dtype):
        """Test _calculate_centroids with per-grouped-channel granularity"""
        weight = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [12.0, 13.0, 14.0, 15.0],
            ],
            dtype=torch.float32,
        )

        spec = PalettizationSpec(
            n_bits=1,  # 2 centroids max to keep simple
            granularity=PerGroupedChannelGranularity(axis=0, group_size=2),
            cluster_dim=1,
            lut_qspec=_make_lut_qspec(lut_dtype),
        )

        palettizer = _KMeansFakePalettize(
            n_bits=spec.n_bits,
            lut_qspec=spec.lut_qspec,
            granularity=spec.granularity,
            cluster_dim=spec.cluster_dim,
            enable_per_channel_scale=spec.enable_per_channel_scale,
        )

        lut, indices = palettizer._calculate_centroids(weight)

        # Test weight reconstruction for per-grouped-channel granularity
        palettized_weight = palettizer._palettize(lut, indices, weight)

        # Verify the reconstructed weight has the same shape
        assert palettized_weight.shape == weight.shape
        assert palettized_weight.dtype == weight.dtype

        # Verify that reconstruction gives a reasonable approximation
        mse = torch.mean((weight - palettized_weight) ** 2)
        print(f"Reconstruction MSE (axis=0): {mse}")
        assert mse < 2.0

    @pytest.mark.parametrize("lut_dtype", [None, *_SUPPORTED_LUT_DTYPES])
    def test__calculate_centroids_per_grouped_channel_axis_1(self, lut_dtype):
        """Test _calculate_centroids with per-grouped-channel granularity"""
        weight = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
                [12.0, 13.0, 14.0, 15.0],
            ],
            dtype=torch.float32,
        )

        spec = PalettizationSpec(
            n_bits=2,  # 4 centroids max
            granularity=PerGroupedChannelGranularity(axis=1, group_size=2),
            cluster_dim=1,
            lut_qspec=_make_lut_qspec(lut_dtype),
        )

        palettizer = _KMeansFakePalettize(
            n_bits=spec.n_bits,
            lut_qspec=spec.lut_qspec,
            granularity=spec.granularity,
            cluster_dim=spec.cluster_dim,
            enable_per_channel_scale=spec.enable_per_channel_scale,
        )

        lut, indices = palettizer._calculate_centroids(weight)

        # Test weight reconstruction for per-grouped-channel granularity
        palettized_weight = palettizer._palettize(lut, indices, weight)

        # Verify the reconstructed weight has the same shape
        assert palettized_weight.shape == weight.shape
        assert palettized_weight.dtype == weight.dtype

        # Verify that reconstruction gives a reasonable approximation
        mse = torch.mean((weight - palettized_weight) ** 2)
        print(f"Reconstruction MSE (axis=0): {mse}")
        assert mse < 2.0

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_cluster_weights_1d_fast_mode_vs_regular(self, dtype):
        """Test that fast mode and regular mode produce reasonable results.

        Hand-crafted weights such that fp16 cast + rounding-to-4-decimals can flip a
        borderline point between clusters, causing a larger centroid shift.
        """
        weight = torch.tensor(
            [
                [0.15312, 0.06250, 0.07050, -0.00505],
                [0.06665, -0.05735, 0.01355, -0.11975],
                [0.04765, -0.01565, 0.27115, 0.06115],
            ],
            dtype=dtype,
        )

        spec = PalettizationSpec(n_bits=2, granularity=PerTensorGranularity(), cluster_dim=1)

        palettizer_fast = _KMeansFakePalettize(
            n_bits=spec.n_bits,
            lut_qspec=spec.lut_qspec,
            granularity=spec.granularity,
            cluster_dim=spec.cluster_dim,
            enable_per_channel_scale=spec.enable_per_channel_scale,
            enable_fast_kmeans_mode=True,
        )

        palettizer_regular = _KMeansFakePalettize(
            n_bits=spec.n_bits,
            lut_qspec=spec.lut_qspec,
            granularity=spec.granularity,
            cluster_dim=spec.cluster_dim,
            enable_per_channel_scale=spec.enable_per_channel_scale,
            enable_fast_kmeans_mode=False,
        )

        centroids_fast, clusters_fast = palettizer_fast._calculate_centroids(weight)
        centroids_regular, clusters_regular = palettizer_regular._calculate_centroids(weight)

        # Both should produce valid results
        assert len(centroids_fast) <= 4
        assert len(centroids_regular) <= 4
        assert clusters_fast.numel() == weight.numel()
        assert clusters_regular.numel() == weight.numel()

        # For fp16 input the fast-mode fp16 cast is a no-op, so centroids
        # match closely. For fp32 input the fp16 cast + rounding can flip a
        # borderline point between clusters, causing a larger centroid shift.
        atol = 0.001 if dtype == torch.float16 else 0.06
        torch.testing.assert_close(centroids_fast, centroids_regular, rtol=atol, atol=atol)

    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_reduce_weights_to_cluster_fast_mode(self, dtype):
        """Test _reduce_weights_to_cluster in fast mode."""
        weight = torch.tensor([1.123456, 1.123456, 2.987654, 2.987654, 3.456789], dtype=dtype)

        spec = PalettizationSpec(n_bits=2, granularity=PerTensorGranularity(), cluster_dim=1)
        palettizer = _KMeansFakePalettize(
            n_bits=spec.n_bits,
            lut_qspec=spec.lut_qspec,
            granularity=spec.granularity,
            cluster_dim=spec.cluster_dim,
            enable_per_channel_scale=spec.enable_per_channel_scale,
            enable_fast_kmeans_mode=True,
            rounding_precision=2,  # Round to 2 decimal places
        )

        values, indices, counts = palettizer._reduce_weights_to_cluster(weight.numpy())

        # Should have reduced unique values due to rounding
        assert len(values) < len(weight)
        assert len(indices) == len(weight)
        assert len(counts) == len(values)
        assert sum(counts) == len(weight)

        reconstructed_weight = torch.from_numpy(values[indices]).to(dtype)

        # Since rounding precision is 2, atol=1e-03 should fail
        assert not torch.all(torch.isclose(weight, reconstructed_weight, atol=0.001))
        # atol=1e-02 should pass
        torch.testing.assert_close(weight, reconstructed_weight, rtol=0.01, atol=0.01)

    class TestGroupSizeValidation:
        @pytest.mark.parametrize(
            "weight_shape, axis, group_size",
            [
                ((4, 7), 0, 2),
                ((4, 7), 0, 4),
                ((2, 6), 1, 3),
                ((2, 6), 1, 6),
            ],
        )
        def test_group_size_divisibility_validation_divisible(self, weight_shape, axis, group_size):
            """Test valid granularity cases."""
            weight = torch.randn(weight_shape)

            # Create fake palettize with per-grouped-channel granularity
            fake_palettize = _KMeansFakePalettize(
                n_bits=2,
                lut_qspec=None,
                granularity=PerGroupedChannelGranularity(axis=axis, group_size=group_size),
                cluster_dim=1,
                enable_per_channel_scale=False,
            )

            lut, indices = fake_palettize._calculate_centroids(weight)
            assert lut is not None
            assert indices is not None
            num_blocks = fake_palettize.granularity.num_blocks_to_cluster(weight)
            assert lut.numel() == num_blocks * 4  # n_bits = 2

        @pytest.mark.parametrize(
            "weight_shape, axis, group_size",
            [
                ((5, 8), 0, 3),
                ((3, 7), 1, 3),
            ],
        )
        def test_group_size_divisibility_validation_indivisible(
            self, weight_shape, axis, group_size
        ):
            """Test that group size divisibility validation warning."""
            weight = torch.randn(weight_shape)

            # Create fake palettize with per-grouped-channel granularity
            fake_palettize = _KMeansFakePalettize(
                n_bits=2,
                lut_qspec=None,
                granularity=PerGroupedChannelGranularity(axis=axis, group_size=group_size),
                cluster_dim=1,
                enable_per_channel_scale=False,
            )

            with pytest.raises(_IncompatibleGranularityError):
                fake_palettize._calculate_centroids(weight)

        def test_group_size_divisibility_validation_insufficient_dimensions(self):
            """Test that validation warns for insufficient parameter dimensions."""

            # Create a 1D weight tensor - insufficient dimensions for axis 1
            weight = torch.randn(10)

            # Create fake palettize with per-grouped-channel granularity on axis 1
            fake_palettize = _KMeansFakePalettize(
                n_bits=2,
                lut_qspec=None,
                granularity=PerGroupedChannelGranularity(axis=1, group_size=2),
                cluster_dim=1,
                enable_per_channel_scale=False,
            )

            # Should warn and skip because parameter is 1D but axis 1 was specified
            with pytest.raises(_IncompatibleGranularityError):
                fake_palettize._calculate_centroids(weight)

    def test_disabled_flag_behavior(self):
        """Test that _disabled flag works correctly and permanently
        disables palettization.
        """
        # Create a weight with incompatible dimensions
        weight = torch.randn(5, 8)  # 5 is not divisible by 3

        # Create fake palettize with incompatible granularity
        fake_palettize = _KMeansFakePalettize(
            n_bits=2,
            lut_qspec=None,
            granularity=PerGroupedChannelGranularity(axis=0, group_size=3),  # 5 % 3 != 0
            cluster_dim=1,
            enable_per_channel_scale=False,
        )

        # Initially not disabled
        assert fake_palettize._disabled is False

        # Spy on _calculate_centroids to verify it's called
        original__calculate_centroids = fake_palettize._calculate_centroids
        fake_palettize._calculate_centroids = Mock(side_effect=original__calculate_centroids)

        # Call forward - this should trigger the _IncompatibleGranularityError
        # and set _disabled = True
        result = fake_palettize.forward(weight)

        # Should now be disabled
        assert fake_palettize._disabled is True
        assert torch.equal(result, weight)  # Should return original tensor

        # _calculate_centroids should have been called once
        assert fake_palettize._calculate_centroids.call_count == 1

        # Call forward again - should immediately return original tensor
        result2 = fake_palettize.forward(weight)

        # Should still be disabled and return original tensor
        assert fake_palettize._disabled is True
        assert torch.equal(result2, weight)

        # _calculate_centroids should still have been called only once
        # (not called the second time)
        assert fake_palettize._calculate_centroids.call_count == 1

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_mps_device_handling(self):
        """
        Test that palettization preserves MPS device for weights while
        using CPU for computation.
        """
        # Create weights on MPS device
        weight = torch.randn(4, 8, dtype=torch.float32, device="mps")

        # Verify input is on MPS
        assert weight.device.type == "mps"

        spec = PalettizationSpec(n_bits=2, granularity=PerTensorGranularity(), cluster_dim=1)

        palettizer = _KMeansFakePalettize(
            n_bits=spec.n_bits,
            lut_qspec=spec.lut_qspec,
            granularity=spec.granularity,
            cluster_dim=spec.cluster_dim,
            enable_per_channel_scale=spec.enable_per_channel_scale,
        )

        # Calculate centroids - this should move computation to CPU internally
        lut, indices = palettizer._calculate_centroids(weight)

        # Verify LUT and indices are on CPU (expected behavior)
        assert lut.device.type == "cpu"
        assert indices.device.type == "cpu"

        # Palettize the weights - this should return result on original device (MPS)
        palettized_weight = palettizer._palettize(lut, indices, weight)

        # Verify the palettized weights are back on MPS
        assert palettized_weight.device.type == "mps"
        assert palettized_weight.shape == weight.shape
        assert palettized_weight.dtype == weight.dtype


class TestPerGroupedChannelAxisDefault:
    """Test per-op axis-default resolution for PerGroupedChannelGranularity."""

    @pytest.mark.parametrize(
        "op_to_optimize",
        [F.linear, F.conv1d, F.conv2d, F.conv3d, F.multi_head_attention_forward, None],
    )
    def test_axis_resolved_from_op_when_unset(self, op_to_optimize):
        """When axis is None, _KMeansFakePalettize resolves it from the op's mixin."""
        palettizer = _KMeansFakePalettize(
            n_bits=2,
            lut_qspec=None,
            granularity=PerGroupedChannelGranularity(group_size=2),
            cluster_dim=1,
            enable_per_channel_scale=False,
            op_to_optimize=op_to_optimize,
        )
        assert palettizer.granularity.axis == palettizer.reshape_strategy.default_axis

    @pytest.mark.parametrize("axis", [0, 1])
    def test_explicit_axis_preserved(self, axis):
        """An explicitly-set axis is not overridden by the op's mixin default."""
        palettizer = _KMeansFakePalettize(
            n_bits=2,
            lut_qspec=None,
            granularity=PerGroupedChannelGranularity(axis=axis, group_size=2),
            cluster_dim=1,
            enable_per_channel_scale=False,
            op_to_optimize=F.linear,
        )
        assert palettizer.granularity.axis == axis

    def test_custom_mixin_with_default_axis_one(self):
        """A custom registered mixin with default_axis=1 resolves axis to 1."""

        def _fake_op(weight, x):
            return x

        registry_key = "_test_custom_axis1_op"

        @_KMeansPalettizerSupportedOpsRegistry.register(registry_key)
        class _CustomAxis1Support(_PalettizationSupportMixin):
            ops = [_fake_op]
            default_axis = 1

            def reshape_for_kmeans(self, weight, axis):
                return weight

            def reshape_to_original(self, clustered_weight, axis, original_shape):
                return clustered_weight

        try:
            palettizer = _KMeansFakePalettize(
                n_bits=2,
                lut_qspec=None,
                granularity=PerGroupedChannelGranularity(group_size=2),
                cluster_dim=1,
                enable_per_channel_scale=False,
                op_to_optimize=_fake_op,
            )
            assert isinstance(palettizer.reshape_strategy, _CustomAxis1Support)
            assert palettizer.granularity.axis == 1
        finally:
            _KMeansPalettizerSupportedOpsRegistry.REGISTRY.pop(registry_key, None)


class TestQuantizedLUT:
    """Test cases for quantized LUT support in _KMeansFakePalettize."""

    @pytest.mark.parametrize("lut_dtype, lut_qscheme", _valid_lut_dtype_qscheme_combinations())
    def test_quantized_lut_centroid(self, lut_dtype, lut_qscheme):
        """Verify centroids are properly quantized/dequantized.

        Quantized centroids should be close to unquantized ones but not identical,
        and should have fewer distinct centroid values than unquantized.
        """

        # Use a large weight to get many distinct float centroids
        weight = torch.randn(32, 64, dtype=torch.float32)

        spec_base = PalettizationSpec(
            n_bits=8,
            granularity=PerTensorGranularity(),
            cluster_dim=1,
        )

        spec_quant = PalettizationSpec(
            n_bits=8,  # 256 clusters for many distinct values
            granularity=PerTensorGranularity(),
            cluster_dim=1,
            lut_qspec=_make_lut_qspec(lut_dtype, lut_qscheme),
        )

        palettizer_base = _KMeansFakePalettize(**spec_base.__dict__)
        palettizer_quant = _KMeansFakePalettize(**spec_quant.__dict__)

        lut_base, indices_base = palettizer_base._calculate_centroids(weight)
        lut_quant, indices_quant = palettizer_quant._calculate_centroids(weight)

        # LUT shapes should be the same
        assert lut_base.shape == lut_quant.shape

        # Quantized centroids should be close to unquantized, but not identical
        assert not torch.equal(lut_base, lut_quant), (
            "Quantized and unquantized LUT should be different."
        )
        assert torch.allclose(lut_base, lut_quant, atol=0.5), (
            f"Quantized LUT too far from unquantized. "
            f"Max diff: {(lut_base - lut_quant).abs().max():.4f}"
        )

        # Quantized centroids should have fewer unique values
        unique_base = len(torch.unique(lut_base))
        unique_quant = len(torch.unique(lut_quant))

        assert unique_quant < unique_base, (
            f"Quantized LUT should have <= unique values. "
            f"Got {unique_quant} (quant) vs {unique_base} (base)"
        )

        # Quantized centroids should have higher reconstruction error
        palettized_base = palettizer_base._palettize(lut_base, indices_base, weight)
        mse_base = torch.mean((weight - palettized_base) ** 2)

        palettized_quant = palettizer_base._palettize(lut_quant, indices_quant, weight)
        mse_quant = torch.mean((weight - palettized_quant) ** 2)
        assert mse_quant > mse_base, (
            f"Reconstruction MSE for quantized centroids: {mse_quant:.4f} should be higher than"
            f"that of unquantized centroids: {mse_base:.4f}"
        )

    @pytest.mark.parametrize(
        "lut_dtype, lut_qscheme, expected_lut",
        [
            pytest.param(
                torch.int8,
                QuantizationScheme.SYMMETRIC,
                # scale = 4.0/127 ≈ 0.031373; q = round(x/s) → [32, 64, 96, 127]
                # dequant = q * s → [1.0039, 2.0078, 3.0118, 3.9843]
                torch.tensor([1.0039, 2.0078, 3.0118, 3.9843]),
                id="int8_symmetric",
            ),
            pytest.param(
                torch.int8,
                QuantizationScheme.ASYMMETRIC,
                # scale ≈ 0.015686; zp = -128; q = [-64, -1, 63, 127]
                # dequant = (q - zp) * s → [1.0039, 1.9922, 2.9961, 4.0]
                torch.tensor([1.0039, 1.9922, 2.9961, 4.0000]),
                id="int8_asymmetric",
            ),
            pytest.param(
                torch.uint8,
                QuantizationScheme.SYMMETRIC,
                # scale ≈ 0.031373; zp = 128; q = [160, 192, 224, 255]
                # dequant = (q - zp) * s → [1.0039, 2.0078, 3.0118, 3.9843]
                torch.tensor([1.0039, 2.0078, 3.0118, 3.9843]),
                id="uint8_symmetric",
            ),
            pytest.param(
                torch.uint8,
                QuantizationScheme.ASYMMETRIC,
                # scale ≈ 0.01569; zp = 0; q = [64, 127, 191, 255]
                # dequant = (q - zp) * s → [1.0039, 1.9922, 2.9961, 4.0]
                torch.tensor([1.0039, 1.9922, 2.9961, 4.0000]),
                id="uint8_asymmetric",
            ),
        ],
    )
    def test_quantized_lut_exact_values(self, lut_dtype, lut_qscheme, expected_lut):
        """Verify exact LUT values after quantize-dequantize for a deterministic case.

        Uses weight [1, 2, 3, 4] with 2-bit (4 clusters) so k-means produces
        exact centroids [1.0, 2.0, 3.0, 4.0]. Then verifies the quantized LUT
        matches the expected quantize→dequantize roundtrip values.
        """
        weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)

        spec = PalettizationSpec(
            n_bits=2,
            granularity=PerTensorGranularity(),
            cluster_dim=1,
            lut_qspec=_make_lut_qspec(lut_dtype, lut_qscheme),
        )

        palettizer = _KMeansFakePalettize(**spec.__dict__)
        lut, indices = palettizer._calculate_centroids(weight)

        # LUT shape: [1, 1, 4, 1] for per-tensor with 2-bit, cluster_dim=1
        assert lut.shape == (1, 1, 4, 1)

        # Extract and sort the 4 centroid values for comparison
        lut_values = lut.flatten().sort().values

        assert torch.allclose(lut_values, expected_lut, atol=1e-4), (
            f"LUT values mismatch for {lut_dtype}/{lut_qscheme}.\n"
            f"  Expected: {expected_lut}\n"
            f"  Got:      {lut_values}"
        )

        # Verify reconstruction works
        palettized = palettizer._palettize(lut, indices, weight)
        assert palettized.shape == weight.shape

        # Reconstruction should map to quantized centroid values
        unique_reconstructed = torch.unique(palettized).sort().values
        assert torch.allclose(unique_reconstructed, expected_lut, atol=1e-4), (
            f"Reconstructed values should match quantized centroids.\n"
            f"  Expected: {expected_lut}\n"
            f"  Got:      {unique_reconstructed}"
        )

    @pytest.mark.parametrize("lut_dtype, lut_qscheme", _valid_lut_dtype_qscheme_combinations())
    def test_quantized_lut_with_sensitivities(self, lut_dtype, lut_qscheme):
        """Test quantized LUT works correctly with sensitivity-weighted k-means."""
        weight = torch.tensor(
            [
                [10.0, 10.1, 10.2, 10.3],
                [1.0, 1.1, 1.2, 1.3],
            ],
            dtype=torch.float32,
        )

        sensitivities = torch.tensor(
            [
                [0.1, 0.1, 0.1, 0.1],
                [10.0, 10.0, 10.0, 10.0],
            ],
            dtype=torch.float32,
        )

        spec = PalettizationSpec(
            n_bits=2,
            granularity=PerTensorGranularity(),
            cluster_dim=1,
        )

        palettizer = _KMeansFakePalettize(
            **spec.__dict__,
            sensitivities=sensitivities,
            enable_fast_kmeans_mode=True,
        )

        lut, indices = palettizer._calculate_centroids(weight)
        palettized = palettizer._palettize(lut, indices, weight)

        # Shape should be preserved
        assert palettized.shape == weight.shape

        # Reconstruction should be reasonable despite quantization
        mse = torch.mean((weight - palettized) ** 2)
        assert mse < 0.5, f"Reconstruction MSE too high: {mse:.4f}"


class TestPerChannelScaling:
    """Test per-channel scaling functionality in _KMeansFakePalettize."""

    @pytest.mark.parametrize(
        "shape",
        [
            (4, 8),  # Basic 2D
            (16, 32),  # Larger 2D
            (8, 4, 3, 3),  # 4D conv weight
            (32, 16, 5),  # 3D conv1d weight
            (1, 10),  # Single channel
            (10, 1),  # Single feature
        ],
    )
    def test_scale_by_per_channel_scale_basic(self, shape):
        """Test basic per-channel scaling functionality."""
        spec = PalettizationSpec(
            n_bits=2, granularity=PerTensorGranularity(), enable_per_channel_scale=True
        )

        palettizer = _KMeansFakePalettize(**spec.__dict__)

        # Create test weight with known values
        weight = torch.randn(shape) * 10  # Scale up for easier testing

        # Apply scaling
        scaled_weight = palettizer._scale_by_per_channel_scale(weight)

        # Check properties
        assert scaled_weight.shape == weight.shape
        assert scaled_weight.dtype == weight.dtype
        assert hasattr(palettizer, "per_channel_scale")
        assert palettizer.per_channel_scale.shape[0] == shape[0]

        # Check that values are in reasonable range [-1, 1]
        assert torch.all(
            torch.abs(scaled_weight) <= 1.0 + 1e-6
        )  # Small tolerance for floating point

    @pytest.mark.parametrize(
        "shape",
        [
            (4, 8),
            (16, 32),
            (8, 4, 3, 3),
            (32, 16, 5),
        ],
    )
    def test_unscale_by_per_channel_scale_basic(self, shape):
        """Test basic per-channel unscaling functionality."""
        spec = PalettizationSpec(
            n_bits=2, granularity=PerTensorGranularity(), enable_per_channel_scale=True
        )

        palettizer = _KMeansFakePalettize(**spec.__dict__)

        # Create test weight
        weight = torch.randn(shape) * 5

        # Scale then unscale
        scaled_weight = palettizer._scale_by_per_channel_scale(weight)
        unscaled_weight = palettizer._unscale_by_per_channel_scale(scaled_weight)

        # Should be approximately equal to original
        assert torch.allclose(unscaled_weight, weight, atol=1e-6)
        assert unscaled_weight.shape == weight.shape
        assert unscaled_weight.dtype == weight.dtype

    def test_zero_weight_handling(self):
        """Test handling of zero weights in per-channel scaling."""
        spec = PalettizationSpec(
            n_bits=2, granularity=PerTensorGranularity(), enable_per_channel_scale=True
        )

        palettizer = _KMeansFakePalettize(**spec.__dict__)

        # Test all-zero weight
        zero_weight = torch.zeros(4, 8)
        scaled = palettizer._scale_by_per_channel_scale(zero_weight)
        unscaled = palettizer._unscale_by_per_channel_scale(scaled)

        assert torch.all(scaled == 0)
        assert torch.all(unscaled == 0)
        assert torch.all(palettizer.per_channel_scale == 1)  # Zero scales should be set to 1

    def test_per_channel_scale_values(self):
        """Test that per_channel_scale contains expected values."""
        spec = PalettizationSpec(
            n_bits=2, granularity=PerTensorGranularity(), enable_per_channel_scale=True
        )

        palettizer = _KMeansFakePalettize(**spec.__dict__)

        # Create weight with known max values per channel
        weight = torch.tensor(
            [
                [1.0, 2.0, -3.0, 0.5],  # max abs = 3.0
                [-5.0, 1.0, 2.0, -1.0],  # max abs = 5.0
                [0.1, -0.2, 0.15, 0.0],  # max abs = 0.2
            ]
        )

        palettizer._scale_by_per_channel_scale(weight)

        expected_scales = torch.tensor([[3.0], [5.0], [0.2]])
        assert torch.allclose(palettizer.per_channel_scale, expected_scales)

    def test_scaling_affects_clustering(self):
        """Test that per-channel scaling improves clustering for unbalanced weights."""
        # Create weight with very different scales per channel
        weight = torch.tensor(
            [
                [100.0, 101.0, 99.0, 102.0],  # Large values
                [0.01, 0.02, 0.015, 0.018],  # Small values
            ]
        )

        spec_with_scale = PalettizationSpec(
            n_bits=2, granularity=PerTensorGranularity(), enable_per_channel_scale=True
        )
        palettizer_with_scale = _KMeansFakePalettize(**spec_with_scale.__dict__)

        lut, indices = palettizer_with_scale._calculate_centroids(weight)
        palettized_with_scale = palettizer_with_scale._palettize(lut, indices, weight)
        mse_with_scale = torch.mean((weight - palettized_with_scale) ** 2)

        assert mse_with_scale < 1


class TestInitializationAndStateDict:
    """Test initialization behavior and state_dict contents."""

    @pytest.mark.parametrize("enable_per_channel_scale", [True, False])
    def test_state_dict_after_initialization(self, enable_per_channel_scale):
        """
        Test state_dict contains correct values after initialization
        without per_channel_scale.
        """
        spec = PalettizationSpec(
            n_bits=2,
            granularity=PerTensorGranularity(),
            enable_per_channel_scale=enable_per_channel_scale,
        )
        palettizer = _KMeansFakePalettize(**spec.__dict__)
        weight = torch.randn(4, 8)

        # Initially, the module should not be initialized
        assert not palettizer._initialized

        # Before initialization, buffers should be None
        # Note: state_dict() only includes buffers that are not None
        initial_state_dict = palettizer.state_dict()
        assert "lut" not in initial_state_dict
        assert "indices" not in initial_state_dict
        assert "per_channel_scale" not in initial_state_dict

        # Initialize by running forward pass
        palettizer.forward(weight)

        # After initialization, check the module is initialized
        assert palettizer._initialized

        # Check state_dict contains proper values
        state_dict = palettizer.state_dict()

        # LUT should be a tensor with shape (1, 1, num_clusters, 1)
        assert state_dict["lut"] is not None
        assert isinstance(state_dict["lut"], torch.Tensor)
        assert state_dict["lut"].shape == (1, 1, 4, 1)  # 2^2 = 4 clusters
        assert state_dict["lut"].dtype == weight.dtype

        # Indices should be a tensor with same shape as weight
        assert state_dict["indices"] is not None
        assert isinstance(state_dict["indices"], torch.Tensor)
        assert state_dict["indices"].shape == weight.shape
        assert state_dict["indices"].dtype == torch.uint8

        if enable_per_channel_scale:
            # per_channel_scale should now be present and properly shaped
            assert state_dict["per_channel_scale"] is not None
            assert isinstance(state_dict["per_channel_scale"], torch.Tensor)
            assert state_dict["per_channel_scale"].shape == (weight.shape[0], 1)
        else:
            # per_channel_scale should still be None since it's disabled
            assert "per_channel_scale" not in state_dict
            assert palettizer.per_channel_scale is None

        # Verify we can access params without error
        assert torch.equal(palettizer.lut, state_dict["lut"])
        assert torch.equal(palettizer.indices, state_dict["indices"])

    def test_observer_modes(self):
        """Test that initialization behavior respects observer modes."""
        spec = PalettizationSpec(
            n_bits=2, granularity=PerTensorGranularity(), enable_per_channel_scale=False
        )
        palettizer = _KMeansFakePalettize(**spec.__dict__)
        weight = torch.randn(3, 4)

        # Test with observer enabled, fake_palett disabled
        palettizer.enable_observer(True)
        palettizer.enable_fake_palett(False)

        # Should initialize but return original tensor
        output = palettizer.forward(weight)
        assert palettizer._initialized
        # Should return original since fake_palett disabled
        assert torch.equal(output, weight)

        # State dict should have initialized values
        state_dict = palettizer.state_dict()
        assert state_dict["lut"] is not None
        assert state_dict["indices"] is not None

        # Test with observer disabled
        palettizer2 = _KMeansFakePalettize(**spec.__dict__)
        palettizer2.enable_observer(False)

        # Should not initialize
        output2 = palettizer2.forward(weight)
        assert not palettizer2._initialized
        # Should return original since not initialized
        assert torch.equal(output2, weight)

        # Test with both disabled
        palettizer2 = _KMeansFakePalettize(**spec.__dict__)
        palettizer2.enable_observer(False)

        # Should not enabled
        palettizer3 = _KMeansFakePalettize(**spec.__dict__)
        palettizer3.enable_observer(True)
        palettizer3.enable_fake_palett(True)

        output3 = palettizer3.forward(weight)
        assert palettizer3._initialized
        # Should return different output
        assert not torch.equal(output3, weight)

    def test_save_load_state_dict_preserves_palettization(self):
        """Test that saving and loading state_dict preserves palettization behavior."""

        # Create test weight
        weight = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ]
        )

        # Test with per_channel_scale enabled to ensure it's also preserved
        spec = PalettizationSpec(
            n_bits=2, granularity=PerTensorGranularity(), enable_per_channel_scale=True
        )

        # Create original palettizer and initialize it
        original_palettizer = _KMeansFakePalettize(**spec.__dict__)
        original_output = original_palettizer.forward(weight)

        # Verify it's initialized
        assert original_palettizer._initialized

        # Disable observer
        original_palettizer.enable_observer(False)

        # Save state_dict to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_file:
            temp_path = tmp_file.name
            torch.save(original_palettizer.state_dict(), temp_path)

        try:
            # Create new palettizer instance
            loaded_palettizer = _KMeansFakePalettize(**spec.__dict__)

            # Verify it starts uninitialized
            assert not loaded_palettizer._initialized

            # Load state_dict
            state_dict = torch.load(temp_path, weights_only=True)
            loaded_palettizer.load_state_dict(state_dict)

            # Verify it's now initialized after loading
            assert loaded_palettizer._initialized

            assert loaded_palettizer.fake_palett_enabled.item() == 1
            assert loaded_palettizer.observer_enabled.item() == 0

            loaded_output = loaded_palettizer.forward(weight)

            assert torch.equal(original_output, loaded_output)

        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestSensitivityBasedPalettization:
    """Test cases for sensitivity-based (weighted) k-means palettization."""

    def test_sensitivities_reduce_error_for_sensitive_values(self):
        """
        Test that providing sensitivities results in lower reconstruction error
        for more sensitive values compared to less sensitive values.
        """
        # Create a small test tensor with two distinct regions:
        # - High-value region (should have low sensitivity)
        # - Low-value region (should have high sensitivity)
        weight = torch.tensor(
            [
                [10.0, 10.1, 10.2, 10.3],  # High values
                [1.0, 1.1, 1.2, 1.3],  # Low values
            ],
            dtype=torch.float32,
        )

        # Create sensitivities where low values are more sensitive
        # Higher sensitivity = more important to represent accurately
        sensitivities = torch.tensor(
            [
                [0.1, 0.1, 0.1, 0.1],  # Low sensitivity for high values
                [10.0, 10.0, 10.0, 10.0],  # High sensitivity for low values
            ],
            dtype=torch.float32,
        )

        # Palettize with sensitivities
        spec = PalettizationSpec(
            n_bits=2,  # Only 4 clusters
            granularity=PerTensorGranularity(),
            cluster_dim=1,
        )
        palettizer_with_sens = _KMeansFakePalettize(**spec.__dict__, sensitivities=sensitivities)

        lut_with_sens, indices_with_sens = palettizer_with_sens._calculate_centroids(weight)
        palettized_with_sens = palettizer_with_sens._palettize(
            lut_with_sens, indices_with_sens, weight
        )

        # Palettize without sensitivities
        palettizer_no_sens = _KMeansFakePalettize(**spec.__dict__, sensitivities=None)
        lut_no_sens, indices_no_sens = palettizer_no_sens._calculate_centroids(weight)
        palettized_no_sens = palettizer_no_sens._palettize(lut_no_sens, indices_no_sens, weight)

        # Calculate reconstruction errors for each region
        high_value_region = weight[0, :]
        low_value_region = weight[1, :]

        # With sensitivities
        error_high_with_sens = torch.mean((high_value_region - palettized_with_sens[0, :]) ** 2)
        error_low_with_sens = torch.mean((low_value_region - palettized_with_sens[1, :]) ** 2)

        # Without sensitivities
        error_high_no_sens = torch.mean((high_value_region - palettized_no_sens[0, :]) ** 2)
        error_low_no_sens = torch.mean((low_value_region - palettized_no_sens[1, :]) ** 2)

        # With sensitivities, the low-value region (high sensitivity) should have
        # lower reconstruction error compared to no sensitivities
        assert error_low_with_sens < error_low_no_sens, (
            f"Expected lower error for high-sensitivity region with sensitivities. "
            f"Got {error_low_with_sens:.6f} vs {error_low_no_sens:.6f}"
        )

        assert error_high_with_sens > error_high_no_sens, (
            f"Expected higher error for low-sensitivity region with sensitivities. "
            f"Got {error_high_with_sens:.6f} vs {error_high_no_sens:.6f}"
        )

    def test_sensitivities_affect_centroid_placement(self):
        """
        Test that sensitivities cause centroids to be placed closer to
        more sensitive values.
        """
        # Create a tensor with widely spaced values
        weight = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0, 100.0, 101.0, 102.0, 103.0]], dtype=torch.float32
        )

        # Make the low values (1-4) highly sensitive
        sensitivities = torch.tensor(
            [[100.0, 100.0, 100.0, 100.0, 0.01, 0.01, 0.01, 0.01]], dtype=torch.float32
        )

        spec = PalettizationSpec(
            n_bits=2,  # 4 clusters
            granularity=PerTensorGranularity(),
            cluster_dim=1,
        )

        palettizer_with_sens = _KMeansFakePalettize(**spec.__dict__, sensitivities=sensitivities)
        lut_with_sens, _ = palettizer_with_sens._calculate_centroids(weight)

        # The centroids should be more concentrated around the low values (1-4)
        # because they have high sensitivity
        centroids = lut_with_sens.flatten()
        centroids_sorted = torch.sort(centroids).values

        # At least 3 out of 4 centroids should be closer to the low range [1, 4]
        # than to the high range [100, 103]
        low_range_centroids = centroids_sorted[centroids_sorted <= 50]
        assert len(low_range_centroids) >= 3, (
            f"Expected at least 3 centroids in low range, "
            f"got {len(low_range_centroids)}. Centroids: {centroids_sorted}"
        )

    def test_fast_vs_regular_kmeans_with_sensitivities_consistency(self):
        """
        Test that fast and regular k-means modes produce similar results
        when sensitivities are provided.
        """
        weight = torch.randn(4, 8, dtype=torch.float32) * 5.0

        # Create some sensitivities with variation
        sensitivities = torch.abs(torch.randn(4, 8, dtype=torch.float32))
        # Normalize sensitivities to [0, 1]
        sensitivities = sensitivities / sensitivities.max()

        spec = PalettizationSpec(
            n_bits=3,  # 8 clusters
            granularity=PerTensorGranularity(),
            cluster_dim=1,
        )

        # Fast mode
        palettizer_fast = _KMeansFakePalettize(
            **spec.__dict__,
            sensitivities=sensitivities.clone(),
            enable_fast_kmeans_mode=True,
        )
        lut_fast, indices_fast = palettizer_fast._calculate_centroids(weight)
        palettized_fast = palettizer_fast._palettize(lut_fast, indices_fast, weight)

        # Regular mode
        palettizer_regular = _KMeansFakePalettize(
            **spec.__dict__,
            sensitivities=sensitivities.clone(),
            enable_fast_kmeans_mode=False,
        )
        lut_regular, indices_regular = palettizer_regular._calculate_centroids(weight)
        palettized_regular = palettizer_regular._palettize(lut_regular, indices_regular, weight)

        # Calculate reconstruction errors
        mse_fast = torch.mean((weight - palettized_fast) ** 2)
        mse_regular = torch.mean((weight - palettized_regular) ** 2)

        print(f"MSE fast: {mse_fast:.6f}, MSE regular: {mse_regular:.6f}")

        # The reconstruction errors should be similar (within reasonable tolerance)
        # Note: They won't be identical due to rounding in fast mode
        assert torch.allclose(mse_fast, mse_regular, rtol=0.5), (
            f"Fast and regular k-means should produce similar reconstruction errors. "
            f"Got {mse_fast:.6f} vs {mse_regular:.6f}"
        )

        # The centroids should also be similar
        # Sort centroids for comparison since order might differ
        centroids_fast_sorted = torch.sort(lut_fast.flatten()).values
        centroids_regular_sorted = torch.sort(lut_regular.flatten()).values

        assert torch.allclose(centroids_fast_sorted, centroids_regular_sorted, atol=0.1), (
            "Fast and regular k-means should produce similar centroids"
        )

    def test_uniform_sensitivities_equivalent_to_no_sensitivities(self):
        """
        Test that uniform sensitivities (all same value) produce results
        similar to no sensitivities.
        """
        weight = torch.randn(3, 6, dtype=torch.float32) * 2.0

        # All sensitivities are the same (uniform)
        sensitivities_uniform = torch.ones_like(weight)

        spec = PalettizationSpec(n_bits=2, granularity=PerTensorGranularity(), cluster_dim=1)

        # With uniform sensitivities
        palettizer_uniform = _KMeansFakePalettize(
            **spec.__dict__, sensitivities=sensitivities_uniform
        )
        lut_uniform, indices_uniform = palettizer_uniform._calculate_centroids(weight)

        # Without sensitivities
        palettizer_none = _KMeansFakePalettize(**spec.__dict__, sensitivities=None)
        lut_none, indices_none = palettizer_none._calculate_centroids(weight)

        # Errors should be very close
        assert torch.equal(lut_none, lut_uniform), (
            f"Uniform sensitivities should behave like no sensitivities. "
            f"Got MSE {lut_none:.6f} vs {lut_uniform:.6f}"
        )

    @pytest.mark.parametrize("enable_fast_kmeans", [True, False])
    def test_sensitivities_with_per_channel_scale(self, enable_fast_kmeans):
        """
        Test that sensitivities work correctly when combined with per-channel scaling.
        """
        # Create weights with different scales per channel
        weight = torch.tensor(
            [
                [100.0, 101.0, 102.0, 103.0],  # Large scale
                [1.0, 1.1, 1.2, 1.3],  # Small scale
            ],
            dtype=torch.float32,
        )

        # High sensitivity for the small-scale channel
        sensitivities = torch.tensor(
            [
                [0.1, 0.1, 0.1, 0.1],
                [10.0, 10.0, 10.0, 10.0],
            ],
            dtype=torch.float32,
        )

        spec = PalettizationSpec(
            n_bits=2,
            granularity=PerTensorGranularity(),
            cluster_dim=1,
            enable_per_channel_scale=True,
        )

        palettizer = _KMeansFakePalettize(
            **spec.__dict__,
            sensitivities=sensitivities,
            enable_fast_kmeans_mode=enable_fast_kmeans,
        )

        lut, indices = palettizer._calculate_centroids(weight)
        palettized = palettizer._palettize(lut, indices, weight)

        # Verify shape is preserved
        assert palettized.shape == weight.shape

        # Calculate reconstruction errors per channel
        error_large_scale = torch.mean((weight[0, :] - palettized[0, :]) ** 2)
        error_small_scale = torch.mean((weight[1, :] - palettized[1, :]) ** 2)

        # Both errors should be reasonable despite different scales
        assert error_large_scale < 5.0, "Large scale channel error should be reasonable"
        assert error_small_scale < 0.5, "Small scale channel error should be low"


class TestVectorPalettization:
    """Test cases for vector palettization (cluster_dim > 1)."""

    @pytest.mark.parametrize("cluster_dim", [2, 3])
    def test_vector_palettization_per_tensor(self, cluster_dim):
        """Test vector palettization with per-tensor granularity."""
        weight = torch.randn(6, 8, dtype=torch.float32)

        spec = PalettizationSpec(
            n_bits=2,
            granularity=PerTensorGranularity(),
            cluster_dim=cluster_dim,
        )

        palettizer = _KMeansFakePalettize(**spec.__dict__)

        lut, indices = palettizer._calculate_centroids(weight)

        # LUT shape: [1, 1, 2^n_bits, cluster_dim]
        num_clusters = 2**spec.n_bits
        assert lut.shape == (1, 1, num_clusters, cluster_dim)

        # Indices should have reduced shape (not matching weight shape)
        # For per-tensor axis=0: (rows/cluster_dim, cols)
        assert indices.shape == (weight.shape[0] // cluster_dim, weight.shape[1])

        # Verify palettization produces correct output shape
        palettized = palettizer._palettize(lut, indices, weight)
        assert palettized.shape == weight.shape

        # Reconstruction should be reasonable
        mse = torch.mean((weight - palettized) ** 2)
        assert mse < 0.5, f"Reconstruction MSE too high: {mse:.4f}"

    @pytest.mark.parametrize("cluster_dim", [2, 4])
    @pytest.mark.parametrize("axis", [0, 1])
    def test_vector_palettization_per_grouped_channel(self, cluster_dim, axis):
        """Test vector palettization with per-grouped-channel granularity."""
        weight = torch.randn(12, 16, dtype=torch.float32)
        group_size = 4

        spec = PalettizationSpec(
            n_bits=2,
            granularity=PerGroupedChannelGranularity(axis=axis, group_size=group_size),
            cluster_dim=cluster_dim,
        )

        palettizer = _KMeansFakePalettize(**spec.__dict__)

        lut, indices = palettizer._calculate_centroids(weight)

        # LUT shape depends on axis
        num_clusters = 2**spec.n_bits
        num_blocks = weight.shape[axis] // group_size
        if axis == 0:
            assert lut.shape == (num_blocks, 1, num_clusters, cluster_dim)
        else:
            assert lut.shape == (1, num_blocks, num_clusters, cluster_dim)

        # Indices are always reduced along axis 0 (output channel),
        # regardless of granularity axis
        assert indices.shape == (
            weight.shape[0] // cluster_dim,
            weight.shape[1],
        )

        # Verify palettization produces correct output shape
        palettized = palettizer._palettize(lut, indices, weight)
        assert palettized.shape == weight.shape

        # Reconstruction should be reasonable
        mse = torch.mean((weight - palettized) ** 2)
        assert mse < 0.5, f"Reconstruction MSE too high: {mse:.4f}"

    def test_vector_palettization_incompatible_dim(self):
        """Test that incompatible cluster_dim gracefully disables palettization."""
        # Weight dim along axis 0 is 3, not divisible by cluster_dim=2
        weight = torch.randn(3, 8, dtype=torch.float32)

        spec = PalettizationSpec(
            n_bits=2,
            granularity=PerTensorGranularity(),
            cluster_dim=2,
        )

        palettizer = _KMeansFakePalettize(**spec.__dict__)

        # forward() should catch _IncompatibleClusterDimError and disable
        output = palettizer.forward(weight)
        assert palettizer._disabled
        assert torch.equal(output, weight)

    @pytest.mark.parametrize(
        "weight_shape, granularity",
        [
            ((3, 8), PerTensorGranularity()),
            ((12, 8), PerGroupedChannelGranularity(axis=0, group_size=3)),
            ((3, 12), PerGroupedChannelGranularity(axis=1, group_size=3)),
        ],
    )
    def test_vector_palettization_incompatible_dim_raises(self, weight_shape, granularity):
        """Test that _calculate_centroids raises _IncompatibleClusterDimError directly."""
        weight = torch.randn(weight_shape, dtype=torch.float32)

        spec = PalettizationSpec(
            n_bits=2,
            granularity=granularity,
            cluster_dim=2,
        )

        palettizer = _KMeansFakePalettize(**spec.__dict__)

        with pytest.raises(_IncompatibleClusterDimError):
            palettizer._calculate_centroids(weight)

    @pytest.mark.parametrize("lut_dtype", _SUPPORTED_LUT_DTYPES)
    def test_vector_palettization_with_quantized_lut(self, lut_dtype):
        """Test vector palettization combined with LUT quantization."""
        weight = torch.randn(8, 8, dtype=torch.float32)

        spec = PalettizationSpec(
            n_bits=2,
            granularity=PerTensorGranularity(),
            cluster_dim=2,
            lut_qspec=_make_lut_qspec(lut_dtype),
        )

        palettizer = _KMeansFakePalettize(**spec.__dict__)

        lut, indices = palettizer._calculate_centroids(weight)

        # LUT shape: [1, 1, num_clusters, cluster_dim]
        assert lut.shape == (1, 1, 2**spec.n_bits, 2)

        palettized = palettizer._palettize(lut, indices, weight)
        assert palettized.shape == weight.shape

        # Reconstruction should be reasonable
        mse = torch.mean((weight - palettized) ** 2)
        print(mse)
        assert mse < 0.5, f"Reconstruction MSE too high: {mse:.4f}"

    @pytest.mark.parametrize(
        "granularity",
        [
            PerTensorGranularity(),
            PerGroupedChannelGranularity(axis=0, group_size=4),
            PerGroupedChannelGranularity(axis=1, group_size=4),
        ],
    )
    def test_vector_palettization_with_per_channel_scale(self, granularity):
        """Test vector palettization with enable_per_channel_scale=True."""
        weight = torch.randn(12, 16, dtype=torch.float32)

        spec = PalettizationSpec(
            n_bits=4,
            granularity=granularity,
            cluster_dim=2,
            enable_per_channel_scale=True,
        )

        palettizer = _KMeansFakePalettize(**spec.__dict__)

        lut, indices = palettizer._calculate_centroids(weight)

        # Per-channel scale should be populated
        assert palettizer.per_channel_scale is not None

        # Verify palettization produces correct output shape
        palettized = palettizer._palettize(lut, indices, weight)
        assert palettized.shape == weight.shape

        # Reconstruction should be reasonable (higher tolerance for aggressive
        # compression with large cluster_dim and per-channel scaling)
        mse = torch.mean((weight - palettized) ** 2)
        print(mse)
        assert mse < 0.5, f"Reconstruction MSE too high: {mse:.4f}"

    def test_vector_palettization_reconstruction_quality(self):
        """Test that vector palettization has reasonable reconstruction quality
        compared to scalar palettization.
        """
        weight = torch.randn(16, 16, dtype=torch.float32)

        # Scalar palettization
        spec_scalar = PalettizationSpec(
            n_bits=3,
            granularity=PerTensorGranularity(),
            cluster_dim=1,
        )
        palettizer_scalar = _KMeansFakePalettize(**spec_scalar.__dict__)
        lut_s, idx_s = palettizer_scalar._calculate_centroids(weight)
        palettized_scalar = palettizer_scalar._palettize(lut_s, idx_s, weight)
        mse_scalar = torch.mean((weight - palettized_scalar) ** 2)

        # Vector palettization with same bits per weight
        spec_vector = PalettizationSpec(
            n_bits=6,
            granularity=PerTensorGranularity(),
            cluster_dim=2,
        )
        palettizer_vector = _KMeansFakePalettize(**spec_vector.__dict__)
        lut_v, idx_v = palettizer_vector._calculate_centroids(weight)
        palettized_vector = palettizer_vector._palettize(lut_v, idx_v, weight)
        mse_vector = torch.mean((weight - palettized_vector) ** 2)

        # Both should have reasonable reconstruction quality
        assert mse_scalar < 0.5, f"Scalar MSE too high: {mse_scalar:.4f}"
        assert mse_vector < 0.5, f"Vector MSE too high: {mse_vector:.4f}"
        assert mse_vector < mse_scalar, (
            f"Expected Vector MSE: {mse_vector:.4f} to be lower than Scale MSE: {mse_scalar:.4f}"
        )
