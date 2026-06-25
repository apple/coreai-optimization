# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for coreai_opt.coreai_utils._utils.sparse_utils."""

import numpy as np
import pytest

from coreai_opt.coreai_utils._utils.sparse_utils import (
    SparseParams,
    _compress_by_magnitude,
    _compress_by_nm_sparsity,
    _produce_sparse_param,
)


class TestProduceSparseParam:
    def test_basic(self) -> None:
        """Correct nonzero_data and uint8 mask for a mixed-zero array."""
        val = np.array([[1.0, 0.0], [0.0, 2.0]])
        result = _produce_sparse_param(val)
        assert isinstance(result, SparseParams)
        np.testing.assert_array_equal(result.nonzero_data, [1.0, 2.0])
        np.testing.assert_array_equal(result.mask, [[1, 0], [0, 1]])
        assert result.mask.dtype == np.uint8

    def test_all_zeros(self) -> None:
        """All-zero input yields empty nonzero_data and an all-zero mask."""
        val = np.zeros((3, 3))
        result = _produce_sparse_param(val)
        assert len(result.nonzero_data) == 0
        np.testing.assert_array_equal(result.mask, np.zeros((3, 3), dtype=np.uint8))

    def test_all_nonzero(self) -> None:
        """No-zero input yields nonzero_data equal to the flattened input."""
        val = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _produce_sparse_param(val)
        np.testing.assert_array_equal(result.nonzero_data, val.flatten())
        np.testing.assert_array_equal(result.mask, np.ones((2, 2), dtype=np.uint8))

    def test_mask_shape_matches_input(self) -> None:
        """Mask shape matches the original (possibly multi-dimensional) input."""
        val = np.arange(1.0, 25.0).reshape(2, 3, 4)
        result = _produce_sparse_param(val)
        assert result.mask.shape == val.shape

    def test_mask_dtype_is_uint8(self) -> None:
        val = np.array([1.0, 0.0, 2.0])
        result = _produce_sparse_param(val)
        assert result.mask.dtype == np.uint8


class TestCompressByMagnitude:
    def test_zero_sparsity_keeps_all(self) -> None:
        """target_sparsity=0 leaves every element intact."""
        val = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _compress_by_magnitude(val, target_sparsity=0.0)
        assert result.mask.sum() == val.size

    def test_full_sparsity_zeros_all(self) -> None:
        """target_sparsity=1 zeros every element."""
        val = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _compress_by_magnitude(val, target_sparsity=1.0)
        assert result.mask.sum() == 0
        assert len(result.nonzero_data) == 0

    def test_half_sparsity_nonzero_count(self) -> None:
        """target_sparsity=0.5 yields half the elements nonzero."""
        val = np.array([1.0, 2.0, 3.0, 4.0])
        result = _compress_by_magnitude(val, target_sparsity=0.5)
        assert result.mask.sum() == 2

    def test_smallest_magnitude_zeroed(self) -> None:
        """The n smallest-magnitude elements are zeroed, not the largest."""
        val = np.array([10.0, 1.0, 20.0, 2.0])
        result = _compress_by_magnitude(val, target_sparsity=0.5)
        np.testing.assert_array_equal(result.nonzero_data, [10.0, 20.0])

    def test_returns_sparse_params(self) -> None:
        val = np.ones((4, 4))
        result = _compress_by_magnitude(val, target_sparsity=0.25)
        assert isinstance(result, SparseParams)

    def test_block_sparsity_zeros_entire_blocks(self) -> None:
        """Block sparsity assigns the same mask to all rows in a block.

        val has two blocks of rows (0-1 and 2-3). Block 0 has smaller L2
        norms per column, so it is zeroed at 50% sparsity.
        """
        val = np.array([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0], [3.0, 4.0]], dtype=np.float32)
        result = _compress_by_magnitude(val, target_sparsity=0.5, block_size=2, dim=0)
        np.testing.assert_array_equal(result.mask, [[0, 0], [0, 0], [1, 1], [1, 1]])
        np.testing.assert_array_equal(result.nonzero_data, [3.0, 4.0, 3.0, 4.0])

    def test_block_sparsity_dim1(self) -> None:
        """Block sparsity along dim=1 returns a SparseParams (not None)."""
        val = np.ones((4, 4), dtype=np.float32)
        result = _compress_by_magnitude(val, target_sparsity=0.5, block_size=2, dim=1)
        assert isinstance(result, SparseParams)

    def test_block_size_without_dim_raises(self) -> None:
        val = np.ones((4, 4))
        with pytest.raises(ValueError, match="`dim` must be provided"):
            _compress_by_magnitude(val, target_sparsity=0.5, block_size=2, dim=None)

    def test_block_size_larger_than_half_channel_returns_none(self) -> None:
        """block_size > channel/2 is not applicable; function returns None."""
        # channel=4 along dim=0, block_size=3: 3 > 4/2=2
        val = np.ones((4, 4), dtype=np.float32)
        result = _compress_by_magnitude(val, target_sparsity=0.5, block_size=3, dim=0)
        assert result is None

    def test_invalid_dim_raises(self) -> None:
        # Use a 3D array so dim=2 is a valid shape index; the ValueError is
        # then raised inside _apply_block_sparsity which checks dim in [0, 1].
        val = np.ones((4, 4, 4))
        with pytest.raises(ValueError, match="block sparsity pruning only supports dim"):
            _compress_by_magnitude(val, target_sparsity=0.5, block_size=2, dim=2)

    def test_invalid_rank_raises(self) -> None:
        val = np.ones((8,))
        with pytest.raises(ValueError, match="block sparsity only supports weights of rank"):
            _compress_by_magnitude(val, target_sparsity=0.5, block_size=2, dim=0)


class TestCompressByNmSparsity:
    def test_1_2_dim1_zeros_smallest_per_pair(self) -> None:
        """1:2 pruning along dim=1 zeros the smaller of each consecutive pair."""
        val = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
        result = _compress_by_nm_sparsity(val, n_m_ratio=(1, 2), dim=1)
        np.testing.assert_array_equal(result.nonzero_data, [2.0, 4.0, 6.0, 8.0])
        np.testing.assert_array_equal(result.mask, [[0, 1, 0, 1], [0, 1, 0, 1]])

    def test_1_2_dim0_zeros_smaller_row_per_block(self) -> None:
        """1:2 pruning along dim=0 zeros the smaller row in each pair of rows."""
        val = np.array(
            [
                [1.0, 3.0, 5.0, 7.0],
                [2.0, 4.0, 6.0, 8.0],
                [9.0, 11.0, 13.0, 15.0],
                [10.0, 12.0, 14.0, 16.0],
            ],
            dtype=np.float32,
        )
        result = _compress_by_nm_sparsity(val, n_m_ratio=(1, 2), dim=0)
        np.testing.assert_array_equal(
            result.mask, [[0, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]]
        )

    def test_n_zero_keeps_all(self) -> None:
        """n=0 means nothing is pruned; all elements are nonzero."""
        val = np.arange(1.0, 9.0, dtype=np.float32).reshape(2, 4)
        result = _compress_by_nm_sparsity(val, n_m_ratio=(0, 2), dim=1)
        assert result.mask.sum() == val.size

    def test_n_equals_m_zeros_all(self) -> None:
        """n=m means all elements are pruned."""
        val = np.arange(1.0, 9.0, dtype=np.float32).reshape(2, 4)
        result = _compress_by_nm_sparsity(val, n_m_ratio=(2, 2), dim=1)
        assert result.mask.sum() == 0
        assert len(result.nonzero_data) == 0

    def test_channel_not_divisible_by_m(self) -> None:
        """Padding zeros consume pruning slots in the last group when channel % m != 0.

        With n=1, m=2, channel=5 the last group per row is [real_elem, 0(pad)].
        The padded zero has magnitude 0, so it takes the single pruning slot and
        the last real element is kept instead of being zeroed.
        """
        val = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]], dtype=np.float32)
        result = _compress_by_nm_sparsity(val, n_m_ratio=(1, 2), dim=1)
        # Groups [1,2] and [3,4] each zero their smaller element.
        # Last group [5, 0(pad)]: padded zero takes the slot, so 5 and 10 survive.
        np.testing.assert_array_equal(result.mask, [[0, 1, 0, 1, 1], [0, 1, 0, 1, 1]])

    def test_m_larger_than_half_channel_returns_none(self) -> None:
        """m > channel/2 is not applicable; function returns None."""
        # channel along dim=1 is 4, m=3: 3 > 4/2=2
        val = np.ones((4, 4), dtype=np.float32)
        result = _compress_by_nm_sparsity(val, n_m_ratio=(1, 3), dim=1)
        assert result is None

    def test_invalid_dim_raises(self) -> None:
        val = np.ones((4, 4))
        with pytest.raises(ValueError, match="n:m pruning only supports dim"):
            _compress_by_nm_sparsity(val, n_m_ratio=(1, 2), dim=2)

    def test_invalid_rank_raises(self) -> None:
        val = np.ones((8,))
        with pytest.raises(ValueError, match="n:m pruning only supports weights of rank"):
            _compress_by_nm_sparsity(val, n_m_ratio=(1, 2), dim=0)
