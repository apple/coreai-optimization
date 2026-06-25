# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np
import pytest
import torch
import torch.nn as nn

from coreai_opt.coreai_utils import DType, sparsify_weights
from tests.export.export_utils import MLIRConverter


@pytest.mark.parametrize("target_sparsity", [0.25, 0.5, 0.75])
def test_mlir_weight_sparsification(
    target_sparsity: float,
    _coreai_program,
) -> None:
    """Test basic sparsification produces sparse ops"""
    coreai_program, _, _ = _coreai_program

    compressed = sparsify_weights(
        coreai_program=coreai_program,
        target_sparsity=target_sparsity,
        weight_num_threshold=0,
        in_place=False,
    )

    ir = str(compressed)
    assert "coreai.build_sparse_with_bitmask" in ir
    assert "coreai.sparse_with_bitmask_to_dense" in ir


def test_mlir_weight_sparsification_n_m_ratio(_coreai_program) -> None:
    """Test n:m structured pruning produces sparse ops."""
    coreai_program, _, _ = _coreai_program

    compressed = sparsify_weights(
        coreai_program=coreai_program,
        target_sparsity=None,
        n_m_ratio=(1, 2),
        weight_num_threshold=0,
        in_place=False,
    )

    ir = str(compressed)
    assert "coreai.build_sparse_with_bitmask" in ir
    assert "coreai.sparse_with_bitmask_to_dense" in ir


@pytest.mark.parametrize(
    "quantize_dtype", [DType.INT8, DType.UINT8, DType.FP8_E4M3FN, DType.FP8_E5M2]
)
def test_mlir_weight_sparsification_quantize_dtype(
    quantize_dtype: DType,
    _coreai_program,
) -> None:
    """Test joint sparsification + quantization produces the expected dtype in the IR."""
    coreai_program, _, _ = _coreai_program

    compressed = sparsify_weights(
        coreai_program=coreai_program,
        target_sparsity=0.5,
        quantize_dtype=quantize_dtype,
        weight_num_threshold=0,
        in_place=False,
    )

    ir = str(compressed)
    assert "coreai.build_sparse_with_bitmask" in ir
    assert "coreai.sparse_with_bitmask_to_dense" in ir
    assert "coreai.blockwise_shift_scale" in ir

    dtype_token = {
        DType.INT8: "si8",
        DType.UINT8: "ui8",
        DType.FP8_E4M3FN: "f8E4M3FN",
        DType.FP8_E5M2: "f8E5M2",
    }[quantize_dtype]
    assert any(
        dtype_token in line for line in ir.splitlines() if "coreai.blockwise_shift_scale" in line
    )


@pytest.mark.parametrize("palettize_nbits", [2, 4, 8])
def test_mlir_weight_sparsification_palettize_nbits(
    palettize_nbits: int,
    _coreai_program,
) -> None:
    """Test joint sparsification + palettization produces lut_to_dense and ui<nbits> indices."""
    coreai_program, _, _ = _coreai_program

    compressed = sparsify_weights(
        coreai_program=coreai_program,
        target_sparsity=0.5,
        palettize_nbits=palettize_nbits,
        weight_num_threshold=0,
        in_place=False,
    )

    ir = str(compressed)
    assert "coreai.build_sparse_with_bitmask" in ir
    assert "coreai.sparse_with_bitmask_to_dense" in ir
    assert "coreai.lut_to_dense" in ir
    assert f"ui{palettize_nbits}" in ir
    assert "blockwise_shift_scale" not in ir


def test_mlir_weight_sparsification_weight_num_threshold(_coreai_program) -> None:
    """Test that weights below weight_num_threshold are not compressed."""
    coreai_program, _, _ = _coreai_program

    compressed = sparsify_weights(
        coreai_program=coreai_program,
        target_sparsity=0.5,
        weight_num_threshold=int(10e6),
        in_place=False,
    )

    # The linear layer weight (2048 * 32 = 65536 elements) is below 10e6,
    # so no compression should have been applied.
    assert "coreai.build_sparse_with_bitmask" not in str(compressed)


def test_mlir_weight_sparsification_in_place(_exported_program) -> None:
    """Test in_place=False leaves the original program unmodified; in_place=True modifies it."""
    exported_program, _, _ = _exported_program

    # in_place=False: result is a deep copy; original is untouched.
    coreai_program = MLIRConverter._lower_to_coreai(exported_program)
    result = sparsify_weights(
        coreai_program=coreai_program,
        target_sparsity=0.5,
        weight_num_threshold=0,
        in_place=False,
    )
    assert result is not coreai_program
    assert "coreai.build_sparse_with_bitmask" not in str(coreai_program)
    assert "coreai.build_sparse_with_bitmask" in str(result)

    # in_place=True: result is the same object; original is modified.
    coreai_program = MLIRConverter._lower_to_coreai(exported_program)
    result = sparsify_weights(
        coreai_program=coreai_program,
        target_sparsity=0.5,
        weight_num_threshold=0,
        in_place=True,
    )
    assert result is coreai_program
    assert "coreai.build_sparse_with_bitmask" in str(coreai_program)


def test_mlir_weight_sparsification_block_size(_coreai_program) -> None:
    """Test block sparsity (block_size=4) produces sparse ops."""
    coreai_program, _, _ = _coreai_program

    compressed = sparsify_weights(
        coreai_program=coreai_program,
        target_sparsity=0.5,
        block_size=4,
        weight_num_threshold=0,
        in_place=False,
    )

    ir = str(compressed)
    assert "coreai.build_sparse_with_bitmask" in ir
    assert "coreai.sparse_with_bitmask_to_dense" in ir


@pytest.mark.parametrize(
    "kwargs, error_match",
    [
        # Both target_sparsity and n_m_ratio set.
        (
            {"target_sparsity": 0.5, "n_m_ratio": (1, 2)},
            "`target_sparsity` and `n_m_ratio` cannot both be set",
        ),
        # Neither target_sparsity nor n_m_ratio set.
        (
            {"target_sparsity": None},
            "One of `target_sparsity` or `n_m_ratio` must be set",
        ),
        # Both quantize_dtype and palettize_nbits set.
        (
            {"target_sparsity": 0.5, "quantize_dtype": DType.INT8, "palettize_nbits": 4},
            "`quantize_dtype` and `palettize_nbits` cannot both be set",
        ),
        # quantize_dtype not in valid set.
        (
            {"target_sparsity": 0.5, "quantize_dtype": DType.INT4},
            "Invalid quantize_dtype",
        ),
        # palettize_nbits not in valid set.
        (
            {"target_sparsity": 0.5, "palettize_nbits": 5},
            "Invalid palettize_nbits",
        ),
        # block_size not greater than 1.
        (
            {"target_sparsity": 0.5, "block_size": 1},
            "`block_size` must be greater than 1",
        ),
    ],
)
def test_mlir_weight_sparsification_validation(
    kwargs: dict,
    error_match: str,
    _coreai_program,
) -> None:
    """Test that invalid parameter combinations raise ValueError."""
    coreai_program, _, _ = _coreai_program
    with pytest.raises(ValueError, match=error_match):
        sparsify_weights(coreai_program=coreai_program, **kwargs)


class _MatmulModel(nn.Module):
    """Model whose fp16 weight buffer lowers to broadcasting_batch_matmul."""

    def __init__(self) -> None:
        super().__init__()
        val = np.array(
            [
                [1, 3, 4, -3],
                [-6, -7, 2, 4],
                [0, 3, 4, 1],
                [-9, 2, -1, 8],
            ],
            dtype=np.float16,
        )
        self.register_buffer("weight", torch.from_numpy(val))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.weight, x)


def _make_coreai_program() -> object:
    x = torch.eye(4, dtype=torch.float16)
    exported = MLIRConverter().trace(_MatmulModel(), x, {})
    return MLIRConverter._lower_to_coreai(exported)


def test_sparsify_weights_n_m_ratio_e2e() -> None:
    """Compressed model output matches expected 1:2 pruned values.

    The weight is a fp16 (4, 4) matrix consumed by broadcasting_batch_matmul,
    so the pass prunes along input_channel_axis (dim=1). With n:m=(1,2), the
    smaller of each consecutive pair of columns is zeroed. Running with an
    identity input isolates the weight in the output.
    """
    compressed = sparsify_weights(
        coreai_program=_make_coreai_program(),
        target_sparsity=None,
        n_m_ratio=(1, 2),
        weight_num_threshold=0,
        in_place=False,
    )
    assert "coreai.build_sparse_with_bitmask" in str(compressed)
    (output,) = MLIRConverter()._run_inference(compressed, torch.eye(4, dtype=torch.float16))
    expected = np.array(
        [
            [0, 3, 4, 0],
            [0, -7, 0, 4],
            [0, 3, 4, 0],
            [-9, 0, 0, 8],
        ],
        dtype=np.float16,
    )
    np.testing.assert_array_equal(output, expected)


def test_sparsify_weights_magnitude_e2e() -> None:
    """Compressed model output matches expected 50%-sparsity pruned values.

    With target_sparsity=0.5, the lowest-magnitude elements are zeroed.
    Running with an identity input isolates the weight in the output.
    """
    compressed = sparsify_weights(
        coreai_program=_make_coreai_program(),
        target_sparsity=0.5,
        weight_num_threshold=0,
        in_place=False,
    )
    assert "coreai.build_sparse_with_bitmask" in str(compressed)
    (output,) = MLIRConverter()._run_inference(compressed, torch.eye(4, dtype=torch.float16))
    expected = np.array(
        [
            [0, 0, 4, 0],
            [-6, -7, 0, 4],
            [0, 0, 4, 0],
            [-9, 0, 0, 8],
        ],
        dtype=np.float16,
    )
    np.testing.assert_array_equal(output, expected)


def test_sparsify_weights_block_e2e() -> None:
    """Compressed model output matches expected block-sparsity pruned values.

    The weight is a fp16 (4, 4) matrix. With block_size=2 and target_sparsity=0.5,
    row pairs are treated as blocks (output_channel_axis=0) and pruned by L2 norm.
    Running with an identity input isolates the weight in the output.
    """
    compressed = sparsify_weights(
        coreai_program=_make_coreai_program(),
        target_sparsity=0.5,
        block_size=2,
        weight_num_threshold=0,
        in_place=False,
    )
    assert "coreai.build_sparse_with_bitmask" in str(compressed)
    (output,) = MLIRConverter()._run_inference(compressed, torch.eye(4, dtype=torch.float16))
    expected = np.array(
        [
            [1, 3, 0, 0],
            [-6, -7, 0, 0],
            [0, 0, 0, 1],
            [-9, 0, 0, 8],
        ],
        dtype=np.float16,
    )
    np.testing.assert_array_equal(output, expected)
