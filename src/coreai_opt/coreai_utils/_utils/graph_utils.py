# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Graph traversal and compression structure utilities for Core AI Optimization passes."""

# TODO: add test enhancements for graph utils.
from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch

from coreai_opt.coreai_utils._coreai_imports import (
    AIProgram,
    BoolAttr,
    DebugInfo,
    DenseElementsAttr,
    FloatAttr,
    IntegerAttr,
    IntegerType,
    Operation,
    RankedTensorType,
    TensorType,
    WalkResult,
    coreai,
    create_elements_attr,
    pack_intx,
    pack_uintx,
    validate_and_cast_numpy_array,
)
from coreai_opt.coreai_utils._utils.type_utils import _is_sub_byte_int
from coreai_opt.coreai_utils.common import CompressionGranularity as _CompressionGranularity

logger = logging.getLogger(__name__)

# Hardcoded CoreAI op name strings (avoids importing coreai.compiler.dialects at module level).
_BATCH_MATMUL_OP = "coreai.batch_matmul"
_BROADCASTING_BATCH_MATMUL_OP = "coreai.decomposable.broadcasting_batch_matmul"
_TRANSPOSE_OP = "coreai.transpose"
_GATHER_ND_OP = "coreai.gather_nd"
_CONV2D_OP = "coreai.conv2d"

# Compression op names used to detect chained (joint) compression.
_WEIGHT_COMPRESSION_OPS = {
    "coreai.blockwise_shift_scale",
}

__all__: list[str] = []


def _select_input_output_channel_axis(
    op: Operation,
) -> tuple[int | None, int | None]:
    """Select the input and output channel axis for a constant op based on its consumers.

    The axis is determined by examining which downstream ops consume this constant:

    - ``batch_matmul`` y operand: output axis ``-1``, input axis ``-2``
    - ``batch_matmul`` x operand: output axis ``-2``, input axis ``-1``
    - ``transpose`` feeding into matmul: axes inferred transitively
    - ``gather_nd``: output axis ``0``, input axis ``1``
    - All other ops: output axis ``0``, input axis ``1`` (defaults)

    Args:
        op (Operation): The constant operation whose downstream consumers determine
            the channel axes.

    Returns:
        tuple[int | None, int | None]: ``(input_channel_axis, output_channel_axis)``.
            Either may be ``None`` if the axis cannot be determined unambiguously.
    """
    output_channel_axis_set: set[int | None] = set()
    input_channel_axis_set: set[int | None] = set()

    for child_op_use in op.result.uses:
        output_channel_axis: int | None = 0
        input_channel_axis: int | None = 1

        child_op: Operation = cast("Operation", child_op_use.owner)
        if child_op.name in [_BATCH_MATMUL_OP, _BROADCASTING_BATCH_MATMUL_OP]:
            child_op_operands = list(child_op.operands)  # type: ignore[call-overload]
            if child_op_operands[1].owner == op:
                # Constant is used as matmul's y operand.
                output_channel_axis = -1
                input_channel_axis = -2
            else:
                # Constant is used as matmul's x operand.
                assert child_op_operands[0].owner == op
                output_channel_axis = -2
                input_channel_axis = -1
        elif child_op.name == _TRANSPOSE_OP:
            # Check if this transpose feeds into a BatchMatmul as the y operand.
            # All uses of the transpose must agree on axes for a determination.
            transpose_output_axes: set[int] = set()
            transpose_input_axes: set[int] = set()

            for transpose_use in child_op.result.uses:
                transpose_child: Operation = cast("Operation", transpose_use.owner)
                if transpose_child.name in [_BATCH_MATMUL_OP, _BROADCASTING_BATCH_MATMUL_OP]:
                    operands = list(transpose_child.operands)  # type: ignore[call-overload]
                    if operands[1].owner == child_op:
                        # Transpose feeds into matmul's y.
                        transpose_output_axes.add(-2)
                        transpose_input_axes.add(-1)
                    else:
                        # Transpose feeds into matmul's x.
                        transpose_output_axes.add(-1)
                        transpose_input_axes.add(-2)

            if len(transpose_output_axes) == 1 and len(transpose_input_axes) == 1:
                output_channel_axis = transpose_output_axes.pop()
                input_channel_axis = transpose_input_axes.pop()
        elif child_op.name == _GATHER_ND_OP:
            output_channel_axis = 0
            input_channel_axis = 1
        elif child_op.name in _WEIGHT_COMPRESSION_OPS:
            # In joint compression, constexpr ops can be chained; recurse.
            input_channel_axis, output_channel_axis = _select_input_output_channel_axis(child_op)

        if output_channel_axis is not None and output_channel_axis < 0:
            output_channel_axis += op.result.type.rank  # type: ignore[attr-defined]
        if input_channel_axis is not None and input_channel_axis < 0:
            input_channel_axis += op.result.type.rank  # type: ignore[attr-defined]
        output_channel_axis_set.add(output_channel_axis)
        input_channel_axis_set.add(input_channel_axis)

    output_channel_axis = 0
    input_channel_axis = 1
    if len(output_channel_axis_set) > 1:
        logger.warning(
            "Can't decide output axis for op %s, because it's fed into multiple "
            "downstream ops which require different output axes.",
            op.name,
        )
        output_channel_axis = None
    elif len(output_channel_axis_set) == 1:
        output_channel_axis = output_channel_axis_set.pop()

    if len(input_channel_axis_set) > 1:
        logger.warning(
            "Can't decide input axis for op %s, because it's fed into multiple "
            "downstream ops which require different input axes.",
            op.name,
        )
        input_channel_axis = None
    elif len(input_channel_axis_set) == 1:
        input_channel_axis = input_channel_axis_set.pop()

    return input_channel_axis, output_channel_axis


def _infer_quantization_block_sizes(
    op: Operation,
    weight_shape: Sequence[int],
    granularity: _CompressionGranularity,
    block_size: int,
) -> Sequence[int]:
    """Infer per-axis block sizes for weight quantization.

    For quantization, ``PER_BLOCK`` applies ``block_size`` to the input
    channel axis and sets output channel block size to 1. This differs from
    palettization, where ``group_size`` applies to the output channel axis.

    Args:
        op (Operation): The constant operation whose downstream consumers determine
            the channel axes.
        weight_shape (Sequence[int]): Shape of the weight tensor.
        granularity (CompressionGranularity): Compression granularity; one of the
            :class:`CompressionGranularity` string values.
        block_size (int): Block size applied to the input channel axis for
            ``PER_BLOCK`` granularity.

    Returns:
        Sequence[int]: ``block_sizes[i]`` is the block size for axis ``i``
            (0 means no blocking on that axis).
    """
    input_channel_axis, output_channel_axis = _select_input_output_channel_axis(op)

    if input_channel_axis is None:
        logger.warning("Cannot determine input_channel_axis for block_sizes, use 1 by default.")
        input_channel_axis = 1
    if output_channel_axis is None:
        logger.warning(
            "Cannot determine output_channel_axis for block_sizes, use 0 by default.",
        )
        output_channel_axis = 0

    block_sizes = [0] * len(weight_shape)
    if granularity == _CompressionGranularity.PER_TENSOR:
        pass
    elif granularity == _CompressionGranularity.PER_CHANNEL:
        if output_channel_axis < len(block_sizes):
            block_sizes[output_channel_axis] = 1
    else:
        assert granularity == _CompressionGranularity.PER_BLOCK
        assert isinstance(block_size, int)
        if output_channel_axis < len(block_sizes):
            block_sizes[output_channel_axis] = 1
        if input_channel_axis < len(block_sizes):
            block_sizes[input_channel_axis] = block_size

    return block_sizes


def _should_compress_op(
    op: Any,
    weight_num_threshold: int,
    ops_weight_need_compression: frozenset[str],
) -> bool:
    """Determine if a constant op should be weight-compressed."""
    if op.name != "coreai.constant":
        return False

    if not (isinstance(op.result.type, RankedTensorType) and op.result.type.has_static_shape):
        return False

    if isinstance(op.result.type.element_type, IntegerType):
        return False

    num_element = np.prod(op.result.type.shape)
    if num_element <= weight_num_threshold:
        return False

    for child_op in op.result.uses:
        if child_op.owner.name not in ops_weight_need_compression:
            # For very large constants, try to compress anyway.
            return bool(num_element >= 1e8)

    return True


def _apply_compression_transform(
    coreai_program: AIProgram,
    compression_fn: Callable[[Operation], WalkResult],
    in_place: bool = False,
) -> AIProgram:
    """Apply a compression transformation to a Core AI program.

    Handles the common boilerplate for compression transforms: optionally deep-copies
    the program, walks operations with the provided function, and returns the
    modified program.

    Args:
        coreai_program (AIProgram): The model to be compressed.
        compression_fn (Callable[[Operation], WalkResult]): Function that takes an
            Operation and returns WalkResult. Should contain the specific compression
            logic.
        in_place (bool): Whether to modify the model in-place or not.

    Returns:
        AIProgram: A compressed Core AI program.
    """
    if not in_place:
        coreai_program = copy.deepcopy(coreai_program)

    coreai_program._module._mlir_module.operation.walk(compression_fn)

    return coreai_program


def _create_constant_value_from_np_array(
    value: npt.NDArray[np.number[Any]] | np.number[Any],
    value_type: Any,
    loc: Any | None = None,
) -> Any:
    """Create a coreai.constant output value from numpy array.

    For sub-byte dtype (such as int4), the value is unpacked (e.g. int8 used to
    represent int4), so it will be packed first during construction.
    If loc is None, the one from the current context manager is used.
    """
    data_tensortype = TensorType(
        shape=list(value.shape) if isinstance(value, np.ndarray) else [],
        dtype=value_type,
    )._to_mlir()
    # Convert to numpy array for consistent handling (before packing, for correct
    # splat check).
    np_value = np.array(value)

    if _is_sub_byte_int(value_type):
        value_type = cast("Any", value_type)
        if value_type.is_signed:
            value = pack_intx(  # type: ignore[assignment]
                torch.tensor(value), value_type.width
            )
        else:
            value = pack_uintx(  # type: ignore[assignment]
                torch.tensor(value), value_type.width
            )

    if loc is None:
        current_debug_info = DebugInfo.current()
        loc = None if current_debug_info is None else current_debug_info._to_mlir()

    # Check if this is a splat (all values are the same).
    # Use np_value (original, unpacked elements) so sub-byte packing does not
    # affect the check.
    data_attr: Any
    if np_value.size > 0 and np.all(np_value == np_value.flat[0]):
        # Use DenseElementsAttr.get_splat for splat values.
        scalar_value = np_value.flat[0]
        # Create appropriate attribute based on the MLIR element type.
        # Use the MLIR type rather than np_value.dtype so that the numpy storage
        # format (e.g. float32 used to hold int4 values) does not determine the
        # attribute kind.
        mlir_element_type = data_tensortype.element_type
        if (
            isinstance(mlir_element_type, IntegerType)
            and mlir_element_type.is_signless
            and mlir_element_type.width == 1
        ):
            # Signless i1 is MLIR bool.
            element_attr: Any = BoolAttr.get(bool(scalar_value))
        elif isinstance(mlir_element_type, IntegerType):
            # Integer types (including sub-byte types like si4, ui4).
            element_attr = IntegerAttr.get(
                mlir_element_type,
                int(scalar_value),
            )
        else:
            # Floating point types.
            element_attr = FloatAttr.get(
                mlir_element_type,
                float(scalar_value),
            )

        data_attr = DenseElementsAttr.get_splat(data_tensortype, element_attr)
    else:
        # Fall back to the custom create_elements_attr for non-splat values.
        # For sub-byte types, pass the packed bytes (value); otherwise pass
        # np_value.
        if isinstance(value, np.ndarray):
            packed_value = value
        elif isinstance(value, torch.Tensor):
            packed_value = value.numpy()
        else:
            packed_value = np.array(value)
        data_attr = create_elements_attr(
            loc,
            data_tensortype,
            validate_and_cast_numpy_array(packed_value),
        )
    return cast("Any", coreai.ConstantOp(value=data_attr, loc=loc).result)
