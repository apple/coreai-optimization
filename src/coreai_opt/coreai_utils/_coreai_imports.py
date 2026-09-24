# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Lazy imports for coreai/coreai-torch optional dependencies used by coreai-opt passes."""

# TODO: refactor the lazy import handling for coreai dependencies.
from __future__ import annotations

import importlib
import math
import typing
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from coreai_opt._utils.import_utils import lazy_import_coreai_torch


class _LazyModule:
    """Import a module on first attribute access."""

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_mod", None)

    def __getattr__(self, item: str) -> Any:
        mod = object.__getattribute__(self, "_mod")
        if mod is None:
            name = object.__getattribute__(self, "_name")
            mod = importlib.import_module(name)
            object.__setattr__(self, "_mod", mod)
        return getattr(mod, item)


ml_dtypes = _LazyModule("ml_dtypes")


def _infer_np_dtype_from_ranked_tensor_type(
    ranked_tensor_type: Any,
) -> type | None:
    from coreai._compiler.ir import F16Type, F32Type, IntegerType  # noqa: PLC0415

    if isinstance(ranked_tensor_type.element_type, F32Type):
        return np.float32
    elif isinstance(ranked_tensor_type.element_type, F16Type):
        return np.float16
    elif isinstance(ranked_tensor_type.element_type, IntegerType):
        dtype_str = f"int{ranked_tensor_type.element_type.width}"
        if ranked_tensor_type.element_type.is_unsigned:
            dtype_str = "u" + dtype_str
        if hasattr(np, dtype_str):
            return cast("type[np.integer[Any]]", getattr(np, dtype_str))
    return None


def _get_constant_value_as_np_array(
    constant_op: Any,
) -> npt.NDArray[np.number[Any]]:
    """Get a constant op's value as a numpy array.

    If the result dtype is supported by numpy (such as float16, float32, etc), the returned
    numpy array will have the corresponding dtype.
    If the result dtype is not supported by numpy (such as int4), the returned numpy will
    be the raw byte array.
    """
    from coreai._compiler._mlir_libs._coreaiIR import _bindings  # noqa: PLC0415
    from coreai._compiler.ir import RankedTensorType  # noqa: PLC0415

    if not constant_op.name.endswith("constant"):
        err_msg = f"The input op should be a constant op, but got {constant_op.name}"
        raise ValueError(err_msg)
    if "value" not in constant_op.attributes:
        err_msg = "The constant op should have `value` in attributes"
        raise ValueError(err_msg)

    result = np.array(constant_op.attributes["value"])
    if result.dtype == np.object_:
        # The attribute is dense resource and needs special handle.
        if not isinstance(constant_op.result.type, RankedTensorType):
            err_msg = "Only support constant op of RankedTensorType."
            raise AssertionError(err_msg)
        if not constant_op.result.type.has_static_shape:
            err_msg = "Only support constant op with static shape."
            raise AssertionError(err_msg)
        target_shape = constant_op.result.type.shape
        bytes_per_element = constant_op.result.type.element_type.width / 8.0  # type: ignore[attr-defined]
        total_bytes = math.ceil(np.prod(target_shape) * bytes_per_element)
        bytes_buffer = np.zeros(total_bytes, dtype=np.uint8)
        # TODO: drop the coreai dependency for copying constant values into arrays.
        _bindings.compiler.copy_constant_value_to_array(  # type: ignore[attr-defined]
            constant_op=constant_op,
            destination=bytes_buffer,
        )
        np_dtype = _infer_np_dtype_from_ranked_tensor_type(constant_op.result.type)
        if np_dtype is not None:
            result = np.frombuffer(bytes_buffer.tobytes(), dtype=np_dtype).reshape(
                target_shape,
            )
        else:
            result = bytes_buffer

    return result


def validate_and_cast_numpy_array(
    arr: np.ndarray[typing.Any, np.dtype[typing.Any]],
) -> np.ndarray[typing.Any, np.dtype[typing.Any]]:
    """Validate and cast a numpy array.

    Only ranked arrays are accepted.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    if arr.shape == ():
        arr = arr[None]
    arr = np.require(
        arr,
        requirements=["C_CONTIGUOUS"],
    )
    arr.flags.writeable = False

    return arr


def _import_coreai() -> tuple:
    # isort: off
    # coreai.authoring must import before coreai._compiler.dialects.coreai: the dialect's
    # constant.py pulls in coreai.authoring.constraints, which (if coreai.authoring isn't
    # already fully initialized) circles back into the still-initializing dialect module
    # and raises ImportError. Keep this order; do not let isort/ruff re-sort it away.
    from coreai.authoring import (  # noqa: PLC0415
        AIProgram,
        Context,
        DebugInfo,
        TensorType,
        blockwise_shift_scale,
        build_sparse_with_bitmask,
        cast as authoring_cast,
        float4_e2m1fn,
        float8_e4m3fn,
        float8_e5m2,
        float8_e8m0fnu,
        float16,
        float32,
        lut_to_dense,
        sparse_with_bitmask_to_dense,
    )
    from coreai.authoring.constant import constant as authoring_constant  # noqa: PLC0415
    # isort: on

    from coreai._compiler.dialects import coreai as coreai_dialect  # noqa: PLC0415
    from coreai._compiler.dialects.coreai.constant import create_elements_attr  # noqa: PLC0415
    from coreai._compiler.ir import (  # noqa: PLC0415
        BoolAttr,
        DenseElementsAttr,
        DenseResourceElementsAttr,
        FloatAttr,
        FloatType,
        InsertionPoint,
        IntegerAttr,
        IntegerType,
        Operation,
        RankedTensorType,
        WalkResult,
    )
    from coreai_torch._compression import _types as compression_types_mod  # noqa: PLC0415
    from coreai_torch._compression._intx import (  # noqa: PLC0415
        pack_intx,
        pack_uintx,
    )

    return (
        create_elements_attr,
        compression_types_mod,
        pack_intx,
        pack_uintx,
        coreai_dialect,
        AIProgram,
        BoolAttr,
        Context,
        DenseElementsAttr,
        DenseResourceElementsAttr,
        FloatAttr,
        FloatType,
        InsertionPoint,
        IntegerAttr,
        IntegerType,
        Operation,
        RankedTensorType,
        WalkResult,
        authoring_cast,
        authoring_constant,
        blockwise_shift_scale,
        build_sparse_with_bitmask,
        float4_e2m1fn,
        float8_e4m3fn,
        float8_e5m2,
        float8_e8m0fnu,
        float16,
        float32,
        lut_to_dense,
        sparse_with_bitmask_to_dense,
        TensorType,
        DebugInfo,
    )


(
    create_elements_attr,
    compression_types,
    pack_intx,
    pack_uintx,
    coreai,
    AIProgram,
    BoolAttr,
    Context,
    DenseElementsAttr,
    DenseResourceElementsAttr,
    FloatAttr,
    FloatType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Operation,
    RankedTensorType,
    WalkResult,
    authoring_cast,
    authoring_constant,
    blockwise_shift_scale,
    build_sparse_with_bitmask,
    float4_e2m1fn,
    float8_e4m3fn,
    float8_e5m2,
    float8_e8m0fnu,
    float16,
    float32,
    lut_to_dense,
    sparse_with_bitmask_to_dense,
    TensorType,
    DebugInfo,
) = lazy_import_coreai_torch(_import_coreai)
