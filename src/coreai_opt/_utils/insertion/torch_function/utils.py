# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Utility functions for eager mode compression."""

import itertools
import logging
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch._ops import OpOverload
from torch.fx.node import map_aggregate
from torch.fx.operator_schemas import create_type_hint, normalize_function

from coreai_opt._utils.config_utils import get_last_matching_spec
from coreai_opt._utils.spec_utils import PartialConstructor
from coreai_opt.config.spec import CompressionSimulatorBase

logger = logging.getLogger(__name__)

_OPTIMIZABLE_DTYPES = {torch.float64, torch.float32, torch.float16, torch.bfloat16}


def normalize_args_kwargs(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Normalize function arguments to keyword-only format.

    Converts positional arguments to keyword arguments using PyTorch's function schema
    introspection. Handles special cases for TensorBase methods by mapping them to their
    corresponding aten operations.

    Args:
        func: The function/operation to normalize arguments for
        args: Positional arguments passed to the function
        kwargs: Keyword arguments passed to the function

    Returns:
        Tuple of (normalized_args, normalized_kwargs) where args are converted to kwargs
        when possible using the function's schema
    """
    if func.__qualname__.startswith("TensorBase"):
        method_name = func.__name__
        # Tensor methods are typically not inspectable, so we need to
        # map them to their corresponding overloaded aten representation first
        func = getattr(torch.ops.aten, method_name)

    if len(args) == 0:
        return args, kwargs

    arg_types = map_aggregate(args, lambda v: type(v))
    assert isinstance(arg_types, tuple)
    arg_types = tuple([create_type_hint(i) for i in arg_types])
    kwarg_types = {k: type(v) for k, v in kwargs.items()}
    normalize_output = None
    try:
        normalize_output = normalize_function(
            func,
            args,
            kwargs,
            arg_types,
            kwarg_types,
            normalize_to_only_use_kwargs=True,
        )
    except (TypeError, RuntimeError, AssertionError) as e:
        logger.warning(f"Couldn't normalize args to kwargs for func {func}: {e}")

    if normalize_output is not None:
        args, kwargs = normalize_output
        # addresses https://github.com/pytorch/pytorch/blob/e880cb2fe0f9742a5cb62b3ef87b308cec01a48e/torch/fx/operator_schemas.py#L74
        if isinstance(func, OpOverload):
            schema = func._schema
            if schema.arguments[0].name == "self":
                kwargs["self"] = kwargs.pop("input")
    return args, kwargs


def get_func_base_name(func: Callable) -> str:
    """
    Return the function base name
    """
    func_name = torch.overrides.resolve_name(func)
    if func_name is None:
        if not hasattr(func, "__name__"):
            error_msg = f"Unable to obtain function name for {func}."
            raise RuntimeError(error_msg)
        return func.__name__
    return func_name.rsplit(".", maxsplit=1)[-1]


def get_func_name(func: Callable, func_count: int) -> str:
    """Return the function name using the base name and func_count.

    For func_count of 0, leave it out when constructing the function name.
    This brings the naming behavior more in line with graph mode.

    """
    if func_count == 0:
        return get_func_base_name(func)
    return f"{get_func_base_name(func)}_{func_count}"


def is_optimizable_tensor(tensor: Any) -> bool:
    """
    Return True if the tensor is optimizable, False otherwise.
    Criteria for optimizable tensor:
    - Is a torch.Tensor
    - Has more than one value (not a scalar)
    - Is floating point dtype
    """
    if (
        not isinstance(tensor, torch.Tensor)
        or torch.numel(tensor) <= 1
        or tensor.dtype not in _OPTIMIZABLE_DTYPES
    ):
        return False
    return True


def any_tensor_optimizable(args: list[Any], kwargs: dict[str, Any]) -> bool:
    """
    Return True if any of the tensors in args and kwargs are optimizable, False otherwise.
    """
    return any(is_optimizable_tensor(tensor) for tensor in itertools.chain(args, kwargs.values()))


def get_optimizer_from_components_dict(
    func: Callable,
    tensor_identifiers: int | str | list[int | str],
    components_dict: Mapping[int | str, PartialConstructor | None],
) -> tuple[CompressionSimulatorBase | None, bool]:
    """Return the appropriate optimizer from ``components_dict``.

    Delegates matching to :func:`~coreai_opt._utils.config_utils.get_last_matching_spec`.
    Returns ``(optimizer, True)`` on match (optimizer may be ``None`` if the
    components-dict entry was explicit-None to disable). Returns
    ``(None, False)`` if no identifier matched.
    """
    if not isinstance(tensor_identifiers, list):
        tensor_identifiers = [tensor_identifiers]
    constructor, found = get_last_matching_spec(tensor_identifiers, components_dict)
    if found:
        return (constructor(op_to_optimize=func), True) if constructor else (None, True)
    return None, False
