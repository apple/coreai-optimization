# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import inspect

from pydantic import BaseModel


def fqn(obj_type: type) -> str:
    """
    Get the fully qualified name (FQN) of a type object.

    The FQN consists of the module name and the qualified name of the type,
    separated by a dot.
    """
    if obj_type is None or not isinstance(obj_type, type):
        raise TypeError(f"Expected a type, got {obj_type}")

    return f"{obj_type.__module__}.{obj_type.__qualname__}"


def get_fn_arg_names(func: callable) -> list[str]:
    signature = inspect.signature(func)
    arg_names = list(signature.parameters.keys())

    return arg_names


def get_generic_type_arg(cls: type[BaseModel], origin: type, arg_index: int = 0) -> type | None:
    """
    Get a generic type argument from a Pydantic model's base class.

    This function extracts type arguments from Pydantic's generic metadata.
    For example, if `cls` inherits from `SomeGeneric[TypeA, TypeB]`, this
    function can extract `TypeA` or `TypeB` based on the `arg_index`.

    Args:
        cls: The class to inspect (should be a Pydantic model subclass)
        origin: The expected generic origin class to match against
        arg_index: Which type argument to return (0-indexed, default 0)

    Returns:
        The type argument at the specified index, or None if not found
    """
    for base in cls.__bases__:
        metadata = getattr(base, "__pydantic_generic_metadata__", None)
        if metadata and metadata.get("origin") is origin:
            args = metadata.get("args", ())
            if len(args) > arg_index and isinstance(args[arg_index], type):
                return args[arg_index]

    return None
