# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def lazy_import_module(import_func: Callable[[], T], error_message: str) -> T:
    """Wrap an import call with a custom ImportError message.

    Wraps import statements to provide custom error messages when optional
    dependencies are not installed.

    Args:
        import_func (Callable[[], T]): A callable that performs the import when
            executed. Should raise ImportError if the dependency is not available.
        error_message (str): Custom error message to display when import fails.

    Returns:
        T: The result of executing import_func (typically imported module(s)).

    Raises:
        ImportError: If the dependency is not installed, with the provided
            custom error message.

    Example:
        >>> def _import_numpy():
        ...     import numpy as np
        ...     return np
        >>> np = lazy_import_module(
        ...     _import_numpy,
        ...     "numpy is required. Install it with: pip install numpy"
        ... )
    """
    try:
        return import_func()
    except ImportError as e:
        raise ImportError(error_message) from e


def lazy_import_coreai_torch(import_func: Callable[[], T]) -> T:
    """Lazily import coreai_torch-dependent modules with standardized error handling.

    Example:
        >>> def _import_custom_layers():
        ...     from coreai_torch._compression.custom_layers import WeightDequantizeModule
        ...     from coreai_torch._compression.utils import wrap_for_parametrization
        ...     return WeightDequantizeModule, wrap_for_parametrization
        >>> modules = lazy_import_coreai_torch(_import_custom_layers)
    """
    error_message = (
        "Core AI export needs the optional coreai and coreai-torch packages; install them "
        "with this package's `coreai` extra. To finalize without them, pass "
        "backend=ExportBackend.Torch."
    )
    return lazy_import_module(import_func, error_message)
