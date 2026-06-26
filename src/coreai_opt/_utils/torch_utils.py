# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Utilities for working with PyTorch types and operations."""

from __future__ import annotations

import logging
import re
from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from os import PathLike
from typing import Any, Final, NamedTuple

import torch
import torch.nn.utils.parametrize as P

from coreai_opt._utils.version_utils import version_ge as _version_ge

logger = logging.getLogger(__name__)

# Mapping from dtype to the largest power-of-2 component of its max value.
# Used by e8m0 scale computation (OCP Microscaling spec, FLOOR mode).
FP_DTYPE_TO_MAX_POW2: dict[torch.dtype, int] = {
    torch.float4_e2m1fn_x2: 2,
    torch.float8_e4m3fn: 8,  # max = 448.0 = 1.75 * 2^8
    torch.float8_e5m2: 15,  # max = 57344.0 = 1.75 * 2^15
}

# Constants for e8m0 scale computation.
E8M0_EXPONENT_BIAS: Final[int] = 127
F32_MIN_NORMAL: Final[float] = 2**-126  # ~1.175e-38


class NamedModule(NamedTuple):
    """NamedTuple for holding name and module info"""

    name: str
    module: torch.nn.Module


def flatten_tensors_to_list(obj: Any) -> list[torch.Tensor]:
    """Flatten nested structure to extract all tensors."""
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        result = []
        for item in obj:
            result.extend(flatten_tensors_to_list(item))
        return result
    if isinstance(obj, dict):
        result = []
        for value in obj.values():
            result.extend(flatten_tensors_to_list(value))
        return result
    return []


def get_module_name(model: torch.nn.Module, module: torch.nn.Module) -> str | None:
    """Find the fully qualified name of a module within a model.

    Args:
        model: The root model to search.
        module: The module to find.

    Returns:
        The fully qualified name (e.g. ``"encoder.block1.conv"``), or
        ``None`` if the module is not found in the model.
    """
    for name, mod in model.named_modules():
        if mod is module:
            return name
    return None


def sanitize_module_name(
    name: str,
    no_leading_underscore: bool = False,
) -> str:
    """Sanitize a string into a safe Python identifier for module names."""
    # Replace invalid identifier characters
    sanitized = re.sub(r"[^0-9a-zA-Z_]", "_", name)

    # Ensure non-empty
    if not sanitized:
        sanitized = "_"

    # Collapse consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)

    # Optionally remove leading underscores
    if no_leading_underscore:
        sanitized = sanitized.lstrip("_")
        if not sanitized:  # must remain valid
            sanitized = "x"

    return sanitized


def get_n_bits_from_dtype(dtype: torch.dtype) -> int:
    """Extract the number of bits from a torch dtype.

    Args:
        dtype: The torch dtype to extract bits from

    Returns:
        Number of bits for the dtype

    Raises:
        RuntimeError: If unable to extract bits from the dtype

    """
    try:
        if dtype == torch.float4_e2m1fn_x2:
            return 4
        info = torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)
        return int(info.bits)
    except (TypeError, AttributeError) as e:
        # Fallback for custom dtypes like int1, int2, etc.
        matches = re.search(r"\d+", str(dtype))
        if matches:
            return int(matches.group())
        msg = f"Unable to extract number of bits from dtype {dtype}"
        raise RuntimeError(msg) from e


def _get_training_state(model: torch.nn.Module) -> bool | None:
    """Return the effective training state of a model, or ``None`` if unknown.

    For regular ``nn.Module``, returns ``model.training``.

    For ``fx.GraphModule``, ``model.training`` is unreliable — it is always
    ``True`` after export regardless of the original model's mode and is never
    updated by ``.train()``/``.eval()``. Instead, the ``_exported_training``
    attribute (set by ``allow_exported_model_train_eval``) tracks the true
    mode. This attribute is only present after ``.train()`` or ``.eval()`` has
    been called at least once on the exported model.

    Returns ``None`` when ``model`` is an ``fx.GraphModule`` without
    ``_exported_training`` set, meaning the training state cannot be
    reliably determined. Callers should handle this case by requiring
    ``original_state`` to be passed explicitly.
    """
    if isinstance(model, torch.fx.GraphModule):
        return getattr(model, "_exported_training", None)
    return model.training


@contextmanager
def move_model_to_train(
    model: torch.nn.Module | torch.fx.GraphModule,
    original_state: bool | None = None,
) -> Generator[None, None, None]:
    """
    Context manager that moves model to training mode and restores
    original state on exit.

    Supports both regular nn.Module and exported FX GraphModule types
    (when ``allow_exported_model_train_eval`` has been applied).

    Args:
        model: The model to modify.
        original_state: Original training state to restore on exit.
            Required for a freshly exported ``fx.GraphModule`` where
            ``.train()``/``.eval()`` has not yet been called.

    Raises:
        RuntimeError: If ``model`` is an ``fx.GraphModule`` whose
            training state cannot be determined (i.e.,
            ``_exported_training`` is not set) and ``original_state``
            is not provided. Call ``.train()`` or ``.eval()`` on the
            model first, or pass ``original_state`` explicitly.

    Yields:
        None

    Example:
        >>> with move_model_to_train(model):
        ...     # model is in training mode
        ...     pass
    """
    # Capture the original training state if not provided
    if original_state is None:
        original_state = _get_training_state(model)
        if original_state is None:
            raise RuntimeError(
                "Cannot determine training state of an fx.GraphModule. "
                "Call .train() or .eval() on the model first, or pass "
                "original_state explicitly."
            )

    model.train()

    try:
        yield
    finally:
        model.train(original_state)


@contextmanager
def move_model_to_eval(
    model: torch.nn.Module | torch.fx.GraphModule,
    original_state: bool | None = None,
) -> Generator[None, None, None]:
    """
    Context manager that moves model to eval mode and restores original
    state on exit.

    Supports both regular nn.Module and exported FX GraphModule types
    (when ``allow_exported_model_train_eval`` has been applied).

    Args:
        model: The model to modify.
        original_state: Original training state to restore on exit.
            Required for a freshly exported ``fx.GraphModule`` where
            ``.train()``/``.eval()`` has not yet been called.

    Raises:
        RuntimeError: If ``model`` is an ``fx.GraphModule`` whose
            training state cannot be determined (i.e.,
            ``_exported_training`` is not set) and ``original_state``
            is not provided. Call ``.train()`` or ``.eval()`` on the
            model first, or pass ``original_state`` explicitly.

    Yields:
        None

    Example:
        >>> with move_model_to_eval(model):
        ...     # model is in eval mode here
        ...     pass
        >>> # model is back to its original state
    """
    # Capture the original training state if not provided
    if original_state is None:
        original_state = _get_training_state(model)
        if original_state is None:
            raise RuntimeError(
                "Cannot determine training state of an fx.GraphModule. "
                "Call .train() or .eval() on the model first, or pass "
                "original_state explicitly."
            )

    model.eval()

    try:
        yield
    finally:
        model.train(original_state)


def is_float8_dtype(dtype: torch.dtype) -> bool:
    """Check if dtype is a float8 variant."""
    return dtype in {
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    }


def is_float4_dtype(dtype: torch.dtype) -> bool:
    """Check if dtype is a float4 variant."""
    return dtype == torch.float4_e2m1fn_x2


def is_float_quant_dtype(dtype: torch.dtype) -> bool:
    """Check if dtype is a floating-point (FP4/FP8) quantization dtype.

    Canonical predicate for "is this a floating-point quantization target dtype" —
    the union of the FP8 and FP4 variants. Prefer it over open-coding
    ``is_float8_dtype(dtype) or is_float4_dtype(dtype)`` so a new floating-point
    quant dtype only has to be recognized in one place.
    """
    return is_float8_dtype(dtype) or is_float4_dtype(dtype)


def remove_compression_parametrizations(
    model: torch.nn.Module,
    modules_to_remove: set[torch.nn.Module],
) -> int:
    """Remove specific compression parametrization instances from a model.

    Unlike ``torch.nn.utils.parametrize.remove_parametrizations`` which removes
    ALL parametrizations on a given parameter, this function removes only the
    specific instances in ``modules_to_remove``, leaving other parametrizations
    on the same parameter intact.

    The current weight values are preserved (unchanged).

    Args:
        model (torch.nn.Module): The model to modify.
        modules_to_remove (set[torch.nn.Module]): Set of parametrization
            module instances to remove.

    Returns:
        int: Number of parametrizations removed.

    """
    ids_to_remove = {id(m) for m in modules_to_remove}
    num_removed = 0

    for module in list(model.modules()):
        if not P.is_parametrized(module):
            continue
        for param_name, parametrizations in list(module.parametrizations.items()):
            to_keep = [p for p in parametrizations if id(p) not in ids_to_remove]
            num_removing = len(parametrizations) - len(to_keep)
            if num_removing == 0:
                continue

            # Remove all parametrizations on this param, restoring original weight
            P.remove_parametrizations(module, param_name, leave_parametrized=False)

            # Re-register the ones we want to keep
            for p in to_keep:
                P.register_parametrization(module, param_name, p, unsafe=True)

            num_removed += num_removing

    return num_removed


def find_parametrization_matching_cls(
    module: torch.nn.Module,
    param_name: str,
    cls: type,
) -> torch.nn.Module | None:
    """Return the first parametrization on ``module.<param_name>`` matching ``cls``.

    Args:
        module (torch.nn.Module): A module that has been parametrized.
        param_name (str): The name of the parameter to inspect.
        cls (type): Class filter. Returns the first parametrization that is an
            instance of this class.

    Returns:
        torch.nn.Module | None: The matching parametrization, or ``None`` if no
        parametrization exists on the parameter or none match ``cls``.
    """
    if not hasattr(module, "parametrizations") or param_name not in module.parametrizations:
        return None
    for p in module.parametrizations[param_name]:
        if isinstance(p, cls):
            return p
    return None


def get_parent_module_and_attr_name(
    model: torch.nn.Module,
    module_name: str,
) -> tuple[torch.nn.Module, str]:
    """Extract parent module and attribute name from a fully qualified module name.

    Args:
        model (torch.nn.Module): The root model.
        module_name (str): Fully qualified module name (e.g., "layer1.conv.relu").

    Returns:
        tuple[torch.nn.Module, str]: Tuple of (parent_module, attr_name).

    """
    if "." in module_name:
        parent_name, attr_name = module_name.rsplit(".", 1)
        parent_module = model.get_submodule(parent_name)
    else:
        parent_module = model
        attr_name = module_name

    return parent_module, attr_name


def is_tensor_on_cpu(tensor: torch.Tensor) -> bool:
    """Return True if ``tensor`` lives on the CPU device.

    Uses ``device.type`` so any CPU device variant (``"cpu"``, ``"cpu:0"``, etc.)
    matches.
    """
    return tensor.device.type == "cpu"


def export_model(
    model: torch.nn.Module,
    example_inputs: tuple[Any, ...],
    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None,
    export_with_no_grad: bool,
) -> torch.fx.GraphModule:
    """Export an ``nn.Module`` to a ``GraphModule``.

    Args:
        model (torch.nn.Module): The model to export.
        example_inputs (tuple[Any, ...]): Example inputs for tracing.
        dynamic_shapes: Dynamic shapes specification for torch.export.
                        Can be a dict mapping input names to dynamic dimensions,
                        a tuple/list of dynamic shapes per input, or None for
                        static shapes. Used to specify which dimensions can vary
                        at runtime during model export.
        export_with_no_grad: Whether to call torch.export.export within a
                             torch.no_grad() context.

    Returns:
        torch.fx.GraphModule: The exported graph module.

    Raises:
        RuntimeError: If ``torch.export.export()`` fails.
    """
    try:
        context = torch.no_grad() if export_with_no_grad else nullcontext()
        with context:
            exported_program = torch.export.export(
                model, example_inputs, dynamic_shapes=dynamic_shapes
            )
            if _version_ge(torch, "2.9"):
                exported_model = exported_program.module(check_guards=False)
            else:
                exported_model = exported_program.module()
        return exported_model
    except Exception as e:
        raise RuntimeError(
            f"Failed to trace the model with torch.export.export(), received error: {e}"
        ) from e


def mmap_module_state_dict(module: torch.nn.Module, path: str | PathLike[str]) -> None:
    """Serialize ``module.state_dict()`` to a safetensors file at ``path`` and
    reload it via mmap, replacing the module's parameters/buffers with mmap
    views so the in-RAM tensors can be released.

    Requires all tensors in ``module.state_dict()`` to be on CPU. Raises
    ``ValueError`` otherwise — mmap is a CPU-only mechanism
    """
    from safetensors.torch import load_file, save_file  # noqa: PLC0415

    state_dict = module.state_dict()
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if not is_tensor_on_cpu(tensor):
            raise ValueError(
                f"mmap_module_state_dict requires CPU tensors; '{name}' is on {tensor.device}."
            )

    save_file(
        {k: v.contiguous() for k, v in state_dict.items() if isinstance(v, torch.Tensor)},
        path,
    )
    mmap_sd = load_file(path, device="cpu")
    module.load_state_dict(mmap_sd, assign=True)
