# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from os import PathLike
from pathlib import Path

import torch

from coreai_opt._utils.torch_utils import is_tensor_on_cpu
from coreai_opt.common import CoreMLExportError, ExportBackend
from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.quantization.spec.granularity import PerTensorGranularity, QuantizationGranularity

COREML_SUPPORTED_WEIGHT_DTYPES: frozenset[torch.dtype] = frozenset(
    {
        torch.int8,
        torch.uint8,
        torch.int4,
        torch.uint4,
    }
)

COREML_SUPPORTED_ACTIVATION_DTYPES: frozenset[torch.dtype] = frozenset(
    {
        torch.int8,
        torch.uint8,
    }
)

COREML_SUPPORTED_LUT_DTYPES: frozenset[torch.dtype] = frozenset(
    {
        torch.int8,
        torch.uint8,
    }
)

COREML_SUPPORTED_ACTIVATION_GRANULARITIES: frozenset[type[QuantizationGranularity]] = frozenset(
    {PerTensorGranularity}
)

_COREML_SUPPORTED_DTYPES_BY_TARGET: dict[CompressionTargetTensor, frozenset[torch.dtype]] = {
    CompressionTargetTensor.WEIGHT: COREML_SUPPORTED_WEIGHT_DTYPES,
    CompressionTargetTensor.ACTIVATION: COREML_SUPPORTED_ACTIVATION_DTYPES,
    CompressionTargetTensor.LUT: COREML_SUPPORTED_LUT_DTYPES,
}


def validate_coreml_compatibility(
    target: CompressionTargetTensor,
    dtype: torch.dtype,
    context: str,
    granularity: QuantizationGranularity | None = None,
) -> None:
    """Raise CoreMLExportError if this weight/activation/LUT config isn't CoreML-exportable.

    Centralizes every reason CoreML export can reject a quantization config, so
    new restrictions are added here once rather than at each call site.

    Args:
        target (CompressionTargetTensor): Which tensor category is being checked.
        dtype (torch.dtype): The quantization dtype to validate.
        context (str): Human-readable description of what's being checked, used
            in the error message (e.g. "weight 'conv.weight' of module 'conv'").
        granularity (QuantizationGranularity | None): The quantization
            granularity, if applicable. Only checked for ACTIVATION — CoreML
            only supports per-tensor activation quantization.

    Raises:
        CoreMLExportError: If the dtype or granularity isn't supported.
    """
    if dtype not in _COREML_SUPPORTED_DTYPES_BY_TARGET[target]:
        raise CoreMLExportError.from_dtype(dtype, context)
    if target == CompressionTargetTensor.ACTIVATION and not isinstance(
        granularity, tuple(COREML_SUPPORTED_ACTIVATION_GRANULARITIES)
    ):
        raise CoreMLExportError.from_config(granularity, context)


def validate_mmap_backend_and_device(
    model: torch.nn.Module,
    backend: ExportBackend,
    mmap_dir: str | PathLike[str] | None,
) -> None:
    """Validate that ``mmap_dir`` is compatible with the target backend and
    model device. No-op when ``mmap_dir is None``.

    Args:
        model (nn.Module): The (already-resolved) model whose parameters and
            buffers will be inspected for non-CPU tensors.
        backend (ExportBackend): Target export backend; only ``CoreAI`` is
            supported with ``mmap_dir``.
        mmap_dir (str | PathLike | None): If set, opt in to mmap-backed
            finalization.

    Raises:
        ValueError: If ``mmap_dir`` is set but the backend is not CoreAI, or
            if any tensor in ``model.state_dict()`` is on a non-CPU device.
    """
    if mmap_dir is None:
        return
    if backend != ExportBackend.CoreAI:
        raise ValueError(
            f"mmap_dir is only supported with backend=ExportBackend.CoreAI, got backend={backend}."
        )
    non_cpu_devices = {
        str(t.device)
        for _, t in model.state_dict().items()
        if isinstance(t, torch.Tensor) and not is_tensor_on_cpu(t)
    }
    if non_cpu_devices:
        raise ValueError(
            "mmap_dir requires the prepared model to be on CPU; "
            f"found tensor(s) on device(s) {non_cpu_devices}. "
            "Call model.cpu() before finalize(mmap_dir=...). "
            "mmap is a CPU-only mechanism"
        )


def prepare_mmap_dir(mmap_dir: str | PathLike[str] | None) -> None:
    """Prepare ``mmap_dir`` for per-layer mmap-backed finalization.

    Creates the directory if needed (including parents) and asserts it is an
    empty directory. No-op when ``mmap_dir is None``.

    Raises:
        NotADirectoryError: If ``mmap_dir`` exists and is not a directory.
        FileExistsError: If ``mmap_dir`` exists and is non-empty.
    """
    if mmap_dir is None:
        return
    mmap_dir_path = Path(mmap_dir)
    if mmap_dir_path.exists() and not mmap_dir_path.is_dir():
        raise NotADirectoryError(f"mmap_dir exists but is not a directory: {mmap_dir}")
    mmap_dir_path.mkdir(parents=True, exist_ok=True)
    if any(mmap_dir_path.iterdir()):
        raise FileExistsError(f"mmap_dir {mmap_dir!r} is non-empty. Pass an empty directory.")


def clear_parametrization_original(
    module: torch.nn.Module,
    param_name: str,
) -> None:
    """Replace the dense ``original`` tensor of a parametrized parameter with a
    zero-size placeholder, freeing its storage.

    ``torch.nn.utils.parametrize`` keeps the pre-parametrization tensor on the
    parametrization list as ``.original``. After finalization the dense weight is
    no longer needed — the quantized representation supersedes it — so we
    replace ``.original`` with a zero-size tensor of the same dtype and device
    to release the underlying storage.

    The original's "kind" is preserved: if ``.original`` was a ``nn.Parameter``,
    the replacement is a zero-size ``nn.Parameter``; if it was a buffer, the
    replacement is a zero-size plain tensor so the slot stays a buffer.

    Args:
        module (nn.Module): The parametrized parent module.
        param_name (str): The name of the parametrized parameter (e.g. ``"weight"``).
    """
    param_list = module.parametrizations[param_name]
    if not hasattr(param_list, "original"):
        return
    orig = param_list.original
    placeholder = torch.empty(0, dtype=orig.dtype, device=orig.device)
    if isinstance(orig, torch.nn.Parameter):
        param_list.original = torch.nn.Parameter(placeholder, requires_grad=False)
    else:
        param_list.original = placeholder
