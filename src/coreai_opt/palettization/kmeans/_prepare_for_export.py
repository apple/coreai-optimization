# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P

from coreai_opt._utils.export_utils import (
    clear_parametrization_original,
    prepare_mmap_dir,
    validate_coreml_palettization_compatibility,
)
from coreai_opt._utils.import_utils import lazy_import_coreai_torch
from coreai_opt._utils.metadata_utils import CompressionType, MILCompressionMetadata
from coreai_opt._utils.torch_utils import mmap_module_state_dict
from coreai_opt.common import ExportBackend
from coreai_opt.palettization.spec.fake_palettize import (
    _FakePalettizeImplBase,
)

_DEFAULT_VECTOR_AXIS = 0


@dataclass(frozen=True)
class LUTQuantizationInfo:
    n_bits: int
    quantized_lut: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor | None = None


@dataclass(frozen=True)
class PalettizationInfo:
    lut: torch.Tensor
    indices: torch.Tensor
    per_channel_scale: torch.Tensor | None = None
    cluster_dim: int = 1
    lut_quantization: LUTQuantizationInfo | None = None


def _expand_rank(
    tensor: torch.Tensor,
    target_rank: int,
    dim: int = -1,
) -> torch.Tensor:
    """Expand tensor rank by inserting size-1 dimensions.

    Args:
        tensor: Tensor to expand.
        target_rank: Desired rank for the tensor.
        dim: Dimension at which to insert size-1 axes. Defaults to -1 (append).
    """
    for _ in range(target_rank - tensor.dim()):
        tensor = tensor.unsqueeze(dim)
    return tensor


def _extract_palettization_params(
    palettized_param_dim: int,
    fake_palett_mod: _FakePalettizeImplBase,
) -> PalettizationInfo:
    """Extract and prepare palettization parameters from fake palettization module.

    All tensors are expanded to the appropriate rank:
    - LUT and quantized LUT: weight_rank + 2 (extra num_clusters and cluster_dim axes)
    - Per-channel scale: weight_rank
    - LUT quantization scale/zero_point: LUT rank (weight_rank + 2)
    - Indices: already at weight_rank as uint8
    """
    lut_rank = palettized_param_dim + 2

    # LUT and quantized LUT to rank K + 2
    lut = _expand_rank(fake_palett_mod.lut, lut_rank, dim=-3)
    quantized_lut = fake_palett_mod.quantized_lut
    if quantized_lut is not None:
        quantized_lut = _expand_rank(quantized_lut, lut_rank, dim=-3)

    # Per-channel scale to weight rank
    per_channel_scale = fake_palett_mod.per_channel_scale
    if per_channel_scale is not None:
        per_channel_scale = _expand_rank(per_channel_scale, palettized_param_dim)

    indices = fake_palett_mod.indices

    # LUT quantization: expand scale/zp to LUT rank, then wrap in LUTQuantizationInfo
    lut_quantization = None
    if quantized_lut is not None:
        lut_quant_scale = fake_palett_mod.lut_quantization_scale
        lut_quant_scale = _expand_rank(lut_quant_scale, lut_rank)
        lut_quant_zp = fake_palett_mod.lut_quantization_zero_point
        if lut_quant_zp is not None:
            lut_quant_zp = _expand_rank(lut_quant_zp.to(quantized_lut.dtype), lut_rank)
        lut_quantization = LUTQuantizationInfo(
            n_bits=fake_palett_mod.lut_qspec.n_bits,
            quantized_lut=quantized_lut,
            scale=lut_quant_scale,
            zero_point=lut_quant_zp,
        )

    return PalettizationInfo(
        lut=lut,
        indices=indices,
        per_channel_scale=per_channel_scale,
        cluster_dim=fake_palett_mod.cluster_dim,
        lut_quantization=lut_quantization,
    )


def _register_mil_compression_metadata(
    module: nn.Module,
    param_name: str,
    palett_info: PalettizationInfo,
) -> None:
    """
    Remove the fake palettization parametrization from the module
    and registers MIL-specific compression metadata as buffers. The metadata includes
    the lookup table (LUT), palettization scale, and other information needed for
    Core ML conversion.
    """
    # We want to leave the weights parametrized (aka palettized). This is because in MIL
    # export, we do not register indices as a metadata, and that is inferred directly
    # from the lut and weight values, which requires the weights to be palettized.
    P.remove_parametrizations(
        module,
        param_name,
        leave_parametrized=True,
    )

    # Determine compression type(s)
    lut_quant = palett_info.lut_quantization
    if lut_quant is not None:
        compression_type = [CompressionType.PALETTIZATION, CompressionType.QUANTIZATION]
    else:
        compression_type = CompressionType.PALETTIZATION

    metadata = MILCompressionMetadata(
        param_name=param_name,
        compression_type=compression_type,
        lut=palett_info.lut,
        palettization_scale=palett_info.per_channel_scale,
        quantization_n_bits=lut_quant.n_bits if lut_quant else None,
        quantization_scale=lut_quant.scale if lut_quant else None,
        zero_point=lut_quant.zero_point if lut_quant else None,
        vector_axis=_DEFAULT_VECTOR_AXIS if palett_info.cluster_dim > 1 else None,
    )
    metadata.register(module)


def _resolve_mlir_lut_and_scale(
    palett_info: PalettizationInfo,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Resolve the LUT, scale, and offset for the MLIR ScaledPalettizeModule.

    Handles four cases based on the presence of LUT quantization and per-channel
    scale (pcs):

    1. LUT quantization only: use quantized LUT with dequantization scale.
    2. LUT quantization + pcs: use quantized LUT with fused scale
       (lut_scale * pcs), expanding lut_scale to per-channel granularity first.
    3. Per-channel scale only: use float LUT with pcs as scale.

    The fourth case (neither) is handled by the caller using PalettizeModule
    directly without scale.

    Returns:
        Tuple of (lut, scale, offset) for _ScaledPalettizeModule.
    """
    has_lut_quantization = palett_info.lut_quantization is not None
    has_per_channel_scale = palett_info.per_channel_scale is not None

    if has_lut_quantization:
        lut_quant = palett_info.lut_quantization
        lut = lut_quant.quantized_lut
        # lut_quantization scale is at LUT rank (weight_rank + 2);
        # constexpr_blockwise_shift_scale needs weight rank, so drop the
        # trailing size-1 num_clusters and cluster_dim dimensions.
        scale = lut_quant.scale.squeeze(-1).squeeze(-1)
        offset = None
        if lut_quant.zero_point is not None:
            offset = lut_quant.zero_point.squeeze(-1).squeeze(-1)
        if has_per_channel_scale:
            # lut_scale may be at block granularity (e.g., shape (16,1,1,1))
            # while per_channel_scale is per output channel (e.g., (32,1,1,1)).
            # Expand lut_scale to per-channel level before fusing.
            pcs = palett_info.per_channel_scale
            repeat_factor = pcs.shape[0] // scale.shape[0]
            scale = scale.repeat_interleave(repeat_factor, dim=0)
            if offset is not None:
                offset = offset.repeat_interleave(repeat_factor, dim=0)
            scale = scale * pcs
    else:
        assert has_per_channel_scale
        lut = palett_info.lut
        scale = palett_info.per_channel_scale
        offset = None

    return lut, scale, offset


def _insert_mlir_custom_op(
    module: nn.Module,
    module_name: str,
    param_name: str,
    palett_info: PalettizationInfo,
    fake_palett_idx: int,
    mmap_dir: str | PathLike[str] | None,
) -> None:
    """
    Replace the _FakePalettizeImplBase parametrization with the appropriate
    MLIR custom op parametrization.

    Uses PalettizeParametrization for basic palettization (with or without vector
    palettization). Uses ScaledPalettizeParametrization when per-channel scaling
    and/or LUT quantization is present, following the op chaining rules:

    1. Palettization only: lut_to_dense
    2. Quantized LUT: lut_to_dense(int LUT) + constexpr_blockwise_shift_scale(lut_scale)
    3. Per-channel scale: lut_to_dense + constexpr_blockwise_shift_scale(pcs)
    4. Both: lut_to_dense(int LUT) + constexpr_blockwise_shift_scale(fused_scale)
       where fused_scale = lut_scale * per_channel_scale

    When ``mmap_dir`` is provided, the new MLIR module is serialized to a
    safetensors file under that directory and reloaded via mmap before being
    swapped in.

    The dense pre-palettization weight stored on the parametrization list is
    always replaced with a zero-size placeholder so its storage can be released.

    Raises:
        ImportError: If coreai-torch package is not installed (required for MLIR export)
    """

    # Lazy import: coreai_torch is required for MLIR export
    def _import_coreai_torch_modules():
        from coreai_torch._compression.custom_layers import (  # noqa: PLC0415
            PalettizeModule,
            ScaledPalettizeModule,
        )
        from coreai_torch._compression.utils import wrap_for_parametrization  # noqa: PLC0415

        return PalettizeModule, ScaledPalettizeModule, wrap_for_parametrization

    PalettizeModule, ScaledPalettizeModule, wrap_for_parametrization = lazy_import_coreai_torch(
        _import_coreai_torch_modules
    )

    PalettizeParametrization = wrap_for_parametrization(PalettizeModule)
    ScaledPalettizeParametrization = wrap_for_parametrization(ScaledPalettizeModule)

    needs_scale = (
        palett_info.lut_quantization is not None or palett_info.per_channel_scale is not None
    )

    vector_axis = _DEFAULT_VECTOR_AXIS if palett_info.cluster_dim > 1 else None

    if needs_scale:
        lut, scale, zero_point = _resolve_mlir_lut_and_scale(palett_info)
        mlir_palett_mod = ScaledPalettizeParametrization(
            indices=palett_info.indices,
            lut=lut,
            scale=scale,
            zero_point=zero_point,
            vector_axis=vector_axis,
        )
    else:
        mlir_palett_mod = PalettizeParametrization(
            indices=palett_info.indices,
            lut=palett_info.lut,
            vector_axis=vector_axis,
        )

    if mmap_dir is not None:
        # ``module_name`` is "" for the root module (torch's named_modules() convention);
        # drop the leading dot in that case so we don't write ".weight.safetensors".
        stem = f"{module_name}.{param_name}" if module_name else param_name
        path = Path(mmap_dir) / f"{stem}.safetensors"
        mmap_module_state_dict(mlir_palett_mod, path)

    # Replace _FakePalettizeImplBase
    module.parametrizations[param_name][fake_palett_idx] = mlir_palett_mod
    clear_parametrization_original(module, param_name)


def _find_fake_palett_parametrization(
    parametrizations: nn.ParameterList,
) -> tuple[int, _FakePalettizeImplBase | None]:
    """
    Find _FakePalettizeImplBase parametrization and its index.
    """
    for idx, p in enumerate(parametrizations):
        if isinstance(p, _FakePalettizeImplBase):
            return idx, p
    return -1, None


def _process_palettized_parameter(
    module: nn.Module,
    module_name: str,
    param_name: str,
    fake_palett_mod: _FakePalettizeImplBase,
    fake_palett_idx: int,
    backend: ExportBackend,
    mmap_dir: str | PathLike[str] | None,
) -> None:
    """
    Process a single palettized parameter for the given backend.

    Args:
        module (nn.Module): The module containing the parameter
        module_name (str): Fully-qualified module name (used to derive per-layer
            mmap file paths)
        param_name (str): Name of the parameter to process
        fake_palett_mod (_FakePalettizeImplBase): The fake palettization module
        fake_palett_idx (int): Index of the fake palettization in the parametrization
            list
        backend (ExportBackend): Target export backend
        mmap_dir (str | None): Directory for per-layer safetensors files when
            mmap-backed finalization is requested. Only consulted for the
            CoreAI backend.
    """
    palettized_param = getattr(module, param_name)
    palett_info = _extract_palettization_params(
        palettized_param_dim=palettized_param.dim(),
        fake_palett_mod=fake_palett_mod,
    )

    if backend == ExportBackend.CoreML:
        _register_mil_compression_metadata(module, param_name, palett_info)
    elif backend == ExportBackend.CoreAI:
        _insert_mlir_custom_op(
            module, module_name, param_name, palett_info, fake_palett_idx, mmap_dir
        )


def _process_weight_palettization(
    model: nn.Module,
    backend: ExportBackend,
    mmap_dir: str | PathLike[str] | None = None,
) -> None:
    """
    This function iterates through all parametrized modules in the model, finds those
    with _FakePalettizeImplBase parametrizations, extracts their palettization
    parameters, and replaces them with backend specific information:
        MIL -> Inserts compression metadata as buffers
        MLIR -> Inserts MLIR custom ops as parametrization

    Args:
        model (nn.Module): The PyTorch model containing parametrized modules with fake
            palettization
        backend (ExportBackend): Target export backend
        mmap_dir (str | None): If provided, serialize each finalized MLIR custom-op
            parametrization to a safetensors file under this directory and reload
            it via mmap, so large-model finalization does not hold full
            palettized weights in RAM. Only honored for the CoreAI backend.
    """
    prepare_mmap_dir(mmap_dir)

    for module_name, module in model.named_modules():
        if not P.is_parametrized(module):
            continue

        for param_name, parametrizations in list(module.parametrizations.items()):
            fake_palett_idx, fake_palett_mod = _find_fake_palett_parametrization(parametrizations)

            # Skip if no FakePalett parametrization found
            if fake_palett_idx == -1 or fake_palett_mod is None:
                continue

            _process_palettized_parameter(
                module,
                module_name,
                param_name,
                fake_palett_mod,
                fake_palett_idx,
                backend,
                mmap_dir,
            )


def prepare_for_mlir_export(
    model: nn.Module, mmap_dir: str | PathLike[str] | None = None
) -> nn.Module:
    """
    Prepare a palettized PyTorch model for Core AI export by replacing fake palettization
    parametrizations and modules with corresponding palettization custom ops.

    Args:
        model (nn.Module): The PyTorch model to be prepared for export
        mmap_dir (str | None): If provided, serialize finalized palettized weights
            to safetensors files under this directory and re-load them via mmap,
            so large-model finalization does not hold full palettized weights in RAM.
            The files in ``mmap_dir`` must remain in place for the lifetime of
            the returned model; removing them invalidates the mmap-backed weights.

    Returns:
        nn.Module: The input model that has now been modified for Core AI export

    Note:
        This function frees the dense pre-palettization weights in place: on each
        parametrized weight, ``parametrizations[...].original`` is replaced with
        a zero-size placeholder so its storage can be released.
    """
    _process_weight_palettization(model, backend=ExportBackend.CoreAI, mmap_dir=mmap_dir)

    return model


def prepare_for_mil_export(model: nn.Module) -> nn.Module:
    """
    Prepare a palettized PyTorch model for CoreML export by removing fake palettization
    parametrizations and registering compression metadata.

    Args:
        model (nn.Module): The PyTorch model to be prepared for export

    Returns:
        nn.Module: The input model that has now been modified for CoreML export
    """
    for module_name, module in model.named_modules():
        if not P.is_parametrized(module):
            continue
        for param_name, parametrizations in module.parametrizations.items():
            _, fake_palett_mod = _find_fake_palett_parametrization(parametrizations)
            if fake_palett_mod is None:
                continue

            context = f"parameter '{param_name}' of module '{module_name}'"
            validate_coreml_palettization_compatibility(
                fake_palett_mod.cluster_dim,
                fake_palett_mod.lut_qspec,
                fake_palett_mod.enable_per_channel_scale,
                context,
            )

    _process_weight_palettization(model, backend=ExportBackend.CoreML)

    # Register metadata version
    MILCompressionMetadata.register_version(model)

    return model
