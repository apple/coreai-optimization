# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import logging
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

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
from coreai_opt._utils.torch_utils import is_float8_dtype, mmap_module_state_dict
from coreai_opt.common import ExportBackend
from coreai_opt.palettization.spec.fake_palettize import (
    _FakePalettizeImplBase,
)
from coreai_opt.palettization.spec.granularity import PerTensorGranularity

_DEFAULT_VECTOR_AXIS = 0

logger = logging.getLogger(__name__)


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


class _SparsePalettizeModule(nn.Module):
    """Parametrization module for a weight that is both sparse and palettized.

    Given a sparse + palettized module, we insert the corresponding ``lut_to_dense`` op
    followed by the ``sparse_to_dense`` op, and optionally followed by
    ``blockwise_shift_scale`` if per-channel scales or LUT quantization is enabled.

    With per-group granularity for palettization, each group should have the same number of
    non zero values (this can be achieved by padding if needed).

    With vector palettization (``cluster_dim = d > 1``), the ``nnz`` stored values of each
    group (the whole tensor for per-tensor) are clustered in vectors of ``d`` consecutive
    values, so there are ``nnz // d`` indices: ``nonzero_indices`` is ``[nnz // d]`` and
    ``lut`` is ``[num_groups, 2**n_bits, d]``. ``lut_to_dense`` expands every index into ``d``
    values, giving ``[nnz]``, and ``sparse_to_dense`` scatters them as above. If a group's
    nonzero count is not a multiple of ``d``, its smallest remaining nonzeros are zeroed out
    before clustering. Groups that hold fewer vectors than the fullest group also store some of
    their zeros, which can share a vector with a nonzero and be reconstructed as small nonzeros.

    For a weight of shape ``[C_out, C_in, ...]``, ``nnz`` nonzero values and ``n_bits`` bits:

    - Per-tensor: ``nonzero_indices`` is ``[nnz]`` and ``lut`` is ``[1, 2**n_bits, 1]``.
      ``lut_to_dense`` gives ``[nnz]`` values, and ``sparse_to_dense`` scatters them into
      ``mask`` (``[C_out, C_in, ...]``), giving the dense weight ``[C_out, C_in, ...]``.
    - Per-group along axis 0, with ``G = C_out / group_size`` groups: ``lut`` is
      ``[G, 2**n_bits, 1]`` and ``nonzero_indices`` is ``[nnz]`` with
      ``nnz = G * nnz_per_group``. ``lut_to_dense`` splits it into ``G`` blocks of
      ``nnz_per_group`` entries and uses ``lut[g]`` for block ``g``.
    """

    def __init__(
        self,
        nonzero_indices: torch.Tensor,
        lut: torch.Tensor,
        mask: torch.Tensor,
        scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.register_buffer("nonzero_indices", nonzero_indices)
        # nonzero_indices is rank 1 (flattened by masking), so lut must be
        # reshaped to rank 3 (lut_to_dense requires lut.rank == indices.rank + 2).
        self.register_buffer("lut", lut.reshape(-1, lut.shape[-2], lut.shape[-1]))
        self.register_buffer("mask", mask)
        self.register_buffer("scale", scale)
        # The nonzero values are a rank-1 list, so their vectors always run along axis 0.
        self.vector_axis = 0

    def forward(self, _: Any) -> torch.Tensor:
        nonzero_values = torch.ops.coreai.lut_to_dense(
            self.nonzero_indices, self.lut, self.vector_axis
        )
        dense = torch.ops.coreai.sparse_to_dense(nonzero_values, self.mask)
        if self.scale is None:
            return dense
        return torch.ops.coreai.constexpr_blockwise_shift_scale(dense, scale=self.scale)


def _build_sparse_palettize_module(
    palett_info: PalettizationInfo, mask: torch.Tensor
) -> _SparsePalettizeModule:
    """Build the sparse + palettized module of a weight from its palettization info.

    Args:
        palett_info (PalettizationInfo): The palettization info holding the indices and LUT.
        mask (torch.Tensor): Boolean nonzero mask with the weight's shape.

    Returns:
        _SparsePalettizeModule: The module that rebuilds the dense weight.

    Raises:
        ValueError: If the LUT quantization has a nonzero zero point, so a zero would not
            dequantize to 0.
    """
    lut_quant = palett_info.lut_quantization
    lut = palett_info.lut if lut_quant is None else lut_quant.quantized_lut
    scale = None
    if lut_quant is not None or palett_info.per_channel_scale is not None:
        _, scale, zero_point = _resolve_mlir_lut_and_scale(palett_info)
        if zero_point is not None and zero_point.any():
            raise ValueError(
                "Joint sparsity requires LUT quantization with a zero point of 0 (use a "
                "symmetric lut_qspec), since masked-out weights must dequantize to exactly 0."
            )

    # Vector palettization already stores one index per vector of nonzero values.
    indices = palett_info.indices
    nonzero_indices = indices if palett_info.cluster_dim > 1 else indices[mask]
    return _SparsePalettizeModule(nonzero_indices, lut, mask, scale)


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
    fake_palett_mod: _FakePalettizeImplBase,
) -> None:
    """
    Remove the fake palettization parametrization from the module
    and registers MIL-specific compression metadata as buffers. The metadata includes
    the lookup table (LUT), palettization scale, and other information needed for
    Core ML conversion. For 8-bit scalar palettization with an int8 or uint8 LUT, the
    weight is registered as plain 8-bit weight quantization instead, without a LUT.
    """
    has_sparsity = fake_palett_mod._sparsity_mask is not None
    if has_sparsity:
        # The CoreML path does not go through _resolve_export_sparsity_mask, so validate here.
        _validate_coreml_sparsity_support(fake_palett_mod)

    # We want to leave the weights parametrized (aka palettized). This is because in MIL
    # export, we do not register indices as a metadata, and that is inferred directly
    # from the lut and weight values, which requires the weights to be palettized.
    P.remove_parametrizations(
        module,
        param_name,
        leave_parametrized=True,
    )

    # PRUNING must be listed first for coremltools to chain the sparse LUT op.
    lut_quant = palett_info.lut_quantization
    if _is_8bit_scalar_quantized_lut_palettization(palett_info):
        # The gathered 8-bit LUT values already form a dense 8-bit weight, so export it as
        # weight quantization only (one op less, no LUT). Each element is dequantized with the
        # same ``(q - zero_point) * scale``, so the decompressed weight is bit-identical.
        # coremltools recovers the 8-bit data from the (palettized) float weight.
        _, scale, zero_point = _resolve_mlir_lut_and_scale(palett_info)
        MILCompressionMetadata(
            param_name=param_name,
            compression_type=CompressionType.QUANTIZATION,
            quantization_n_bits=lut_quant.n_bits,
            quantization_scale=scale,
            zero_point=zero_point,
        ).register(module)
        return

    if lut_quant is not None:
        compression_type = [CompressionType.PALETTIZATION, CompressionType.QUANTIZATION]
    else:
        compression_type = [CompressionType.PALETTIZATION]
    if has_sparsity:
        compression_type = [CompressionType.PRUNING, *compression_type]

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


def _validate_coreml_sparsity_support(fake_palett_mod: _FakePalettizeImplBase) -> None:
    """Reject CoreML joint sparsity with vector palettization or LUT quantization.

    ``constexpr_lut_to_sparse`` takes its mask on the grid of vectors, so a vector is kept or
    dropped as a whole; it cannot express nonzero values grouped into vectors.

    Raises:
        ValueError: If ``cluster_dim != 1`` or ``lut_qspec`` is set.
    """
    if fake_palett_mod.cluster_dim != 1:
        raise ValueError(
            f"cluster_dim={fake_palett_mod.cluster_dim} (vector palettization) "
            "not supported for joint sparsity on CoreML."
        )
    if fake_palett_mod.lut_qspec is not None:
        raise ValueError("lut_qspec not supported for joint sparsity on CoreML.")


def _validate_coreai_sparsity_support(
    fake_palett_mod: _FakePalettizeImplBase, context: str
) -> None:
    """Reject joint sparsity configurations the CoreAI sparse op chain cannot represent.

    The chain maps contiguous axis-0 blocks to LUTs. Vector palettization does not support LUT
    quantization.

    Args:
        fake_palett_mod (_FakePalettizeImplBase): The fake palettization module.
        context (str): Prefix identifying the parameter, prepended to error messages.

    Raises:
        ValueError: If the granularity is neither per-tensor nor axis-0 grouped, or
            ``cluster_dim != 1`` (vector palettization) is combined with ``lut_qspec``.
    """
    granularity = fake_palett_mod.granularity
    is_per_tensor = isinstance(granularity, PerTensorGranularity)
    if not (is_per_tensor or fake_palett_mod._axis0_group_size() is not None):
        raise ValueError(f"{context}granularity={granularity} not supported for joint sparsity.")
    if fake_palett_mod.cluster_dim != 1 and fake_palett_mod.lut_qspec is not None:
        raise ValueError(
            f"{context}lut_qspec not supported for joint sparsity with vector palettization."
        )


def _pad_mask_to_equal_groups(
    mask: torch.Tensor,
    indices: torch.Tensor,
    lut: torch.Tensor,
    group_size: int,
    context: str,
) -> torch.Tensor:
    """Pad the mask so every axis-0 channel group stores as many elements as the fullest one.

    Groups with fewer nonzeros also store some of their zeros. The stored zeros must look up
    exactly 0.0 in the LUT, so the weight is unchanged. K-means reserves a zero LUT entry for
    the groups that get padded, while a palette reused from an already-clustered block holds 0.0
    at another position, so the padded positions' own indices are what gets checked.

    Args:
        mask (torch.Tensor): Boolean nonzero mask of the weight.
        indices (torch.Tensor): Per-element LUT indices, same shape as ``mask``.
        lut (torch.Tensor): LUTs of shape ``(num_groups, num_clusters, 1)``.
        group_size (int): Number of channels per group.
        context (str): Prefix identifying the parameter, prepended to error messages.

    Returns:
        torch.Tensor: The padded boolean mask.

    Raises:
        ValueError: If a group has too few zeros to pad, or a padded zero does not look up 0.0.
    """
    groups = mask.split(group_size, dim=0)
    target = max(int(group.sum()) for group in groups)
    padded = []
    for group_idx, group in enumerate(groups):
        missing = target - int(group.sum())
        flat = group.flatten().clone()
        zero_positions = (~flat).nonzero().flatten()[:missing]
        if zero_positions.numel() < missing:
            raise ValueError(
                f"{context}group {group_idx} of {group.numel()} elements cannot hold "
                f"{target} nonzeros."
            )
        flat[zero_positions] = True
        padded.append(flat.view_as(group))
    padded_mask = torch.cat(padded, dim=0)

    nonzero_indices = indices[padded_mask].long()
    # Group index of each stored element.
    block = torch.arange(nonzero_indices.numel()) // (nonzero_indices.numel() // lut.shape[0])
    stored_values = lut[block, nonzero_indices, 0]
    padded_only = (~mask)[padded_mask]
    if (stored_values[padded_only] != 0).any():
        raise ValueError(
            f"{context}padded zeros must look up exactly 0.0 in the LUT, but some look up a "
            "nonzero value, so the LUT or indices changed since prepare(). Do not modify the "
            "centroids or indices between prepare() and finalize(), or use an even sparsity "
            "pattern across channel groups."
        )
    return padded_mask


def _resolve_export_sparsity_mask(
    fake_palett_mod: _FakePalettizeImplBase,
    palett_info: PalettizationInfo,
    module_name: str,
    param_name: str,
) -> torch.Tensor | None:
    """Return the CPU boolean mask of the elements to store jointly, or None to export densely.

    The weight is exported densely when no sparsity was detected or the weight is entirely
    zero (its palette is all zeros, so this is exact). Grouped LUTs need the same number of
    stored elements in every axis-0 channel group, so the groups with fewer nonzeros also
    store some of their zeros (see ``_pad_mask_to_equal_groups``).

    Args:
        fake_palett_mod (_FakePalettizeImplBase): The fake palettization module.
        palett_info (PalettizationInfo): The palettization info holding the indices and LUT.
        module_name (str): Fully-qualified module name, used in warnings and errors.
        param_name (str): Name of the palettized parameter, used in warnings and errors.

    Returns:
        torch.Tensor | None: Boolean mask on CPU, or ``None`` for dense export.

    Raises:
        ValueError: If sparsity is combined with an unsupported configuration, or a group
            cannot be padded to the fullest group's size.
    """
    if fake_palett_mod._sparsity_mask is None:
        return None
    context = f"'{module_name}.{param_name}': "
    _validate_coreai_sparsity_support(fake_palett_mod, context)

    # indices live on CPU, so the mask used to index them must too.
    mask = fake_palett_mod._sparsity_mask.cpu().to(torch.bool)
    if not mask.any():
        logger.warning(
            "'%s.%s': all weights are zero; exporting without joint sparsity.",
            module_name,
            param_name,
        )
        return None
    if fake_palett_mod._has_uneven_groups():
        lut = palett_info.lut.reshape(-1, palett_info.lut.shape[-2], palett_info.lut.shape[-1])
        padded_mask = _pad_mask_to_equal_groups(
            mask,
            palett_info.indices,
            lut,
            fake_palett_mod._axis0_group_size(),
            context,
        )
        logger.info(
            "'%s.%s': channel groups hold different numbers of nonzeros; storing %d extra zeros "
            "so every group holds %d.",
            module_name,
            param_name,
            int(padded_mask.sum() - mask.sum()),
            int(padded_mask.sum()) // lut.shape[0],
        )
        return padded_mask
    return mask


def _is_8bit_scalar_quantized_lut_palettization(palett_info: PalettizationInfo) -> bool:
    """Return whether the weight is 8-bit scalar palettized with an int8, uint8 or float8 LUT."""
    lut_quant = palett_info.lut_quantization
    return (
        lut_quant is not None
        and lut_quant.n_bits == 8
        and (
            lut_quant.quantized_lut.dtype in (torch.int8, torch.uint8)
            or is_float8_dtype(lut_quant.quantized_lut.dtype)
        )
        # Scalar palettization: 256 LUT entries, each of size 1 (no vector palettization)
        and lut_quant.quantized_lut.shape[-2:] == (2**8, 1)
    )


def _decompress_scalar_palettization(indices: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Return the dense weight ``lut[block, indices]`` without per-element LUT copies.

    Args:
        indices (torch.Tensor): Palette indices with the weight's shape.
        lut (torch.Tensor): Scalar LUT of shape ``(*num_blocks, num_palettes, 1)``.

    Returns:
        torch.Tensor: Dense weight with the shape of ``indices`` and the dtype of ``lut``.
    """
    num_blocks = lut.shape[:-2]
    # Split each weight axis into (block, position in block).
    blocked_indices = indices.to(device=lut.device, dtype=torch.long).reshape(
        [s for n, d in zip(num_blocks, indices.shape, strict=True) for s in (n, d // n)]
    )
    # gather has no float8 kernel; a uint8 view keeps the 1-byte values bit for bit.
    # The size-1 dims broadcast each block's LUT over its elements without copying.
    blocked_lut = lut.view(torch.uint8).reshape(
        [s for n in num_blocks for s in (n, 1)] + [lut.shape[-2]]
    )
    dense = torch.take_along_dim(blocked_lut, blocked_indices.unsqueeze(-1), dim=-1)
    return dense.reshape(indices.shape).view(lut.dtype)


def _insert_mlir_custom_op(
    module: nn.Module,
    module_name: str,
    param_name: str,
    palett_info: PalettizationInfo,
    fake_palett_mod: _FakePalettizeImplBase,
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
       (for 8-bit scalar palettization with an int8, uint8 or float8 LUT, the gathered values
       already form a dense 8-bit weight, so it is exported as an 8-bit weight quantization
       instead: constexpr_blockwise_shift_scale(8-bit weight, scale), with no LUT, where scale
       is lut_scale, or lut_scale * per_channel_scale when both are present. Vector
       palettization keeps the LUT: a dense weight would need 8 bits per weight instead of
       8 / cluster_dim)
    3. Per-channel scale: lut_to_dense + constexpr_blockwise_shift_scale(pcs)
    4. Both: lut_to_dense(int LUT) + constexpr_blockwise_shift_scale(fused_scale)
       where fused_scale = lut_scale * per_channel_scale
    5. Detected sparsity: lut_to_dense on the nonzero-only indices + sparse_to_dense
       (+ constexpr_blockwise_shift_scale if a scale is present), as a _SparsePalettizeModule
       instead of a plain Palettize/ScaledPalettize parametrization. Weights that cannot
       be exported jointly (see ``_resolve_export_sparsity_mask``) use cases 1-4.

    When ``mmap_dir`` is provided, the new MLIR module is serialized to a
    safetensors file under that directory and reloaded via mmap before being
    swapped in.

    The dense pre-palettization weight stored on the parametrization list is
    always replaced with a zero-size placeholder so its storage can be released.

    Raises:
        ImportError: If coreai-torch package is not installed (required for MLIR export)
        ValueError: If detected sparsity is combined with an unsupported configuration
    """

    # Lazy import: coreai_torch is required for MLIR export
    def _import_coreai_torch_modules():
        from coreai_torch._compression.custom_layers import (  # noqa: PLC0415
            PalettizeModule,
            ScaledPalettizeModule,
            WeightDequantizeModule,
        )
        from coreai_torch._compression.utils import wrap_for_parametrization  # noqa: PLC0415

        return (
            PalettizeModule,
            ScaledPalettizeModule,
            WeightDequantizeModule,
            wrap_for_parametrization,
        )

    (
        PalettizeModule,
        ScaledPalettizeModule,
        WeightDequantizeModule,
        wrap_for_parametrization,
    ) = lazy_import_coreai_torch(_import_coreai_torch_modules)

    PalettizeParametrization = wrap_for_parametrization(PalettizeModule)
    ScaledPalettizeParametrization = wrap_for_parametrization(ScaledPalettizeModule)
    WeightDequantizeParametrization = wrap_for_parametrization(WeightDequantizeModule)

    needs_scale = (
        palett_info.lut_quantization is not None or palett_info.per_channel_scale is not None
    )

    # Reuses the mask detected in prepare()'s forward pass.
    mask = _resolve_export_sparsity_mask(fake_palett_mod, palett_info, module_name, param_name)
    vector_axis = _DEFAULT_VECTOR_AXIS if palett_info.cluster_dim > 1 else None

    if mask is not None:
        mlir_palett_mod = _build_sparse_palettize_module(palett_info, mask)
    elif _is_8bit_scalar_quantized_lut_palettization(palett_info):
        lut, scale, zero_point = _resolve_mlir_lut_and_scale(palett_info)
        dense_8bit_weight = _decompress_scalar_palettization(palett_info.indices, lut)
        mlir_palett_mod = WeightDequantizeParametrization(
            quantized_data=dense_8bit_weight,
            scale=scale,
            zero_point=zero_point,
        )
    elif needs_scale:
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
        _register_mil_compression_metadata(module, param_name, palett_info, fake_palett_mod)
    elif backend == ExportBackend.CoreAI:
        _insert_mlir_custom_op(
            module, module_name, param_name, palett_info, fake_palett_mod, fake_palett_idx, mmap_dir
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
