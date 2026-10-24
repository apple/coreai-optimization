# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Module-tree walking and storage-cost helpers for ``bits_per_weight``."""

import math
from collections.abc import Iterator

import torch
from torch.nn.utils.parametrize import ParametrizationList as _ParametrizationList

from coreai_opt.config.spec import (
    CompressionSimulatorBase as _CompressionSimulatorBase,
    CompressionTargetTensor as _CompressionTargetTensor,
)
from coreai_opt.palettization.spec.fake_palettize import _FakePalettizeImplBase
from coreai_opt.pruning.spec import PruneImplBase as _PruneImplBase
from coreai_opt.quantization.spec import QuantizationScheme as _QuantizationScheme
from coreai_opt.quantization.spec.fake_quantize import FakeQuantizeImplBase as _FakeQuantizeImplBase
from coreai_opt.quantization.spec.qformulation import (
    QuantizationFormulation as _QuantizationFormulation,
)

_BITS_PER_BYTE = 8
_DEFAULT_SCALE_BITS = 32

# Weight and scale dtypes whose storage cost this module knows how to compute.
_SUPPORTED_QUANT_DTYPES = frozenset(
    {
        torch.int8,
        torch.uint8,
        torch.int4,
        torch.uint4,
        torch.int2,
        torch.uint2,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float4_e2m1fn_x2,
    }
)
_SUPPORTED_SCALE_DTYPES = frozenset({None, torch.float8_e8m0fnu})
_SUPPORTED_QSCHEMES = frozenset(
    {
        _QuantizationScheme.SYMMETRIC,
        _QuantizationScheme.ASYMMETRIC,
        _QuantizationScheme.SYMMETRIC_WITH_CLIPPING,
    }
)
_SUPPORTED_QFORMULATIONS = frozenset({_QuantizationFormulation.MINVAL, _QuantizationFormulation.ZP})

WeightCompressor = _FakeQuantizeImplBase | _FakePalettizeImplBase


def named_modules_excluding_compression_machinery(
    module: torch.nn.Module, name: str = ""
) -> Iterator[tuple[str, torch.nn.Module]]:
    """Yield ``(fqn, module)`` for every module that owns logical tensors.

    Compression machinery owns no weights of its own, so a machinery
    module and everything below it is left out by not descending into it.

    Args:
        module (torch.nn.Module): The subtree root to walk.
        name (str): Fully qualified name of ``module`` from the model root.

    Yields:
        tuple[str, torch.nn.Module]: Name and module, parents before children.
    """
    if isinstance(module, (_CompressionSimulatorBase, _ParametrizationList)):
        return

    yield name, module

    for child_name, child in module.named_children():
        yield from named_modules_excluding_compression_machinery(
            child, f"{name}.{child_name}" if name else child_name
        )


def get_weight_compressor(param_list: _ParametrizationList) -> WeightCompressor | None:
    """Return the weight-targeting compressor in a parametrization list, if any.

    Args:
        param_list (ParametrizationList): Parametrizations registered on a parameter.

    Returns:
        WeightCompressor | None: The first ``_FakePalettizeImplBase`` or
        weight-target ``FakeQuantizeImplBase`` in the list, or ``None`` if the
        list contains no recognized weight compressor.
    """
    for entry in param_list:
        if isinstance(entry, _FakePalettizeImplBase):
            return entry
        if (
            isinstance(entry, _FakeQuantizeImplBase)
            and entry.quantization_target == _CompressionTargetTensor.WEIGHT
        ):
            return entry
    return None


def ensure_single_original(
    param_list: _ParametrizationList, module_name: str, tensor_name: str
) -> None:
    """Raise if a parametrization stores its dense tensor as multiple originals.

    A ``right_inverse`` makes PyTorch register ``original0``,
    ``original1``, ... instead of a single ``original`` (as
    ``torch.nn.utils.parametrizations.weight_norm`` does for instance), so there is no one
    dense tensor whose storage cost we can attribute.
    """
    if not param_list.is_tensor:
        raise NotImplementedError(
            f"bits_per_weight cannot size the parametrization on "
            f"'{module_name}.{tensor_name}': it stores multiple original tensors "
            f"rather than a single dense one."
        )


def ensure_not_pruned(param_list: _ParametrizationList, module_name: str, tensor_name: str) -> None:
    """Raise if a weight carries a pruning parametrization."""
    for entry in param_list:
        if isinstance(entry, _PruneImplBase):
            raise NotImplementedError(
                f"bits_per_weight cannot compute the storage cost of a pruned "
                f"weight '{module_name}.{tensor_name}'."
            )


def ensure_supported_quantization(
    compressor: WeightCompressor | None, module_name: str, tensor_name: str
) -> None:
    """Raise if a compressor's quantization settings have no known storage cost.

    Args:
        compressor (WeightCompressor | None): The weight compressor, if any.
        module_name (str): Dotted name of the owning module, for the error message.
        tensor_name (str): Name of the parametrized tensor, for the error message.
    """
    if isinstance(compressor, _FakePalettizeImplBase):
        if compressor.lut_qspec is not None:
            _ensure_supported_qscheme(compressor.lut_qspec.qscheme, module_name, tensor_name)
        return

    if not isinstance(compressor, _FakeQuantizeImplBase):
        return

    if compressor.dtype not in _SUPPORTED_QUANT_DTYPES:
        raise NotImplementedError(
            f"bits_per_weight cannot compute the storage cost of quantization dtype "
            f"{compressor.dtype} on '{module_name}.{tensor_name}'. Supported dtypes: "
            f"{sorted(str(dtype) for dtype in _SUPPORTED_QUANT_DTYPES)}."
        )

    scale_dtype = compressor.qparams_calculator.scale_dtype
    if scale_dtype not in _SUPPORTED_SCALE_DTYPES:
        raise NotImplementedError(
            f"bits_per_weight cannot compute the storage cost of scale dtype "
            f"{scale_dtype} on '{module_name}.{tensor_name}'."
        )

    _ensure_supported_qscheme(compressor.qscheme, module_name, tensor_name)

    if compressor.qformulation not in _SUPPORTED_QFORMULATIONS:
        raise NotImplementedError(
            f"bits_per_weight cannot compute the storage cost of quantization formulation "
            f"{compressor.qformulation} on '{module_name}.{tensor_name}'."
        )


def _ensure_supported_qscheme(
    qscheme: _QuantizationScheme, module_name: str, tensor_name: str
) -> None:
    if qscheme not in _SUPPORTED_QSCHEMES:
        raise NotImplementedError(
            f"bits_per_weight cannot compute the storage cost of quantization scheme "
            f"{qscheme} on '{module_name}.{tensor_name}'."
        )


def tensor_storage_bits(weight: torch.Tensor, compressor: WeightCompressor | None) -> int:
    """Return the storage cost in bits of a (possibly compressed) weight tensor."""
    if compressor is None:
        return full_precision_bits(weight)
    if isinstance(compressor, _FakePalettizeImplBase):
        return _palettized_bits(weight, compressor)
    return _quantized_tensor_bits(weight.shape, compressor)


def full_precision_bits(tensor: torch.Tensor) -> int:
    """Return the dense storage cost of a tensor in bits."""
    return int(tensor.numel() * tensor.element_size() * _BITS_PER_BYTE)


def _quantized_tensor_bits(
    shape: torch.Size, fq: _FakeQuantizeImplBase, qparams_nonnegative: bool = False
) -> int:
    """Return the storage cost of a tensor quantized by ``fq``, including its qparams.

    Qparams quantized per ``qparam_qspec`` cost their own quantized storage, recursively.

    Args:
        shape (torch.Size): Shape of the quantized tensor.
        fq (FakeQuantizeImplBase): The fake quantizer of the tensor.
        qparams_nonnegative (bool): Whether the tensor holds non-negative qparams
            quantized per ``qparam_qspec``, for :func:`_offset_bits`.

    Returns:
        int: ``payload_bits + scale_bits + offset_bits`` in bits.

    Note:
        The per-block qparam grid shape is read directly from the materialized
        ``qparams_calculator.scale`` buffer. When that buffer is not yet materialized,
        it is derived analytically from ``granularity.get_block_size``.
    """
    scale = fq.qparams_calculator.scale
    if scale is not None and scale.numel() > 0:
        qparam_shape = scale.shape
    else:
        block_size = fq.granularity.get_block_size(shape, fq.quantization_target)
        qparam_shape = torch.Size(
            dim // block for dim, block in zip(shape, block_size, strict=True)
        )

    payload_bits = math.prod(shape) * fq.n_bits
    scale_fq = fq.qparams_calculator.scale_fake_quantize
    if scale_fq is not None:
        scale_bits = _quantized_tensor_bits(qparam_shape, scale_fq, qparams_nonnegative=True)
    else:
        scale_bits = math.prod(qparam_shape) * _scale_storage_bits(fq)

    return int(payload_bits + scale_bits + _offset_bits(fq, qparam_shape, qparams_nonnegative))


def _scale_storage_bits(fq: _FakeQuantizeImplBase) -> int:
    """Return the per-element bit width of the scale as it is stored."""
    # ``scale_dtype`` takes precedence when set
    dtype = fq.qparams_calculator.scale_dtype or fq.qparams_calculator._compute_dtype_for_export
    return int(dtype.itemsize * _BITS_PER_BYTE)


def _float_qparam_bits(fq: _FakeQuantizeImplBase) -> int:
    """Return the per-element bit width of the float qparams this weight stores."""
    dtype = fq.qparams_calculator._compute_dtype_for_export
    return int(dtype.itemsize * _BITS_PER_BYTE)


def _offset_bits(
    fq: _FakeQuantizeImplBase, qparam_shape: torch.Size, qparams_nonnegative: bool
) -> int:
    """Return the per-block dequantization offset cost of a quantized tensor, in bits.

    - ``ZP``: an asymmetric tensor stores a per-block ``zero_point``, packed at ``n_bits``.
    - ``MINVAL``: the formulation stores a per-block ``minval`` float instead of a
      zero-point, so it costs a full float per block, unless ``qparam_qspec`` quantizes
      it.

    Qparams quantized per ``qparam_qspec`` are non-negative, so an asymmetric nested spec
    whose range minimum is not pinned below 0 by ``float_range`` always gets the offset
    ``quant_min`` (ZP) or 0 (MINVAL), which needs no storage.
    """
    if (
        qparams_nonnegative
        and fq.qscheme == _QuantizationScheme.ASYMMETRIC
        and fq.qparams_calculator.float_range[0] in (None, 0)
    ):
        return 0
    num_blocks = int(math.prod(qparam_shape))
    if fq.qformulation == _QuantizationFormulation.MINVAL:
        minval_fq = fq.qparams_calculator.minval_fake_quantize
        if minval_fq is not None:
            return _quantized_tensor_bits(qparam_shape, minval_fq, qparams_nonnegative=True)
        return num_blocks * _float_qparam_bits(fq)
    if fq.qscheme == _QuantizationScheme.ASYMMETRIC:
        return num_blocks * fq.n_bits
    return 0


def _palettized_bits(weight: torch.Tensor, palettization_fq: _FakePalettizeImplBase) -> int:
    """Return the storage cost of a palettized weight including LUT / per-channel-scale overhead.

    Args:
        weight (torch.Tensor): The dense original weight tensor.
        palettization_fq (_FakePalettizeImplBase): The weight fake-palettize parametrization.

    Returns:
        int: Effective storage cost of a palettized weight tensor
    """
    num_elements = weight.numel()

    # Indices: one n_bits index per cluster_dim-sized group along axis 0.
    indices_bits = (num_elements // palettization_fq.cluster_dim) * palettization_fq.n_bits

    # LUT (centroids): shape after _reshape_lut_tensor is
    # (num_blocks_axis0, num_blocks_axis1, 2**n_bits, cluster_dim).
    lut = palettization_fq.lut
    if lut is not None and lut.numel() > 0:
        lut_elements = lut.numel()
    else:
        lut_elements = (
            palettization_fq.granularity.num_blocks_to_cluster(weight)
            * (2**palettization_fq.n_bits)
            * palettization_fq.cluster_dim
        )

    if palettization_fq.lut_qspec is not None:
        lut_dtype_bits = palettization_fq.lut_qspec.n_bits
    else:
        lut_dtype_bits = _buffer_dtype_bits(lut, weight.element_size() * _BITS_PER_BYTE)
    lut_bits = lut_elements * lut_dtype_bits

    # Per-channel scale: one weight-dtype value per output channel
    # (weight.shape[0])
    num_channels = weight.shape[0]
    per_channel_scale_bits = 0
    if palettization_fq.enable_per_channel_scale:
        per_channel_scale = palettization_fq.per_channel_scale
        if per_channel_scale is not None and per_channel_scale.numel() > 0:
            num_channels = per_channel_scale.numel()
            per_channel_scale_bits = (
                num_channels * per_channel_scale.element_size() * _BITS_PER_BYTE
            )
        else:
            per_channel_scale_bits = num_channels * weight.element_size() * _BITS_PER_BYTE

    return int(
        indices_bits + lut_bits + per_channel_scale_bits + _lut_quant_bits(weight, palettization_fq)
    )


def _lut_quant_bits(weight: torch.Tensor, palettization_fq: _FakePalettizeImplBase) -> int:
    """Return the qparams cost of a quantized LUT, in bits.

    Dequantizing a quantized LUT needs one scale (and, when asymmetric, one zero-point)
    per palettization block: the LUT fake-quantizer overrides ``lut_qspec``'s per-tensor
    granularity to per-channel over the stacked LUT (in
    ``palettization/kmeans/kmeans_fake_palettize.py``).
    """
    if palettization_fq.lut_qspec is None:
        return 0

    lut_scale = palettization_fq.lut_quantization_scale
    if lut_scale is not None and lut_scale.numel() > 0:
        num_lut_blocks = lut_scale.numel()
    else:
        num_lut_blocks = palettization_fq.granularity.num_blocks_to_cluster(weight)

    bits = num_lut_blocks * _buffer_dtype_bits(lut_scale, _DEFAULT_SCALE_BITS)
    if palettization_fq.lut_qspec.qscheme == _QuantizationScheme.ASYMMETRIC:
        bits += num_lut_blocks * palettization_fq.lut_qspec.n_bits
    return int(bits)


def _buffer_dtype_bits(buffer: torch.Tensor | None, default: int) -> int:
    """Return the per-element bit width of a buffer, or a default if unmaterialized.

    Args:
        buffer (torch.Tensor | None): A scale, zero-point, or LUT buffer.
        default (int): Bit width to assume when the buffer is missing or empty.

    Returns:
        int: ``buffer.element_size() * _BITS_PER_BYTE`` if the buffer holds data,
        else ``default``.
    """
    if buffer is not None and buffer.numel() > 0:
        return int(buffer.element_size() * _BITS_PER_BYTE)
    return default
