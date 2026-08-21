# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Compute the average bits-per-weight (bpw) of a prepared ``coreai-opt`` model.

Estimates the average bit width of the tensors a model carries, amortizing
compression overhead (quantization scales / zero-points, palettization LUTs and
per-channel scales). Compressed
tensors count at their effective compressed cost, everything else (biases,
norms, untargeted weights, buffers such as BatchNorm running stats) counts at
its full-precision dtype cost.

This is an analytical estimate, not a bit-exact proxy of the model asset.
It answers "what does this compression config cost, in principle?" from
a prepared model.

Only eager-mode prepared models are supported currently.

Supported model shapes and compression:

- Full-precision ``torch.nn.Module``.
- Eager-mode integer weight quantization (int8 / int4 / int2 and their unsigned
  variants), symmetric or asymmetric, at any granularity. Sub-byte payloads and
  zero-points are packed at ``n_bits`` with no padding, matching the export.
- Palettization at any spec-supported ``n_bits`` (1, 2, 3, 4, 6, 8), including a
  quantized LUT (``lut_qspec``).

Unsupported (raises ``NotImplementedError``):

- Floating-point weight quantization (FP8 / FP4): the deploy-time storage math
  is not yet validated for these formats.
- Pruned models
- Weight parametrizations whose dense tensor is not a single ``original``
  tensor (e.g. ``torch.nn.utils.parametrizations.weight_norm``).
- Graph-mode / ``torch.fx.GraphModule`` models.

**Notes:** This is an estimate, not a measurement. A real export may differ:

- It may be *smaller*, because a backend ships only what its graph needs while
  this counts every parameter and buffer the model owns, and because a tensor may
  be representable more compactly than its shape and dtype imply. A tensor that
  ``forward`` never reads or a module that is never called both are accounted for
  here but may be skipped in the export.
- It may be *larger*, because a serialized artifact carries structural metadata
  that is not modelled here.
- This utility is intended to be used with a prepared ``coreai-opt`` model. Passing
  a finalized model to it may result in unpredictable behavior or a wrong bpw value.

Example:
    >>> from coreai_opt.inspection import bits_per_weight
    >>> result = bits_per_weight(prepared_model)
    >>> result.bpw
    8.86
"""

import math
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch.nn.utils import parametrize as _parametrize
from torch.nn.utils.parametrize import ParametrizationList as _ParametrizationList

from coreai_opt._utils.torch_utils import is_float_quant_dtype as _is_float_quant_dtype
from coreai_opt.base_model_compressor import _COREAI_OPT_PREPARED_ATTR as _PREPARED_MARKER
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

__all__ = ["BitsPerWeightResult", "bits_per_weight"]

_DEFAULT_SCALE_BITS = 32

_WeightCompressor = _FakeQuantizeImplBase | _FakePalettizeImplBase


@dataclass
class BitsPerWeightResult:
    """Result of a bits-per-weight computation.

    Attributes:
        bpw (float): Overall average bits per weight across all parameters
            (``total_bits / total_weights``); ``0.0`` if the model has no
            parameters.
        per_module_map (dict[str, float]): Map from module name to that module's own
            average bits per weight. Modules with no logical tensors are omitted.
        total_bits (int): Total storage cost in bits, including amortized
            compression overhead.
        total_weights (int): Total number of logical parameter elements.
    """

    bpw: float
    per_module_map: dict[str, float]
    total_bits: int
    total_weights: int

    def __repr__(self) -> str:
        return (
            f"BitsPerWeightResult(bpw={self.bpw:.4f}, "
            f"total_bits={self.total_bits}, total_weights={self.total_weights})"
        )


def bits_per_weight(model: torch.nn.Module) -> BitsPerWeightResult:
    """Compute the average bits-per-weight of a prepared ``coreai-opt`` model.

    Walks the module tree once. For each parametrized weight, the dense original
    tensor is counted at its effective compressed cost (quantization or
    palettization). Every other directly-owned parameter (biases, norms,
    untargeted weights) and every buffer (BatchNorm running stats, RoPE caches,
    etc.) are counted at their full-precision dtype cost, regardless of
    ``persistent=``, i.e., the metric covers every tensor the model carries.

    Args:
        model (torch.nn.Module): A full-precision, eager-mode integer-quantized,
            or palettized prepared model.

    Returns:
        BitsPerWeightResult: Overall bpw, per-module breakdown, and the totals
        used to derive them.

    Raises:
        NotImplementedError: If ``model`` is a graph-mode prepared model (a
            ``torch.fx.GraphModule``) or a ``torch.export.ExportedProgram``, or if it
            contains a weight compression whose storage cost this utility cannot
            compute: floating-point (FP8 / FP4) quantization, pruning, or a
            parametrization storing multiple original tensors.
    """
    if isinstance(model, (torch.fx.GraphModule, torch.export.ExportedProgram)):
        raise NotImplementedError(
            f"Graph mode prepared models are not supported currently, got {type(model)}. "
            "Only full-precision, eager-mode integer quantized, and palettized "
            "nn.Modules are handled."
        )

    module_bits: dict[str, int] = defaultdict(int)
    module_weights: dict[str, int] = defaultdict(int)

    # id() of every Parameter / Buffer already counted, so a tied tensor
    # is counted once
    seen_ids: set[int] = set()

    for name, module in _named_modules_excluding_compression_machinery(model):
        # Parametrized weights: count the dense original at its compressed cost.
        if _parametrize.is_parametrized(module):
            for tensor_name, param_list in module.parametrizations.items():
                _ensure_single_original(param_list, name, tensor_name)
                _ensure_not_pruned(param_list, name, tensor_name)

                original = param_list.original
                if id(original) in seen_ids:
                    continue
                seen_ids.add(id(original))

                compressor = _get_weight_compressor(param_list)

                if isinstance(compressor, _FakeQuantizeImplBase) and _is_float_quant_dtype(
                    compressor.target_dtype
                ):
                    raise NotImplementedError(
                        f"bits_per_weight cannot compute the storage cost of floating-point "
                        f"weight quantization (dtype {compressor.target_dtype}) on module "
                        f"'{name}'. Only integer quantization (int8 / int4 / int2 and "
                        f"their unsigned variants) and palettization are supported currently."
                    )

                module_bits[name] += _tensor_storage_bits(original, compressor)
                module_weights[name] += original.numel()

        # Directly-owned plain parameters: bias, untargeted weights, norms, etc.
        # A parametrized weight is no longer in _parameters (PyTorch moves it into
        # the ParametrizationList), so it is not re-counted here.
        for param in module.parameters(recurse=False):
            if id(param) in seen_ids:
                continue
            seen_ids.add(id(param))
            module_bits[name] += _full_precision_bits(param)
            module_weights[name] += param.numel()

        # Buffers (BatchNorm running stats, RoPE caches, ...)
        # recurse=False keeps buf_name un-prefixed, so the marker comparison is
        # bare-to-bare.
        for buf_name, buf in module.named_buffers(recurse=False):
            if buf_name == _PREPARED_MARKER or id(buf) in seen_ids:
                continue
            seen_ids.add(id(buf))
            module_bits[name] += _full_precision_bits(buf)
            module_weights[name] += buf.numel()

    total_bits = sum(module_bits.values())
    total_weights = sum(module_weights.values())
    per_module_map = {
        name: bits / module_weights[name]
        for name, bits in module_bits.items()
        if module_weights.get(name, 0) > 0
    }
    bpw = total_bits / total_weights if total_weights > 0 else 0.0
    return BitsPerWeightResult(
        bpw=bpw,
        per_module_map=per_module_map,
        total_bits=total_bits,
        total_weights=total_weights,
    )


def _named_modules_excluding_compression_machinery(
    module: torch.nn.Module, name: str = ""
) -> Iterator[tuple[str, torch.nn.Module]]:
    """Yield ``(dotted_name, module)`` for every module that owns logical tensors.

    Compression machinery owns no logical weights of its own, so a machinery
    module and everything below it is left out by not descending into it.

    Args:
        module (torch.nn.Module): The subtree root to walk.
        name (str): Dotted path of ``module`` from the model root

    Yields:
        tuple[str, torch.nn.Module]: Name and module, parents before children.
    """
    if isinstance(module, (_CompressionSimulatorBase, _ParametrizationList)):
        return

    yield name, module

    for child_name, child in module.named_children():
        yield from _named_modules_excluding_compression_machinery(
            child, f"{name}.{child_name}" if name else child_name
        )


def _get_weight_compressor(param_list: _ParametrizationList) -> _WeightCompressor | None:
    """Return the weight-targeting compressor in a parametrization list, if any.

    Args:
        param_list (ParametrizationList): Parametrizations registered on a parameter.

    Returns:
        _WeightCompressor | None: The first ``_FakePalettizeImplBase`` or
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


def _ensure_single_original(
    param_list: _ParametrizationList, module_name: str, tensor_name: str
) -> None:
    """Raise if a parametrization stores its dense tensor as multiple originals.

    A ``right_inverse`` returning a sequence makes PyTorch register ``original0``,
    ``original1``, ... instead of a single ``original`` (as
    ``torch.nn.utils.parametrizations.weight_norm`` does), so there is no one
    dense tensor whose storage cost we can attribute.
    """
    if not param_list.is_tensor:
        raise NotImplementedError(
            f"bits_per_weight cannot size the parametrization on "
            f"'{module_name}.{tensor_name}': it stores multiple original tensors "
            f"(e.g. weight_norm / spectral_norm) rather than a single dense one."
        )


def _ensure_not_pruned(
    param_list: _ParametrizationList, module_name: str, tensor_name: str
) -> None:
    """Raise if a weight carries a pruning parametrization."""
    for entry in param_list:
        if isinstance(entry, _PruneImplBase):
            raise NotImplementedError(
                f"bits_per_weight cannot compute the storage cost of a pruned "
                f"weight '{module_name}.{tensor_name}'."
            )


def _tensor_storage_bits(weight: torch.Tensor, compressor: _WeightCompressor | None) -> int:
    """Return the storage cost in bits of a (possibly compressed) weight tensor."""
    if compressor is None:
        return _full_precision_bits(weight)
    if isinstance(compressor, _FakePalettizeImplBase):
        return _palettized_bits(weight, compressor)
    return _quantized_bits(weight, compressor)


def _full_precision_bits(tensor: torch.Tensor) -> int:
    """Return the dense storage cost of a tensor in bits."""
    return int(tensor.numel() * tensor.element_size() * 8)


def _quantized_bits(weight: torch.Tensor, quantization_fq: _FakeQuantizeImplBase) -> int:
    """Return the storage cost of a quantized weight including scale / offset overhead.

    Args:
        weight (torch.Tensor): The dense original weight tensor.
        fq (FakeQuantizeImplBase): The weight fake-quantize parametrization.

    Returns:
        int: ``payload_bits + scale_bits + offset_bits`` in bits, where the payload
        is ``numel * n_bits`` and the per-block scale and offset overhead is
        amortized across the weight.

    Note:
        ``num_blocks`` is read directly from the materialized
        ``qparams_calculator.scale`` buffer. This is the canonical
        per-granularity block count (per-tensor, per-channel, per-block, and
        multi-axis per-block all reduce to ``scale.numel()``). When that buffer
        is not yet materialized it is empty,``num_blocks`` is
        derived analytically from ``granularity.get_block_size``.
    """
    num_elements = weight.numel()

    scale = quantization_fq.qparams_calculator.scale
    if scale is not None and scale.numel() > 0:
        num_blocks = scale.numel()
    else:
        block_size = quantization_fq.granularity.get_block_size(weight.shape)
        num_blocks = num_elements // math.prod(block_size)

    payload_bits = num_elements * quantization_fq.n_bits
    scale_bits = num_blocks * _float_qparam_bits(quantization_fq)

    return int(payload_bits + scale_bits + _offset_bits(quantization_fq, num_blocks))


def _float_qparam_bits(fq: _FakeQuantizeImplBase) -> int:
    """Return the per-element bit width of the float qparams this weight exports with."""
    dtype = fq.qparams_calculator._compute_dtype_for_export
    return int(dtype.itemsize * 8)


def _offset_bits(fq: _FakeQuantizeImplBase, num_blocks: int) -> int:
    """Return the per-block dequantization offset cost of a quantized weight, in bits.

    - ``ZP``: the export ships ``zero_point``, packed at ``n_bits``.
    - ``MINVAL``: the export ships ``minval`` instead, and drops the zero-point.
      ``minval`` is a float, so it costs a full float per block.
    """
    if fq.qformulation == _QuantizationFormulation.MINVAL:
        return num_blocks * _float_qparam_bits(fq)
    if fq.qscheme == _QuantizationScheme.ASYMMETRIC:
        return num_blocks * fq.n_bits
    return 0


def _palettized_bits(weight: torch.Tensor, palettization_fq: _FakePalettizeImplBase) -> int:
    """Return the storage cost of a palettized weight including LUT / per-channel-scale overhead.

    Args:
        weight (torch.Tensor): The dense original weight tensor.
        pal (_FakePalettizeImplBase): The weight fake-palettize parametrization.

    Returns:
        int: Effective storage cost of a palettized weight tensor along with
        overhead.
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
        lut_dtype_bits = _buffer_dtype_bits(lut, weight.element_size() * 8)
    lut_bits = lut_elements * lut_dtype_bits

    # Per-channel scale: one weight-dtype value per output channel
    # (weight.shape[0]); amortized when enabled, regardless of calibration.
    num_channels = weight.shape[0]
    per_channel_scale_bits = 0
    if palettization_fq.enable_per_channel_scale:
        per_channel_scale = palettization_fq.per_channel_scale
        if per_channel_scale is not None and per_channel_scale.numel() > 0:
            num_channels = per_channel_scale.numel()
            per_channel_scale_bits = num_channels * per_channel_scale.element_size() * 8
        else:
            per_channel_scale_bits = num_channels * weight.element_size() * 8

    return int(
        indices_bits
        + lut_bits
        + per_channel_scale_bits
        + _lut_quant_bits(weight, palettization_fq, num_channels)
    )


def _lut_quant_bits(
    weight: torch.Tensor, palettization_fq: _FakePalettizeImplBase, num_channels: int
) -> int:
    """Return the qparams cost of a quantized LUT, in bits.

    Dequantizing a quantized LUT needs one scale per palettization block: the LUT
    fake-quantizer overrides ``lut_qspec``'s per-tensor granularity to per-channel
    over the stacked LUT. Two behaviors of the export shape the cost:

    - A symmetric zero-point is a single repeated value, which the export emits as
      one shared constant at no per-element cost, so only asymmetric zero-points
      are counted.
    - With per-channel scaling enabled, the LUT scale is fused into the per-channel
      scale (see ``palettization/kmeans/_prepare_for_export.py``), so a single scale
      tensor ships and is already accounted for by the caller. only the zero-point,
      expanded the same way, is extra.
    """
    if palettization_fq.lut_qspec is None:
        return 0

    lut_scale = palettization_fq.lut_quantization_scale
    if lut_scale is not None and lut_scale.numel() > 0:
        num_lut_blocks = lut_scale.numel()
    else:
        num_lut_blocks = palettization_fq.granularity.num_blocks_to_cluster(weight)

    if palettization_fq.enable_per_channel_scale:
        scale_elements, zero_point_elements = 0, num_channels
    else:
        scale_elements = zero_point_elements = num_lut_blocks

    bits = scale_elements * _buffer_dtype_bits(lut_scale, _DEFAULT_SCALE_BITS)
    if palettization_fq.lut_qspec.qscheme == _QuantizationScheme.ASYMMETRIC:
        bits += zero_point_elements * palettization_fq.lut_qspec.n_bits
    return int(bits)


def _buffer_dtype_bits(buffer: torch.Tensor | None, default: int) -> int:
    """Return the per-element bit width of a buffer, or a default if unmaterialized.

    Args:
        buffer (torch.Tensor | None): A scale, zero-point, or LUT buffer.
        default (int): Bit width to assume when the buffer is missing or empty.

    Returns:
        int: ``buffer.element_size() * 8`` if the buffer holds data, else
        ``default``.
    """
    if buffer is not None and buffer.numel() > 0:
        return int(buffer.element_size() * 8)
    return default
