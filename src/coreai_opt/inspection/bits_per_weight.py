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

from collections import defaultdict
from dataclasses import dataclass

import torch
from torch.nn.utils import parametrize as _parametrize

from coreai_opt._utils.torch_utils import is_float_quant_dtype as _is_float_quant_dtype
from coreai_opt.base_model_compressor import _COREAI_OPT_PREPARED_ATTR as _PREPARED_MARKER
from coreai_opt.quantization.spec.fake_quantize import FakeQuantizeImplBase as _FakeQuantizeImplBase

from ._bpw_utils import (
    ensure_not_pruned as _ensure_not_pruned,
    ensure_single_original as _ensure_single_original,
    full_precision_bits as _full_precision_bits,
    get_weight_compressor as _get_weight_compressor,
    named_modules_excluding_compression_machinery as _named_modules_excluding_compression_machinery,
    tensor_storage_bits as _tensor_storage_bits,
)

__all__ = ["BitsPerWeightResult", "bits_per_weight"]


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
