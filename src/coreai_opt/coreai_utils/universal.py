# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Universal assets: N compressed graphs that share one fp16 copy of the weights.

Prototype 2a of the Universal Asset project (AOT graph transformation). For each
scheme, the existing coreai-opt function rewrites a copy of the graph ahead of
time. The weight constants it computed are then replaced by
``udml.placeholder "uasset.weight_transform"`` ops that read the fp16 source
weight, and a JSON file per scheme (the IR) records how to recompute them. The
compiler's ``coreai-materialize-weight-transforms`` pass executes that JSON at
JIT compile time.
"""

from __future__ import annotations

import copy
import inspect
import json
import re
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from coreai_opt.coreai_utils._coreai_imports import (
    AIProgram as _AIProgram,
    DenseElementsAttr as _DenseElementsAttr,
    InsertionPoint as _InsertionPoint,
    IntegerAttr as _IntegerAttr,
    IntegerType as _IntegerType,
)
from coreai_opt.coreai_utils._utils.graph_utils import (
    _infer_quantization_block_sizes,
    _select_input_output_channel_axis,
    _should_compress_op,
)
from coreai_opt.coreai_utils._utils.palettize_utils import (
    _infer_palettization_block_sizes_and_channel_axis,
)
from coreai_opt.coreai_utils.common import DType, QScheme
from coreai_opt.coreai_utils.passes import _OPS_WEIGHT_NEED_COMPRESSION
from coreai_opt.coreai_utils.passes.weight_palettization import palettize_weights
from coreai_opt.coreai_utils.passes.weight_quantization import quantize_weights
from coreai_opt.coreai_utils.passes.weight_sparsification import sparsify_weights

IR_NAME = "coreai.weight_transforms"
IR_VERSION = 1
PLACEHOLDER_OPNAME = "uasset.weight_transform"
TRANSFORM_ID_ATTR = "uasset.transform_id"
RESULT_ATTR = "uasset.result"
WEIGHT_TRANSFORMS_DIR = "weight_transforms"

# A scheme is a coreai-opt compression function plus its keyword arguments.
Scheme = tuple[Callable[..., _AIProgram], dict[str, Any]]

# Role of a deferred constant, keyed by the (consumer op, operand index) it feeds.
_ROLES = {
    ("coreai.blockwise_shift_scale", 0): "data",
    ("coreai.blockwise_shift_scale", 1): "scale",
    ("coreai.blockwise_shift_scale", 2): "zero_point",
    ("coreai.blockwise_shift_scale", 3): "offset2",
    ("coreai.lut_to_dense", 0): "indices",
    ("coreai.lut_to_dense", 1): "lut",
    ("coreai.lut_to_dense", 2): "axis",
    ("coreai.build_sparse_with_bitmask", 0): "nonzero_values",
    ("coreai.build_sparse_with_bitmask", 1): "mask",
}
# Structural constants that never depend on weight values.
_INLINE_ROLES = frozenset({"offset2", "axis"})


def _effective_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Bind ``kwargs`` to ``fn`` and fill in its defaults, so the IR is explicit."""
    bound = inspect.signature(fn).bind(coreai_program=None, **kwargs)
    bound.apply_defaults()
    resolved = dict(bound.arguments)
    resolved.pop("coreai_program")
    resolved.pop("in_place", None)
    return resolved


def _quantize_params(op: Any, kw: dict[str, Any]) -> dict[str, Any]:
    shape = list(op.result.type.shape)
    dtype = DType(kw["dtype"])
    scale_dtype = kw["scale_dtype"]
    if dtype == DType.FP4_E2M1FN:
        scale_dtype = DType.FP8_E8M0FNU  # quantize_weights always stores MXFP4 scales in e8m0.
    block_sizes = _infer_quantization_block_sizes(op, shape, kw["granularity"], kw["block_size"])
    return {
        "dtype": dtype.value,
        "qscheme": QScheme(kw["qscheme"]).value,
        "block_sizes": list(block_sizes),
        "scale_dtype": None if scale_dtype is None else DType(scale_dtype).value,
    }


def _palettize_params(op: Any, kw: dict[str, Any]) -> dict[str, Any]:
    shape = list(op.result.type.shape)
    block_sizes, channel_axis = _infer_palettization_block_sizes_and_channel_axis(
        op, shape, kw["granularity"], kw["group_size"]
    )
    lut_dtype = kw["lut_dtype"]
    return {
        "n_bits": kw["n_bits"],
        "lut_dtype": None if lut_dtype is None else DType(lut_dtype).value,
        "block_sizes": list(block_sizes),
        "channel_axis": channel_axis,
        "cluster_dim": kw["cluster_dim"],
        "enable_per_channel_scale": kw["enable_per_channel_scale"],
        "enable_fast_kmeans_mode": kw["enable_fast_kmeans_mode"],
        "rounding_precision": kw["rounding_precision"],
    }


def _percentile_index_dtype() -> str:
    """How this numpy computes np.percentile's virtual index: in the array's dtype
    (numpy < 2.4; fp16 then overflows above 65504 elements) or in float64."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        probe = np.percentile(np.ones(70000, dtype=np.float16), 50)
    return "weight" if np.isnan(probe) else "float64"


def _sparsify_params(op: Any, kw: dict[str, Any]) -> dict[str, Any]:
    input_axis, output_axis = _select_input_output_channel_axis(op)
    quantize_dtype = kw["quantize_dtype"]
    n_m_ratio = kw["n_m_ratio"]
    return {
        "target_sparsity": kw["target_sparsity"],
        "n_m_ratio": None if n_m_ratio is None else list(n_m_ratio),
        "block_size": kw["block_size"],
        "input_channel_axis": 1 if input_axis is None else input_axis,
        "output_channel_axis": 0 if output_axis is None else output_axis,
        "quantize_dtype": None if quantize_dtype is None else DType(quantize_dtype).value,
        "palettize_nbits": kw["palettize_nbits"],
        "percentile_index_dtype": _percentile_index_dtype(),
    }


@dataclass(frozen=True)
class _Kernel:
    name: str
    params: Callable[[Any, dict[str, Any]], dict[str, Any]]


_KERNELS: dict[Callable[..., Any], _Kernel] = {
    quantize_weights: _Kernel("quantize", _quantize_params),
    palettize_weights: _Kernel("palettize", _palettize_params),
    sparsify_weights: _Kernel("sparsify", _sparsify_params),
}


@dataclass
class DeferredScheme:
    """What ``defer_weights`` produced for one scheme."""

    ir: dict[str, Any]
    # Value attribute of every constant replaced by a placeholder, keyed by
    # (transform id, role). Lets tests rebuild the direct twin exactly.
    removed_values: dict[tuple[int, str], Any] = field(default_factory=dict)


@dataclass
class UniversalAsset:
    path: Path
    twins: dict[str, Path]
    schemes: dict[str, DeferredScheme]
    program: _AIProgram  # The in-memory universal program, as saved to ``path``.


def _iter_ops(op: Any) -> Iterator[Any]:
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield child
                yield from _iter_ops(child)


def _graphs(program: _AIProgram) -> list[Any]:
    return list(program._module._mlir_module.body.operations)  # noqa: SLF001


def _is_splat_zero(attr: Any) -> bool:
    if not isinstance(attr, _DenseElementsAttr) or not attr.is_splat:
        return False
    return float(attr.get_splat_value().value) == 0.0


def _role(constant: Any) -> str:
    uses = list(constant.result.uses)
    if len(uses) != 1:
        raise ValueError(f"expected one use of a compression constant, got {len(uses)}")
    key = (uses[0].owner.name, uses[0].operand_number)
    if key not in _ROLES:
        raise ValueError(f"unknown compression constant role {key}")
    return _ROLES[key]


def defer_weights(
    graph: Any,
    eligible: list[tuple[Any, dict[str, Any]]],
    preexisting: set[Any],
    kernel: str,
    scheme: str,
    graph_name: str,
) -> DeferredScheme:
    """Replace the weight-derived constants coreai-opt created in ``graph`` by placeholders.

    coreai-opt inserts each weight's compression chain right before the source
    constant and leaves the source dead. For each such source, every new leaf
    constant of its chain becomes a placeholder at the same position and location,
    except structural constants and splat zeros. The source is moved above the
    chain so the placeholders can read it.
    """
    from coreai._compiler.dialects import udml  # noqa: PLC0415
    from coreai._compiler.ir import StringAttr  # noqa: PLC0415

    deferred = DeferredScheme(
        ir={
            "ir": IR_NAME,
            "version": IR_VERSION,
            "scheme": scheme,
            "graph": graph_name,
            "transforms": [],
        }
    )
    for source, params in eligible:
        if len(list(source.result.uses)) != 0:
            continue  # coreai-opt skipped this weight.
        block_ops = list(source.operation.block.operations)
        index = block_ops.index(source.operation)
        chain = []
        while index > 0 and block_ops[index - 1].operation not in preexisting:
            index -= 1
            chain.insert(0, block_ops[index])
        if not chain:
            continue
        transform_id = len(deferred.ir["transforms"])
        results: list[str] = []
        first_placeholder = None
        for op in chain:
            if op.name != "coreai.constant":
                continue
            role = _role(op)
            value = op.attributes["value"]
            if role in _INLINE_ROLES or _is_splat_zero(value):
                continue
            with op.context, op.location, _InsertionPoint(op):
                placeholder = udml.PlaceholderOp(
                    [op.result.type], PLACEHOLDER_OPNAME, [source.result]
                )
                placeholder.attributes[TRANSFORM_ID_ATTR] = _IntegerAttr.get(
                    _IntegerType.get_signless(64), transform_id
                )
                placeholder.attributes[RESULT_ATTR] = StringAttr.get(role)
            op.result.replace_all_uses_with(placeholder.results[0])
            op.erase()
            deferred.removed_values[(transform_id, role)] = value
            results.append(role)
            if first_placeholder is None:
                first_placeholder = placeholder
        if first_placeholder is None:
            continue
        source.operation.move_before(first_placeholder.operation)
        deferred.ir["transforms"].append(
            {"id": transform_id, "kernel": kernel, "params": params, "results": results}
        )
    return deferred


def make_universal_asset(
    program: _AIProgram,
    schemes: dict[str, Scheme],
    out_dir: Path,
) -> UniversalAsset:
    """Write ``out_dir/universal.aimodel`` and each scheme's direct twin.

    ``universal.aimodel`` holds the fp16 ``@main`` plus ``@main__<scheme>`` per
    scheme, and ``weight_transforms/<scheme>.json``. ``out_dir/twins/<scheme>.aimodel``
    is coreai-opt's direct output for the scheme, used for comparison.
    """
    graphs = _graphs(program)
    if len(graphs) != 1 or graphs[0].attributes["sym_name"].value != "main":
        raise ValueError("expected a program with a single graph named main")

    universal = copy.deepcopy(program)
    universal_body_end = _graphs(universal)[-1]
    twins: dict[str, Path] = {}
    deferred_schemes: dict[str, DeferredScheme] = {}
    for scheme, (fn, kwargs) in schemes.items():
        if not re.fullmatch(r"[A-Za-z0-9_]+", scheme):
            raise ValueError(f"scheme name must be alphanumeric/underscore, got {scheme!r}")
        if fn not in _KERNELS:
            raise ValueError(f"unsupported compression function {fn!r}")
        if "in_place" in kwargs or "coreai_program" in kwargs:
            raise ValueError("scheme kwargs must not set coreai_program or in_place")
        kernel = _KERNELS[fn]
        kw = _effective_kwargs(fn, kwargs)

        work = copy.deepcopy(program)
        (graph,) = _graphs(work)
        preexisting = set()
        eligible = []
        for op in _iter_ops(graph):
            preexisting.add(op.operation)
            if _should_compress_op(op, kw["weight_num_threshold"], _OPS_WEIGHT_NEED_COMPRESSION):
                eligible.append((op, kernel.params(op, kw)))

        fn(coreai_program=work, in_place=True, **kwargs)
        twins[scheme] = out_dir / "twins" / f"{scheme}.aimodel"
        work.save_asset(twins[scheme])

        graph_name = f"main__{scheme}"
        deferred = defer_weights(graph, eligible, preexisting, kernel.name, scheme, graph_name)
        from coreai._compiler.ir import StringAttr  # noqa: PLC0415

        with graph.context:
            graph.attributes["sym_name"] = StringAttr.get(graph_name)
        graph.move_after(universal_body_end)
        universal_body_end = graph
        deferred_schemes[scheme] = deferred

    path = out_dir / "universal.aimodel"
    universal.save_asset(path)
    ir_dir = path / WEIGHT_TRANSFORMS_DIR
    ir_dir.mkdir(exist_ok=True)
    for scheme, deferred in deferred_schemes.items():
        (ir_dir / f"{scheme}.json").write_text(json.dumps(deferred.ir, indent=1) + "\n")
    return UniversalAsset(path=path, twins=twins, schemes=deferred_schemes, program=universal)


__all__ = [
    "IR_NAME",
    "IR_VERSION",
    "PLACEHOLDER_OPNAME",
    "RESULT_ATTR",
    "TRANSFORM_ID_ATTR",
    "WEIGHT_TRANSFORMS_DIR",
    "DeferredScheme",
    "Scheme",
    "UniversalAsset",
    "defer_weights",
    "make_universal_asset",
]
