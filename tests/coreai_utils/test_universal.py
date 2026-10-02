# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import copy
import json
import re
from pathlib import Path

import pytest

from coreai_opt.coreai_utils import (
    CompressionGranularity,
    DType,
    palettize_weights,
    quantize_weights,
    sparsify_weights,
)
from coreai_opt.coreai_utils.common import QScheme
from coreai_opt.coreai_utils.universal import (
    PLACEHOLDER_OPNAME,
    RESULT_ATTR,
    TRANSFORM_ID_ATTR,
    WEIGHT_TRANSFORMS_DIR,
    make_universal_asset,
)

_SCHEMES = {
    "w8": (quantize_weights, {"dtype": DType.INT8}),
    "w4_block": (
        quantize_weights,
        {"dtype": DType.INT4, "granularity": CompressionGranularity.PER_BLOCK},
    ),
    "uw8_asym": (quantize_weights, {"dtype": DType.UINT8, "qscheme": QScheme.ASYMMETRIC}),
    "fp8": (quantize_weights, {"dtype": DType.FP8_E4M3FN}),
    "lut4": (palettize_weights, {"n_bits": 4, "lut_dtype": None}),
    "lut4_int8_pcs": (
        palettize_weights,
        {
            "n_bits": 4,
            "lut_dtype": DType.INT8,
            "granularity": CompressionGranularity.PER_GROUPED_CHANNEL,
            "group_size": 16,
            "enable_per_channel_scale": True,
        },
    ),
    "s50": (sparsify_weights, {"target_sparsity": 0.5}),
    "s50_w8": (sparsify_weights, {"target_sparsity": 0.5, "quantize_dtype": DType.INT8}),
    "s50_lut4": (sparsify_weights, {"target_sparsity": 0.5, "palettize_nbits": 4}),
}


def _asm(program) -> str:
    return program._module._mlir_module.operation.get_asm(enable_debug_info=True)


def _graph(program, name: str):
    for op in program._module._mlir_module.body.operations:
        if op.attributes["sym_name"].value == name:
            return op
    raise KeyError(name)


def _placeholders(graph):
    return [
        op
        for op in graph.regions[0].blocks[0].operations
        if op.name == "udml.placeholder" and op.attributes["opname"].value == PLACEHOLDER_OPNAME
    ]


@pytest.fixture
def _universal(_coreai_program, tmp_path: Path):
    program, _, _ = _coreai_program
    return program, make_universal_asset(program, _SCHEMES, tmp_path)


def test_one_copy_of_each_weight(_universal) -> None:
    program, universal = _universal
    from coreai.authoring import AIModelAsset

    loaded = _asm(AIModelAsset.load(universal.path).program)
    resources = set(re.findall(r"resource_\d+", loaded))
    assert resources == set(re.findall(r"resource_\d+", _asm(program)))
    graph_names = re.findall(r"coreai\.graph @(\w+)", loaded)
    assert graph_names == ["main"] + [f"main__{s}" for s in _SCHEMES]


def test_json_entry_per_placeholder(_universal) -> None:
    _, universal = _universal
    for scheme in _SCHEMES:
        ir = json.loads((universal.path / WEIGHT_TRANSFORMS_DIR / f"{scheme}.json").read_text())
        assert ir["graph"] == f"main__{scheme}"
        expected = {(t["id"], role) for t in ir["transforms"] for role in t["results"]}
        placeholders = _placeholders(_graph(universal.program, f"main__{scheme}"))
        actual = [
            (op.attributes[TRANSFORM_ID_ATTR].value, op.attributes[RESULT_ATTR].value)
            for op in placeholders
        ]
        assert ir["transforms"], scheme
        assert sorted(actual) == sorted(expected)


@pytest.mark.parametrize("scheme", list(_SCHEMES))
def test_refilled_graph_is_direct_twin(_universal, scheme: str, tmp_path: Path) -> None:
    """Filling the placeholders with coreai-opt's own values gives back the twin exactly."""
    from coreai._compiler.dialects import coreai
    from coreai._compiler.ir import InsertionPoint, StringAttr
    from coreai.authoring import AIModelAsset

    _, universal = _universal
    program = copy.deepcopy(universal.program)
    graph = _graph(program, f"main__{scheme}")
    for op in list(program._module._mlir_module.body.operations):
        if op != graph:
            op.erase()
    removed = universal.schemes[scheme].removed_values
    sources = set()
    for placeholder in _placeholders(graph):
        key = (placeholder.attributes[TRANSFORM_ID_ATTR].value, placeholder.attributes[RESULT_ATTR].value)
        sources.add(placeholder.operands[0].owner)
        with placeholder.context, placeholder.location, InsertionPoint(placeholder):
            constant = coreai.ConstantOp(value=removed[key])
        placeholder.results[0].replace_all_uses_with(constant.result)
        placeholder.erase()
    for source in sources:
        assert not list(source.result.uses)
        source.erase()
    with graph.context:
        graph.attributes["sym_name"] = StringAttr.get("main")
    program.save_asset(tmp_path / "refilled.aimodel")

    refilled = _asm(AIModelAsset.load(tmp_path / "refilled.aimodel").program)
    twin = _asm(AIModelAsset.load(universal.twins[scheme]).program)
    assert refilled == twin
