# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
KV-cache update export handler tests

``register_cache_update_export_handler`` lets a quantization whose cache update is not a
static ``coreai.quantize -> coreai.dequantize`` pair (e.g. one whose qparams vary per update
and must be stored next to the cache) own its export. These tests check that finalize's
cache pass hands such writes, and the read-only fetches of their slots, to the handler,
and leaves everything else to the built-in relocation.
"""

import pytest
import torch

import coreai_opt.quantization._export_utils as export_utils
from coreai_opt.quantization._export_utils import (
    CacheUpdateRead,
    CacheUpdateWrite,
    get_cache_update_export_handler,
    register_cache_update_export_handler,
)
from coreai_opt.quantization._graph._prepare_for_export import _move_cache_dequant_to_output

# The ``coreai_opt_test_cache_export`` namespace is reserved for this file
# (``torch.library`` registrations are process-global and cannot be removed).
_NS = "coreai_opt_test_cache_export"


@torch.library.custom_op(f"{_NS}::produce_update", mutates_args=())
def _produce_update(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


@_produce_update.register_fake
def _(x):
    return torch.empty_like(x)


@torch.library.custom_op(f"{_NS}::cache_update", mutates_args=["cache"])
def _cache_update(
    cache: torch.Tensor, update: torch.Tensor, begin: torch.Tensor, end: torch.Tensor, slot: int
) -> torch.Tensor:
    return cache[slot].clone()


@_cache_update.register_fake
def _(cache, update, begin, end, slot):
    return torch.empty_like(cache[slot])


def _graph(slots_written: tuple[int, ...], slots_read: tuple[int, ...]) -> torch.fx.GraphModule:
    """A cache written through ``produce_update`` and read through empty updates."""
    graph = torch.fx.Graph()

    def placeholder(name: str, val: torch.Tensor) -> torch.fx.Node:
        node = graph.placeholder(name)
        node.meta["val"] = val
        return node

    cache = placeholder("cache", torch.empty(4, 8, 2))
    x = placeholder("x", torch.empty(8, 2))
    # Computed, as in a real graph: the cache must be the op's only placeholder argument.
    index = graph.call_function(torch.ops.aten.zeros.default, ([3],), {"dtype": torch.int32})
    index.meta["val"] = torch.empty(3, dtype=torch.int32)

    def cache_op(update: torch.fx.Node, slot: int) -> torch.fx.Node:
        node = graph.call_function(
            torch.ops.coreai_opt_test_cache_export.cache_update.default,
            (cache, update, index, index, slot),
        )
        node.meta["torch_fn"] = ("cache_update", f"{_NS}.cache_update")
        node.meta["val"] = torch.empty(8, 2)
        return node

    outs = []
    for slot in slots_written:
        update = graph.call_function(
            torch.ops.coreai_opt_test_cache_export.produce_update.default, (x,)
        )
        update.meta["val"] = torch.empty(8, 2)
        outs.append(cache_op(update, slot))
    for slot in slots_read:
        empty = graph.call_function(torch.ops.aten.empty.memory_format, ([8, 0],))
        empty.meta["val"] = torch.empty(8, 0)
        outs.append(cache_op(empty, slot))
    graph.output(tuple(outs))
    return torch.fx.GraphModule(torch.nn.Module(), graph)


@pytest.fixture
def recording_handler(monkeypatch):
    """Register a handler for ``produce_update`` that records what it is given."""
    calls: list[tuple[CacheUpdateWrite, list[CacheUpdateRead], int]] = []

    def handler(model, write, reads, quant_input_idx):
        calls.append((write, reads, quant_input_idx))

    monkeypatch.setattr(export_utils, "_CACHE_UPDATE_EXPORT_HANDLERS", {})
    register_cache_update_export_handler(f"{_NS}::produce_update", handler)
    return calls


class TestCacheUpdateExportHandler:
    def test_lookup_is_by_the_producing_op(self, recording_handler):
        model = _graph(slots_written=(0,), slots_read=())
        update = next(n for n in model.graph.nodes if "produce_update" in str(n.target))
        placeholder = next(n for n in model.graph.nodes if n.op == "placeholder")

        assert get_cache_update_export_handler(update) is not None
        assert get_cache_update_export_handler(placeholder) is None

    def test_each_handled_slot_is_one_call_with_its_reads(self, recording_handler):
        """One call per written slot, carrying that slot's read-only fetches only."""
        model = _graph(slots_written=(0, 1), slots_read=(1,))

        _move_cache_dequant_to_output(model, op_type="cache_update", quant_input_idx=1)

        assert [(w.op.args[4], len(reads), idx) for w, reads, idx in recording_handler] == [
            (0, 0, 1),
            (1, 1, 1),
        ]
        write, (read,), _ = recording_handler[1]
        assert "produce_update" in str(write.update.target)
        assert read.op.args[4] == 1
        assert read.empty_update is read.op.args[1]

    def test_without_a_handler_an_unknown_update_still_raises(self, monkeypatch):
        monkeypatch.setattr(export_utils, "_CACHE_UPDATE_EXPORT_HANDLERS", {})
        model = _graph(slots_written=(0,), slots_read=())

        with pytest.raises(RuntimeError, match="Expected coreai.dequantize"):
            _move_cache_dequant_to_output(model, op_type="cache_update", quant_input_idx=1)

    def test_two_writes_of_a_handled_slot_raise(self, recording_handler):
        model = _graph(slots_written=(0, 0), slots_read=())

        with pytest.raises(RuntimeError, match="Two quantized cache_update ops write slot"):
            _move_cache_dequant_to_output(model, op_type="cache_update", quant_input_idx=1)
        assert not recording_handler
