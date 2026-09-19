# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for casting_utils.

Tests for the cleanup pipeline (fuse chains, remove noops, bypass branching,
deduplicate) and the individual utility functions (FP16 helpers, INT16 unsafe
node detection, anchor insertion).
"""

import numpy as np
import pytest
import torch

from coreai_opt._utils.casting_utils import (
    anchor_int16_reshape_inputs,
    build_castable_int16_nodes,
    build_unsafe_to_cast_nodes,
    build_user_reachable_nodes,
    cast_scalar_to_fp16,
    cast_tensor_to_fp16,
    check_tensor_overflow_fp16,
    classify_float_args,
    cleanup_casts,
    insert_cast_after,
    is_ignored_op,
)


def _make_fake_val(dtype: torch.dtype, shape: tuple[int, ...] = (1,)) -> torch.Tensor:
    """Create a fake tensor for node metadata."""
    return torch.empty(shape, dtype=dtype)


def _get_placeholder(graph: torch.fx.Graph) -> torch.fx.Node:
    """Return the first placeholder node in the graph."""
    return next(n for n in graph.nodes if n.op == "placeholder")


def _get_cast_nodes(graph: torch.fx.Graph) -> list[torch.fx.Node]:
    """Return all _to_copy cast nodes in the graph."""
    return [
        node
        for node in graph.nodes
        if node.op == "call_function" and node.target == torch.ops.aten._to_copy.default
    ]


def _add_cast(
    graph: torch.fx.Graph,
    input_node: torch.fx.Node,
    dtype: torch.dtype,
    name: str,
    pass_inserted: set[torch.fx.Node] | None = None,
) -> torch.fx.Node:
    """Add a _to_copy cast node to the graph."""
    node = graph.create_node(
        "call_function",
        torch.ops.aten._to_copy.default,
        args=(input_node,),
        kwargs={"dtype": dtype},
        name=name,
    )
    node.meta["val"] = _make_fake_val(dtype)
    if pass_inserted is not None:
        pass_inserted.add(node)
    return node


def _add_relu(graph: torch.fx.Graph, input_node: torch.fx.Node, name: str) -> torch.fx.Node:
    """Add a relu node to the graph."""
    node = graph.create_node(
        "call_function",
        torch.ops.aten.relu.default,
        args=(input_node,),
        name=name,
    )
    node.meta["val"] = input_node.meta["val"].clone()
    return node


# =============================================================================
# Branching cast tests
# =============================================================================
def _build_graph_with_branching_casts(child_3_is_cast: bool):
    """Build a graph where a pass-inserted fp32->fp16 cast branches into 3 children.

    Structure before cleanup::

        source(fp32) -> pass_cast(fp32->fp16) -+-> pass_cast_1(fp16->fp32) -> relu_1
                                                |-> pass_cast_2(fp16->fp32) -> relu_2
                                                +-> child_3                 -> relu_3

    If child_3_is_cast=True:  child_3 = original cast fp16->fp32
    If child_3_is_cast=False: child_3 = relu (non-cast op consuming fp16)

    Returns (graph, pass_inserted).
    """
    graph = torch.fx.Graph()
    pass_inserted: set[torch.fx.Node] = set()

    source = graph.placeholder("source")
    source.meta["val"] = _make_fake_val(torch.float32)

    parent_cast = _add_cast(graph, source, torch.float16, "pass_fp32_to_fp16", pass_inserted)
    child_1 = _add_cast(graph, parent_cast, torch.float32, "pass_1_fp16_to_fp32", pass_inserted)
    child_2 = _add_cast(graph, parent_cast, torch.float32, "pass_2_fp16_to_fp32", pass_inserted)

    if child_3_is_cast:
        child_3 = _add_cast(graph, parent_cast, torch.float32, "orig_fp16_to_fp32")
    else:
        child_3 = _add_relu(graph, parent_cast, "relu_on_fp16")

    relu_1 = _add_relu(graph, child_1, "relu_1")
    relu_2 = _add_relu(graph, child_2, "relu_2")
    relu_3 = _add_relu(graph, child_3, "relu_3")
    graph.output((relu_1, relu_2, relu_3))

    return graph, pass_inserted


class TestCleanupBranchingCasts:
    """Test cleanup_casts on graphs where a cast branches into multiple children."""

    def test_all_three_children_are_casts(self):
        """When parent branches into 3 cast children (2 pass-inserted + 1 original),
        all are fp32->fp16->fp32 roundtrips.

        Expected: all roundtrips bypassed, all relus read directly from source.
        """
        graph, pass_inserted = _build_graph_with_branching_casts(child_3_is_cast=True)
        assert len(_get_cast_nodes(graph)) == 4

        cleanup_casts(graph, pass_inserted)
        graph.eliminate_dead_code()

        assert len(_get_cast_nodes(graph)) == 0
        source = _get_placeholder(graph)
        nodes = {n.name: n for n in graph.nodes}
        for name in ("relu_1", "relu_2", "relu_3"):
            assert nodes[name].args[0] is source

    def test_third_child_is_non_cast_op(self):
        """When parent branches into 2 pass-inserted cast children + 1 non-cast op.

        Expected: two roundtrips bypassed, parent cast survives for the relu child.
        """
        graph, pass_inserted = _build_graph_with_branching_casts(child_3_is_cast=False)
        assert len(_get_cast_nodes(graph)) == 3

        cleanup_casts(graph, pass_inserted)
        graph.eliminate_dead_code()

        remaining_casts = _get_cast_nodes(graph)
        assert len(remaining_casts) == 1
        assert remaining_casts[0].kwargs.get("dtype") == torch.float16

        source = _get_placeholder(graph)
        nodes = {n.name: n for n in graph.nodes}
        for name in ("relu_1", "relu_2"):
            assert nodes[name].args[0] is source

        # relu_3 -> relu_on_fp16 -> surviving parent cast
        relu_3_input = nodes["relu_3"].args[0]
        assert relu_3_input.target == torch.ops.aten.relu.default
        assert relu_3_input.args[0] is remaining_casts[0]


# =============================================================================
# Chain fusion tests
# =============================================================================
class TestFuseCastChains:
    """Test chain fusion and barrier behavior in cleanup_casts."""

    def test_same_category_roundtrip_eliminated(self):
        """A pass-inserted fp32->fp16->fp32 chain is a roundtrip and gets eliminated.

        source(fp32) -> cast(fp32->fp16) -> cast(fp16->fp32) -> relu
        =>
        source(fp32) -> relu
        """
        graph = torch.fx.Graph()
        pass_inserted: set[torch.fx.Node] = set()

        source = graph.placeholder("source")
        source.meta["val"] = _make_fake_val(torch.float32)
        c1 = _add_cast(graph, source, torch.float16, "cast_fp32_fp16", pass_inserted)
        c2 = _add_cast(graph, c1, torch.float32, "cast_fp16_fp32", pass_inserted)
        relu = _add_relu(graph, c2, "relu")
        graph.output(relu)

        cleanup_casts(graph, pass_inserted)
        graph.eliminate_dead_code()

        assert len(_get_cast_nodes(graph)) == 0
        nodes = {n.name: n for n in graph.nodes}
        assert nodes["relu"].args[0] is _get_placeholder(graph)

    def test_same_category_collapses_to_single_cast(self):
        """A pass-inserted fp32->fp16->bf16 chain collapses to fp32->bf16.

        source(fp32) -> cast(fp32->fp16) -> cast(fp16->bf16) -> relu
        =>
        source(fp32) -> cast(fp32->bf16) -> relu
        """
        graph = torch.fx.Graph()
        pass_inserted: set[torch.fx.Node] = set()

        source = graph.placeholder("source")
        source.meta["val"] = _make_fake_val(torch.float32)
        c1 = _add_cast(graph, source, torch.float16, "cast_fp32_fp16", pass_inserted)
        c2 = _add_cast(graph, c1, torch.bfloat16, "cast_fp16_bf16", pass_inserted)
        relu = _add_relu(graph, c2, "relu")
        graph.output(relu)

        cleanup_casts(graph, pass_inserted)
        graph.eliminate_dead_code()

        remaining = _get_cast_nodes(graph)
        assert len(remaining) == 1
        assert remaining[0].kwargs.get("dtype") == torch.bfloat16
        assert remaining[0].all_input_nodes[0] is _get_placeholder(graph)

    def test_three_cast_chain_collapses(self):
        """A 3-cast pass-inserted chain fp32->fp16->bf16->fp32 is a roundtrip.

        source(fp32) -> cast(fp32->fp16) -> cast(fp16->bf16) -> cast(bf16->fp32) -> relu
        =>
        source(fp32) -> relu
        """
        graph = torch.fx.Graph()
        pass_inserted: set[torch.fx.Node] = set()

        source = graph.placeholder("source")
        source.meta["val"] = _make_fake_val(torch.float32)
        c1 = _add_cast(graph, source, torch.float16, "cast_1", pass_inserted)
        c2 = _add_cast(graph, c1, torch.bfloat16, "cast_2", pass_inserted)
        c3 = _add_cast(graph, c2, torch.float32, "cast_3", pass_inserted)
        relu = _add_relu(graph, c3, "relu")
        graph.output(relu)

        cleanup_casts(graph, pass_inserted)
        graph.eliminate_dead_code()

        assert len(_get_cast_nodes(graph)) == 0
        nodes = {n.name: n for n in graph.nodes}
        assert nodes["relu"].args[0] is _get_placeholder(graph)

    def test_cross_category_blocks_fusion(self):
        """A chain crossing dtype categories cannot be fused.

        source(fp32) -> cast(fp32->int32) -> cast(int32->fp32) -> relu

        Both casts are pass-inserted but cross float/signed_int categories,
        so neither can be included in the other's chain. Both survive.
        """
        graph = torch.fx.Graph()
        pass_inserted: set[torch.fx.Node] = set()

        source = graph.placeholder("source")
        source.meta["val"] = _make_fake_val(torch.float32)
        c1 = _add_cast(graph, source, torch.int32, "cast_fp32_int32", pass_inserted)
        c2 = _add_cast(graph, c1, torch.float32, "cast_int32_fp32", pass_inserted)
        relu = _add_relu(graph, c2, "relu")
        graph.output(relu)

        cleanup_casts(graph, pass_inserted)
        graph.eliminate_dead_code()

        remaining = _get_cast_nodes(graph)
        assert len(remaining) == 2

    def test_original_narrowing_cast_blocks_fusion(self):
        """An original narrowing cast acts as a chain barrier.

        source(fp32) -> orig_cast(fp32->fp16) -> pass_cast(fp16->fp32) -> relu

        The original fp32->fp16 is narrowing, so it's not includable in the chain.
        The pass-inserted fp16->fp32 can't fuse with it. Both survive.
        """
        graph = torch.fx.Graph()
        pass_inserted: set[torch.fx.Node] = set()

        source = graph.placeholder("source")
        source.meta["val"] = _make_fake_val(torch.float32)
        c1 = _add_cast(graph, source, torch.float16, "orig_fp32_fp16")  # original, narrowing
        c2 = _add_cast(graph, c1, torch.float32, "pass_fp16_fp32", pass_inserted)
        relu = _add_relu(graph, c2, "relu")
        graph.output(relu)

        cleanup_casts(graph, pass_inserted)
        graph.eliminate_dead_code()

        remaining = _get_cast_nodes(graph)
        assert len(remaining) == 2

    def test_original_widening_cast_allows_fusion(self):
        """An original widening cast is includable and can be fused.

        source(fp16) -> orig_cast(fp16->fp32) -> pass_cast(fp32->fp16) -> relu

        The original fp16->fp32 is widening (same category), so the chain
        fp16->fp32->fp16 is a roundtrip and gets eliminated.
        """
        graph = torch.fx.Graph()
        pass_inserted: set[torch.fx.Node] = set()

        source = graph.placeholder("source")
        source.meta["val"] = _make_fake_val(torch.float16)
        c1 = _add_cast(graph, source, torch.float32, "orig_fp16_fp32")  # original, widening
        c2 = _add_cast(graph, c1, torch.float16, "pass_fp32_fp16", pass_inserted)
        relu = _add_relu(graph, c2, "relu")
        graph.output(relu)

        cleanup_casts(graph, pass_inserted)
        graph.eliminate_dead_code()

        assert len(_get_cast_nodes(graph)) == 0
        nodes = {n.name: n for n in graph.nodes}
        assert nodes["relu"].args[0] is _get_placeholder(graph)

    def test_copy_flag_blocks_fusion(self):
        """An original cast with copy=True blocks chain fusion even if widening.

        source(fp16) -> orig_cast(fp16->fp32, copy=True) -> pass_cast(fp32->fp16) -> relu

        Despite being widening and same-category, the copy flag makes the
        original cast a barrier. Both casts survive.
        """
        graph = torch.fx.Graph()
        pass_inserted: set[torch.fx.Node] = set()

        source = graph.placeholder("source")
        source.meta["val"] = _make_fake_val(torch.float16)
        c1 = graph.create_node(
            "call_function",
            torch.ops.aten._to_copy.default,
            args=(source,),
            kwargs={"dtype": torch.float32, "copy": True},
            name="orig_fp16_fp32_copy",
        )
        c1.meta["val"] = _make_fake_val(torch.float32)
        c2 = _add_cast(graph, c1, torch.float16, "pass_fp32_fp16", pass_inserted)
        relu = _add_relu(graph, c2, "relu")
        graph.output(relu)

        cleanup_casts(graph, pass_inserted)
        graph.eliminate_dead_code()

        remaining = _get_cast_nodes(graph)
        assert len(remaining) == 2

    def test_int_category_roundtrip_eliminated(self):
        """A pass-inserted int32->int16->int32 roundtrip gets eliminated.

        source(int32) -> cast(int32->int16) -> cast(int16->int32) -> relu
        =>
        source(int32) -> relu
        """
        graph = torch.fx.Graph()
        pass_inserted: set[torch.fx.Node] = set()

        source = graph.placeholder("source")
        source.meta["val"] = _make_fake_val(torch.int32)
        c1 = _add_cast(graph, source, torch.int16, "cast_i32_i16", pass_inserted)
        c2 = _add_cast(graph, c1, torch.int32, "cast_i16_i32", pass_inserted)
        relu = _add_relu(graph, c2, "relu")
        graph.output(relu)

        cleanup_casts(graph, pass_inserted)
        graph.eliminate_dead_code()

        assert len(_get_cast_nodes(graph)) == 0
        nodes = {n.name: n for n in graph.nodes}
        assert nodes["relu"].args[0] is _get_placeholder(graph)


# =============================================================================
# Noop cast removal tests
# =============================================================================
class TestRemoveNoopCasts:
    """Test removal of casts where input dtype already matches target dtype."""

    def test_same_dtype_cast_removed(self):
        """A pass-inserted fp32->fp32 cast is a noop and gets removed.

        source(fp32) -> cast(fp32->fp32) -> relu
        =>
        source(fp32) -> relu
        """
        graph = torch.fx.Graph()
        pass_inserted: set[torch.fx.Node] = set()

        source = graph.placeholder("source")
        source.meta["val"] = _make_fake_val(torch.float32)
        c1 = _add_cast(graph, source, torch.float32, "noop_cast", pass_inserted)
        relu = _add_relu(graph, c1, "relu")
        graph.output(relu)

        cleanup_casts(graph, pass_inserted)
        graph.eliminate_dead_code()

        assert len(_get_cast_nodes(graph)) == 0
        nodes = {n.name: n for n in graph.nodes}
        assert nodes["relu"].args[0] is _get_placeholder(graph)

    def test_copy_flag_preserves_noop_cast(self):
        """A same-dtype cast with copy=True is intentional and must survive.

        source(fp32) -> cast(fp32->fp32, copy=True) -> relu
        =>
        (unchanged)
        """
        graph = torch.fx.Graph()
        pass_inserted: set[torch.fx.Node] = set()

        source = graph.placeholder("source")
        source.meta["val"] = _make_fake_val(torch.float32)
        c1 = graph.create_node(
            "call_function",
            torch.ops.aten._to_copy.default,
            args=(source,),
            kwargs={"dtype": torch.float32, "copy": True},
            name="copy_cast",
        )
        c1.meta["val"] = _make_fake_val(torch.float32)
        relu = _add_relu(graph, c1, "relu")
        graph.output(relu)

        cleanup_casts(graph, pass_inserted)
        graph.eliminate_dead_code()

        remaining = _get_cast_nodes(graph)
        assert len(remaining) == 1
        assert remaining[0].kwargs.get("copy") is True


# =============================================================================
# Deduplication tests
# =============================================================================
class TestDeduplicateCasts:
    """Test deduplication of identical cast nodes."""

    def test_duplicate_pass_inserted_casts_merged(self):
        """Two pass-inserted casts with same input and dtype are deduplicated.

        source(fp32) -+-> cast_1(fp32->fp16) -> relu_1
                       +-> cast_2(fp32->fp16) -> relu_2
        =>
        source(fp32) -> cast(fp32->fp16) -+-> relu_1
                                           +-> relu_2
        """
        graph = torch.fx.Graph()
        pass_inserted: set[torch.fx.Node] = set()

        source = graph.placeholder("source")
        source.meta["val"] = _make_fake_val(torch.float32)
        c1 = _add_cast(graph, source, torch.float16, "cast_1", pass_inserted)
        c2 = _add_cast(graph, source, torch.float16, "cast_2", pass_inserted)
        relu_1 = _add_relu(graph, c1, "relu_1")
        relu_2 = _add_relu(graph, c2, "relu_2")
        graph.output((relu_1, relu_2))

        cleanup_casts(graph, pass_inserted)
        graph.eliminate_dead_code()

        remaining = _get_cast_nodes(graph)
        assert len(remaining) == 1
        # Both relus should share the same cast
        nodes = {n.name: n for n in graph.nodes}
        assert nodes["relu_1"].args[0] is remaining[0]
        assert nodes["relu_2"].args[0] is remaining[0]

    def test_original_narrowing_casts_not_deduplicated(self):
        """Two original narrowing casts are not eligible for dedup.

        source(fp32) -+-> orig_cast_1(fp32->fp16) -> relu_1
                       +-> orig_cast_2(fp32->fp16) -> relu_2
        =>
        (unchanged — both are original narrowing, not chain-includable)
        """
        graph = torch.fx.Graph()
        pass_inserted: set[torch.fx.Node] = set()

        source = graph.placeholder("source")
        source.meta["val"] = _make_fake_val(torch.float32)
        c1 = _add_cast(graph, source, torch.float16, "orig_cast_1")  # original, narrowing
        c2 = _add_cast(graph, source, torch.float16, "orig_cast_2")  # original, narrowing
        relu_1 = _add_relu(graph, c1, "relu_1")
        relu_2 = _add_relu(graph, c2, "relu_2")
        graph.output((relu_1, relu_2))

        cleanup_casts(graph, pass_inserted)
        graph.eliminate_dead_code()

        remaining = _get_cast_nodes(graph)
        assert len(remaining) == 2

    def test_original_widening_casts_deduplicated(self):
        """Two original widening casts with same input/dtype are deduplicated.

        source(fp16) -+-> orig_cast_1(fp16->fp32) -> relu_1
                       +-> orig_cast_2(fp16->fp32) -> relu_2
        =>
        source(fp16) -> orig_cast(fp16->fp32) -+-> relu_1
                                                +-> relu_2
        """
        graph = torch.fx.Graph()
        pass_inserted: set[torch.fx.Node] = set()

        source = graph.placeholder("source")
        source.meta["val"] = _make_fake_val(torch.float16)
        c1 = _add_cast(graph, source, torch.float32, "orig_cast_1")  # original, widening
        c2 = _add_cast(graph, source, torch.float32, "orig_cast_2")  # original, widening
        relu_1 = _add_relu(graph, c1, "relu_1")
        relu_2 = _add_relu(graph, c2, "relu_2")
        graph.output((relu_1, relu_2))

        cleanup_casts(graph, pass_inserted)
        graph.eliminate_dead_code()

        remaining = _get_cast_nodes(graph)
        assert len(remaining) == 1
        nodes = {n.name: n for n in graph.nodes}
        assert nodes["relu_1"].args[0] is remaining[0]
        assert nodes["relu_2"].args[0] is remaining[0]


# =============================================================================
# FP16 scalar casting tests
# =============================================================================
_FP16_MAX = float(np.finfo(np.float16).max)  # 65504.0
_FP16_MIN = float(np.nextafter(0.0, 1.0, dtype=np.float16))  # ~5.96e-08


class TestCastScalarToFp16:
    """Test cast_scalar_to_fp16 across its code paths."""

    @pytest.mark.parametrize(
        "val, expected",
        [
            (0.0, 0.0),
            (1.0, 1.0),
            (-1.0, -1.0),
            (0.5, 0.5),
        ],
        ids=["zero", "one", "neg_one", "half"],
    )
    def test_normal_values_rounded(self, val, expected):
        assert cast_scalar_to_fp16(val) == expected

    @pytest.mark.parametrize(
        "val",
        [70000.0, -70000.0, _FP16_MAX + 1.0],
        ids=["positive", "negative", "just_over_max"],
    )
    def test_overflow_returns_none(self, val):
        """Values in (FP16_MAX, 1e38) are unrepresentable -> None."""
        assert cast_scalar_to_fp16(val) is None

    @pytest.mark.parametrize(
        "val, expected_sign",
        [(1e38, 1), (-1e38, -1), (float("inf"), 1), (float("-inf"), -1)],
        ids=["pos_inf_like", "neg_inf_like", "pos_inf", "neg_inf"],
    )
    def test_inf_like_clamped_to_fp16_max(self, val, expected_sign):
        """Values >= 1e38 are treated as inf and clamped to ±FP16_MAX."""
        result = cast_scalar_to_fp16(val)
        assert result == expected_sign * _FP16_MAX

    @pytest.mark.parametrize(
        "val, expected",
        [(1e-10, _FP16_MIN), (-1e-10, -_FP16_MIN)],
        ids=["pos_underflow", "neg_underflow"],
    )
    def test_underflow_snapped_to_fp16_min(self, val, expected):
        """Near-zero values snapped to ±FP16_MIN."""
        assert cast_scalar_to_fp16(val) == expected


# =============================================================================
# FP16 tensor overflow tests
# =============================================================================
class TestCheckTensorOverflowFp16:
    """Test check_tensor_overflow_fp16."""

    @pytest.mark.parametrize(
        "tensor, expected",
        [
            (torch.tensor([1.0, 100.0, 65504.0]), False),
            (torch.tensor([1.0, 70000.0]), True),
            (torch.tensor([float("inf"), float("-inf")]), False),
            (torch.tensor([1.0], dtype=torch.float16), False),
            (torch.tensor([70000.0, float("inf")]), True),
        ],
        ids=["in_range", "overflow", "all_inf", "non_fp32", "mixed_overflow_and_inf"],
    )
    def test_overflow_detection(self, tensor, expected):
        assert check_tensor_overflow_fp16(tensor) == expected


# =============================================================================
# FP16 tensor casting tests
# =============================================================================
class TestCastTensorToFp16:
    """Test cast_tensor_to_fp16 underflow correction and casting."""

    def test_normal_values_cast_correctly(self):
        """Values well within fp16 range are cast without modification."""
        tensor = torch.tensor([1.0, -1.0, 0.5, 100.0])
        result = cast_tensor_to_fp16(tensor)
        assert result.dtype == torch.float16
        torch.testing.assert_close(result, tensor.half())

    def test_zero_preserved(self):
        """Exact zeros stay zero (not snapped to FP16_MIN)."""
        tensor = torch.tensor([0.0, 0.0])
        result = cast_tensor_to_fp16(tensor)
        assert (result == 0).all()

    @pytest.mark.parametrize(
        "val, expected_sign",
        [(1e-10, 1), (-1e-10, -1), (1e-40, 1), (-1e-40, -1)],
        ids=["small_pos", "small_neg", "tiny_pos", "tiny_neg"],
    )
    def test_underflow_snapped_to_fp16_min(self, val, expected_sign):
        """Near-zero nonzero values are snapped to ±FP16_MIN instead of zero."""
        tensor = torch.tensor([val])
        result = cast_tensor_to_fp16(tensor)
        assert result.item() != 0.0, "Should not underflow to zero"
        assert (result.item() > 0) == (expected_sign > 0), "Sign should be preserved"

    def test_mixed_normal_and_underflow(self):
        """Tensor with both normal and underflow values handles each correctly."""
        normal_val = 1.0
        underflow_val = 1e-10
        tensor = torch.tensor([normal_val, underflow_val, 0.0, -underflow_val])
        result = cast_tensor_to_fp16(tensor)

        assert result[0] == torch.tensor(normal_val).half()  # normal: cast normally
        assert result[1] != 0.0  # underflow: snapped, not zero
        assert result[2] == 0.0  # zero: stays zero
        assert result[3] != 0.0  # negative underflow: snapped, not zero

    def test_original_tensor_not_mutated(self):
        """The input tensor should not be modified in place."""
        tensor = torch.tensor([1e-10, 1.0])
        original = tensor.clone()
        cast_tensor_to_fp16(tensor)
        torch.testing.assert_close(tensor, original)


# =============================================================================
# classify_float_args tests
# =============================================================================
class TestClassifyFloatArgs:
    """Test classify_float_args detection of float inputs and overflow."""

    @staticmethod
    def _make_node_with_args(*args, **kwargs) -> torch.fx.Node:
        """Create a minimal FX node with the given args/kwargs."""
        graph = torch.fx.Graph()
        # Create placeholder nodes for any torch.fx.Node args we need
        node_args = []
        for arg in args:
            if isinstance(arg, torch.dtype):
                p = graph.placeholder("p")
                p.meta["val"] = _make_fake_val(arg)
                node_args.append(p)
            else:
                node_args.append(arg)
        node = graph.create_node(
            "call_function", torch.ops.aten.relu.default, args=tuple(node_args), kwargs=kwargs
        )
        node.meta["val"] = _make_fake_val(torch.float32)
        graph.output(node)
        return node

    @pytest.mark.parametrize(
        "args, expected",
        [
            # (has_float_input, has_overflow_or_fp32_node)
            ((torch.float16,), (True, False)),
            ((torch.float32,), (True, True)),
            ((torch.int32,), (False, False)),
            ((torch.float16, 1.0), (True, False)),
            ((torch.float16, 70000.0), (True, True)),
            ((torch.float16, 1e38), (True, False)),  # inf-like, not overflow
        ],
        ids=[
            "fp16_node",
            "fp32_node",
            "int_node_no_float",
            "fp16_node_and_normal_scalar",
            "fp16_node_and_overflow_scalar",
            "fp16_node_and_inf_scalar",
        ],
    )
    def test_classification(self, args, expected):
        node = self._make_node_with_args(*args)
        assert classify_float_args(node) == expected


# =============================================================================
# build_unsafe_to_cast_nodes tests
# =============================================================================
class TestBuildUnsafeToCastNodes:
    """Test that index-feeding nodes are correctly marked unsafe."""

    @pytest.mark.parametrize(
        "index_op, index_arg_pos",
        [
            (torch.ops.aten.embedding.default, 1),
            (torch.ops.aten.gather.default, 2),
            (torch.ops.aten.index_select.default, 2),
            (torch.ops.aten.scatter.src, 2),
            (torch.ops.aten.scatter.value, 2),
            (torch.ops.aten.scatter_add.default, 2),
            (torch.ops.aten.scatter_reduce.two, 2),
            (torch.ops.aten.index_copy.default, 2),
            (torch.ops.aten.index_add.default, 2),
            (torch.ops.aten.index_fill.int_Scalar, 2),
            (torch.ops.aten.index_fill.int_Tensor, 2),
            (torch.ops.aten.take_along_dim.default, 1),
        ],
        ids=[
            "embedding",
            "gather",
            "index_select",
            "scatter_src",
            "scatter_value",
            "scatter_add",
            "scatter_reduce",
            "index_copy",
            "index_add",
            "index_fill_scalar",
            "index_fill_tensor",
            "take_along_dim",
        ],
    )
    def test_index_arg_marked_unsafe(self, index_op, index_arg_pos):
        """The index argument and its upstream chain are marked unsafe."""
        graph = torch.fx.Graph()

        data = graph.placeholder("data")
        data.meta["val"] = _make_fake_val(torch.float32, (10, 4))

        index_source = graph.placeholder("index_source")
        index_source.meta["val"] = _make_fake_val(torch.int32, (3,))

        # Add an intermediate op upstream of the index to test transitivity
        intermediate = graph.create_node(
            "call_function", torch.ops.aten.relu.default, args=(index_source,), name="intermediate"
        )
        intermediate.meta["val"] = _make_fake_val(torch.int32, (3,))

        # Build args: pad with None/int for the non-index positions
        args: list = [None] * (index_arg_pos + 1)
        args[0] = data
        if index_arg_pos == 2:
            args[1] = 0  # dim argument for gather/index_select
        args[index_arg_pos] = intermediate

        op_node = graph.create_node("call_function", index_op, args=tuple(args), name="index_op")
        op_node.meta["val"] = _make_fake_val(torch.float32, (3, 4))
        graph.output(op_node)

        unsafe = build_unsafe_to_cast_nodes(graph)

        # intermediate and index_source should be unsafe, data should not
        assert intermediate in unsafe
        assert index_source in unsafe
        assert data not in unsafe

    def test_index_tensor_list_args_marked_unsafe(self):
        """index.Tensor: all index tensors in arg[1] list are marked unsafe."""
        graph = torch.fx.Graph()

        data = graph.placeholder("data")
        data.meta["val"] = _make_fake_val(torch.float32, (10, 10))

        idx_0 = graph.placeholder("idx_0")
        idx_0.meta["val"] = _make_fake_val(torch.int64, (3,))

        idx_1 = graph.placeholder("idx_1")
        idx_1.meta["val"] = _make_fake_val(torch.int64, (3,))

        op_node = graph.create_node(
            "call_function",
            torch.ops.aten.index.Tensor,
            args=(data, [idx_0, idx_1]),
            name="index_tensor",
        )
        op_node.meta["val"] = _make_fake_val(torch.float32, (3,))
        graph.output(op_node)

        unsafe = build_unsafe_to_cast_nodes(graph)

        assert idx_0 in unsafe
        assert idx_1 in unsafe
        assert data not in unsafe

    def test_index_put_list_args_marked_unsafe(self):
        """index_put: all index tensors in arg[1] list are marked unsafe."""
        graph = torch.fx.Graph()

        data = graph.placeholder("data")
        data.meta["val"] = _make_fake_val(torch.float32, (10, 10))

        idx_0 = graph.placeholder("idx_0")
        idx_0.meta["val"] = _make_fake_val(torch.int64, (3,))

        values = graph.placeholder("values")
        values.meta["val"] = _make_fake_val(torch.float32, (3,))

        op_node = graph.create_node(
            "call_function",
            torch.ops.aten.index_put.default,
            args=(data, [idx_0], values),
            name="index_put",
        )
        op_node.meta["val"] = _make_fake_val(torch.float32, (10, 10))
        graph.output(op_node)

        unsafe = build_unsafe_to_cast_nodes(graph)

        assert idx_0 in unsafe
        assert data not in unsafe
        assert values not in unsafe


# =============================================================================
# insert_cast_after tests
# =============================================================================
class TestInsertCastAfter:
    """Test insert_cast_after rewiring behavior."""

    def test_rewires_all_users(self):
        """insert_cast_after should insert a cast and rewire all users to it."""
        graph = torch.fx.Graph()
        pass_inserted: set[torch.fx.Node] = set()

        source = graph.placeholder("source")
        source.meta["val"] = _make_fake_val(torch.float32)
        relu_1 = _add_relu(graph, source, "relu_1")
        relu_2 = _add_relu(graph, source, "relu_2")
        graph.output((relu_1, relu_2))

        insert_cast_after(graph, source, torch.float16, pass_inserted=pass_inserted)

        # Both relus should now read from the cast, not source
        casts = _get_cast_nodes(graph)
        assert len(casts) == 1
        assert casts[0].kwargs.get("dtype") == torch.float16
        nodes = {n.name: n for n in graph.nodes}
        assert nodes["relu_1"].args[0] is casts[0]
        assert nodes["relu_2"].args[0] is casts[0]

    def test_noop_when_no_users(self):
        """insert_cast_after should do nothing when node has no users."""
        graph = torch.fx.Graph()
        source = graph.placeholder("source")
        source.meta["val"] = _make_fake_val(torch.float32)
        graph.output(source)

        # source's only "user" is the output node, but let's create a truly unused node
        unused = graph.create_node(
            "call_function", torch.ops.aten.relu.default, args=(source,), name="unused"
        )
        unused.meta["val"] = _make_fake_val(torch.float32)
        # Remove the user link by replacing unused's output with source in a new output
        # Actually, just test with source directly — output is a user though.
        # Create a node with no users by making it, then removing its user.
        graph2 = torch.fx.Graph()
        s = graph2.placeholder("s")
        s.meta["val"] = _make_fake_val(torch.float32)
        leaf = graph2.create_node(
            "call_function", torch.ops.aten.relu.default, args=(s,), name="leaf"
        )
        leaf.meta["val"] = _make_fake_val(torch.float32)
        graph2.output(s)  # leaf has no users

        insert_cast_after(graph2, leaf, torch.float16)
        assert len(_get_cast_nodes(graph2)) == 0


# =============================================================================
# anchor_int16_reshape_inputs tests
# =============================================================================
class TestAnchorInt16ReshapeInputs:
    """Test anchor_int16_reshape_inputs inserts anchors in the right places."""

    def test_anchor_inserted_after_compute_op(self):
        """A reshape fed by a compute op (e.g. mul) with int16 metadata gets an anchor.

        source(int16) -> mul(int16) -> reshape(int16)
        =>
        source(int16) -> mul(int16) -> anchor_cast(int16) -> reshape(int16)
        """
        graph = torch.fx.Graph()
        pass_inserted: set[torch.fx.Node] = set()

        source = graph.placeholder("source")
        source.meta["val"] = _make_fake_val(torch.int16, (2, 3))

        mul_node = graph.create_node(
            "call_function",
            torch.ops.aten.mul.Tensor,
            args=(source, source),
            name="mul",
        )
        mul_node.meta["val"] = _make_fake_val(torch.int16, (2, 3))

        reshape = graph.create_node(
            "call_function",
            torch.ops.aten.reshape.default,
            args=(mul_node, [6]),
            name="reshape",
        )
        reshape.meta["val"] = _make_fake_val(torch.int16, (6,))
        graph.output(reshape)

        count = anchor_int16_reshape_inputs(graph, pass_inserted)

        assert count == 1
        casts = _get_cast_nodes(graph)
        assert len(casts) == 1
        assert casts[0].kwargs.get("dtype") == torch.int16
        assert casts[0].all_input_nodes[0] is mul_node
        assert reshape.args[0] is casts[0]

    def test_no_anchor_for_placeholder_input(self):
        """A reshape fed directly by a placeholder does not get an anchor."""
        graph = torch.fx.Graph()
        pass_inserted: set[torch.fx.Node] = set()

        source = graph.placeholder("source")
        source.meta["val"] = _make_fake_val(torch.int16, (2, 3))

        reshape = graph.create_node(
            "call_function",
            torch.ops.aten.reshape.default,
            args=(source, [6]),
            name="reshape",
        )
        reshape.meta["val"] = _make_fake_val(torch.int16, (6,))
        graph.output(reshape)

        count = anchor_int16_reshape_inputs(graph, pass_inserted)
        assert count == 0
        assert len(_get_cast_nodes(graph)) == 0

    def test_no_anchor_for_reshape_chain(self):
        """A reshape fed by another reshape does not get an anchor."""
        graph = torch.fx.Graph()
        pass_inserted: set[torch.fx.Node] = set()

        source = graph.placeholder("source")
        source.meta["val"] = _make_fake_val(torch.int16, (2, 3))

        reshape_1 = graph.create_node(
            "call_function",
            torch.ops.aten.reshape.default,
            args=(source, [6]),
            name="reshape_1",
        )
        reshape_1.meta["val"] = _make_fake_val(torch.int16, (6,))

        reshape_2 = graph.create_node(
            "call_function",
            torch.ops.aten.reshape.default,
            args=(reshape_1, [2, 3]),
            name="reshape_2",
        )
        reshape_2.meta["val"] = _make_fake_val(torch.int16, (2, 3))
        graph.output(reshape_2)

        count = anchor_int16_reshape_inputs(graph, pass_inserted)
        assert count == 0
        assert len(_get_cast_nodes(graph)) == 0

    def test_no_anchor_when_already_cast(self):
        """A reshape fed by an explicit int16 cast does not get a redundant anchor."""
        graph = torch.fx.Graph()
        pass_inserted: set[torch.fx.Node] = set()

        source = graph.placeholder("source")
        source.meta["val"] = _make_fake_val(torch.int32, (2, 3))

        cast_node = _add_cast(graph, source, torch.int16, "explicit_cast", pass_inserted)

        reshape = graph.create_node(
            "call_function",
            torch.ops.aten.reshape.default,
            args=(cast_node, [6]),
            name="reshape",
        )
        reshape.meta["val"] = _make_fake_val(torch.int16, (6,))
        graph.output(reshape)

        count = anchor_int16_reshape_inputs(graph, pass_inserted)
        assert count == 0
        # Only the explicit cast exists, no anchor added
        assert len(_get_cast_nodes(graph)) == 1

    def test_no_anchor_for_non_int16_reshape(self):
        """A reshape with fp32 output does not get an anchor."""
        graph = torch.fx.Graph()
        pass_inserted: set[torch.fx.Node] = set()

        source = graph.placeholder("source")
        source.meta["val"] = _make_fake_val(torch.float32, (2, 3))

        mul_node = graph.create_node(
            "call_function",
            torch.ops.aten.mul.Tensor,
            args=(source, source),
            name="mul",
        )
        mul_node.meta["val"] = _make_fake_val(torch.float32, (2, 3))

        reshape = graph.create_node(
            "call_function",
            torch.ops.aten.reshape.default,
            args=(mul_node, [6]),
            name="reshape",
        )
        reshape.meta["val"] = _make_fake_val(torch.float32, (6,))
        graph.output(reshape)

        count = anchor_int16_reshape_inputs(graph, pass_inserted)
        assert count == 0
        assert len(_get_cast_nodes(graph)) == 0


# =============================================================================
# build_castable_int16_nodes tests
# =============================================================================
class TestBuildCastableInt16Nodes:
    """Test that castable candidate set is built correctly from computation ops + propagation."""

    def test_computation_op_included(self):
        """An integer add downstream of a user input is castable."""
        graph = torch.fx.Graph()

        user_input = graph.placeholder("user_input")
        user_input.meta["val"] = _make_fake_val(torch.int32, (3,))

        add_node = graph.create_node(
            "call_function",
            torch.ops.aten.add.Tensor,
            args=(user_input, user_input),
            name="add",
        )
        add_node.meta["val"] = _make_fake_val(torch.int32, (3,))
        graph.output(add_node)

        user_reachable = build_user_reachable_nodes([user_input])
        unsafe = build_unsafe_to_cast_nodes(graph)
        castable = build_castable_int16_nodes(graph, user_reachable, unsafe)

        assert add_node in castable

    def test_view_op_propagated_from_computation_op(self):
        """A reshape between two computation ops is included via propagation."""
        graph = torch.fx.Graph()

        user_input = graph.placeholder("user_input")
        user_input.meta["val"] = _make_fake_val(torch.int32, (2, 3))

        add_node = graph.create_node(
            "call_function",
            torch.ops.aten.add.Tensor,
            args=(user_input, user_input),
            name="add",
        )
        add_node.meta["val"] = _make_fake_val(torch.int32, (2, 3))

        reshape = graph.create_node(
            "call_function",
            torch.ops.aten.reshape.default,
            args=(add_node, [6]),
            name="reshape",
        )
        reshape.meta["val"] = _make_fake_val(torch.int32, (6,))

        mul_node = graph.create_node(
            "call_function",
            torch.ops.aten.mul.Tensor,
            args=(reshape, reshape),
            name="mul",
        )
        mul_node.meta["val"] = _make_fake_val(torch.int32, (6,))
        graph.output(mul_node)

        user_reachable = build_user_reachable_nodes([user_input])
        unsafe = build_unsafe_to_cast_nodes(graph)
        castable = build_castable_int16_nodes(graph, user_reachable, unsafe)

        assert add_node in castable
        assert reshape in castable
        assert mul_node in castable

    def test_isolated_view_op_not_castable(self):
        """A reshape with no adjacent computation op is NOT castable."""
        graph = torch.fx.Graph()

        user_input = graph.placeholder("user_input")
        user_input.meta["val"] = _make_fake_val(torch.int32, (2, 3))

        reshape = graph.create_node(
            "call_function",
            torch.ops.aten.reshape.default,
            args=(user_input, [6]),
            name="reshape",
        )
        reshape.meta["val"] = _make_fake_val(torch.int32, (6,))
        graph.output(reshape)

        user_reachable = build_user_reachable_nodes([user_input])
        unsafe = build_unsafe_to_cast_nodes(graph)
        castable = build_castable_int16_nodes(graph, user_reachable, unsafe)

        assert reshape not in castable

    def test_multi_view_chain_without_computation_op_not_castable(self):
        """A chain of view ops with no computation op anywhere is NOT castable."""
        graph = torch.fx.Graph()

        user_input = graph.placeholder("user_input")
        user_input.meta["val"] = _make_fake_val(torch.int32, (2, 3))

        reshape = graph.create_node(
            "call_function",
            torch.ops.aten.reshape.default,
            args=(user_input, [6]),
            name="reshape",
        )
        reshape.meta["val"] = _make_fake_val(torch.int32, (6,))

        unsqueeze = graph.create_node(
            "call_function",
            torch.ops.aten.unsqueeze.default,
            args=(reshape, 0),
            name="unsqueeze",
        )
        unsqueeze.meta["val"] = _make_fake_val(torch.int32, (1, 6))
        graph.output(unsqueeze)

        user_reachable = build_user_reachable_nodes([user_input])
        unsafe = build_unsafe_to_cast_nodes(graph)
        castable = build_castable_int16_nodes(graph, user_reachable, unsafe)

        assert reshape not in castable
        assert unsqueeze not in castable

    def test_constant_foldable_computation_op_not_castable(self):
        """An add op in a constant-foldable subgraph (no user input ancestry) is NOT castable."""
        graph = torch.fx.Graph()

        user_input = graph.placeholder("user_input")
        user_input.meta["val"] = _make_fake_val(torch.float32, (3,))

        arange = graph.create_node(
            "call_function",
            torch.ops.aten.arange.default,
            args=(5,),
            name="arange",
        )
        arange.meta["val"] = _make_fake_val(torch.int64, (5,))

        const_add = graph.create_node(
            "call_function",
            torch.ops.aten.add.Tensor,
            args=(arange, arange),
            name="const_add",
        )
        const_add.meta["val"] = _make_fake_val(torch.int64, (5,))
        graph.output((user_input, const_add))

        user_reachable = build_user_reachable_nodes([user_input])
        unsafe = build_unsafe_to_cast_nodes(graph)
        castable = build_castable_int16_nodes(graph, user_reachable, unsafe)

        assert const_add not in castable

    def test_unsafe_computation_op_not_castable(self):
        """A computation op in the unsafe set (feeds embedding index) is NOT castable."""
        graph = torch.fx.Graph()

        user_input = graph.placeholder("user_input")
        user_input.meta["val"] = _make_fake_val(torch.int32, (3,))

        add_node = graph.create_node(
            "call_function",
            torch.ops.aten.add.Tensor,
            args=(user_input, user_input),
            name="add",
        )
        add_node.meta["val"] = _make_fake_val(torch.int32, (3,))

        weight = graph.placeholder("weight")
        weight.meta["val"] = _make_fake_val(torch.float32, (100, 16))

        embedding = graph.create_node(
            "call_function",
            torch.ops.aten.embedding.default,
            args=(weight, add_node),
            name="embedding",
        )
        embedding.meta["val"] = _make_fake_val(torch.float32, (3, 16))
        graph.output(embedding)

        user_reachable = build_user_reachable_nodes([user_input])
        unsafe = build_unsafe_to_cast_nodes(graph)
        castable = build_castable_int16_nodes(graph, user_reachable, unsafe)

        assert add_node not in castable

    def test_non_integer_computation_op_not_castable(self):
        """A float add is NOT a computation op (only integer arithmetic qualifies)."""
        graph = torch.fx.Graph()

        user_input = graph.placeholder("user_input")
        user_input.meta["val"] = _make_fake_val(torch.float32, (3,))

        add_node = graph.create_node(
            "call_function",
            torch.ops.aten.add.Tensor,
            args=(user_input, user_input),
            name="add",
        )
        add_node.meta["val"] = _make_fake_val(torch.float32, (3,))
        graph.output(add_node)

        user_reachable = build_user_reachable_nodes([user_input])
        unsafe = build_unsafe_to_cast_nodes(graph)
        castable = build_castable_int16_nodes(graph, user_reachable, unsafe)

        assert add_node not in castable

    def test_user_input_placeholder_castable_when_feeds_computation_op(self):
        """A user input placeholder feeding a computation op is included."""
        graph = torch.fx.Graph()

        user_input = graph.placeholder("user_input")
        user_input.meta["val"] = _make_fake_val(torch.int32, (3,))

        add_node = graph.create_node(
            "call_function",
            torch.ops.aten.add.Tensor,
            args=(user_input, user_input),
            name="add",
        )
        add_node.meta["val"] = _make_fake_val(torch.int32, (3,))
        graph.output(add_node)

        user_reachable = build_user_reachable_nodes([user_input])
        unsafe = build_unsafe_to_cast_nodes(graph)
        castable = build_castable_int16_nodes(graph, user_reachable, unsafe)

        assert user_input in castable

    def test_passthrough_placeholder_not_castable(self):
        """A user input that passes through untouched (no computation op) is NOT castable."""
        graph = torch.fx.Graph()

        user_input = graph.placeholder("user_input")
        user_input.meta["val"] = _make_fake_val(torch.int32, (3,))

        graph.output(user_input)

        user_reachable = build_user_reachable_nodes([user_input])
        unsafe = build_unsafe_to_cast_nodes(graph)
        castable = build_castable_int16_nodes(graph, user_reachable, unsafe)

        assert user_input not in castable

    def test_scalar_operand_computation_op_not_castable(self):
        """An add(tensor, scalar) cannot run in int16 and is NOT castable."""
        graph = torch.fx.Graph()

        user_input = graph.placeholder("user_input")
        user_input.meta["val"] = _make_fake_val(torch.int32, (3,))

        # add with scalar operand: add(tensor, 1)
        add_node = graph.create_node(
            "call_function",
            torch.ops.aten.add.Tensor,
            args=(user_input, 1),
            name="add_scalar",
        )
        add_node.meta["val"] = _make_fake_val(torch.int32, (3,))
        graph.output(add_node)

        user_reachable = build_user_reachable_nodes([user_input])
        unsafe = build_unsafe_to_cast_nodes(graph)
        castable = build_castable_int16_nodes(graph, user_reachable, unsafe)

        assert add_node not in castable

    def test_scalar_operand_blocks_propagation(self):
        """Propagation stops at a computation op with a scalar operand.

        user_input -> add(tensor, tensor) -> mul(tensor, 2) -> reshape
        The mul(tensor, 2) has a scalar so it blocks propagation.
        reshape should not be castable.
        """
        graph = torch.fx.Graph()

        user_input = graph.placeholder("user_input")
        user_input.meta["val"] = _make_fake_val(torch.int32, (2, 3))

        add_node = graph.create_node(
            "call_function",
            torch.ops.aten.add.Tensor,
            args=(user_input, user_input),
            name="add",
        )
        add_node.meta["val"] = _make_fake_val(torch.int32, (2, 3))

        # mul with scalar — blocks propagation
        mul_node = graph.create_node(
            "call_function",
            torch.ops.aten.mul.Tensor,
            args=(add_node, 2),
            name="mul_scalar",
        )
        mul_node.meta["val"] = _make_fake_val(torch.int32, (2, 3))

        reshape = graph.create_node(
            "call_function",
            torch.ops.aten.reshape.default,
            args=(mul_node, [6]),
            name="reshape",
        )
        reshape.meta["val"] = _make_fake_val(torch.int32, (6,))
        graph.output(reshape)

        user_reachable = build_user_reachable_nodes([user_input])
        unsafe = build_unsafe_to_cast_nodes(graph)
        castable = build_castable_int16_nodes(graph, user_reachable, unsafe)

        assert add_node in castable
        assert mul_node not in castable
        assert reshape not in castable


# =============================================================================
# is_ignored_op tests
# =============================================================================
class TestIsIgnoredOp:
    """Tests for the is_ignored_op helper."""

    @pytest.fixture
    def exp_node(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        node = graph.create_node(
            "call_function",
            torch.ops.aten.exp.default,
            args=(x,),
            name="exp_1",
        )
        return node

    @pytest.fixture
    def add_node(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        node = graph.create_node(
            "call_function",
            torch.ops.aten.add.Tensor,
            args=(x, x),
            name="add_1",
        )
        return node

    def test_empty_or_none_ignored_ops(self, exp_node):
        assert not is_ignored_op(exp_node, None)
        assert not is_ignored_op(exp_node, [])
        assert not is_ignored_op(exp_node, set())

    def test_match_op_overload(self, exp_node, add_node):
        assert is_ignored_op(exp_node, {torch.ops.aten.exp.default})
        assert is_ignored_op(exp_node, [torch.ops.aten.exp.default])
        assert not is_ignored_op(add_node, {torch.ops.aten.exp.default})

    def test_match_op_overload_packet(self, exp_node, add_node):
        assert is_ignored_op(exp_node, {torch.ops.aten.exp})
        assert is_ignored_op(exp_node, [torch.ops.aten.exp])
        assert not is_ignored_op(add_node, {torch.ops.aten.exp})

    def test_match_single_op_instance(self, exp_node, add_node):
        assert is_ignored_op(exp_node, torch.ops.aten.exp)
        assert is_ignored_op(exp_node, torch.ops.aten.exp.default)
        assert not is_ignored_op(add_node, torch.ops.aten.exp)

    def test_match_predicate_function(self, exp_node, add_node):
        # Match by node name via custom predicate
        assert is_ignored_op(exp_node, lambda node: node.name == "exp_1")
        assert not is_ignored_op(exp_node, lambda node: node.name == "exp_2")
        assert not is_ignored_op(add_node, lambda node: node.name == "exp_1")

        # Match by target inspection via custom predicate
        assert is_ignored_op(exp_node, lambda node: node.target == torch.ops.aten.exp.default)
        assert not is_ignored_op(add_node, lambda node: node.target == torch.ops.aten.exp.default)
