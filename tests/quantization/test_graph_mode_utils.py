# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from functools import wraps

import torch
import torch.nn as nn

from coreai_opt.quantization import Quantizer, QuantizerConfig
from coreai_opt.quantization._graph._utils import (
    _has_no_disallowed_kwargs,
    restore_kwargs,
    strip_non_aten_metadata_kwargs,
)


def test_quantization_with_wraps_decorator():
    """Verify quantization with @wraps decorator and module exclusion."""

    def casting_decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapper

    class TwoLayerModelWithWraps(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 10, bias=False)
            self.linear2 = nn.Linear(10, 10, bias=False)

        @casting_decorator
        def forward(self, x):
            x = self.linear1(x)
            x = torch.relu(x)
            x = self.linear2(x)
            return x

    model = TwoLayerModelWithWraps()
    example_inputs = (torch.randn(1, 10),)

    quantization_config = {
        "global_config": {
            "op_state_spec": {
                "weight": {
                    "dtype": "int4",
                    "qscheme": "symmetric",
                    "granularity": {"type": "per_tensor"},
                    "fake_quantize_cls": "default",
                    "qparam_calculator_cls": "default",
                    "range_calculator_cls": "minmax",
                }
            },
        },
        "module_name_configs": {
            "linear2": None,
        },
    }

    config = QuantizerConfig.from_dict({"quantization_config": quantization_config})
    quantizer = Quantizer(model, config)
    prepared_model = quantizer.prepare(example_inputs)

    # Check that linear2.weight does NOT go through activation_post_process
    linear2_weight_node = None
    for node in prepared_model.graph.nodes:
        if node.op == "get_attr" and node.target == "linear2.weight":
            linear2_weight_node = node
            # Check all users of linear2.weight
            for user in node.users:
                if user.op == "call_module":
                    assert "activation_post_process" not in user.target, (
                        "linear2.weight should NOT go through "
                        "activation_post_process (module is excluded)"
                    )
            break

    # Check that linear2's output does NOT go through activation_post_process
    assert linear2_weight_node is not None, "Should find linear2.weight node"
    for node in prepared_model.graph.nodes:
        if node.op == "call_function" and "linear" in str(node.target):
            # Check if this linear op uses linear2.weight
            if linear2_weight_node in node.all_input_nodes:
                # Check output does NOT go through activation_post_process
                for user in node.users:
                    if user.op == "call_module":
                        assert "activation_post_process" not in user.target, (
                            "linear2 output activation should NOT go through "
                            "activation_post_process (module is excluded)"
                        )
                break


torch.library.define("test_ns::identity", "(Tensor x) -> Tensor")


@torch.library.impl("test_ns::identity", "cpu")
def identity_impl(x):
    return x


def build_graph(aten_kwargs=None, custom_kwargs=None):
    """Build an FX graph with one aten op and one custom op."""
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    aten_node = graph.call_function(torch.ops.aten.relu.default, (x,), aten_kwargs or {})
    custom_node = graph.call_function(
        torch.ops.test_ns.identity.default, (aten_node,), custom_kwargs or {}
    )
    graph.output(custom_node)
    return graph, aten_node, custom_node


class TestHasNoDisallowedKwargs:
    def test_metadata_kwargs(self):
        _, _, node = build_graph(
            custom_kwargs={"s": "val", "i": 42, "f": 3.14, "b": True, "n": None}
        )
        assert _has_no_disallowed_kwargs(node) is True

    def test_node_kwargs(self):
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        aten_node = graph.call_function(torch.ops.aten.relu.default, (x,))
        custom_node = graph.call_function(
            torch.ops.test_ns.identity.default, (aten_node,), {"mask": aten_node}
        )
        graph.output(custom_node)
        assert _has_no_disallowed_kwargs(custom_node) is False

    def test_tensor_kwargs(self):
        _, _, node = build_graph(custom_kwargs={"mask": torch.zeros(1)})
        assert _has_no_disallowed_kwargs(node) is False

    def test_empty_kwargs(self):
        _, _, node = build_graph()
        assert _has_no_disallowed_kwargs(node) is True


class TestStripAndRestoreKwargs:
    def test_strip_skips_aten_ops(self):
        aten_kwargs = {"memory_format": torch.contiguous_format}
        graph, aten_node, _ = build_graph(aten_kwargs=aten_kwargs)

        strip_non_aten_metadata_kwargs(graph)

        assert aten_node.kwargs == aten_kwargs

    def test_strip_skips_non_aten_op_without_kwargs(self):
        graph, _, custom_node = build_graph()

        saved = strip_non_aten_metadata_kwargs(graph)

        assert custom_node.kwargs == {}
        assert custom_node.name not in saved

    def test_strip_skips_node_with_tensor_kwarg(self):
        kwargs_with_tensor = {"name": "query", "mask": torch.zeros(1)}
        graph, _, custom_node = build_graph(custom_kwargs=kwargs_with_tensor)

        saved = strip_non_aten_metadata_kwargs(graph)

        assert custom_node.kwargs == kwargs_with_tensor
        assert custom_node.name not in saved

    def test_strip_and_restore_round_trip(self):
        metadata = {"name": "query", "op_name": "sdpa", "id": "abc", "index": 0}
        graph, _, custom_node = build_graph(custom_kwargs=metadata)

        saved = strip_non_aten_metadata_kwargs(graph)

        assert custom_node.kwargs == {}
        assert custom_node.name in saved
        assert saved[custom_node.name] == metadata

        restore_kwargs(graph, saved)
        assert custom_node.kwargs == metadata
