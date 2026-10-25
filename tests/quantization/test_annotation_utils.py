# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for _annotation_utils."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._ops import OpOverloadPacket
from torchao.quantization.pt2e import (
    WrapperModule,
    find_sequential_partitions as torchao_find_sequential_partitions,
)

from coreai_opt._utils.torch_utils import export_model
from coreai_opt.quantization._graph._annotation_utils import (
    _find_sequential_partitions,
    _get_aten_graph_module_for_pattern,
    match_pattern_with_subgraph_matcher,
)
from coreai_opt.quantization._graph._qspec_constraint_generation import (
    _PASSTHROUGH_OP_OVERLOADS,
)


def _overloadpackets_in_graph(model: nn.Module, example_inputs: tuple) -> set[OpOverloadPacket]:
    """Return the set of OpOverloadPackets for all call_function nodes in the exported graph."""
    exported = torch.export.export(model, example_inputs, strict=False)
    return {
        node.target.overloadpacket
        for node in exported.graph_module.graph.nodes
        if node.op == "call_function" and hasattr(node.target, "overloadpacket")
    }


class _Permute(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 1)


class _Reshape(nn.Module):
    def forward(self, x):
        return x.reshape(-1)


class _Squeeze(nn.Module):
    def forward(self, x):
        return x.squeeze(1)


class _Transpose(nn.Module):
    def forward(self, x):
        return x.transpose(0, 1)


class _Unsqueeze(nn.Module):
    def forward(self, x):
        return x.unsqueeze(0)


class _View(nn.Module):
    def forward(self, x):
        return x.view(-1)


class _Dropout(nn.Module):
    def forward(self, x):
        # training=True ensures the op is not folded away at export time
        return F.dropout(x, p=0.5, training=True)


class _FeatureDropout(nn.Module):
    def forward(self, x):
        # training=True ensures the op is not folded away at export time
        return F.dropout2d(x, p=0.5, training=True)


class _Clone(nn.Module):
    def forward(self, x):
        return x.clone()


class _Expand(nn.Module):
    def forward(self, x):
        return x.expand(3, -1)


class _Select(nn.Module):
    def forward(self, x):
        return x.select(1, 0)


class _Slice(nn.Module):
    def forward(self, x):
        return x[:, :2]


class _T(nn.Module):
    def forward(self, x):
        return x.t()


# Maps each member of _PASSTHROUGH_OP_OVERLOADS to a (model, example_inputs) pair
# that exercises it. If an op is added to _PASSTHROUGH_OP_OVERLOADS without a
# corresponding entry here, test_passthrough_ops_all_covered will fail.
_PASSTHROUGH_OP_TEST_CASES: dict[OpOverloadPacket, tuple[nn.Module, tuple]] = {
    torch.ops.aten.clone: (_Clone(), (torch.randn(1, 4),)),
    torch.ops.aten.dropout: (
        _Dropout(),
        (
            torch.randn(
                4,
            ),
        ),
    ),
    torch.ops.aten.expand: (_Expand(), (torch.randn(1, 4),)),
    torch.ops.aten.feature_dropout: (_FeatureDropout(), (torch.randn(1, 4, 4, 4),)),
    torch.ops.aten.permute: (_Permute(), (torch.randn(1, 4, 8),)),
    torch.ops.aten.reshape: (_Reshape(), (torch.randn(1, 4),)),
    torch.ops.aten.select: (_Select(), (torch.randn(1, 4, 4),)),
    torch.ops.aten.slice: (_Slice(), (torch.randn(1, 4),)),
    torch.ops.aten.squeeze: (_Squeeze(), (torch.randn(1, 1, 4),)),
    torch.ops.aten.t: (_T(), (torch.randn(2, 4),)),
    torch.ops.aten.transpose: (_Transpose(), (torch.randn(2, 4),)),
    torch.ops.aten.unsqueeze: (
        _Unsqueeze(),
        (
            torch.randn(
                4,
            ),
        ),
    ),
    torch.ops.aten.view: (_View(), (torch.randn(1, 4),)),
}


def test_passthrough_ops_all_covered():
    """Every op in _PASSTHROUGH_OP_OVERLOADS must have a test case in _PASSTHROUGH_OP_TEST_CASES."""
    assert set(_PASSTHROUGH_OP_TEST_CASES.keys()) == _PASSTHROUGH_OP_OVERLOADS


@pytest.mark.parametrize(
    "overloadpacket, model_inputs",
    _PASSTHROUGH_OP_TEST_CASES.items(),
    ids=list(_PASSTHROUGH_OP_TEST_CASES.keys()),
)
def test_passthrough_op_produces_expected_overloadpacket(overloadpacket, model_inputs):
    """Each passthrough op must lower to its expected ATen OpOverloadPacket after export."""
    model, inputs = model_inputs
    assert overloadpacket in _overloadpackets_in_graph(model, inputs)


def test_subgraph_matcher_requires_weight_in_pattern():
    """A pattern matched with the subgraph matcher must name its weight."""

    def _relu_pattern(input):
        output = F.relu(input)
        return output, {"input": input, "output": output}

    pattern_gm = _get_aten_graph_module_for_pattern(WrapperModule(_relu_pattern), (torch.randn(2),))
    model = export_model(nn.ReLU(), (torch.randn(2),), None, False)
    with pytest.raises(RuntimeError, match='"weight" entry'):
        match_pattern_with_subgraph_matcher(model, pattern_gm)


class TestFindSequentialPartitions:
    """Tests for ``_find_sequential_partitions``."""

    class AddReluMul(nn.Module):
        def forward(self, x1, x2):
            out = torch.relu(x1 + x2) * x2
            return out

    class AddOperatorAndFunction(nn.Module):
        """One add written as ``+`` and one as ``torch.add``: both get ``torch_fn`` "add_1"."""

        def forward(self, x1, x2):
            out = x1 + x2
            out = torch.add(out, x2)
            return out

    @staticmethod
    def _partition_node_names(sequences) -> list[list[list[str]]]:
        """Return the node names of each partition, for each sequence."""
        names = []
        for sequence in sequences:
            sequence_names = []
            for partition in sequence:
                sequence_names.append([node.name for node in partition.nodes])
            names.append(sequence_names)
        return names

    @pytest.mark.parametrize(
        "op_types",
        [["add"], ["add", "relu"], ["relu", "mul"], ["mul"]],
        ids=["add", "add_relu", "relu_mul", "mul"],
    )
    def test_matches_torchao(self, op_types):
        """``_find_sequential_partitions`` mostly returns what torchao's version does.

        They differ only when torch merges unrelated ops into one partition (see
        ``test_torch_fn_name_collision``), and in torchao also matching equivalent types,
        e.g. "add_" for "add", which the op-list patterns list explicitly instead.
        """
        model = export_model(
            self.AddReluMul().eval(), (torch.randn(2, 4), torch.randn(2, 4)), None, False
        )

        expected = self._partition_node_names(torchao_find_sequential_partitions(model, op_types))
        actual = self._partition_node_names(_find_sequential_partitions(model, op_types))

        assert len(actual) == 1
        assert actual == expected

    def test_torch_fn_name_collision(self):
        """Each op gets its own partition, even when two ops share a ``torch_fn`` name."""
        model = export_model(
            self.AddOperatorAndFunction().eval(),
            (torch.randn(2, 4), torch.randn(2, 4)),
            None,
            False,
        )

        sequences = _find_sequential_partitions(model, ["add"])

        # torchao's find_sequential_partitions returns [[["add", "add_1"]]] here: one
        # sequence whose single partition holds both adds, since torch's
        # get_source_partitions groups nodes by torch_fn name.
        assert self._partition_node_names(sequences) == [[["add"]], [["add_1"]]]
