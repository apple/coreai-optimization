# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for _is_state_node in PT2E annotation utils.

Verifies that _is_state_node correctly identifies state nodes (weights,
parameters, compressed weight decompression ops) and rejects activation nodes.
"""

from unittest.mock import Mock

import pytest
import torch

from coreai_opt._utils.fx_utils import is_coreai_compressed_state_node as is_state_node
from tests.test_utils.general import COREAI_AVAILABLE


def _make_node(
    op: str,
    namespace: str | None = None,
    op_name: str | None = None,
    args: tuple = (),
) -> Mock:
    """Create a mock FX node with the attributes needed by _is_state_node.

    Args:
        op (str): The FX node op type (e.g., "get_attr", "call_function").
        namespace (str | None): OpOverload namespace (e.g., "coreai", "aten").
            When provided, creates a mock OpOverload as target.
        op_name (str | None): OpOverload operation name (e.g., "lut_to_dense").
        args (tuple): Node args (other mock nodes or values).

    Returns:
        Mock: A mock node suitable for passing to _is_state_node.
    """
    node = Mock(spec=torch.fx.Node)
    node.op = op
    if namespace is not None:
        target = Mock(spec=torch._ops.OpOverload)
        target.namespace = namespace
        target._opname = op_name
        node.target = target
    else:
        node.target = "some.attr.path"
    node.args = args
    return node


class TestIsStateNode:
    def test_get_attr_is_state(self):
        """get_attr nodes (direct parameter access) are state."""
        node = _make_node("get_attr")
        assert is_state_node(node) is True

    def test_placeholder_is_not_state(self):
        """Placeholder nodes (model inputs) are not state."""
        node = _make_node("placeholder")
        assert is_state_node(node) is False

    def test_lut_to_dense_is_state(self):
        """coreai.lut_to_dense call_function is state (palettized weights)."""
        indices = _make_node("get_attr")
        lut = _make_node("get_attr")
        node = _make_node("call_function", "coreai", "lut_to_dense", args=(indices, lut))
        assert is_state_node(node) is True

    def test_shift_scale_with_lut_input_is_state(self):
        """constexpr_blockwise_shift_scale fed by lut_to_dense is state."""
        indices = _make_node("get_attr")
        lut = _make_node("get_attr")
        lut_node = _make_node("call_function", "coreai", "lut_to_dense", args=(indices, lut))
        scale = _make_node("get_attr")
        node = _make_node(
            "call_function", "coreai", "constexpr_blockwise_shift_scale", args=(lut_node, scale)
        )
        assert is_state_node(node) is True

    def test_shift_scale_is_state(self):
        """constexpr_blockwise_shift_scale is always state. This op is only
        intended for weights.
        """
        data = _make_node("get_attr")
        scale = _make_node("get_attr")
        node = _make_node(
            "call_function", "coreai", "constexpr_blockwise_shift_scale", args=(data, scale)
        )
        assert is_state_node(node) is True

    def test_aten_op_with_all_state_inputs_is_not_state(self):
        """An aten call_function whose inputs are all get_attr is NOT state.

        Only recognize specific coreai ops as state producing not others.
        """
        weight = _make_node("get_attr")
        bias = _make_node("get_attr")
        node = _make_node("call_function", "aten", "add", args=(weight, bias))
        assert is_state_node(node) is False


def _find_coreai_nodes(gm: torch.fx.GraphModule, op_name: str) -> list[torch.fx.Node]:
    """Return all coreai call_function nodes matching the given op name."""
    return [
        node
        for node in gm.graph.nodes
        if (
            node.op == "call_function"
            and isinstance(node.target, torch._ops.OpOverload)
            and node.target.namespace == "coreai"
            and node.target._opname == op_name
        )
    ]


@pytest.mark.skipif(not COREAI_AVAILABLE, reason="Requires coreai")
class TestIsStateNodeIntegration:
    @pytest.mark.seed
    def test_joint_compression_lut_to_dense_not_quantized(
        self, simple_conv_linear_model, simple_model_input
    ):
        """In joint compression (palettize then quantize), lut_to_dense nodes
        must be treated as state and their outputs must not be quantized.

        This is a full end-to-end test: palettize the model, then apply
        activation-only quantization, and verify that lut_to_dense outputs
        do not feed into any quantize op in the final graph.
        """
        from coreai_opt import ExportBackend  # noqa: PLC0415
        from coreai_opt.palettization import KMeansPalettizerConfig  # noqa: PLC0415
        from coreai_opt.palettization.kmeans import KMeansPalettizer  # noqa: PLC0415
        from coreai_opt.quantization import (  # noqa: PLC0415
            ModuleQuantizerConfig,
            Quantizer,
            QuantizerConfig,
        )
        from coreai_opt.quantization.spec import (  # noqa: PLC0415
            default_activation_quantization_spec,
        )

        model = simple_conv_linear_model
        input_data = simple_model_input

        # Palettize
        palettizer = KMeansPalettizer(model, KMeansPalettizerConfig())
        palettizer.prepare((input_data,))
        palettized = palettizer.finalize(backend=ExportBackend.MLIR)

        # Quantize activations only (no weight quantization)
        act_spec = default_activation_quantization_spec()
        quant_config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec=None,
                op_input_spec={"*": act_spec},
                op_output_spec={"*": act_spec},
            )
        )
        quantizer = Quantizer(palettized, quant_config)
        quantizer.prepare((input_data,))
        joint_model = quantizer.finalize(backend=ExportBackend.MLIR)

        # Verify lut_to_dense nodes exist (conv + linear = at least 2)
        lut_nodes = _find_coreai_nodes(joint_model, "lut_to_dense")
        assert len(lut_nodes) >= 2, (
            f"Expected at least 2 lut_to_dense nodes, found {len(lut_nodes)}"
        )

        # Each lut_to_dense must be recognized as state
        for node in lut_nodes:
            assert is_state_node(node) is True, (
                f"lut_to_dense node {node.name} not identified as state"
            )

        # lut_to_dense outputs must NOT feed into quantize ops
        for node in lut_nodes:
            for user in node.users:
                if hasattr(user.target, "name"):
                    user_op_name = user.target.name()
                else:
                    user_op_name = str(user.target)
                assert "quantize" not in user_op_name.lower(), (
                    f"lut_to_dense output feeds into {user_op_name}, but should not be quantized"
                )

    @pytest.mark.seed
    def test_joint_compression_lut_quantized_shift_scale_not_quantized(
        self, simple_conv_linear_model, simple_model_input
    ):
        """When palettization uses int8 LUT quantization, the graph contains
        constexpr_blockwise_shift_scale chained after lut_to_dense. Both ops
        must be identified as state and neither should be quantized.
        """
        from coreai_opt import ExportBackend  # noqa: PLC0415
        from coreai_opt.palettization import (  # noqa: PLC0415
            KMeansPalettizerConfig,
            ModuleKMeansPalettizerConfig,
        )
        from coreai_opt.palettization.kmeans import KMeansPalettizer  # noqa: PLC0415
        from coreai_opt.palettization.spec import PalettizationSpec  # noqa: PLC0415
        from coreai_opt.quantization import (  # noqa: PLC0415
            ModuleQuantizerConfig,
            Quantizer,
            QuantizerConfig,
        )
        from coreai_opt.quantization.spec import (  # noqa: PLC0415
            QuantizationScheme,
            QuantizationSpec,
            default_activation_quantization_spec,
        )

        model = simple_conv_linear_model
        input_data = simple_model_input

        # Palettize with int8 LUT quantization
        palett_spec = PalettizationSpec(
            n_bits=4,
            lut_qspec=QuantizationSpec(
                dtype=torch.int8,
                qscheme=QuantizationScheme.SYMMETRIC,
            ),
        )
        palett_config = KMeansPalettizerConfig(
            global_config=ModuleKMeansPalettizerConfig(
                op_state_spec={"weight": palett_spec},
            ),
        )
        palettizer = KMeansPalettizer(model, palett_config)
        palettizer.prepare((input_data,))
        palettized = palettizer.finalize(backend=ExportBackend.MLIR)

        # Quantize activations only
        act_spec = default_activation_quantization_spec()
        quant_config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec=None,
                op_input_spec={"*": act_spec},
                op_output_spec={"*": act_spec},
            )
        )
        quantizer = Quantizer(palettized, quant_config)
        quantizer.prepare((input_data,))
        joint_model = quantizer.finalize(backend=ExportBackend.MLIR)

        # Verify both op types exist in the graph
        lut_nodes = _find_coreai_nodes(joint_model, "lut_to_dense")
        shift_scale_nodes = _find_coreai_nodes(joint_model, "constexpr_blockwise_shift_scale")
        assert len(lut_nodes) >= 2, (
            f"Expected at least 2 lut_to_dense nodes, found {len(lut_nodes)}"
        )
        assert len(shift_scale_nodes) >= 2, (
            f"Expected at least 2 constexpr_blockwise_shift_scale nodes, "
            f"found {len(shift_scale_nodes)}"
        )

        # Both op types must be recognized as state
        for node in lut_nodes + shift_scale_nodes:
            assert is_state_node(node) is True, (
                f"{node.target._opname} node {node.name} not identified as state"
            )

        # Neither should feed into quantize ops
        for node in lut_nodes + shift_scale_nodes:
            for user in node.users:
                if hasattr(user.target, "name"):
                    user_op_name = user.target.name()
                else:
                    user_op_name = str(user.target)
                assert "quantize" not in user_op_name.lower(), (
                    f"{node.target._opname} output feeds into {user_op_name}, "
                    "but should not be quantized"
                )
