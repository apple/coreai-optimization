# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Comprehensive unit tests for annotation pattern registry with PT2E execution mode.

This module tests all patterns in _annotation_pattern_registry.py with various
input/output specifications to ensure fake quantize layers are inserted correctly
in different parts of the model.
"""

from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from torchao.quantization.pt2e import disable_fake_quant
from torchao.quantization.pt2e.export_utils import _EXPORTED_TRAINING_ATTR

from coreai_opt._utils.config_utils import ALL_TENSORS as _ALL_TENSORS
from coreai_opt.quantization import (
    ModuleQuantizerConfig,
    Quantizer,
    QuantizerConfig,
)
from coreai_opt.quantization._graph._annotation_pattern_registry import (
    NAryActPattern,
    SharedObserverModulePattern,
    WeightedModulePattern,
    _AnnotationPatternRegistry,
)
from coreai_opt.quantization._graph._annotation_utils import (
    OpsListPattern as _OpsListPattern,
)
from coreai_opt.quantization.config import OpQuantizerConfig
from coreai_opt.quantization.spec import (
    PerChannelGranularity,
    PerTensorGranularity,
    QuantizationScheme,
    QuantizationSpec,
    default_activation_quantization_spec,
    default_weight_quantization_spec,
)
from coreai_opt.quantization.spec.fake_quantize import _DefaultFakeQuantizeImpl
from tests.quantization.test_quantization_spec import (
    expanded_dtype_allowlist,  # noqa: F401
)


def per_tensor_int4_qspec():
    return QuantizationSpec(
        dtype=torch.int4,
        qscheme=QuantizationScheme.SYMMETRIC,
        granularity=PerTensorGranularity(),
        fake_quantize_cls="default",
        qparam_calculator_cls="default",
        range_calculator_cls="minmax",
    )


def per_tensor_int8_qspec():
    return QuantizationSpec(
        dtype=torch.int8,
        qscheme=QuantizationScheme.SYMMETRIC,
        granularity=PerTensorGranularity(),
        fake_quantize_cls="default",
        qparam_calculator_cls="default",
        range_calculator_cls="minmax",
    )


class SimpleConvModel(nn.Module):
    """Simple model with conv operation."""

    def __init__(self, use_bn=False, use_act=False):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.use_bn = use_bn
        self.use_act = use_act

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_act:
            x = self.relu(x)
        return x


class SimpleLinearModel(nn.Module):
    """Simple model with linear operation."""

    def __init__(self, use_bn=False, use_act=False):
        super().__init__()
        self.linear = nn.Linear(10, 20)
        self.bn = nn.BatchNorm1d(20)
        self.relu = nn.ReLU()
        self.use_bn = use_bn
        self.use_act = use_act

    def forward(self, x):
        x = self.linear(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_act:
            x = self.relu(x)
        return x


class SimpleEmbeddingModel(nn.Module):
    """Simple model with embedding operation."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 3)

    def forward(self, x):
        return self.embedding(x)


class SimpleBinaryModel(nn.Module):
    """Simple model with binary callable."""

    def __init__(self, binary_fn, use_act=False):
        super().__init__()
        self.binary_fn = binary_fn
        self.use_act = use_act
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x = self.binary_fn(x1, x2)
        if self.use_act:
            x = self.relu(x)
        return x


class SimpleAddOperatorModel(nn.Module):
    """Simple model with add operation."""

    def __init__(self, use_act=False):
        super().__init__()
        self.use_act = use_act
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x = x1 + x2
        if self.use_act:
            x = self.relu(x)
        return x


class SimpleMulOperatorModel(nn.Module):
    """Simple model with mul operation."""

    def __init__(self, use_act=False):
        super().__init__()
        self.use_act = use_act
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x = x1 * x2
        if self.use_act:
            x = self.relu(x)
        return x


class SimpleSubOperatorModel(nn.Module):
    """Simple model with sub operation."""

    def __init__(self, use_act=False):
        super().__init__()
        self.use_act = use_act
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x = x1 - x2
        if self.use_act:
            x = self.relu(x)
        return x


class SimpleMatMulOperatorModel(nn.Module):
    """Simple model with matmul operation using @ operator."""

    def __init__(self, use_act=False):
        super().__init__()
        self.use_act = use_act
        self.relu = nn.ReLU()

    def forward(self, x1, x2):
        x = x1 @ x2
        if self.use_act:
            x = self.relu(x)
        return x


class SimpleFlattenModel(nn.Module):
    """Simple model with flatten operation."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, (1, 1), bias=False)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(12, 2, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class SimpleCatModel(nn.Module):
    """Simple model with concat operation."""

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2, bias=False)
        self.linear2 = torch.nn.Linear(2, 2, bias=False)
        self.linear3 = torch.nn.Linear(2, 2, bias=False)

    def forward(self, x1, x2):
        x1 = self.linear1(x1)
        x2 = self.linear2(x2)
        x = torch.cat([x1, x2])
        x = self.linear3(x)
        return x


class SimpleConcatModel(nn.Module):
    """Simple model with concat operation."""

    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2, bias=False)
        self.linear2 = torch.nn.Linear(2, 2, bias=False)
        self.linear3 = torch.nn.Linear(2, 2, bias=False)

    def forward(self, x1, x2):
        x1 = self.linear1(x1)
        x2 = self.linear2(x2)
        x = torch.concat([x1, x2])
        x = self.linear3(x)
        return x


class AddLinearMulModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, bias=False)
        self.simple_mul_model = SimpleMulOperatorModel(use_act=True)

    def forward(self, inp1, inp2):
        x = inp1 + inp2
        x = self.linear(x)
        x = self.simple_mul_model(x, inp2)
        return x


class InnerModel1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_middle = torch.nn.Linear(2, 2, False)
        self.linear_out1 = torch.nn.Linear(2, 2, False)
        self.linear_out2 = torch.nn.Linear(2, 2, False)

    def forward(self, inp1, inp2):
        x = inp1 + inp2
        x = self.linear_middle(x)
        x1 = self.linear_out1(x)
        x2 = self.linear_out2(x)
        return x1, x2


class NestedModel1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 2, bias=False)
        self.inner_module = InnerModel1()

    def forward(self, inp1, inp2, inp3):
        x = self.linear1(inp2)
        x1, x2 = self.inner_module(inp3, x)
        x = inp1 - x2
        return x, x1


class NestedModel2InnerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.linear_weight node name: inner_linear_weight_1
        self.linear_weight = torch.nn.Parameter(torch.randn(1, 2))
        self.relu = torch.nn.ReLU()

        # self.linear.weight node name: inner_linear_weight_2
        self.linear = torch.nn.Linear(2, 2, bias=False)

    def forward(self, inp, inp2):
        x = inp + inp2  # node name: add
        x = self.linear(x)  # node name: linear
        return x


class NestedModel2SubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inp, inp2):
        x = inp - inp2  # node name: sub
        return x


class NestedModel2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = NestedModel2InnerModel()
        # self.inner_linear_weight node name: inner_linear_weight_3
        self.inner_linear = torch.nn.Linear(2, 2, bias=False)

        # self.inner_linear_weight node name: inner_linear_weight
        self.inner_linear_weight = torch.nn.Parameter(torch.randn(1, 2))
        self.sub_model = NestedModel2SubModel()

    def forward(self, inp):
        x = self.inner(self.inner_linear_weight, inp)
        x = self.inner_linear(x)  # node name: linear_1
        sub = self.sub_model(self.inner_linear.weight, self.inner.linear_weight)
        x = x + sub  # node name: add_1
        return x


def analyze_graph_structure(model: torch.fx.GraphModule) -> dict[str, any]:
    """
    Analyze the graph structure to find fake quantize placement relative to operations.

    Returns:
        Dict containing:
        - operation types mapped to their fake quantize placement patterns
        - weight_fq_count: number of weight fake quantize nodes
    """
    nodes = list(model.graph.nodes)

    def is_fake_quantize(node):
        return "activation_post_process" in str(node.target)

    def is_weight(node):
        """
        Return True if this is a weight node or a fake quantize op for a weight node.
        """
        if not is_fake_quantize(node):
            return node.op == "get_attr"
        # Check if any of the inputs to this FQ node are weight parameters
        for arg in node.args:
            if isinstance(arg, torch.fx.Node) and arg.op == "get_attr":
                return True

        # torchao behavior for fused conv2d and bn ops leads to additional ops
        # between conv weight and the weight quantizer
        users = node.users
        if len(users) == 1:
            user = list(users.keys())[0]
            if "conv2d" in user.name and node == user.args[1]:
                return True

        return False

    def get_node_type(node):
        if is_fake_quantize(node):
            return "fake_quantize"
        else:
            return "other"

    # Count weight fake quantize nodes
    weight_fq_count = sum(1 for node in nodes if is_fake_quantize(node) and is_weight(node))

    # Find patterns: what comes before and after each operation
    patterns = {"other": [], "fake_quantize": []}

    for node in nodes:
        node_type = get_node_type(node)
        # Check what comes before (inputs) - exclude weight fake quantize
        # from activation inputs
        before_types = []
        weights = {}
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                arg_type = get_node_type(arg)
                if is_weight(arg):
                    if is_fake_quantize(arg):
                        # This is a weight fake quantize, don't count it as
                        # input activation FQ
                        # Use the original model weight name for easier writing of unit
                        # tests.
                        weights[arg.all_input_nodes[0].target] = arg_type
                    else:
                        weights[arg.target] = arg_type
                else:
                    before_types.append(arg_type)

        # Check what comes after (users)
        after_types = []
        for user in node.users:
            after_types.append(get_node_type(user))

        pattern = {
            "node_name": node.name,
            "before": before_types,
            "after": after_types,
            "weights": weights,
            "target": str(node.target),
        }
        patterns[node_type].append(pattern)

    # Add weight_fq_count to the result
    patterns["weight_fq_count"] = weight_fq_count
    return patterns


def check_fake_quantize_placement(
    patterns: dict[str, list], expected_placements: dict[str, dict[str, bool]]
):
    """
    Check if fake quantize operations are placed as expected and assert validation.

    Args:
        patterns: Graph structure analysis from analyze_graph_structure
        expected_placements: Expected placement patterns. Example:
            {'specific_node_name': {
                'input_fq': True, 'weight_fq': True, 'output_fq': False
            }}
    """
    # Count total fake quantize nodes
    total_fq_nodes = len(patterns.get("fake_quantize", []))

    for key, expectations in expected_placements.items():
        found_pattern = None
        for op_type, op_patterns in patterns.items():
            if op_type in ["fake_quantize", "weight_fq_count"]:
                continue
            for pattern in op_patterns:
                if pattern["node_name"] == key:
                    found_pattern = pattern
                    break
            if found_pattern:
                break

        assert found_pattern is not None, f"Node '{key}' not found in graph"
        _check_single_node_placement(found_pattern, expectations, patterns, total_fq_nodes)


def _check_single_node_placement(
    pattern: dict, expectations: dict, patterns: dict, total_fq_nodes: int
):
    """Helper function to check fake quantize placement for a single node."""
    node_name = pattern["node_name"]

    # Check for input fake quantize
    input_fq_expectation = expectations.get("input_fq")
    if input_fq_expectation:
        if isinstance(input_fq_expectation, list):
            for idx, input_fq_bool in enumerate(input_fq_expectation):
                has_quantizer = pattern["before"][idx] == "fake_quantize"
                assert has_quantizer == input_fq_bool, (
                    f" Expected presence of quantizer at {node_name} input index {idx} to "
                    f"be {input_fq_bool} but got {has_quantizer}."
                )
        else:
            has_input_fq = "fake_quantize" in pattern["before"]
            assert has_input_fq, (
                f"Expected input fake quantize before {node_name}, "
                f"but not found. Before: {pattern['before']}"
            )
    elif input_fq_expectation is False:
        has_input_fq = "fake_quantize" in pattern["before"]
        assert not has_input_fq, (
            f"Expected NO input fake quantize before {node_name}, "
            f"but found one. Before: {pattern['before']}"
        )

    # Check for output fake quantize
    output_fq_expectation = expectations.get("output_fq")
    if output_fq_expectation:
        has_output_fq = "fake_quantize" in pattern["after"]
        assert has_output_fq, (
            f"Expected output fake quantize after {node_name}, "
            f"but not found. After: {pattern['after']}"
        )
    elif output_fq_expectation is False:
        has_output_fq = "fake_quantize" in pattern["after"]
        assert not has_output_fq, (
            f"Expected NO output fake quantize after {node_name}, "
            f"but found one. After: {pattern['after']}"
        )

    weight_fq_expectation = expectations.get("weight_fq")
    if weight_fq_expectation is None:
        return
    if isinstance(weight_fq_expectation, dict):
        # More fine-grained check allows checking specific weight names
        for name, expected_quantized in weight_fq_expectation.items():
            assert (pattern["weights"].get(name) == "fake_quantize") is expected_quantized, (
                f"Expected weight with name {name} quantized = {expected_quantized}"
            )
    else:
        assert isinstance(weight_fq_expectation, bool), (
            "weight_fq should either be a bool or a dict mapping weight names to bool. "
            f"Got {weight_fq_expectation}."
        )
        # Falling back to expecting at least one weight input to be quantized if
        # weight_fq_expectation is True.
        found_quantized = False
        for arg_type in pattern["weights"].values():
            if arg_type == "fake_quantize":
                found_quantized = True

        assert found_quantized == weight_fq_expectation, (
            f"{node_name} weight check failed. Weight expectation: {weight_fq_expectation}."
        )


class TestAnnotationPatternRegistry:
    """Test suite for annotation pattern registry."""

    @pytest.fixture
    def weight_spec(self):
        """Default weight quantization spec."""
        return default_weight_quantization_spec()

    @pytest.fixture
    def activation_spec(self):
        """Default activation quantization spec."""
        return default_activation_quantization_spec()

    @pytest.fixture
    def relu_pattern(self):
        """Temporarily register a relu pattern if not already registered."""
        already_registered = "relu" in _AnnotationPatternRegistry.REGISTRY

        if not already_registered:

            @_AnnotationPatternRegistry.register("relu")
            class ReluPattern(NAryActPattern):
                """Annotates input -> relu -> output."""

                @classmethod
                def generate_patterns(cls) -> list[_OpsListPattern]:
                    """Return relu pattern."""
                    return [_OpsListPattern(["relu"])]

        yield

        if not already_registered:
            del _AnnotationPatternRegistry.REGISTRY["relu"]

    def _build_config_dict(
        self,
        quantization_type,
        module_name,
        config_type,
        weight_spec=None,
        activation_spec=None,
    ):
        config_dict = {}
        assert config_type in ["global_config", "module_type_configs", "module_name_configs"], (
            "Config type should be one of 'global_config', 'module_type_configs', "
            "'module_name_configs'"
        )

        module_key = module_name
        if module_key is None:
            assert config_type == "global_config", (
                "Config type should be 'global_config' for module_name of None."
            )

        # Map module name to module type for type based configs
        module_type_config_mapping = {
            "conv": torch.nn.Conv2d,
            "linear": torch.nn.Linear,
            "embedding": torch.nn.Embedding,
        }
        if config_type == "module_type_configs":
            module_key = module_type_config_mapping.get(module_name)
            if module_name is None:
                raise ValueError("Unrecognized module name: {module_name}")

        op_state_spec = None
        op_input_spec = None
        op_output_spec = None
        if quantization_type == "weight_only":
            op_state_spec = {"weight": weight_spec}
        elif quantization_type == "input_activation_only":
            op_input_spec = {"*": activation_spec}
        elif quantization_type == "output_activation_only":
            op_output_spec = {"*": activation_spec}
        elif quantization_type == "activation_only":
            op_input_spec = {"*": activation_spec}
            op_output_spec = {"*": activation_spec}
        elif quantization_type == "weight_and_activation":
            op_state_spec = {"weight": weight_spec}
            op_input_spec = {"*": activation_spec}
            op_output_spec = {"*": activation_spec}

        config_dict = {
            "op_state_spec": op_state_spec,
            "op_input_spec": op_input_spec,
            "op_output_spec": op_output_spec,
        }

        if config_type == "global_config":
            return {config_type: config_dict}
        else:
            return {config_type: {module_key: config_dict}}

    @pytest.mark.parametrize(
        "model,example_inputs,quantization_type,module_name,expected,is_training",
        [
            # Conv pattern tests
            (
                SimpleConvModel(),
                (torch.randn(1, 3, 32, 32),),
                "weight_only",
                "conv",
                {"conv2d": {"weight_fq": True, "input_fq": False, "output_fq": False}},
                False,
            ),
            (
                SimpleConvModel(),
                (torch.randn(1, 3, 32, 32),),
                "input_activation_only",
                "conv",
                {"conv2d": {"weight_fq": False, "input_fq": True, "output_fq": False}},
                False,
            ),
            (
                SimpleConvModel(),
                (torch.randn(1, 3, 32, 32),),
                "output_activation_only",
                "conv",
                {"conv2d": {"weight_fq": False, "input_fq": False, "output_fq": True}},
                False,
            ),
            (
                SimpleConvModel(),
                (torch.randn(1, 3, 32, 32),),
                "activation_only",
                "conv",
                {"conv2d": {"weight_fq": False, "input_fq": True, "output_fq": True}},
                False,
            ),
            (
                SimpleConvModel(),
                (torch.randn(1, 3, 32, 32),),
                "weight_and_activation",
                "conv",
                {
                    "conv2d": {
                        "weight_fq": {"conv.weight": True, "conv.bias": False},
                        "input_fq": True,
                        "output_fq": True,
                    }
                },
                False,
            ),
            # Conv+BN pattern tests
            (
                SimpleConvModel(use_bn=True),
                (torch.randn(1, 3, 32, 32),),
                "weight_only",
                "conv",
                {"conv2d_1": {"weight_fq": True, "input_fq": False, "output_fq": False}},
                False,
            ),
            (
                SimpleConvModel(use_bn=True),
                (torch.randn(1, 3, 32, 32),),
                "weight_and_activation",
                "conv",
                {
                    "conv2d_1": {"weight_fq": True, "input_fq": True, "output_fq": False},
                    "batch_norm_2": {"input_fq": False, "output_fq": True},
                },
                False,
            ),
            (
                SimpleConvModel(use_bn=True),
                (torch.randn(1, 3, 32, 32),),
                "weight_and_activation",
                "conv",
                {
                    "conv2d_1": {"weight_fq": True, "input_fq": True, "output_fq": False},
                    "batch_norm_3": {"input_fq": False, "output_fq": True},
                },
                True,
            ),
            # Conv+ReLU pattern tests
            (
                SimpleConvModel(use_act=True),
                (torch.randn(1, 3, 32, 32),),
                "weight_only",
                "conv",
                {"conv2d": {"weight_fq": True, "input_fq": False, "output_fq": False}},
                False,
            ),
            (
                SimpleConvModel(use_act=True),
                (torch.randn(1, 3, 32, 32),),
                "weight_and_activation",
                "conv",
                {
                    "conv2d": {"weight_fq": True, "input_fq": True, "output_fq": False},
                    "relu": {"input_fq": False, "output_fq": True},
                },
                False,
            ),
            # Conv+BN+ReLU pattern tests
            (
                SimpleConvModel(use_bn=True, use_act=True),
                (torch.randn(1, 3, 32, 32),),
                "weight_only",
                "conv",
                {
                    "conv2d_1": {"weight_fq": True, "input_fq": False, "output_fq": False},
                    "batch_norm_2": {"input_fq": False, "output_fq": False},
                    "relu": {"input_fq": False, "output_fq": False},
                },
                False,
            ),
            (
                SimpleConvModel(use_bn=True, use_act=True),
                (torch.randn(1, 3, 32, 32),),
                "weight_and_activation",
                "conv",
                {
                    "conv2d_1": {"weight_fq": True, "input_fq": True, "output_fq": False},
                    "batch_norm_2": {"input_fq": False, "output_fq": False},
                    "relu": {"input_fq": False, "output_fq": True},
                },
                False,
            ),
            (
                SimpleConvModel(use_bn=True, use_act=True),
                (torch.randn(1, 3, 32, 32),),
                "weight_and_activation",
                "conv",
                {
                    "conv2d_1": {"weight_fq": True, "input_fq": True, "output_fq": False},
                    "batch_norm_3": {"input_fq": False, "output_fq": False},
                    "relu": {"input_fq": False, "output_fq": True},
                },
                True,
            ),
            # Linear pattern tests
            (
                SimpleLinearModel(),
                (torch.randn(1, 10),),
                "weight_only",
                "linear",
                {"linear": {"weight_fq": True, "input_fq": False, "output_fq": False}},
                False,
            ),
            (
                SimpleLinearModel(),
                (torch.randn(1, 10),),
                "input_activation_only",
                "linear",
                {"linear": {"weight_fq": False, "input_fq": True, "output_fq": False}},
                False,
            ),
            (
                SimpleLinearModel(),
                (torch.randn(1, 10),),
                "output_activation_only",
                "linear",
                {"linear": {"weight_fq": False, "input_fq": False, "output_fq": True}},
                False,
            ),
            (
                SimpleLinearModel(),
                (torch.randn(1, 10),),
                "activation_only",
                "linear",
                {"linear": {"weight_fq": False, "input_fq": True, "output_fq": True}},
                False,
            ),
            (
                SimpleLinearModel(),
                (torch.randn(1, 10),),
                "weight_and_activation",
                "linear",
                {"linear": {"weight_fq": True, "input_fq": True, "output_fq": True}},
                False,
            ),
            # Linear+BN pattern tests
            (
                SimpleLinearModel(use_bn=True),
                (torch.randn(2, 10),),
                "weight_only",
                "linear",
                {
                    "linear": {"weight_fq": True, "input_fq": False, "output_fq": False},
                    "batch_norm_1": {"input_fq": False, "output_fq": False},
                },
                False,
            ),
            (
                SimpleLinearModel(use_bn=True),
                (torch.randn(2, 10),),
                "weight_and_activation",
                "linear",
                {
                    "linear": {"weight_fq": True, "input_fq": True, "output_fq": False},
                    "batch_norm_1": {"input_fq": False, "output_fq": True},
                },
                False,
            ),
            (
                SimpleLinearModel(use_bn=True),
                (torch.randn(2, 10),),
                "weight_and_activation",
                "linear",
                {
                    "linear": {"weight_fq": True, "input_fq": True, "output_fq": False},
                    "batch_norm_2": {"input_fq": False, "output_fq": True},
                },
                True,
            ),
            # Linear+ReLU pattern tests
            (
                SimpleLinearModel(use_act=True),
                (torch.randn(1, 10),),
                "weight_only",
                "linear",
                {"linear": {"weight_fq": True, "input_fq": False, "output_fq": False}},
                False,
            ),
            (
                SimpleLinearModel(use_act=True),
                (torch.randn(1, 10),),
                "weight_and_activation",
                "linear",
                {
                    "linear": {"weight_fq": True, "input_fq": True, "output_fq": False},
                    "relu": {"input_fq": False, "output_fq": True},
                },
                False,
            ),
            # Linear+BN+ReLU pattern tests
            (
                SimpleLinearModel(use_bn=True, use_act=True),
                (torch.randn(2, 10),),
                "weight_only",
                "linear",
                {
                    "linear": {"weight_fq": True, "input_fq": False, "output_fq": False},
                    "batch_norm_1": {"input_fq": False, "output_fq": False},
                    "relu": {"input_fq": False, "output_fq": False},
                },
                False,
            ),
            (
                SimpleLinearModel(use_bn=True, use_act=True),
                (torch.randn(2, 10),),
                "weight_and_activation",
                "linear",
                {
                    "linear": {"weight_fq": True, "input_fq": True, "output_fq": False},
                    "batch_norm_1": {"input_fq": False, "output_fq": False},
                    "relu": {"input_fq": False, "output_fq": True},
                },
                False,
            ),
            (
                SimpleLinearModel(use_bn=True, use_act=True),
                (torch.randn(2, 10),),
                "weight_and_activation",
                "linear",
                {
                    "linear": {"weight_fq": True, "input_fq": True, "output_fq": False},
                    "batch_norm_2": {"input_fq": False, "output_fq": False},
                    "relu": {"input_fq": False, "output_fq": True},
                },
                True,
            ),
            # Embedding pattern test
            (
                SimpleEmbeddingModel(),
                (torch.tensor([[1, 0, 2, 4]]),),
                "weight_only",
                "embedding",
                {
                    "embedding": {
                        "weight_fq": True,
                        "input_fq": False,
                        "output_fq": False,
                    },
                },
                False,
            ),
            # MatMul pattern tests
            pytest.param(
                SimpleBinaryModel(binary_fn=torch.matmul),
                (torch.randn(1, 2), torch.randn(2, 3)),
                "activation_only",
                None,
                {"matmul": {"input_fq": [True, True], "output_fq": True}},
                False,
            ),
            pytest.param(
                SimpleBinaryModel(binary_fn=torch.matmul, use_act=True),
                (torch.randn(1, 2), torch.randn(2, 3)),
                "activation_only",
                None,
                {
                    "matmul": {"input_fq": [True, True], "output_fq": False},
                    "relu": {"input_fq": False, "output_fq": True},
                },
                False,
            ),
            # Add pattern tests
            pytest.param(
                SimpleBinaryModel(binary_fn=torch.add),
                (torch.randn(1, 2), torch.randn(1, 2)),
                "activation_only",
                None,
                {"add": {"input_fq": [True, True], "output_fq": True}},
                False,
            ),
            pytest.param(
                SimpleBinaryModel(binary_fn=torch.add, use_act=True),
                (torch.randn(1, 2), torch.randn(1, 2)),
                "activation_only",
                None,
                {
                    "add": {"input_fq": [True, True], "output_fq": False},
                    "relu": {"input_fq": False, "output_fq": True},
                },
                False,
            ),
            pytest.param(
                SimpleAddOperatorModel(use_act=True),
                (torch.randn(1, 2), torch.randn(1, 2)),
                "activation_only",
                None,
                {
                    "add": {"input_fq": [True, True], "output_fq": False},
                    "relu": {"input_fq": False, "output_fq": True},
                },
                False,
            ),
            # Mul pattern tests
            pytest.param(
                SimpleBinaryModel(binary_fn=torch.mul),
                (torch.randn(1, 2), torch.randn(1, 2)),
                "activation_only",
                None,
                {"mul": {"input_fq": [True, True], "output_fq": True}},
                False,
            ),
            pytest.param(
                SimpleBinaryModel(binary_fn=torch.mul, use_act=True),
                (torch.randn(1, 2), torch.randn(1, 2)),
                "activation_only",
                None,
                {
                    "mul": {"input_fq": [True, True], "output_fq": False},
                    "relu": {"input_fq": False, "output_fq": True},
                },
                False,
            ),
            pytest.param(
                SimpleMulOperatorModel(use_act=True),
                (torch.randn(1, 2), torch.randn(1, 2)),
                "activation_only",
                None,
                {
                    "mul": {"input_fq": [True, True], "output_fq": False},
                    "relu": {"input_fq": False, "output_fq": True},
                },
                False,
            ),
            # Sub pattern tests
            pytest.param(
                SimpleBinaryModel(binary_fn=torch.sub),
                (torch.randn(1, 2), torch.randn(1, 2)),
                "activation_only",
                None,
                {"sub": {"input_fq": [True, True], "output_fq": True}},
                False,
            ),
            pytest.param(
                SimpleSubOperatorModel(),
                (torch.randn(1, 2), torch.randn(1, 2)),
                "activation_only",
                None,
                {"sub": {"input_fq": [True, True], "output_fq": True}},
                False,
            ),
            pytest.param(
                SimpleMatMulOperatorModel(),
                (torch.randn(1, 2), torch.randn(2, 3)),
                "activation_only",
                None,
                {"matmul": {"input_fq": [True, True], "output_fq": True}},
                False,
            ),
            pytest.param(
                SimpleMatMulOperatorModel(use_act=True),
                (torch.randn(1, 2), torch.randn(2, 3)),
                "activation_only",
                None,
                {
                    "matmul": {"input_fq": [True, True], "output_fq": False},
                    "relu": {"input_fq": False, "output_fq": True},
                },
                False,
            ),
        ],
    )
    def test_quantization_patterns(
        self,
        model,
        example_inputs,
        quantization_type,
        module_name,
        expected,
        is_training,
        weight_spec,
        activation_spec,
    ):
        config_dicts_to_test = []

        if module_name is None:
            config_dicts_to_test.append(
                self._build_config_dict(
                    quantization_type,
                    module_name,
                    "global_config",
                    weight_spec,
                    activation_spec,
                )
            )

        else:
            config_dicts_to_test.append(
                self._build_config_dict(
                    quantization_type,
                    module_name,
                    "module_type_configs",
                    weight_spec,
                    activation_spec,
                )
            )
            config_dicts_to_test.append(
                self._build_config_dict(
                    quantization_type,
                    module_name,
                    "module_name_configs",
                    weight_spec,
                    activation_spec,
                )
            )

        for config_dict in config_dicts_to_test:
            if is_training:
                model.train()
            else:
                model.eval()

            original_out = model(*example_inputs)
            # Create config with resolved parameters
            config = QuantizerConfig.from_dict({QuantizerConfig._CONFIG_KEY: config_dict})

            quantizer = Quantizer(model, config)
            prepared_model = quantizer.prepare(example_inputs)

            patterns = analyze_graph_structure(prepared_model)
            check_fake_quantize_placement(patterns, expected)

            # Disable all quantizers to check output of original vs. prepared models
            prepared_model.apply(disable_fake_quant)
            prepared_out = prepared_model(*example_inputs)

            # Check that training mode is set properly. For exported models,
            # _EXPORTED_TRAINING_ATTR must be checked instead of "training" attribute
            if is_training:
                assert getattr(prepared_model, _EXPORTED_TRAINING_ATTR)
            else:
                assert not getattr(prepared_model, _EXPORTED_TRAINING_ATTR)
            assert torch.allclose(original_out, prepared_out, atol=1e-5)

    @pytest.mark.parametrize(
        "branch_op, expected",
        [
            (
                None,
                {
                    "linear": {"input_fq": [True], "weight_fq": True, "output_fq": False},
                    "relu": {"input_fq": False, "output_fq": True},
                },
            ),
            (
                torch.nn.Linear(2, 2, bias=False),
                {
                    "linear": {"input_fq": [True], "weight_fq": True, "output_fq": True},
                    "relu": {"input_fq": True, "output_fq": False},
                    "linear_1": {"input_fq": True, "weight_fq": True, "output_fq": True},
                },
            ),
            (
                torch.nn.Hardtanh(),
                {
                    "linear": {"input_fq": [True], "weight_fq": True, "output_fq": True},
                    "relu": {"input_fq": True, "output_fq": False},
                    "hardtanh": {"input_fq": True, "output_fq": False},
                },
            ),
        ],
    )
    def test_subgraph_pattern_with_branching(self, branch_op, expected):
        class SubgraphBranchingModel(torch.nn.Module):
            def __init__(self, branch_op):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2, bias=False)
                self.relu = torch.nn.ReLU()
                self.hardtanh = torch.nn.Hardtanh()
                self.branch_op = branch_op

            def forward(self, inp):
                x = self.linear(inp)
                x1 = self.relu(x)
                if self.branch_op:
                    x2 = self.branch_op(x)
                    return x1, x2
                return x1

        model = SubgraphBranchingModel(branch_op)
        example_inputs = (torch.randn(1, 2),)
        quantizer = Quantizer(model, QuantizerConfig())
        prepared_model = quantizer.prepare(example_inputs)
        patterns = analyze_graph_structure(prepared_model)
        check_fake_quantize_placement(patterns, expected)

    @pytest.mark.parametrize(
        "branch_op, expected",
        [
            (
                None,
                {
                    "add": {"input_fq": [True, True], "output_fq": False},
                    "relu": {"input_fq": False, "output_fq": True},
                },
            ),
            (
                torch.nn.Linear(2, 2, bias=False),
                {
                    "add": {"input_fq": [True, True], "output_fq": True},
                    "relu": {"input_fq": True, "output_fq": False},
                    "linear": {"input_fq": True, "weight_fq": True, "output_fq": True},
                },
            ),
            (
                torch.nn.Hardtanh(),
                {
                    "add": {"input_fq": [True, True], "output_fq": True},
                    "relu": {"input_fq": True, "output_fq": False},
                    "hardtanh": {"input_fq": True, "output_fq": False},
                },
            ),
        ],
    )
    def test_sequential_pattern_with_branching(self, branch_op, expected):
        class SequentialBranchingModel(torch.nn.Module):
            def __init__(self, branch_op):
                super().__init__()
                self.relu = torch.nn.ReLU()
                self.hardtanh = torch.nn.Hardtanh()
                self.branch_op = branch_op

            def forward(self, inp, inp2):
                x = inp + inp2
                x1 = self.relu(x)
                if self.branch_op:
                    x2 = self.branch_op(x)
                    return x1, x2
                return x1

        model = SequentialBranchingModel(branch_op)
        example_inputs = (torch.randn(1, 2), torch.randn(1, 2))
        quantizer = Quantizer(model, QuantizerConfig())
        prepared_model = quantizer.prepare(example_inputs)
        patterns = analyze_graph_structure(prepared_model)
        check_fake_quantize_placement(patterns, expected)

    def test_shared_observer_output_qspec_set(self):
        """
        Check that output qspec is shared when config has output quantization set.
        """
        model = SimpleFlattenModel()
        inp = torch.randn(1, 3, 2, 2)
        config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec=None,
            )
        )

        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_inputs=(inp,))

        flatten_node = [node for node in prepared_model.graph.nodes if node.name == "flatten"][0]
        assert flatten_node.all_input_nodes[0].name == "activation_post_process_1"
        assert list(flatten_node.users.keys())[0].name == "activation_post_process_2"
        assert prepared_model.activation_post_process_1 == prepared_model.activation_post_process_2

    def test_shared_observer_output_qspec_not_set(self):
        """
        Test that output qspec is shared no matter whether output qspec in the config is
        None
        """
        model = SimpleFlattenModel()
        inp = torch.randn(1, 3, 2, 2)
        config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(op_state_spec=None, op_output_spec=None)
        )

        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_inputs=(inp,))

        flatten_node = [node for node in prepared_model.graph.nodes if node.name == "flatten"][0]
        assert flatten_node.all_input_nodes[0].name == "activation_post_process_1"
        assert list(flatten_node.users.keys())[0].name == "activation_post_process_2"
        assert prepared_model.activation_post_process_1 == prepared_model.activation_post_process_2

    def test_shared_observer_qspec_propagation(self):
        """
        If input node to shared observer has output qspec set, and child node of
        shared observer has input qspec set, then even if shared observer itself
        does not have quantization annotations, it should have the effect of sharing
        the quantizer before it with the quantizer after it.
        """
        model = SimpleFlattenModel()
        inp = torch.randn(1, 3, 2, 2)
        config = QuantizerConfig(
            module_type_configs={
                torch.nn.Conv2d: ModuleQuantizerConfig(op_state_spec=None, op_input_spec=None),
                torch.nn.Linear: ModuleQuantizerConfig(op_state_spec=None, op_output_spec=None),
            }
        )

        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_inputs=(inp,))

        flatten_node = [node for node in prepared_model.graph.nodes if node.name == "flatten"][0]
        assert flatten_node.all_input_nodes[0].name == "activation_post_process_0"
        assert list(flatten_node.users.keys())[0].name == "activation_post_process_1"
        assert prepared_model.activation_post_process_0 == prepared_model.activation_post_process_1

    def test_shared_observer_with_standard_output_qspec(self):
        """
        If input qspec is None and config has output qspec, then there will be a
        regular qspec set for the shared observer node's output.
        """
        model = SimpleFlattenModel()
        inp = torch.randn(1, 3, 2, 2)
        config = QuantizerConfig(
            global_config=None,
            module_name_configs={
                "flatten": ModuleQuantizerConfig(
                    op_state_spec=None,
                    op_input_spec=None,
                ),
            },
        )

        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_inputs=(inp,))
        flatten_node = [node for node in prepared_model.graph.nodes if node.name == "flatten"][0]
        quantize_nodes = [
            node for node in prepared_model.graph.nodes if "activation_post_process" in node.name
        ]
        assert len(quantize_nodes) == 1
        assert quantize_nodes[0] in flatten_node.users.keys()

    def test_shared_observer_no_qspec_set(self):
        """
        If input qspec is None and config has output qspec set to None, the shared
        observer node's output qspec should be None.
        """
        model = SimpleFlattenModel()
        inp = torch.randn(1, 3, 2, 2)
        config = QuantizerConfig(global_config=None)

        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_inputs=(inp,))
        for node in prepared_model.graph.nodes:
            assert "activation_post_process" not in node.name

    def test_shared_observer_forces_per_tensor_for_flatten(self):
        """
        When per-channel activation granularity is used globally, shared observer
        ops that alter channel semantics (e.g., flatten) should have their shared
        fake quantize modules forced to per-tensor granularity. The conv and linear
        activation quantizers (which are separate objects) should remain per-channel.
        """
        model = SimpleFlattenModel()
        inp = torch.randn(2, 3, 2, 2)

        per_channel_activation_spec = QuantizationSpec(
            dtype=torch.int8,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerChannelGranularity(axis=0),
            fake_quantize_cls="default",
            qparam_calculator_cls="default",
            range_calculator_cls="minmax",
        )

        config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec=None,
                op_input_spec={"*": per_channel_activation_spec},
                op_output_spec={"*": per_channel_activation_spec},
            ),
        )

        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_inputs=(inp,))

        # Find the flatten node and its adjacent fake quantize modules
        flatten_node = [node for node in prepared_model.graph.nodes if node.name == "flatten"][0]
        input_fq_name = flatten_node.all_input_nodes[0].name
        output_fq_name = list(flatten_node.users.keys())[0].name

        input_fq = getattr(prepared_model, input_fq_name)
        output_fq = getattr(prepared_model, output_fq_name)

        # Shared observer: input and output should be the same object
        assert input_fq is output_fq

        # The shared fake quantize should have been forced to per-tensor
        assert isinstance(input_fq.granularity, PerTensorGranularity)
        assert isinstance(input_fq.qparams_calculator.granularity, PerTensorGranularity)
        assert isinstance(
            input_fq.qparams_calculator.range_calculator.granularity,
            PerTensorGranularity,
        )

        # Run a forward pass to initialize observer parameters, then verify
        # the scale is a scalar (numel == 1), confirming per-tensor behavior.
        prepared_model.eval()
        with torch.no_grad():
            prepared_model(inp)
        assert input_fq.qparams_calculator.scale.numel() == 1

        # Conv input and linear output fake quantizers should remain per-channel
        # since they are separate objects (not shared across a shape-destroying op).
        conv_node = [node for node in prepared_model.graph.nodes if node.name == "conv2d"][0]
        conv_input_fq_node = [
            n for n in conv_node.all_input_nodes if "activation_post_process" in n.name
        ][0]
        conv_input_fq = getattr(prepared_model, conv_input_fq_node.name)
        assert isinstance(conv_input_fq.granularity, PerChannelGranularity)
        assert conv_input_fq.qparams_calculator.scale.numel() > 1

        linear_node = [node for node in prepared_model.graph.nodes if node.name == "linear"][0]
        linear_output_fq_node = [
            n for n in linear_node.users if "activation_post_process" in n.name
        ][0]
        linear_output_fq = getattr(prepared_model, linear_output_fq_node.name)
        assert isinstance(linear_output_fq.granularity, PerChannelGranularity)
        assert linear_output_fq.qparams_calculator.scale.numel() > 1

    def test_cat(self):
        """
        Given a model with

        input1 -> relu1
                        \
                          ->  Cat -> Linear
        input2 -> relu2 /

        Check that concat inputs and outputs share the same quantizers.
        """
        model = SimpleCatModel()
        inp = (torch.randn(1, 2), torch.randn(1, 2))

        config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(op_state_spec=None, op_input_spec=None),
        )

        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_inputs=inp)
        quantizers = [
            module
            for module in prepared_model.modules()
            if isinstance(module, _DefaultFakeQuantizeImpl)
        ]
        assert len(quantizers) == 2
        assert prepared_model.activation_post_process_0 is prepared_model.activation_post_process_1
        assert prepared_model.activation_post_process_0 is prepared_model.activation_post_process_2

        # Final linear output quantizer is not associated with the others
        assert prepared_model.activation_post_process_0 != prepared_model.activation_post_process_3

    def test_concat(self):
        """
        Given a model with

        input1 -> relu1
                        \
                          ->  Concat -> Linear
        input2 -> relu2 /

        Check that concat inputs and outputs share the same quantizers.
        """
        model = SimpleConcatModel()
        inp = (torch.randn(1, 2), torch.randn(1, 2))

        config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(op_state_spec=None, op_input_spec=None),
        )

        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_inputs=inp)
        quantizers = [
            module
            for module in prepared_model.modules()
            if isinstance(module, _DefaultFakeQuantizeImpl)
        ]
        assert len(quantizers) == 2
        assert prepared_model.activation_post_process_0 is prepared_model.activation_post_process_1
        assert prepared_model.activation_post_process_0 is prepared_model.activation_post_process_2

        # Final linear output quantizer is not associated with the others
        assert prepared_model.activation_post_process_0 != prepared_model.activation_post_process_3

    def test_no_quantization_config(self):
        """Test that no fake quantize is inserted when no config is provided."""
        model = SimpleConvModel()
        config = QuantizerConfig(global_config=None)  # Empty config
        quantizer = Quantizer(model, config)

        example_inputs = (torch.randn(1, 3, 32, 32),)
        prepared_model = quantizer.prepare(example_inputs)

        patterns = analyze_graph_structure(prepared_model)

        # Expected: no fake quantize operations
        assert len(patterns.get("fake_quantize", [])) == 0

    def test_multiple_operations_model(self, weight_spec, activation_spec):
        """Test model with multiple different operations."""

        class MultiOpModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3, padding=1)
                self.linear = nn.Linear(16 * 32 * 32, 128)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.conv(x)
                x = self.relu(x)
                x = x.view(x.size(0), -1)
                x = self.linear(x)
                return x

        model = MultiOpModel()
        config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec={"weight": weight_spec},
                op_input_spec={"*": activation_spec},
                op_output_spec={"*": activation_spec},
            )
        )
        quantizer = Quantizer(model, config)

        example_inputs = (torch.randn(1, 3, 32, 32),)
        prepared_model = quantizer.prepare(example_inputs)

        patterns = analyze_graph_structure(prepared_model)

        # Expected: both conv and linear should be quantized
        expected = {
            "conv2d": {"weight_fq": True, "input_fq": True, "output_fq": False},
            "linear": {"weight_fq": True, "input_fq": True, "output_fq": True},
            "relu": {"input_fq": False, "output_fq": True},
        }
        check_fake_quantize_placement(patterns, expected)

    def test_correct_output_spec_precedence(self):
        class TwoLinearReluModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(2, 2, bias=False)
                self.relu1 = nn.ReLU()
                self.linear2 = nn.Linear(2, 2, bias=False)
                self.relu2 = nn.ReLU()

            def forward(self, x):
                x = self.linear1(x)
                x = self.relu1(x)
                x = self.linear2(x)
                x = self.relu2(x)
                return x

        model = TwoLinearReluModel()

        int4_activation_spec = QuantizationSpec(
            dtype=torch.int4,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerTensorGranularity(),
            fake_quantize_cls="default",
            qparam_calculator_cls="default",
            range_calculator_cls="minmax",
        )

        config = QuantizerConfig(
            global_config=None,
            module_name_configs={
                "linear2": ModuleQuantizerConfig(
                    op_input_spec={"*": int4_activation_spec},
                    op_output_spec=None,
                    op_state_spec=None,
                )
            },
            module_type_configs={
                torch.nn.Linear: ModuleQuantizerConfig(
                    op_input_spec=None,
                    op_output_spec={"*": default_activation_quantization_spec()},
                    op_state_spec=None,
                )
            },
        )

        model = TwoLinearReluModel()
        quantizer = Quantizer(model, config)
        example_inputs = (torch.randn(1, 2),)
        prepared_model = quantizer.prepare(example_inputs)
        assert prepared_model.activation_post_process_0.qparams_calculator.dtype == torch.int4

    def test_overlapping_quantization_configs_precedence(self, weight_spec, activation_spec):
        """
        Test when both output of preceding node and input of succeeding node
        are both annotated. Verify only one fake_quant is inserted and the
        later config (linear2) takes precedence over the earlier config (linear1).
        """

        class TwoLinearModel(nn.Module):
            """Model with two consecutive linear operations."""

            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 20)
                self.linear2 = nn.Linear(20, 30)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                return x

        model = TwoLinearModel()

        # First activation spec with different parameters
        activation_spec_int8 = QuantizationSpec(
            dtype=torch.int8,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerTensorGranularity(),
            fake_quantize_cls="default",
            qparam_calculator_cls="default",
            range_calculator_cls="minmax",
        )

        # Second activation spec with different parameters
        activation_spec_int4 = QuantizationSpec(
            dtype=torch.int4,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerTensorGranularity(),
            fake_quantize_cls="default",
            qparam_calculator_cls="default",
            range_calculator_cls="minmax",
        )

        # Configure first linear with output activation quantization
        # Configure second linear with input activation quantization
        # This creates an overlap where both nodes want to quantize the same edge
        config = QuantizerConfig(
            global_config=None,
            module_name_configs={
                "linear1": ModuleQuantizerConfig(
                    op_state_spec={"weight": weight_spec},
                    op_input_spec=None,
                    op_output_spec={"*": activation_spec_int8},
                ),
                "linear2": ModuleQuantizerConfig(
                    op_state_spec={"weight": weight_spec},
                    op_input_spec={"*": activation_spec_int4},
                    op_output_spec=None,
                ),
            },
        )

        quantizer = Quantizer(model, config)
        example_inputs = (torch.randn(1, 10),)
        prepared_model = quantizer.prepare(example_inputs)

        patterns = analyze_graph_structure(prepared_model)

        # Count total FQ nodes
        total_fq_nodes = len(patterns.get("fake_quantize", []))
        weight_fq_count = patterns.get("weight_fq_count", 0)
        activation_fq_count = total_fq_nodes - weight_fq_count

        # Should have exactly 2 weight FQ + 1 activation FQ (not 2 due to merging)
        assert weight_fq_count == 2, f"Expected 2 weight FQ nodes, found {weight_fq_count}"
        assert activation_fq_count == 1, (
            f"Expected exactly 1 activation FQ node, found {activation_fq_count}. "
            f"Total FQ: {total_fq_nodes}, Weight FQ: {weight_fq_count}. "
        )

        # Find the activation fake quantize node to inspect its properties
        activation_fq_node = None
        for node in prepared_model.graph.nodes:
            if "activation_post_process" in str(node.target):
                # Check if this is not a weight fake quantize
                is_weight_fq = any(
                    isinstance(arg, torch.fx.Node)
                    and arg.op == "get_attr"
                    and "weight" in arg.target
                    for arg in node.args
                )
                if not is_weight_fq:
                    activation_fq_node = node
                    break

        assert activation_fq_node is not None, "Should find an activation fake quantize node"

        # Get the actual fake quantize module to inspect its configuration
        fq_module = getattr(prepared_model, activation_fq_node.target)

        # Check that later config (linear2) takes precedence
        actual_dtype = fq_module.dtype
        assert actual_dtype == activation_spec_int4.dtype

    def test_nested_module_structure_quantization(self, weight_spec, activation_spec):
        """
        Test quantization on a model with nested module structure.
        Apply quantization to only one of the leaf modules to verify
        selective quantization works correctly.
        """

        class NestedBlock(nn.Module):
            """A nested block containing multiple layers."""

            def __init__(self, in_features, out_features):
                super().__init__()
                self.linear1 = nn.Linear(in_features, out_features)
                self.relu = nn.ReLU()
                self.linear2 = nn.Linear(out_features, out_features)

            def forward(self, x):
                x = self.linear1(x)
                x = self.relu(x)
                x = self.linear2(x)
                return x

        class NestedModel(nn.Module):
            """Model with nested module structure."""

            def __init__(self):
                super().__init__()
                self.input_layer = nn.Linear(10, 32)
                self.block1 = NestedBlock(32, 64)
                self.block2 = NestedBlock(64, 64)
                self.output_layer = nn.Linear(64, 5)

            def forward(self, x):
                x = self.input_layer(x)
                x = self.block1(x)
                x = self.block2(x)
                x = self.output_layer(x)
                return x

        model = NestedModel()

        # Apply quantization only to the first linear layer in block1
        # This tests selective quantization of a leaf module in nested structure
        config = QuantizerConfig(
            global_config=None,
            module_name_configs={
                "block1.linear1": ModuleQuantizerConfig(
                    op_state_spec={"weight": weight_spec},
                    op_input_spec={"*": activation_spec},
                    op_output_spec={"*": activation_spec},
                )
            },
        )

        quantizer = Quantizer(model, config)
        example_inputs = (torch.randn(1, 10),)
        prepared_model = quantizer.prepare(example_inputs)

        patterns = analyze_graph_structure(prepared_model)

        # Count total FQ nodes
        total_fq_nodes = len(patterns.get("fake_quantize", []))
        weight_fq_count = patterns.get("weight_fq_count", 0)
        activation_fq_count = total_fq_nodes - weight_fq_count

        # Should have exactly 1 weight FQ + 2 activation FQ nodes
        # (input and output activation for the quantized linear layer)
        assert weight_fq_count == 1, f"Expected 1 weight FQ node, found {weight_fq_count}"
        assert activation_fq_count == 2, (
            f"Expected exactly 2 activation FQ nodes, found {activation_fq_count}. "
            f"Total FQ: {total_fq_nodes}, Weight FQ: {weight_fq_count}."
        )

        expected = {
            "linear_1": {"weight_fq": True, "input_fq": True, "output_fq": False},
            "relu": {"input_fq": False, "output_fq": True},
        }
        check_fake_quantize_placement(patterns, expected)

    def test_input_quantization_between_patterns(self, activation_spec):
        class TwoLinearModel(nn.Module):
            """Model with two consecutive linear operations."""

            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 20)
                self.linear2 = nn.Linear(20, 30)

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                return x

        model = TwoLinearModel()
        # Apply input quantization only
        config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec=None,
                op_input_spec={"*": activation_spec},
                op_output_spec=None,
            ),
        )
        quantizer = Quantizer(model, config)
        example_inputs = (torch.randn(1, 10),)
        prepared_model = quantizer.prepare(example_inputs)

        patterns = analyze_graph_structure(prepared_model)

        # Count total FQ nodes
        total_fq_nodes = len(patterns.get("fake_quantize", []))
        assert total_fq_nodes == 2

        # Check that both linear ops have input quantization
        expected = {
            "linear": {"input_fq": True, "output_fq": True},
            "linear_1": {"input_fq": True, "output_fq": False},
        }
        check_fake_quantize_placement(patterns, expected)

    def test_hierarchical_config_precedence(self, expanded_dtype_allowlist):  # noqa: F811
        """
        Test hierarchical config settings precedence: module_name > module_type >
        global.

        This test verifies that quantization configs are applied correctly based on
        their specificity, with more specific configs (module_name) taking precedence
        over less specific ones (module_type and global). Uses different dtypes for
        each config level to easily verify which config was applied.
        """

        class HierarchicalConfigTestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.linear = nn.Linear(32 * 8 * 8, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = x.view(x.size(0), -1)
                x = self.linear(x)
                return x

        model = HierarchicalConfigTestModel()

        # Create quantization specs with different dtypes for each hierarchy level
        # Global: int8
        global_weight_spec = QuantizationSpec(
            dtype=torch.int8,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerTensorGranularity(),
            fake_quantize_cls="default",
            qparam_calculator_cls="default",
            range_calculator_cls="minmax",
        )
        global_activation_spec = QuantizationSpec(
            dtype=torch.int8,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerTensorGranularity(),
            fake_quantize_cls="default",
            qparam_calculator_cls="default",
            range_calculator_cls="minmax",
        )

        # Module type (Conv2d): int4
        module_type_weight_spec = QuantizationSpec(
            dtype=torch.int4,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerTensorGranularity(),
            fake_quantize_cls="default",
            qparam_calculator_cls="default",
            range_calculator_cls="minmax",
        )
        module_type_activation_spec = QuantizationSpec(
            dtype=torch.int4,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerTensorGranularity(),
            fake_quantize_cls="default",
            qparam_calculator_cls="default",
            range_calculator_cls="minmax",
        )

        # Module name (conv1): int6
        module_name_weight_spec = QuantizationSpec(
            dtype=torch.int6,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerTensorGranularity(),
            fake_quantize_cls="default",
            qparam_calculator_cls="default",
            range_calculator_cls="minmax",
        )
        module_name_activation_spec = QuantizationSpec(
            dtype=torch.int6,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerTensorGranularity(),
            fake_quantize_cls="default",
            qparam_calculator_cls="default",
            range_calculator_cls="minmax",
        )

        # Create hierarchical config:
        # - Global: int8 weight + input activation
        # - Module type Conv2d: int4 weight + input activation
        # - Module name "conv1": uint8 weight + input activation
        config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec={"weight": global_weight_spec},
                op_input_spec={"*": global_activation_spec},
                op_output_spec=None,
            ),
            module_type_configs={
                torch.nn.Conv2d: ModuleQuantizerConfig(
                    op_state_spec={"weight": module_type_weight_spec},
                    op_input_spec={"*": module_type_activation_spec},
                    op_output_spec=None,
                )
            },
            module_name_configs={
                "conv1": ModuleQuantizerConfig(
                    op_state_spec={"weight": module_name_weight_spec},
                    op_input_spec={"*": module_name_activation_spec},
                    op_output_spec=None,
                )
            },
        )

        quantizer = Quantizer(model, config)
        example_inputs = (torch.randn(1, 3, 8, 8),)
        prepared_model = quantizer.prepare(example_inputs)
        for node in prepared_model.graph.nodes:
            if node.name == "conv2d":
                activation_mod = getattr(prepared_model, node.all_input_nodes[0].target)
                assert activation_mod.dtype == torch.int6
                weight_mod = getattr(prepared_model, node.all_input_nodes[1].target)
                assert weight_mod.dtype == torch.int6
            elif node.name == "conv2d_1":
                activation_mod = getattr(prepared_model, node.all_input_nodes[0].target)
                assert activation_mod.dtype == torch.int4
                weight_mod = getattr(prepared_model, node.all_input_nodes[1].target)
                assert weight_mod.dtype == torch.int4
            elif node.name == "linear":
                activation_mod = getattr(prepared_model, node.all_input_nodes[0].target)
                assert activation_mod.dtype == torch.int8
                weight_mod = getattr(prepared_model, node.all_input_nodes[1].target)
                assert weight_mod.dtype == torch.int8

    def test_module_name_precedence_over_pattern(self, relu_pattern):
        """
        Test that setting module_name for a relu node within a conv -> relu pattern
        leads to conv being annotated by itself (as opposed to conv -> relu fused op).
        """

        config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec=None,
                op_input_spec={
                    "*": QuantizationSpec(
                        dtype=torch.int8,
                        qscheme=QuantizationScheme.SYMMETRIC,
                        granularity=PerTensorGranularity(),
                        fake_quantize_cls="default",
                        qparam_calculator_cls="default",
                        range_calculator_cls="minmax",
                    )
                },
                op_output_spec=None,
            ),
            module_type_configs={
                torch.nn.ReLU: ModuleQuantizerConfig(
                    op_state_spec=None,
                    op_input_spec={
                        "*": QuantizationSpec(
                            dtype=torch.int4,
                            qscheme=QuantizationScheme.SYMMETRIC,
                            granularity=PerTensorGranularity(),
                            fake_quantize_cls="default",
                            qparam_calculator_cls="default",
                            range_calculator_cls="minmax",
                        )
                    },
                    op_output_spec=None,
                )
            },
        )

        model = SimpleConvModel(use_act=True)
        quantizer = Quantizer(model, config)
        example_inputs = (torch.randn(1, 3, 32, 32),)
        prepared_model = quantizer.prepare(example_inputs)
        node_dict = {node.name: node for node in prepared_model.graph.nodes}
        conv2d_input = node_dict["conv2d"].all_input_nodes[0].name
        relu_input = node_dict["relu"].all_input_nodes[0].name
        assert getattr(quantizer._model, conv2d_input).qparams_calculator.dtype == torch.int8
        assert getattr(quantizer._model, relu_input).qparams_calculator.dtype == torch.int4

    def test_multiple_matching_name_configs(self):
        """
        Test that when a leaf level module matches multiple module name settings, the
        last matching module name config is the setting to follow.
        """

        class InnerModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 2, bias=False)
                self.linear2 = torch.nn.Linear(2, 2, bias=False)

            def forward(self, inp):
                x = self.linear1(inp)
                x = self.linear2(x)
                return x

        class OuterModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner_model = InnerModel()
                self.relu = torch.nn.ReLU()

            def forward(self, inp):
                x = self.inner_model(inp)
                x = self.relu(x)
                return x

        model = OuterModel()

        int8_spec = QuantizationSpec(
            dtype=torch.int8,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerTensorGranularity(),
            fake_quantize_cls="default",
            qparam_calculator_cls="default",
            range_calculator_cls="minmax",
        )
        int4_spec = QuantizationSpec(
            dtype=torch.int4,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerTensorGranularity(),
            fake_quantize_cls="default",
            qparam_calculator_cls="default",
            range_calculator_cls="minmax",
        )
        config = QuantizerConfig(
            global_config=None,
            module_name_configs={
                "inner_model.linear2": ModuleQuantizerConfig(
                    op_state_spec=None,
                    op_input_spec={"*": int4_spec},
                    op_output_spec=None,
                ),
                # Both linears should end up matching this
                "inner_model": ModuleQuantizerConfig(
                    op_state_spec=None,
                    op_input_spec={"*": int8_spec},
                    op_output_spec=None,
                ),
            },
        )

        quantizer = Quantizer(model, config)
        example_inputs = (torch.randn(1, 2),)
        _ = model(*example_inputs)
        prepared_model = quantizer.prepare(example_inputs)
        for node in prepared_model.graph.nodes:
            if node.name == "linear":  # Node for linear1
                input_quantizer_name = node.all_input_nodes[0].name
                input_quantizer = getattr(prepared_model, input_quantizer_name)
                assert input_quantizer.dtype == torch.int8
            if node.name == "linear_1":  # Node for linear2
                input_quantizer_name = node.all_input_nodes[0].name
                input_quantizer = getattr(prepared_model, input_quantizer_name)
                assert input_quantizer.dtype == torch.int8

    def test_complex_nested_config_precedence_integration(
        self,
        expanded_dtype_allowlist,  # noqa: F811
    ):
        """
        Comprehensive integration test for config precedence with complex nested
        modules.

        Tests all corner cases:
        1. Modules matching multiple module_name patterns (later should win)
        2. Modules matching across config levels (name > type > global)
        3. Recursive config propagation to child modules
        4. Priority override with nested hierarchies
        5. Full annotation pipeline integration

        Uses different dtypes to track which config was actually applied.
        """

        class ComplexNestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Top-level modules
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

                # Nested module structures
                self.backbone = nn.Sequential(
                    nn.Conv2d(32, 64, 3),  # backbone.0
                    nn.ReLU(),  # backbone.1
                    nn.Conv2d(64, 128, 3),  # backbone.2
                )

                # Calculate flattened size: after conv operations on 10x10 input
                # conv1: 10x10 -> 10x10 (padding=1)
                # conv2: 10x10 -> 10x10 (padding=1)
                # backbone conv: 10x10 -> 8x8 -> 6x6 (no padding)
                # So final size: 128 * 6 * 6 = 4608
                flattened_size = 128 * 6 * 6

                # Deeper nesting with name conflicts
                self.features = nn.Module()
                self.features.encoder = nn.Sequential(
                    nn.Linear(flattened_size, 256),  # features.encoder.0
                    nn.ReLU(),  # features.encoder.1
                    nn.Linear(256, 512),  # features.encoder.2
                )
                self.features.decoder = nn.Sequential(
                    nn.Linear(512, 256),  # features.decoder.0
                    nn.Dropout(0.5),  # features.decoder.1
                    nn.Linear(256, 128),  # features.decoder.2
                )

                # Final classifier with conflicting names
                self.classifier = nn.Linear(128, 10)

            def forward(self, x):
                # Use all modules to ensure they appear in exported graph
                x = x + x
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.backbone(x)

                # Flatten for linear layers
                x = x.view(x.size(0), -1)

                # Go through encoder
                encoded = self.features.encoder(x)

                # Go through decoder
                decoded = self.features.decoder(encoded)

                # Final classifier
                output = self.classifier(decoded)
                return output

        model = ComplexNestedModel()

        # Create quantization specs with different dtypes for easy verification
        global_spec = QuantizationSpec(
            dtype=torch.int8,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerTensorGranularity(),
            fake_quantize_cls="default",
            qparam_calculator_cls="default",
            range_calculator_cls="minmax",
        )

        # Module type configs
        conv_type_spec = QuantizationSpec(
            dtype=torch.int4,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerTensorGranularity(),
            fake_quantize_cls="default",
            qparam_calculator_cls="default",
            range_calculator_cls="minmax",
        )

        linear_type_spec = QuantizationSpec(
            dtype=torch.int6,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerTensorGranularity(),
            fake_quantize_cls="default",
            qparam_calculator_cls="default",
            range_calculator_cls="minmax",
        )

        # Module name configs
        backbone_spec = QuantizationSpec(
            dtype=torch.int5,
            qscheme=QuantizationScheme.ASYMMETRIC,
            granularity=PerTensorGranularity(),
            fake_quantize_cls="default",
            qparam_calculator_cls="default",
            range_calculator_cls="minmax",
        )

        backbone_2_spec = QuantizationSpec(
            dtype=torch.int2,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerTensorGranularity(),
            fake_quantize_cls="default",
            qparam_calculator_cls="default",
            range_calculator_cls="minmax",
        )

        features_decoder_pattern_spec = QuantizationSpec(
            dtype=torch.int16,
            qscheme=QuantizationScheme.ASYMMETRIC,
            granularity=PerTensorGranularity(),
            fake_quantize_cls="default",
            qparam_calculator_cls="default",
            range_calculator_cls="minmax",
        )

        features_pattern_spec = QuantizationSpec(
            dtype=torch.int7,
            qscheme=QuantizationScheme.ASYMMETRIC,
            granularity=PerTensorGranularity(),
            fake_quantize_cls="default",
            qparam_calculator_cls="default",
            range_calculator_cls="minmax",
        )

        features_encoder_pattern_spec = QuantizationSpec(
            dtype=torch.int3,
            qscheme=QuantizationScheme.ASYMMETRIC,
            granularity=PerTensorGranularity(),
            fake_quantize_cls="default",
            qparam_calculator_cls="default",
            range_calculator_cls="minmax",
        )

        linear_pattern_spec = QuantizationSpec(
            dtype=torch.int32,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerTensorGranularity(),
            fake_quantize_cls="default",
            qparam_calculator_cls="default",
            range_calculator_cls="minmax",
        )

        # Build comprehensive config with overlapping patterns
        config = QuantizerConfig(
            # Global fallback
            global_config=ModuleQuantizerConfig(
                op_state_spec={"weight": global_spec},
                op_input_spec={"*": global_spec},
                op_output_spec=None,
            ),
            # Module type configs
            module_type_configs={
                nn.Conv2d: ModuleQuantizerConfig(
                    op_state_spec={"weight": conv_type_spec},
                    op_input_spec={"*": conv_type_spec},
                    op_output_spec=None,
                ),
                # Below linear spec won't end up being used
                nn.Linear: ModuleQuantizerConfig(
                    op_state_spec={"weight": linear_type_spec},
                    op_input_spec={"*": linear_type_spec},
                    op_output_spec=None,
                ),
            },
            # Module name configs with complex precedence rules
            module_name_configs={
                # Pattern 1: pattern for features.decoder modules
                r"features\.decoder\..*": ModuleQuantizerConfig(
                    op_state_spec={"weight": features_decoder_pattern_spec},
                    op_input_spec={"*": features_decoder_pattern_spec},
                    op_output_spec=None,
                ),
                # Pattern 2: Parent module config (should propagate to children)
                "backbone": ModuleQuantizerConfig(
                    op_state_spec={"weight": backbone_spec},
                    op_input_spec={"*": backbone_spec},
                    op_output_spec=None,
                ),
                # Pattern 3: Should completely overwrite previous features.decoder spec
                r"features*": ModuleQuantizerConfig(
                    op_state_spec={"weight": features_pattern_spec},
                    op_input_spec={"*": features_pattern_spec},
                    op_output_spec=None,
                ),
                # Pattern 3: Should overwrite previous features spec
                r"features\.encoder\..*": ModuleQuantizerConfig(
                    op_state_spec={"weight": features_encoder_pattern_spec},
                    op_input_spec={"*": features_encoder_pattern_spec},
                    op_output_spec=None,
                ),
                # Pattern 4: Specific override for backbone.2
                "backbone.2": ModuleQuantizerConfig(
                    op_state_spec={"weight": backbone_2_spec},
                    op_input_spec={"*": backbone_2_spec},
                    op_output_spec=None,
                ),
                # Pattern 5: Classifier specific config
                "classifier": ModuleQuantizerConfig(
                    op_state_spec={"weight": linear_pattern_spec},
                    op_input_spec={"*": linear_pattern_spec},
                    op_output_spec=None,
                ),
            },
        )

        # Run the quantizer
        quantizer = Quantizer(model, config)
        example_inputs = (torch.randn(1, 3, 10, 10),)
        prepared_model = quantizer.prepare(example_inputs)

        # Verify config precedence by checking dtypes of fake quantize modules
        expected_configs = {
            "add": global_spec.dtype,
            # Top-level convs: module_type (int4) - no name matches
            "conv2d": conv_type_spec.dtype,  # conv1
            "conv2d_1": conv_type_spec.dtype,  # conv2
            # backbone.0: uses "backbone" module name config
            "conv2d_2": backbone_spec.dtype,  # backbone.0
            # backbone.2: uses "backbone.2" config
            "conv2d_3": backbone_2_spec.dtype,  # backbone.2
            # features.encoder.*: uses "features.encoder" module name config
            "linear": features_encoder_pattern_spec.dtype,  # features.encoder.0
            "linear_1": features_encoder_pattern_spec.dtype,  # features.encoder.2
            # features.decoder.*: uses "features" module name config and not
            # "features.decoder"
            "linear_2": features_pattern_spec.dtype,  # features.decoder.0
            "linear_3": features_pattern_spec.dtype,  # features.decoder.2
            # classifier: "classifier" specific config wins over linear type config
            "linear_4": linear_pattern_spec.dtype,  # classifier
        }

        # Check that each node has the expected configuration
        actual_configs = {}
        for node in prepared_model.graph.nodes:
            if node.op == "call_function" and node.name in expected_configs:
                if node.name == "add":
                    input_node = node.all_input_nodes[0]
                    quantizer_mod = getattr(prepared_model, input_node.target)
                    actual_configs[node.name] = quantizer_mod.dtype
                    continue
                # Get weight fake quantize module
                weight_input_node = node.all_input_nodes[1]
                weight_quantizer_mod = getattr(prepared_model, weight_input_node.target)
                actual_configs[node.name] = weight_quantizer_mod.dtype

        # Verify all expected configs were found and match
        for node_name, expected_dtype in expected_configs.items():
            assert node_name in actual_configs, f"Node {node_name} not found in prepared model"
            actual_dtype = actual_configs[node_name]
            assert actual_dtype == expected_dtype, (
                f"Config precedence failed for {node_name}: "
                f"expected {expected_dtype}, got {actual_dtype}"
            )

        # Additional verification: check that we found all the expected nodes
        assert len(actual_configs) == len(expected_configs), (
            f"Expected {len(expected_configs)} configured nodes, "
            f"but found {len(actual_configs)}: {list(actual_configs.keys())}"
        )

    @pytest.mark.parametrize(
        "config, expected",
        [
            (
                QuantizerConfig(
                    global_config=ModuleQuantizerConfig(
                        op_input_spec={1: default_activation_quantization_spec()},
                        op_output_spec=None,
                        op_type_config={
                            "linear": OpQuantizerConfig(op_input_spec=None, op_output_spec=None)
                        },
                    ),
                ),
                {
                    "add": {
                        "input_fq": [False, True],
                        "output_fq": False,
                    },
                    "linear_weight": {
                        "output_fq": True,
                    },
                    "linear": {
                        "input_fq": [False],
                        "output_fq": False,
                    },
                    "mul": {
                        "input_fq": [False, True],
                        "output_fq": False,
                    },
                    "relu": {
                        "input_fq": [False],
                        "output_fq": False,
                    },
                },
            ),
            (
                QuantizerConfig(
                    global_config=ModuleQuantizerConfig(
                        op_input_spec=None,
                        op_output_spec=None,
                        op_state_spec=None,
                        op_type_config={
                            "add": OpQuantizerConfig(
                                op_input_spec={1: default_activation_quantization_spec()},
                                op_output_spec=None,
                            ),
                            "linear": OpQuantizerConfig(
                                op_input_spec={0: default_activation_quantization_spec()},
                                op_output_spec=None,
                                op_state_spec=None,
                            ),
                            "mul": OpQuantizerConfig(
                                op_input_spec=None,
                                # Not specifying op_output_spec defaults it to
                                # quantizing with default
                                # Since the model has mul->relu pattern, this leads
                                # to relu output being quantized.
                            ),
                        },
                    )
                ),
                {
                    "add": {"input_fq": [False, True], "output_fq": True},
                    "linear_weight": {
                        "output_fq": False,
                    },
                    "linear": {"input_fq": [True], "output_fq": False},
                    "mul": {"input_fq": [False, False], "output_fq": False},
                    "relu": {"input_fq": [False], "output_fq": True},
                },
            ),
            (
                QuantizerConfig(
                    global_config=None,
                    module_type_configs={
                        SimpleMulOperatorModel: ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec=None,
                            op_name_config={
                                "mul": OpQuantizerConfig(),
                            },
                        )
                    },
                ),
                {
                    "add": {"input_fq": [False, False], "output_fq": False},
                    "linear_weight": {
                        "output_fq": False,
                    },
                    "linear": {"input_fq": [False], "output_fq": True},
                    "mul": {"input_fq": [True, True], "output_fq": False},
                    "relu": {"input_fq": [False], "output_fq": True},
                },
            ),
        ],
    )
    def test_specific_index_annotation(self, config, expected):
        model = AddLinearMulModel()
        example_inputs = (torch.randn(1, 2), torch.randn(1, 2))
        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_inputs)

        patterns = analyze_graph_structure(prepared_model)
        check_fake_quantize_placement(patterns, expected)

    def test_op_name_annotation(self):
        class AddModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, inp1, inp2, inp3):
                x = inp1 + inp2
                x = x + inp3
                return x

        config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_input_spec=None,
                op_output_spec=None,
                op_name_config={
                    "add_1": OpQuantizerConfig(
                        op_input_spec={1: default_activation_quantization_spec()},
                        op_output_spec={_ALL_TENSORS: default_activation_quantization_spec()},
                    )
                },
            )
        )

        model = AddModel()
        example_inputs = (torch.randn(1, 2), torch.randn(1, 2), torch.randn(1, 2))
        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_inputs)

        patterns = analyze_graph_structure(prepared_model)

        expected = {
            "add": {"input_fq": [False, False], "output_fq": False},
            "add_1": {"input_fq": [False, True], "output_fq": True},
        }
        check_fake_quantize_placement(patterns, expected)

    def test_op_level_precedence(self):
        class TwoLinearModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(2, 3, bias=False)
                self.linear2 = torch.nn.Linear(3, 4, bias=False)

            def forward(self, inp):
                x = self.linear1(inp)
                x = self.linear2(x)
                return x

        config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_input_spec=None,
                op_output_spec=None,
                op_state_spec=None,
                op_type_config={
                    "linear": OpQuantizerConfig(
                        op_input_spec={"*": per_tensor_int4_qspec()},
                        op_output_spec={"*": per_tensor_int4_qspec()},
                        op_state_spec=None,
                    ),
                },
                op_name_config={
                    "linear_1": OpQuantizerConfig(
                        op_input_spec={"*": per_tensor_int8_qspec()},
                        op_output_spec={"*": per_tensor_int8_qspec()},
                        op_state_spec=None,
                    )
                },
            )
        )
        model = TwoLinearModel()
        example_inputs = (torch.randn(1, 2),)
        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_inputs)

        assert prepared_model.activation_post_process_0.qparams_calculator.dtype == torch.int4
        assert prepared_model.activation_post_process_1.qparams_calculator.dtype == torch.int8

    def test_error_on_input_idx_being_state_tensor(self):
        model = SimpleLinearModel()
        config = QuantizerConfig(
            global_config=None,
            module_type_configs={
                torch.nn.Linear: ModuleQuantizerConfig(
                    op_input_spec={1: default_activation_quantization_spec()}
                )
            },
        )
        example_inputs = (torch.randn(1, 10),)
        quantizer = Quantizer(model, config)
        with pytest.raises(RuntimeError, match="Config is attempting to set op_input_spec."):
            _ = quantizer.prepare(example_inputs)

    def test_state_name_annotation(self):
        class InnerModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # self.linear_weight node name: inner_linear_weight_1
                self.linear_weight = torch.nn.Parameter(torch.randn(1, 2))
                self.relu = torch.nn.ReLU()

                # self.linear.weight node name: inner_linear_weight_2
                self.linear = torch.nn.Linear(2, 2, bias=False)

            def forward(self, inp, inp2):
                x = inp + inp2  # node name: add
                x = self.linear(x)  # node name: linear
                return x

        class SubModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, inp, inp2):
                x = inp - inp2  # node name: sub
                return x

        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = InnerModel()
                # self.inner_linear_weight node name: inner_linear_weight_3
                self.inner_linear = torch.nn.Linear(2, 2, bias=False)

                # self.inner_linear_weight node name: inner_linear_weight
                self.inner_linear_weight = torch.nn.Parameter(torch.randn(1, 2))
                self.sub_model = SubModel()

            def forward(self, inp):
                x = self.inner(self.inner_linear_weight, inp)
                x = self.inner_linear(x)  # node name: linear_1
                sub = self.sub_model(self.inner_linear.weight, self.inner.linear_weight)
                x = x + sub  # node name: add_1
                return x

        model = MyModel()
        example_inputs = (torch.randn(1, 2),)

        config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_input_spec=None,
                op_output_spec=None,
                op_state_spec=None,
                op_name_config={
                    "sub": OpQuantizerConfig(
                        # Since both inputs to sub are weights, this input spec should
                        # not affect any inputs. We will still expect sub op's second input to be
                        # quantized due to the "linear_weight" setting below.
                        op_input_spec={_ALL_TENSORS: None},
                        op_output_spec=None,
                    ),
                },
            ),
            module_type_configs={
                InnerModel: ModuleQuantizerConfig(
                    op_input_spec=None,
                    op_output_spec=None,
                    op_state_spec=None,
                    module_state_spec={"linear_weight": default_weight_quantization_spec()},
                    # Using "*" will allow for MyModel.inner_linear_weight to be quantized when used
                    # as add's first input.
                    op_name_config={
                        "add": OpQuantizerConfig(
                            op_input_spec={1: default_activation_quantization_spec()},
                            op_output_spec=None,
                            op_state_spec={_ALL_TENSORS: default_weight_quantization_spec()},
                        ),
                    },
                ),
            },
            module_name_configs={
                "inner_linear": ModuleQuantizerConfig(
                    op_input_spec=None,
                    op_output_spec=None,
                ),
            },
        )
        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_inputs)
        node_dict = {node.name: node for node in prepared_model.graph.nodes}

        assert "activation_post_process" in node_dict["add"].all_input_nodes[0].name
        weight_quantizer = getattr(prepared_model, node_dict["add"].all_input_nodes[0].name)
        act_quantizer = getattr(prepared_model, node_dict["add"].all_input_nodes[1].name)
        assert isinstance(weight_quantizer.qparams_calculator.granularity, PerChannelGranularity)
        assert isinstance(act_quantizer.qparams_calculator.granularity, PerTensorGranularity)

        assert "activation_post_process" in node_dict["sub"].all_input_nodes[0].name
        assert "activation_post_process" in node_dict["sub"].all_input_nodes[1].name
        assert "activation_post_process" in node_dict["linear_1"].all_input_nodes[1].name

        # Sub first input should share the same quantizer as linear_1's weight input
        assert (
            node_dict["sub"].all_input_nodes[0].name
            == node_dict["linear_1"].all_input_nodes[1].name
        )

        # Check that there are no other quantizers in the model
        assert (
            len([node_name for node_name in node_dict if "activation_post_process" in node_name])
            == 4
        )

    @pytest.mark.parametrize(
        "config, expected",
        [
            (
                QuantizerConfig(
                    global_config=None,
                    module_type_configs=None,
                ),
                {
                    "linear": {
                        "input_fq": [False],
                        "output_fq": False,
                    },
                },
            ),
            (
                QuantizerConfig(
                    global_config=None,
                    module_type_configs={torch.nn.Linear: None},
                ),
                {
                    "linear": {
                        "input_fq": [False],
                        "output_fq": False,
                    },
                },
            ),
            (
                QuantizerConfig(
                    global_config=None,
                    module_type_configs={
                        torch.nn.Linear: ModuleQuantizerConfig(
                            op_input_spec=None,
                        )
                    },
                ),
                {
                    "linear": {
                        "input_fq": [False],
                        "output_fq": True,
                    },
                },
            ),
            (
                QuantizerConfig(
                    global_config=None,
                    module_type_configs={
                        torch.nn.Linear: ModuleQuantizerConfig(
                            op_input_spec={_ALL_TENSORS: None},
                        )
                    },
                ),
                {
                    "linear": {
                        "input_fq": [False],
                        "output_fq": True,
                    },
                },
            ),
            (
                QuantizerConfig(
                    global_config=None,
                    module_type_configs={
                        torch.nn.Linear: ModuleQuantizerConfig(
                            op_output_spec={_ALL_TENSORS: None},
                        )
                    },
                ),
                {
                    "linear": {
                        "input_fq": [True],
                        "output_fq": False,
                    },
                },
            ),
        ],
    )
    def test_none_configs(self, config, expected):
        model = SimpleLinearModel()
        example_inputs = (torch.randn(1, 10),)

        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_inputs)

        patterns = analyze_graph_structure(prepared_model)
        check_fake_quantize_placement(patterns, expected)

    @pytest.mark.parametrize(
        "config, expected",
        [
            pytest.param(
                QuantizerConfig(
                    global_config=None,
                    module_name_configs={
                        "inner_module": ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec=None,
                            module_input_spec={1: default_activation_quantization_spec()},
                            module_output_spec={1: default_activation_quantization_spec()},
                        )
                    },
                ),
                {
                    "linear": {  # model.linear1
                        "input_fq": [False],
                        "output_fq": True,
                    },
                    "add": {"input_fq": [False, True], "output_fq": False},
                    "linear_1": {  # model.inner_module.linear_middle
                        "input_fq": [False],
                        "output_fq": False,
                    },
                    "linear_2": {  # model.inner_module.linear_out1
                        "input_fq": [False],
                        "output_fq": False,
                    },
                    "linear_3": {  # model.inner_module.linear_out2
                        "input_fq": [False],
                        "output_fq": True,
                    },
                    "sub": {"input_fq": [False, True], "output_fq": False},
                },
                id="all_disable_except_module_boundaries",
            ),
            pytest.param(
                QuantizerConfig(
                    module_name_configs={
                        "inner_module": ModuleQuantizerConfig(
                            module_input_spec={0: None},
                            module_output_spec={0: None},
                        )
                    }
                ),
                {
                    "linear": {  # model.linear1
                        "input_fq": [True],
                        "output_fq": True,
                    },
                    "add": {"input_fq": [False, True], "output_fq": True},
                    "linear_1": {  # model.inner_module.linear_middle
                        "input_fq": [True],
                        "output_fq": True,
                    },
                    "linear_2": {  # model.inner_module.linear_out1
                        "input_fq": [True],
                        "output_fq": False,
                    },
                    "linear_3": {  # model.inner_module.linear_out2
                        "input_fq": [True],
                        "output_fq": True,
                    },
                    "sub": {"input_fq": [True, True], "output_fq": True},
                },
                id="all_enabled_except_module_boundaries",
            ),
            pytest.param(
                QuantizerConfig(
                    global_config=None,
                    module_name_configs={
                        "inner_module\.linear_out.*": ModuleQuantizerConfig(
                            op_state_spec=None,
                            module_state_spec={"weight": default_weight_quantization_spec()},
                        )
                    },
                ),
                {
                    "linear": {  # model.linear1
                        "weight_fq": False
                    },
                    "linear_1": {  # model.inner_module.linear_middle
                        "weight_fq": False
                    },
                    "linear_2": {  # model.inner_module.linear_out1
                        "weight_fq": True
                    },
                    "linear_3": {  # model.inner_module.linear_out2
                        "weight_fq": True
                    },
                },
                id="inner_module_weight_quantization",
            ),
            pytest.param(
                QuantizerConfig(
                    global_config=None,
                    module_name_configs={
                        # By using a name which matches all modules, this will have the
                        # effect of quantizing all nested layers
                        ".*": ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec=None,
                            module_input_spec={"*": default_activation_quantization_spec()},
                            module_output_spec={"*": default_activation_quantization_spec()},
                        )
                    },
                ),
                {
                    "linear": {  # model.linear1
                        "input_fq": [True],
                        "output_fq": True,
                    },
                    "add": {"input_fq": [True, True], "output_fq": True},
                    "linear_1": {  # model.inner_module.linear_middle
                        "input_fq": [True],
                        "output_fq": True,
                    },
                    "linear_2": {  # model.inner_module.linear_out1
                        "input_fq": [True],
                        "output_fq": True,
                    },
                    "linear_3": {  # model.inner_module.linear_out2
                        "input_fq": [True],
                        "output_fq": True,
                    },
                    "sub": {"input_fq": [True, True], "output_fq": True},
                },
                id="equivalent_to_all_ops_enabled",
            ),
        ],
    )
    def test_module_level_settings(self, config, expected):
        model = NestedModel1()
        example_inputs = (torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2))
        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_inputs)

        patterns = analyze_graph_structure(prepared_model)
        check_fake_quantize_placement(patterns, expected)

    def test_module_level_setting_precedence(self):
        model = NestedModel1()
        example_inputs = (torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2))
        int4_spec = QuantizationSpec(
            dtype=torch.int4,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerTensorGranularity(),
            fake_quantize_cls="default",
            qparam_calculator_cls="default",
            range_calculator_cls="minmax",
        )

        config = QuantizerConfig(
            global_config=None,
            module_type_configs={
                InnerModel1: ModuleQuantizerConfig(
                    op_input_spec=None,
                    op_output_spec=None,
                    op_state_spec=None,
                    module_input_spec={"*": default_activation_quantization_spec()},
                    module_output_spec={"*": default_activation_quantization_spec()},
                )
            },
            module_name_configs={
                "inner_module\.linear_out1": ModuleQuantizerConfig(
                    op_input_spec=None,
                    op_output_spec=None,
                    op_state_spec=None,
                    module_input_spec={"*": int4_spec},
                    module_output_spec={"*": int4_spec},
                )
            },
        )
        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_inputs)

        expected = {
            "linear": {  # model.linear1
                "input_fq": [False],
                "output_fq": True,
            },
            "add": {"input_fq": [True, True], "output_fq": False},
            "linear_1": {  # model.inner_module.linear_middle
                "input_fq": [False],
                "output_fq": True,
            },
            "linear_2": {  # model.inner_module.linear_out1
                "input_fq": [True],
                "output_fq": True,
            },
            "linear_3": {  # model.inner_module.linear_out2
                "input_fq": [False],
                "output_fq": True,
            },
            "sub": {"input_fq": [False, True], "output_fq": False},
        }

        patterns = analyze_graph_structure(prepared_model)
        node_dict = {node.name: node for node in prepared_model.graph.nodes}
        check_fake_quantize_placement(patterns, expected)

        # Check for specific dtypes. The more specific linear_out1 module_name setting
        # should take precedence over the lower priority InnerModule module type
        # setting.
        inner_module_inp1_name = node_dict["add"].all_input_nodes[0].name
        inner_module_inp2_name = node_dict["add"].all_input_nodes[0].name
        inner_module_out1_name = list(node_dict["linear_2"].users.keys())[0].name
        inner_module_out2_name = list(node_dict["linear_3"].users.keys())[0].name
        linear_out1_inp1_name = node_dict["linear_2"].all_input_nodes[0].name
        # linear_out1's output is the same as inner_module_out1

        assert (
            getattr(prepared_model, inner_module_inp1_name).qparams_calculator.dtype == torch.int8
        )
        assert (
            getattr(prepared_model, inner_module_inp2_name).qparams_calculator.dtype == torch.int8
        )
        assert (
            getattr(prepared_model, inner_module_out1_name).qparams_calculator.dtype == torch.int4
        )
        assert (
            getattr(prepared_model, inner_module_out2_name).qparams_calculator.dtype == torch.int8
        )
        assert getattr(prepared_model, linear_out1_inp1_name).qparams_calculator.dtype == torch.int4

    @pytest.mark.parametrize(
        "config, expected",
        [
            pytest.param(
                QuantizerConfig(
                    global_config=None,
                    module_type_configs={
                        NestedModel2: ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec=None,
                            module_state_spec={"*": default_weight_quantization_spec()},
                        ),
                    },
                ),
                {
                    "add": {
                        "input_fq": [False],
                        "output_fq": False,
                        "weight_fq": {"inner_linear_weight": True},
                    },
                    "linear": {
                        "input_fq": [False],
                        "output_fq": False,
                        "weight_fq": {"inner.linear.weight": False},
                    },
                    "linear_1": {
                        "input_fq": [False],
                        "output_fq": False,
                        "weight_fq": {"weight": False},
                    },
                    "sub": {
                        "weight_fq": {"inner.linear_weight": False, "inner_linear.weight": False},
                        "output_fq": False,
                    },
                    "add_1": {"input_fq": [False, False], "output_fq": False},
                },
                id="nested_model_2_type_state",
            ),
            pytest.param(
                QuantizerConfig(
                    global_config=None,
                    module_type_configs={
                        NestedModel2InnerModel: ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec=None,
                            module_state_spec={"*": default_weight_quantization_spec()},
                        ),
                    },
                ),
                {
                    "add": {
                        "input_fq": [False],
                        "output_fq": False,
                        "weight_fq": {"inner_linear_weight": False},
                    },
                    "linear": {
                        "input_fq": [False],
                        "output_fq": False,
                        "weight_fq": {"inner.linear.weight": False},
                    },
                    "linear_1": {
                        "input_fq": [False],
                        "output_fq": False,
                        "weight_fq": {"weight": False},
                    },
                    "sub": {
                        "weight_fq": {"inner.linear_weight": True, "inner_linear.weight": False},
                        "output_fq": False,
                    },
                    "add_1": {"input_fq": [False, False], "output_fq": False},
                },
                id="inner_model_2_type_state",
            ),
            pytest.param(
                QuantizerConfig(
                    global_config=None,
                    module_type_configs={
                        NestedModel2SubModel: ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec=None,
                            module_state_spec={"*": default_weight_quantization_spec()},
                        ),
                    },
                ),
                {
                    "add": {
                        "input_fq": [False],
                        "output_fq": False,
                        "weight_fq": {"inner_linear_weight": False},
                    },
                    "linear": {
                        "input_fq": [False],
                        "output_fq": False,
                        "weight_fq": {"inner.linear.weight": False},
                    },
                    "linear_1": {
                        "input_fq": [False],
                        "output_fq": False,
                        "weight_fq": {"weight": False},
                    },
                    "sub": {
                        "weight_fq": {"inner.linear_weight": False, "inner_linear.weight": False},
                        "output_fq": False,
                    },
                    "add_1": {"input_fq": [False, False], "output_fq": False},
                },
                id="sub_model_2_type_state",
            ),
        ],
    )
    def test_module_state_name_annotation(self, config, expected):
        model = NestedModel2()
        example_inputs = (torch.randn(1, 2),)
        _ = model(*example_inputs)

        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_inputs)

        patterns = analyze_graph_structure(prepared_model)
        check_fake_quantize_placement(patterns, expected)

    @pytest.mark.parametrize(
        "config, expected",
        [
            pytest.param(
                QuantizerConfig(
                    global_config=ModuleQuantizerConfig(op_state_spec=None, op_output_spec=None)
                ),
                {
                    "flatten": {"weight_fq": {"param": False}, "output_fq": True},
                    "add": {
                        "input_fq": [True, True],
                        "output_fq": False,
                    },
                },
                id="only_input_quantized",
            ),
            pytest.param(
                QuantizerConfig(
                    global_config=ModuleQuantizerConfig(
                        op_state_spec={
                            "*": QuantizationSpec(
                                dtype=torch.int4,
                                qscheme=QuantizationScheme.SYMMETRIC,
                                granularity=PerChannelGranularity(axis=0),
                                fake_quantize_cls="default",
                                qparam_calculator_cls="default",
                                range_calculator_cls="minmax",
                            )
                        },
                        op_output_spec=None,
                    )
                ),
                {
                    "flatten": {"weight_fq": {"param": True}, "output_fq": True},
                    "add": {
                        "input_fq": [True, True],
                        "output_fq": False,
                    },
                },
                id="both_input_and_state_quantized",
            ),
        ],
    )
    def test_shared_observer_op_after_state(self, config, expected):
        class StateWithSharedObserver(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param = torch.nn.Parameter(torch.ones(1, 2, 3))
                self.flatten = torch.nn.Flatten()

            def forward(self, inp):
                flattened_param = self.flatten(self.param)
                return inp + flattened_param

        model = StateWithSharedObserver()
        example_inputs = (torch.zeros(1, 6),)
        _ = model(*example_inputs)

        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_inputs)
        patterns = analyze_graph_structure(prepared_model)
        check_fake_quantize_placement(patterns, expected)

        node_dict = {node.name: node for node in prepared_model.graph.nodes}
        flatten_preceding_quantizer_name = node_dict["flatten"].all_input_nodes[0].name
        flatten_succeeding_node_name = node_dict["add"].all_input_nodes[1].name
        add_first_input_node_name = node_dict["add"].all_input_nodes[0].name

        flatten_succeeding_quantizer = getattr(prepared_model, flatten_succeeding_node_name)
        add_first_input_quantizer = getattr(prepared_model, add_first_input_node_name)
        assert add_first_input_quantizer.dtype == torch.int8

        if config.global_config.op_state_spec:
            flatten_preceding_quantizer = getattr(prepared_model, flatten_preceding_quantizer_name)
            assert flatten_preceding_quantizer is flatten_succeeding_quantizer
            assert flatten_preceding_quantizer.dtype == torch.int4
        else:
            assert "activation_post_process" not in flatten_preceding_quantizer_name
            assert flatten_succeeding_quantizer.dtype == torch.int8

    @pytest.mark.parametrize(
        "config, expected",
        [
            pytest.param(
                QuantizerConfig(
                    global_config=None,
                    module_name_configs={
                        "": ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec={"my_buf": default_weight_quantization_spec()},
                        )
                    },
                ),
                {
                    "add": {"input_fq": [False], "output_fq": False, "weight_fq": {"my_buf": True}},
                },
                id="op_state_spec",
            ),
            pytest.param(
                QuantizerConfig(
                    global_config=None,
                    module_name_configs={
                        "": ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec=None,
                            module_state_spec={"my_buf": default_weight_quantization_spec()},
                        )
                    },
                ),
                {
                    "add": {"input_fq": [False], "output_fq": False, "weight_fq": {"my_buf": True}},
                },
                id="module_state_spec",
            ),
        ],
    )
    def test_buffer_state(self, config, expected):
        class ModelWithBuffer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("my_buf", torch.randn(1, 2))

            def forward(self, inp):
                return inp + self.my_buf

        model = ModelWithBuffer()
        example_inputs = (torch.randn(1, 2),)
        _ = model(*example_inputs)

        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_inputs)
        patterns = analyze_graph_structure(prepared_model)
        check_fake_quantize_placement(patterns, expected)

    @pytest.mark.parametrize(
        "config, expected",
        [
            pytest.param(
                QuantizerConfig(
                    global_config=None,
                    module_name_configs={
                        "": ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec=None,
                            module_state_spec={
                                "param1": default_activation_quantization_spec(),
                                "buffer2": default_activation_quantization_spec(),
                            },
                        )
                    },
                ),
                {
                    "add": {"input_fq": [False], "output_fq": False, "weight_fq": True},
                    "div": {"input_fq": [False], "output_fq": False, "weight_fq": True},
                    "sub": {"input_fq": [False], "output_fq": False, "weight_fq": True},
                    "mul": {"input_fq": [False], "output_fq": False, "weight_fq": True},
                    "add_1": {"input_fq": [False], "output_fq": False, "weight_fq": False},
                    "add_2": {"input_fq": [False], "output_fq": False, "weight_fq": False},
                    "mul_1": {"input_fq": [False], "output_fq": False, "weight_fq": False},
                },
                id="param1_and_buffer2",
            ),
            pytest.param(
                QuantizerConfig(
                    global_config=None,
                    module_name_configs={
                        "": ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec=None,
                            module_state_spec={
                                "param2": default_activation_quantization_spec(),
                                "buffer1": default_activation_quantization_spec(),
                            },
                        )
                    },
                ),
                {
                    "add": {"input_fq": [False], "output_fq": False, "weight_fq": True},
                    "div": {"input_fq": [False], "output_fq": False, "weight_fq": True},
                    "sub": {"input_fq": [False], "output_fq": False, "weight_fq": True},
                    "mul": {"input_fq": [False], "output_fq": False, "weight_fq": True},
                    "add_1": {"input_fq": [False], "output_fq": False, "weight_fq": False},
                    "add_2": {"input_fq": [False], "output_fq": False, "weight_fq": False},
                    "mul_1": {"input_fq": [False], "output_fq": False, "weight_fq": False},
                },
                id="param2_and_buffer1",
            ),
            pytest.param(
                QuantizerConfig(
                    global_config=None,
                    module_name_configs={
                        "inner_module_1": ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec=None,
                            module_state_spec={
                                "inner_param": default_activation_quantization_spec()
                            },
                        )
                    },
                ),
                {
                    "add_1": {"input_fq": [False], "output_fq": False, "weight_fq": True},
                    "add_2": {"input_fq": [False], "output_fq": False, "weight_fq": False},
                    "mul_1": {"input_fq": [False], "output_fq": False, "weight_fq": True},
                },
                id="inner_module_1_inner_param",
            ),
            pytest.param(
                QuantizerConfig(
                    global_config=None,
                    module_name_configs={
                        "inner_module_1": ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec=None,
                            module_state_spec={
                                # inner_module_1 doesn't have "a", so this should not
                                # have any effect, even though inner_param is the same
                                # object as inner_module_2.a
                                "a": default_activation_quantization_spec()
                            },
                        )
                    },
                ),
                {
                    "add_1": {"input_fq": [False], "output_fq": False, "weight_fq": False},
                    "add_2": {"input_fq": [False], "output_fq": False, "weight_fq": False},
                    "mul_1": {"input_fq": [False], "output_fq": False, "weight_fq": False},
                },
                id="inner_module_1_a_no_effect",
            ),
            pytest.param(
                QuantizerConfig(
                    global_config=None,
                    module_name_configs={
                        "inner_module_2": ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec=None,
                            module_state_spec={
                                # inner_module_1 doesn't have "a", so this should not
                                # have any effect, even though inner_param is the same
                                # object as inner_module_2.a
                                "a": default_activation_quantization_spec()
                            },
                        )
                    },
                ),
                {
                    "add_1": {"input_fq": [False], "output_fq": False, "weight_fq": True},
                    "add_2": {"input_fq": [False], "output_fq": False, "weight_fq": False},
                    "mul_1": {"input_fq": [False], "output_fq": False, "weight_fq": True},
                },
                id="inner_module_2_a",
            ),
            pytest.param(
                QuantizerConfig(
                    global_config=None,
                    module_name_configs={
                        "inner_module_2": ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec=None,
                            module_state_spec={
                                # inner_module_1 doesn't have "a", so this should not
                                # have any effect, even though inner_param is the same
                                # object as inner_module_2.a
                                "inner_param": default_activation_quantization_spec()
                            },
                        )
                    },
                ),
                {
                    "add_1": {"input_fq": [False], "output_fq": False, "weight_fq": False},
                    "add_2": {"input_fq": [False], "output_fq": False, "weight_fq": True},
                    "mul_1": {"input_fq": [False], "output_fq": False, "weight_fq": False},
                },
                id="inner_module_2_inner_param",
            ),
        ],
    )
    def test_shared_param_module_state(self, config, expected):
        """Test that shared params are quantized as expected for module state"""

        class InnerModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner_param = torch.nn.Parameter(torch.randn(1, 2))

            def forward(self, inp):
                x = inp + self.inner_param
                return x

        class InnerModel2(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.a = torch.nn.Parameter(torch.randn(1, 2))
                self.inner_param = torch.nn.Parameter(torch.randn(1, 2))

            def forward(self, inp):
                x = inp + self.inner_param
                x = x * self.a
                return x

        class ModelWithDuplicateStates(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.param1 = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
                self.param2 = self.param1
                self.register_buffer("buffer1", torch.tensor([3.0, 4.0]))
                self.register_buffer("buffer2", self.buffer1)
                self.inner_module_1 = InnerModel()
                self.inner_module_2 = InnerModel2()
                self.inner_module_2.a = self.inner_module_1.inner_param

            def forward(self, inp):
                x = inp + self.param1
                x = x / self.param2
                x = x - self.buffer1
                x = x * self.buffer2
                x = self.inner_module_1(x)
                x = self.inner_module_2(x)
                return x

        model = ModelWithDuplicateStates()
        example_inputs = (torch.randn(1, 2),)

        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_inputs)
        patterns = analyze_graph_structure(prepared_model)
        check_fake_quantize_placement(patterns, expected)

    @pytest.mark.parametrize(
        "config, expected",
        [
            pytest.param(
                QuantizerConfig(
                    global_config=None,
                    module_name_configs={
                        "layer2": ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec=None,
                            module_state_spec={"weight": per_tensor_int8_qspec()},
                        ),
                        "layer1": ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec=None,
                            module_state_spec={"weight": per_tensor_int4_qspec()},
                        ),
                    },
                ),
                {
                    "linear_1": {"input_fq": [False], "output_fq": False, "weight_fq": True},
                    "linear_2": {"input_fq": [False], "output_fq": False, "weight_fq": True},
                },
                id="module_state",
            ),
            pytest.param(
                QuantizerConfig(
                    global_config=None,
                    module_name_configs={
                        "layer2": ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec={"weight": per_tensor_int8_qspec()},
                        ),
                        "layer1": ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec={"weight": per_tensor_int4_qspec()},
                        ),
                    },
                ),
                {
                    "linear_1": {"input_fq": [False], "output_fq": False, "weight_fq": True},
                    "linear_2": {"input_fq": [False], "output_fq": False, "weight_fq": True},
                },
                id="op_state_multiple_specs_for_shared_state",
            ),
            pytest.param(
                QuantizerConfig(
                    global_config=None,
                    module_name_configs={
                        "layer2": ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec={"weight": per_tensor_int8_qspec()},
                        ),
                        "layer1": ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec={"weight": None},
                        ),
                    },
                ),
                {
                    "linear_1": {"input_fq": [False], "output_fq": False, "weight_fq": None},
                    "linear_2": {"input_fq": [False], "output_fq": False, "weight_fq": None},
                },
                id="op_state_multiple_specs_with_None",
            ),
            pytest.param(
                QuantizerConfig(
                    global_config=None,
                    module_name_configs={
                        "layer1": ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec={"weight": per_tensor_int4_qspec()},
                        ),
                    },
                ),
                {
                    "linear_1": {"input_fq": [False], "output_fq": False, "weight_fq": True},
                    "linear_2": {"input_fq": [False], "output_fq": False, "weight_fq": True},
                },
                id="op_state_shared_state_single_qspec",
            ),
        ],
    )
    def test_shared_param_precedence(
        self, shared_params_model, shared_params_model_input, config, expected
    ):
        """
        Test that the last applicable config is used for a model with shared params with
        multiple specs for the shared param.
        """
        quantizer = Quantizer(shared_params_model, config)
        prepared_model = quantizer.prepare((shared_params_model_input,))
        patterns = analyze_graph_structure(prepared_model)
        check_fake_quantize_placement(patterns, expected)

        fq_nodes = [
            node for node in prepared_model.graph.nodes if "activation_post_process" in node.name
        ]
        if config.module_name_configs["layer1"].op_state_spec.get(
            "weight"
        ) or config.module_name_configs["layer1"].module_state_spec.get("weight"):
            assert len(fq_nodes) == 1
            fq_module = getattr(prepared_model, fq_nodes[0].name)
            assert fq_module.qparams_calculator.dtype == torch.int4
        else:
            assert len(fq_nodes) == 0

    @pytest.mark.parametrize(
        "config, match",
        [
            (
                QuantizerConfig(
                    global_config=None,
                    module_type_configs={
                        torch.nn.Linear: ModuleQuantizerConfig(
                            op_input_spec={"input": default_activation_quantization_spec()},
                        )
                    },
                ),
                "Only integer indices or '\*' are supported",
            ),
            (
                QuantizerConfig(
                    global_config=None,
                    module_type_configs={
                        torch.nn.Linear: ModuleQuantizerConfig(
                            module_input_spec={"input": default_activation_quantization_spec()},
                        )
                    },
                ),
                "Only integer indices or '\*' are supported",
            ),
            (
                QuantizerConfig(
                    global_config=None,
                    module_type_configs={
                        torch.nn.Linear: ModuleQuantizerConfig(
                            op_output_spec={1: default_activation_quantization_spec()},
                        )
                    },
                ),
                "op_output_qspec currently supports",
            ),
        ],
    )
    def test_validate_unsupported_configs(self, config, match):
        model = SimpleLinearModel()
        with pytest.raises(NotImplementedError, match=match):
            _ = Quantizer(model, config)


class TestAnnotationPattern:
    """Test suite for annotation pattern class."""

    def test_pattern_length_validation(self):
        """Test pattern validation and get pattern length behaviors"""

        class EmptyPattern(WeightedModulePattern):
            @classmethod
            def generate_patterns(cls) -> list[torch.fx.GraphModule]:
                patterns = []
                return patterns

        # Test that an empty pattern raises runtime error
        pattern = EmptyPattern()
        with pytest.raises(RuntimeError):
            pattern.get_pattern_length()

        mock_call_function_node = Mock()
        mock_call_function_node.op = "call_function"
        mock_call_module_node = Mock()
        mock_call_module_node.op = "call_module"

        class ValidPattern(WeightedModulePattern):
            @classmethod
            def generate_patterns(cls) -> list[torch.fx.GraphModule]:
                mock_pattern_1 = Mock()
                mock_pattern_1.graph.nodes = [mock_call_function_node, mock_call_module_node]
                mock_pattern_2 = Mock()
                mock_pattern_2.graph.nodes = [mock_call_function_node, mock_call_function_node]

                patterns = [mock_pattern_1, mock_pattern_2]
                return patterns

        # Test normal pattern class behavior with 2 patterns of length 2
        pattern = ValidPattern()
        assert pattern.get_pattern_length() == 2

        # Test that a pattern class with different pattern lengths fails validation
        class PatternWithDifferentPatternLengths(WeightedModulePattern):
            @classmethod
            def generate_patterns(cls) -> list[torch.fx.GraphModule]:
                mock_pattern_1 = Mock()
                mock_pattern_1.graph.nodes = [mock_call_function_node]
                mock_pattern_2 = Mock()
                mock_pattern_2.graph.nodes = [mock_call_function_node, mock_call_function_node]

                patterns = [mock_pattern_1, mock_pattern_2]
                return patterns

        pattern = PatternWithDifferentPatternLengths()
        with pytest.raises(RuntimeError):
            pattern.get_patterns()

        # Test that a pattern class with patterns of same number of call functions with
        # other nodes passes.
        mock_other_node = Mock()
        mock_other_node.op = "placeholder"

        class PatternWithOtherNodeOpType(WeightedModulePattern):
            @classmethod
            def generate_patterns(cls) -> list[torch.fx.GraphModule]:
                mock_pattern_1 = Mock()
                mock_pattern_1.graph.nodes = [mock_call_function_node]
                mock_pattern_2 = Mock()
                mock_pattern_2.graph.nodes = [mock_call_function_node, mock_other_node]

                patterns = [mock_pattern_1, mock_pattern_2]
                return patterns

        pattern = PatternWithOtherNodeOpType()
        assert pattern.get_pattern_length() == 1

    def test_shared_pattern_length_validation(self):
        """
        Test that shared patterns with pattern length > 1 raise an error.
        """

        class Pattern(SharedObserverModulePattern):
            @classmethod
            def generate_patterns(cls) -> list[torch.fx.GraphModule]:
                mock_pattern_1 = Mock()
                mock_call_function_node = Mock()
                mock_call_function_node.op = "call_function"
                mock_pattern_1.graph.nodes = [mock_call_function_node, mock_call_function_node]

                patterns = [mock_pattern_1]
                return patterns

        pattern = Pattern()
        with pytest.raises(RuntimeError):
            pattern.get_patterns()
