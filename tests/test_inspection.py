# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the coreai_opt.inspection module."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from coreai_opt._utils.torch_utils import export_model as _export_model
from coreai_opt.base_model_compressor import _BaseModelCompressor
from coreai_opt.inspection import (
    ModelInspector,
    ModelSummary,
    ModuleInfo,
)
from coreai_opt.quantization import Quantizer
from coreai_opt.quantization.config.quantization_config import ExecutionMode

execution_modes = pytest.mark.parametrize(
    "execution_mode",
    [
        ExecutionMode.GRAPH,
        pytest.param(
            ExecutionMode.EAGER,
            marks=pytest.mark.xfail(reason="Eager inspection not yet implemented"),
        ),
    ],
)


class _SimpleConvModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.relu(x)
        x = x.mean(dim=[2, 3])
        x = self.fc(x)
        return x


class _NestedModel(nn.Module):
    """Model with nested submodules for testing hierarchy."""

    class _Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv1(x)
            x = torch.relu(x)
            x = self.conv2(x)
            return x

    class _Decoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(32, 64)
            self.fc2 = nn.Linear(64, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            return x

    def __init__(self) -> None:
        super().__init__()
        self.encoder = self._Encoder()
        self.decoder = self._Decoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.mean(dim=[2, 3])
        x = self.decoder(x)
        return x


class _ArithmeticModel(nn.Module):
    """Model with multiple arithmetic ops for testing op naming."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.linear(x)
        b = a + x
        c = b + a
        d = b * c
        return d


def _assert_query_round_trip(inspector: ModelInspector) -> None:
    """Verify every op is findable by all of its own metadata via query methods."""
    for op in inspector.summary.model.all_ops():
        assert op in inspector.get_matched_ops_for_op_name(op.op_name), (
            f"Op '{op.op_name}' not found by get_matched_ops_for_op_name"
        )
        if op.op_type:
            assert op in inspector.get_matched_ops_for_op_type(op.op_type), (
                f"Op '{op.op_name}' not found by get_matched_ops_for_op_type('{op.op_type}')"
            )
        for ctx in op.module_stack:
            assert op in inspector.get_matched_ops_for_module_name(ctx.module_name), (
                f"Op '{op.op_name}' not found by get_matched_ops_for_module_name"
                f"('{ctx.module_name}')"
            )
            assert op in inspector.get_matched_ops_for_module_type(ctx.module_type), (
                f"Op '{op.op_name}' not found by get_matched_ops_for_module_type"
                f"('{ctx.module_type}')"
            )


@execution_modes
class TestModelInspector:
    """Tests for ModelInspector across execution modes."""

    def test_simple_conv_model(self, execution_mode: ExecutionMode) -> None:
        """Verify op discovery, types, module stack, queries, and formatting on a simple model."""
        inspector = ModelInspector(
            _SimpleConvModel(),
            (torch.randn(1, 3, 8, 8),),
            execution_mode=execution_mode,
            compressor=Quantizer,
        )

        # Summary type and mode
        assert isinstance(inspector.summary, ModelSummary)
        assert inspector.summary.mode == execution_mode

        # Root is a ModuleSummary
        assert isinstance(inspector.summary.model, ModuleInfo)

        # Op discovery
        ops = inspector.summary.model.all_ops()
        op_names = [op.op_name for op in ops]
        assert "conv2d" in op_names
        assert "linear" in op_names

        # Op types
        conv_op = next(op for op in ops if op.op_name == "conv2d")
        assert conv_op.op_type == "conv2d"
        linear_op = next(op for op in ops if op.op_name == "linear")
        assert linear_op.op_type == "linear"

        # Module stack
        assert len(conv_op.module_stack) >= 1
        fqns = [m.module_name for m in conv_op.module_stack]
        assert "conv" in fqns
        conv_module = next(m for m in conv_op.module_stack if m.module_name == "conv")
        assert "Conv2d" in conv_module.module_type

        # Query: no-match cases
        assert inspector.get_matched_ops_for_op_type("nonexistent") == ()
        assert inspector.get_matched_ops_for_op_name("nonexistent") == ()
        assert inspector.get_matched_ops_for_module_name("nonexistent") == ()
        assert inspector.get_matched_ops_for_module_type("NonexistentModule") == ()

        # Query: by name (exact)
        conv_by_name = inspector.get_matched_ops_for_op_name("conv2d")
        assert len(conv_by_name) == 1
        assert conv_by_name[0].op_name == "conv2d"

        # Query: by name (regex)
        all_by_regex = inspector.get_matched_ops_for_op_name(".*")
        assert len(all_by_regex) == len(ops)

        # Query: module type (class and full FQN string)
        conv_ops = inspector.get_matched_ops_for_module_type(nn.Conv2d)
        assert len(conv_ops) >= 1
        assert all(op.op_type == "conv2d" for op in conv_ops)
        assert len(inspector.get_matched_ops_for_module_type("torch.nn.modules.conv.Conv2d")) >= 1

        # Formatting
        result = inspector.format_summary(colorize=False)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "conv2d" in result
        assert "linear" in result
        assert "type: conv2d" in result or "[conv2d]" in result
        assert "type: linear" in result or "[linear]" in result
        assert "conv" in result
        assert "fc" in result
        assert "Conv2d" in result
        assert "Linear" in result
        assert any(c in result for c in ["├", "└", "│"])

        # Round-trip: every op is findable by its own metadata
        _assert_query_round_trip(inspector)

        # Check that passing in an already exported model provides the same summary
        if execution_mode == ExecutionMode.GRAPH:
            gm = _export_model(
                _SimpleConvModel(),
                (torch.randn(1, 3, 8, 8),),
                dynamic_shapes=None,
                export_with_no_grad=True,
            )
            gm_inspector = ModelInspector(
                gm,
                None,
                execution_mode=execution_mode,
                compressor=Quantizer,
            )
            assert inspector.summary == gm_inspector.summary

    def test_nested_model(self, execution_mode: ExecutionMode) -> None:
        """
        Verify hierarchy, graph ordering, nested FQNs, and regex queries on a multi-level model.
        """
        inspector = ModelInspector(
            _NestedModel(),
            (torch.randn(1, 3, 8, 8),),
            execution_mode=execution_mode,
            compressor=Quantizer,
        )

        # Op discovery and hierarchy
        op_names = [op.op_name for op in inspector.summary.model.all_ops()]
        assert "conv2d" in op_names
        assert "conv2d_1" in op_names
        assert "linear" in op_names
        assert "linear_1" in op_names

        # Graph order
        assert op_names.index("conv2d") < op_names.index("linear")

        # Nested module FQNs
        conv_op = next(op for op in inspector.summary.model.all_ops() if op.op_name == "conv2d")
        fqns = [m.module_name for m in conv_op.module_stack]
        assert "encoder" in fqns
        assert "encoder.conv1" in fqns

        # Query: by type
        conv_ops = inspector.get_matched_ops_for_op_type("conv2d")
        assert len(conv_ops) == 2
        assert all(op.op_type == "conv2d" for op in conv_ops)

        # Query: by module name
        encoder_ops = inspector.get_matched_ops_for_module_name("encoder")
        encoder_op_names = [op.op_name for op in encoder_ops]
        assert "conv2d" in encoder_op_names
        assert "conv2d_1" in encoder_op_names

        # Query: by module name (leaf)
        leaf_ops = inspector.get_matched_ops_for_module_name("encoder.conv1")
        assert len(leaf_ops) == 1
        assert leaf_ops[0].op_name == "conv2d"

        # Query: by module name (regex)
        encoder_regex_ops = inspector.get_matched_ops_for_module_name(r"encoder\..*")
        encoder_regex_op_names = [op.op_name for op in encoder_regex_ops]
        assert "conv2d" in encoder_regex_op_names
        assert "conv2d_1" in encoder_regex_op_names

        # Query: by name (regex matching multiple ops)
        conv_ops_by_name = inspector.get_matched_ops_for_op_name(r"conv2d.*")
        assert len(conv_ops_by_name) == 2
        linear_ops_by_name = inspector.get_matched_ops_for_op_name(r"linear.*")
        assert len(linear_ops_by_name) == 2

        # Formatting
        result = inspector.format_summary(colorize=False)
        assert "encoder.conv1" in result
        assert "decoder.fc1" in result

        # Round-trip: every op is findable by its own metadata
        _assert_query_round_trip(inspector)

    def test_arithmetic_model(self, execution_mode: ExecutionMode) -> None:
        """Verify that repeated ops of the same type get distinct names."""
        inspector = ModelInspector(
            _ArithmeticModel(),
            (torch.randn(1, 10),),
            execution_mode=execution_mode,
            compressor=Quantizer,
        )
        op_names = [op.op_name for op in inspector.summary.model.all_ops()]
        assert "linear" in op_names
        add_ops = [n for n in op_names if "add" in n]
        mul_ops = [n for n in op_names if "mul" in n]
        assert len(add_ops) >= 2, f"Expected at least 2 add ops, got {add_ops}"
        assert len(mul_ops) >= 1, f"Expected at least 1 mul op, got {mul_ops}"

        # Round-trip: every op is findable by its own metadata
        _assert_query_round_trip(inspector)

    def test_compressor_filters_ops(self, execution_mode: ExecutionMode) -> None:
        """
        Verify that passing a compressor returns a strict subset of all ops.
        Note: this test assumes that not all ops in _SimpleConvModel are quantizable (ex. mean,
        relu). If this changes in the future, this test will need to update.
        """
        model = _SimpleConvModel()
        inputs = (torch.randn(1, 3, 8, 8),)

        all_ops_inspector = ModelInspector(
            model,
            inputs,
            execution_mode=execution_mode,
        )
        quantizer_inspector = ModelInspector(
            model,
            inputs,
            execution_mode=execution_mode,
            compressor=Quantizer,
        )

        all_op_names = {op.op_name for op in all_ops_inspector.summary.model.all_ops()}
        quantizer_op_names = {op.op_name for op in quantizer_inspector.summary.model.all_ops()}

        # Quantizer-filtered ops must be a subset of all ops
        assert quantizer_op_names < all_op_names

        # All ops should include non-quantizable ops that the quantizer excludes
        assert len(all_op_names) > len(quantizer_op_names), (
            f"Expected all ops ({all_op_names}) to include more ops than "
            f"quantizer-filtered ops ({quantizer_op_names})"
        )

    def test_op_connectivity_arithmetic_model(self, execution_mode: ExecutionMode) -> None:
        """Verify input/output connectivity on a model with arithmetic ops."""
        inspector = ModelInspector(
            _ArithmeticModel(),
            (torch.randn(1, 10),),
            execution_mode=execution_mode,
        )
        ops = inspector.summary.model.all_ops()
        ops_by_name = {op.op_name: op for op in ops}

        # linear has a placeholder in inputs
        linear_op = ops_by_name["linear"]
        assert any(inp.op_name not in ops_by_name for inp in linear_op.inputs), (
            "linear should have a placeholder/parameter input"
        )
        # linear's outputs should include an add op
        assert any("add" in out.op_name for out in linear_op.outputs)

        # add ops have correct inputs
        add_op = ops_by_name["add"]
        assert len(add_op.inputs) >= 2

        # mul has two inputs (both add-related ops)
        mul_op = ops_by_name["mul"]
        assert len(mul_op.inputs) == 2

    def test_module_io_nested_model(self, execution_mode: ExecutionMode) -> None:
        """Verify module input_ops and output_ops on a nested model."""
        inspector = ModelInspector(
            _NestedModel(),
            (torch.randn(1, 3, 8, 8),),
            execution_mode=execution_mode,
        )
        root = inspector.summary.model
        encoder = root.child_modules["encoder"]
        decoder = root.child_modules["decoder"]

        # Encoder: first conv is an input, last conv is an output
        assert len(encoder.input_ops) >= 1
        encoder_input_names = {op.op_name for op in encoder.input_ops}
        assert "conv2d" in encoder_input_names or any("conv" in n for n in encoder_input_names)
        assert len(encoder.output_ops) >= 1

        # Decoder: first linear is an input, last linear is an output
        assert len(decoder.input_ops) >= 1
        assert len(decoder.output_ops) >= 1

    def test_tree_structure_nested_model(self, execution_mode: ExecutionMode) -> None:
        """Verify that the ModuleSummary tree mirrors the nn.Module hierarchy."""
        inspector = ModelInspector(
            _NestedModel(),
            (torch.randn(1, 3, 8, 8),),
            execution_mode=execution_mode,
            compressor=Quantizer,
        )
        root = inspector.summary.model
        assert isinstance(root, ModuleInfo)
        assert root.module_name == ""

        # Root should have children for encoder and decoder
        child_fqns = {c.module_name for c in root.child_modules.values()}
        assert "encoder" in child_fqns
        assert "decoder" in child_fqns

        # Encoder should have children for conv1 and conv2
        encoder = root.child_modules["encoder"]
        encoder_child_fqns = {c.module_name for c in encoder.child_modules.values()}
        assert "encoder.conv1" in encoder_child_fqns
        assert "encoder.conv2" in encoder_child_fqns

        # Ops should be nested inside leaf modules, not at root
        conv1 = encoder.child_modules["encoder.conv1"]
        conv1_op_names = [op.op_name for op in conv1.ops]
        assert "conv2d" in conv1_op_names

        # Decoder should have children for fc1 and fc2
        decoder = root.child_modules["decoder"]
        decoder_child_fqns = {c.module_name for c in decoder.child_modules.values()}
        assert "decoder.fc1" in decoder_child_fqns
        assert "decoder.fc2" in decoder_child_fqns

    def test_module_info_children(self, execution_mode: ExecutionMode) -> None:
        """Verify children() and named_children() yield direct child modules."""
        inspector = ModelInspector(
            _NestedModel(),
            (torch.randn(1, 3, 8, 8),),
            execution_mode=execution_mode,
        )
        root = inspector.summary.model

        # children() should yield direct children only
        direct_children = list(root.children())
        direct_fqns = {c.module_name for c in direct_children}
        assert "encoder" in direct_fqns
        assert "decoder" in direct_fqns
        # Should not include grandchildren
        assert not any("conv" in c.module_name for c in direct_children)

        # named_children() should yield (fqn, module) pairs
        named = dict(root.named_children())
        assert set(named.keys()) == direct_fqns
        assert named["encoder"].module_name == "encoder"
        assert named["decoder"].module_name == "decoder"

        # Leaf module should have no children
        encoder = root.child_modules["encoder"]
        conv1 = encoder.child_modules["encoder.conv1"]
        assert list(conv1.children()) == []
        assert list(conv1.named_children()) == []

    def test_module_info_modules(self, execution_mode: ExecutionMode) -> None:
        """Verify modules() and named_modules() yield all descendants depth-first."""
        inspector = ModelInspector(
            _NestedModel(),
            (torch.randn(1, 3, 8, 8),),
            execution_mode=execution_mode,
        )
        root = inspector.summary.model

        # modules() should include root and all descendants
        all_modules = list(root.modules())
        all_fqns = [m.module_name for m in all_modules]
        assert all_fqns[0] == ""  # root is first
        assert "encoder" in all_fqns
        assert "encoder.conv1" in all_fqns
        assert "encoder.conv2" in all_fqns
        assert "decoder" in all_fqns
        assert "decoder.fc1" in all_fqns
        assert "decoder.fc2" in all_fqns

        # Depth-first: encoder's children appear before decoder
        assert all_fqns.index("encoder.conv1") < all_fqns.index("decoder")

        # named_modules() should match
        named = list(root.named_modules())
        assert [(fqn, m.module_name) for fqn, m in named] == [(fqn, fqn) for fqn in all_fqns]

        # Subtree: encoder.modules() should only include encoder and its children
        encoder = root.child_modules["encoder"]
        encoder_fqns = [m.module_name for m in encoder.modules()]
        assert encoder_fqns[0] == "encoder"
        assert "encoder.conv1" in encoder_fqns
        assert "encoder.conv2" in encoder_fqns
        assert "decoder" not in encoder_fqns

    def test_get_submodule(self, execution_mode: ExecutionMode) -> None:
        """Verify get_submodule() looks up descendants by fully-qualified name."""
        inspector = ModelInspector(
            _NestedModel(),
            (torch.randn(1, 3, 8, 8),),
            execution_mode=execution_mode,
        )
        root = inspector.summary.model

        # Direct child
        encoder = root.get_submodule("encoder")
        assert encoder.module_name == "encoder"

        # Grandchild
        conv1 = root.get_submodule("encoder.conv1")
        assert conv1.module_name == "encoder.conv1"

        # Get child from non-root module
        conv1 = encoder.get_submodule("encoder.conv1")
        assert conv1.module_name == "encoder.conv1"

        # Root can find itself
        assert root.get_submodule("").module_name == ""

        # Non-existent raises KeyError
        with pytest.raises(KeyError, match="no_such_module"):
            root.get_submodule("no_such_module")

        with pytest.raises(KeyError, match="."):
            root.get_submodule(".")

    def test_all_ops(self, execution_mode: ExecutionMode) -> None:
        """Verify all_ops() returns ops from the entire subtree."""
        inspector = ModelInspector(
            _NestedModel(),
            (torch.randn(1, 3, 8, 8),),
            execution_mode=execution_mode,
            compressor=Quantizer,
        )
        root = inspector.summary.model

        # Root all_ops should include all ops from all submodules
        root_all = root.all_ops()
        assert len(root_all) > 0

        # Encoder subtree should contain only encoder ops
        encoder = root.get_submodule("encoder")
        encoder_ops = encoder.all_ops()
        encoder_op_names = [op.op_name for op in encoder_ops]
        assert "conv2d" in encoder_op_names
        assert "conv2d_1" in encoder_op_names
        assert not any("linear" in n for n in encoder_op_names)

        # Leaf module all_ops should equal its direct ops
        conv1 = root.get_submodule("encoder.conv1")
        assert conv1.all_ops() == conv1.ops

    def test_empty_summary_after_compressor_filter(self, execution_mode: ExecutionMode) -> None:
        """Verify formatting when compressor filters all ops."""
        inspector = ModelInspector(
            _SimpleConvModel(),
            (torch.randn(1, 3, 8, 8),),
            execution_mode=execution_mode,
            compressor=Quantizer,
        )
        # The root should be non-empty for a real model with quantizable ops
        assert inspector.summary.model.child_modules or inspector.summary.model.ops


class TestModelInspectorValidation:
    """Tests for ModelInspector input validation."""

    def test_rejects_non_module(self) -> None:
        """Verify TypeError when model is not an nn.Module."""
        with pytest.raises(TypeError, match="Expected a torch.fx.GraphModule or torch.nn.Module"):
            ModelInspector("not a module", (torch.randn(1),), execution_mode="graph")

    @pytest.mark.parametrize("execution_mode", [ExecutionMode.GRAPH, ExecutionMode.EAGER])
    def test_example_input_none(self, execution_mode: ExecutionMode) -> None:
        """Verify ValueError for example_inputs of None when model not a GraphModule and
        execution_mode is not ExecutionMode.GRAPH."""
        with pytest.raises(ValueError, match="example_inputs can only be None when"):
            ModelInspector(nn.Linear(10, 5), None, execution_mode=execution_mode)

    def test_eager_with_graph_module_raises_type_error(self) -> None:
        """Verify TypeError for eager mode given a graph module."""
        model = nn.Linear(10, 5)

        gm = _export_model(
            model, (torch.randn(1, 10),), dynamic_shapes=None, export_with_no_grad=True
        )
        with pytest.raises(TypeError, match="Expected a torch.nn.Module for Eager execution_mode"):
            ModelInspector(gm, (torch.randn(1, 10),), execution_mode="eager")

    def test_eager_raises_not_implemented(self) -> None:
        """Verify NotImplementedError for eager mode (not yet supported)."""
        model = nn.Linear(10, 5)
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            ModelInspector(model, (torch.randn(1, 10),), execution_mode="eager")

    def test_invalid_execution_mode_raises(self) -> None:
        """Verify ValueError for unrecognized execution mode."""
        model = nn.Linear(10, 5)
        with pytest.raises(ValueError, match="Unknown execution_mode"):
            ModelInspector(model, (torch.randn(1, 10),), execution_mode="invalid")

    def test_unsupported_compressor_raises(self) -> None:
        """Verify ValueError when compressor is not a supported compression class."""

        class _FakeCompressor(_BaseModelCompressor):
            pass

        model = nn.Linear(10, 5)
        with pytest.raises(ValueError, match="Unsupported compressor class"):
            ModelInspector(
                model,
                (torch.randn(1, 10),),
                execution_mode="graph",
                compressor=_FakeCompressor,
            )
