# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the coreai_opt.inspection module."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from coreai_opt._utils.insertion.torch_function.module_boundary_tracker import (
    TensorIdVersion,
)
from coreai_opt._utils.torch_utils import export_model as _export_model
from coreai_opt.base_model_compressor import _BaseModelCompressor
from coreai_opt.inspection import (
    ModelInspector,
    ModelSummary,
    ModuleInfo,
)
from coreai_opt.inspection._eager_mode import _EagerOpDiscoveryMode
from coreai_opt.inspection.types import BoundaryEdge, InputEdge, OpInfo
from coreai_opt.palettization import KMeansPalettizer
from coreai_opt.quantization import Quantizer
from coreai_opt.quantization.config.quantization_config import ExecutionMode

execution_modes = pytest.mark.parametrize(
    "execution_mode",
    [ExecutionMode.GRAPH, ExecutionMode.EAGER],
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

        # Op discovery — names differ by mode
        ops = inspector.summary.model.all_ops()
        op_names = [op.op_name for op in ops]
        if execution_mode == ExecutionMode.GRAPH:
            assert "conv2d" in op_names
            assert "linear" in op_names
            conv_op_name = "conv2d"
            linear_op_name = "linear"
        else:
            assert "conv.conv2d" in op_names
            assert "fc.linear" in op_names
            conv_op_name = "conv.conv2d"
            linear_op_name = "fc.linear"

        # Op types
        conv_op = next(op for op in ops if op.op_name == conv_op_name)
        assert conv_op.op_type == "conv2d"
        linear_op = next(op for op in ops if op.op_name == linear_op_name)
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
        conv_by_name = inspector.get_matched_ops_for_op_name(conv_op_name)
        assert len(conv_by_name) == 1
        assert conv_by_name[0].op_name == conv_op_name

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

        # Op discovery and hierarchy — names differ by mode
        op_names = [op.op_name for op in inspector.summary.model.all_ops()]
        if execution_mode == ExecutionMode.GRAPH:
            conv_first, conv_second = "conv2d", "conv2d_1"
            linear_first, linear_second = "linear", "linear_1"
        else:
            conv_first, conv_second = "encoder.conv1.conv2d", "encoder.conv2.conv2d"
            linear_first, linear_second = "decoder.fc1.linear", "decoder.fc2.linear"
        assert conv_first in op_names
        assert conv_second in op_names
        assert linear_first in op_names
        assert linear_second in op_names

        # Execution order: convs before linears
        assert op_names.index(conv_first) < op_names.index(linear_first)

        # Nested module FQNs
        conv_op = next(op for op in inspector.summary.model.all_ops() if op.op_name == conv_first)
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
        assert conv_first in encoder_op_names
        assert conv_second in encoder_op_names

        # Query: by module name (leaf)
        leaf_ops = inspector.get_matched_ops_for_module_name("encoder.conv1")
        assert len(leaf_ops) == 1
        assert leaf_ops[0].op_name == conv_first

        # Query: by module name (regex)
        encoder_regex_ops = inspector.get_matched_ops_for_module_name(r"encoder\..*")
        encoder_regex_op_names = [op.op_name for op in encoder_regex_ops]
        assert conv_first in encoder_regex_op_names
        assert conv_second in encoder_regex_op_names

        # Query: by name (regex matching multiple ops)
        conv_ops_by_name = inspector.get_matched_ops_for_op_name(r".*conv2d.*")
        assert len(conv_ops_by_name) == 2
        linear_ops_by_name = inspector.get_matched_ops_for_op_name(r".*linear.*")
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
        # Linear should be present (module-qualified in eager)
        if execution_mode == ExecutionMode.GRAPH:
            assert "linear" in op_names
        else:
            assert "linear.linear" in op_names
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

        linear_name = "linear" if execution_mode == ExecutionMode.GRAPH else "linear.linear"
        linear_op = ops_by_name[linear_name]

        # linear's outputs should include an add op
        assert any(
            "add" in out.op_name for consumers in linear_op.outputs.values() for out in consumers
        )

        # add ops have correct inputs
        add_name = "add"
        add_op = ops_by_name[add_name]
        assert len(add_op.inputs) == 2

        # mul has inputs (both add-related ops)
        mul_name = "mul"
        mul_op = ops_by_name[mul_name]
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

        # Encoder: has input and output ops
        assert len(encoder.input_ops) >= 1
        encoder_input_names = {e.op.op_name for edges in encoder.input_ops.values() for e in edges}
        assert any("conv" in n for n in encoder_input_names)
        assert len(encoder.output_ops) >= 1

        # Decoder: has input and output ops
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
        if execution_mode == ExecutionMode.GRAPH:
            assert "conv2d" in conv1_op_names
        else:
            assert "encoder.conv1.conv2d" in conv1_op_names

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
        if execution_mode == ExecutionMode.GRAPH:
            assert "conv2d" in encoder_op_names
            assert "conv2d_1" in encoder_op_names
        else:
            assert "encoder.conv1.conv2d" in encoder_op_names
            assert "encoder.conv2.conv2d" in encoder_op_names
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

    def test_source_frames(self, execution_mode: ExecutionMode) -> None:
        """Verify source frames are captured from forward() methods in both modes."""
        inspector = ModelInspector(
            _SimpleConvModel(),
            (torch.randn(1, 3, 8, 8),),
            execution_mode=execution_mode,
            compressor=Quantizer,
        )
        conv_op = next(op for op in inspector.summary.model.all_ops() if op.op_type == "conv2d")
        assert len(conv_op.source_frames) >= 1
        assert all(f.function_name == "forward" for f in conv_op.source_frames)
        assert all(f.filename != "" for f in conv_op.source_frames)

    def test_connectivity_through_non_captured_ops(self, execution_mode: ExecutionMode) -> None:
        """Verify filtered ops still provide connectivity edges between tree ops."""

        class _ReluBetweenLinears(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = nn.Linear(10, 10)
                self.linear2 = nn.Linear(10, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.linear1(x)
                x = torch.relu(x)
                x = self.linear2(x)
                return x

        inspector = ModelInspector(
            _ReluBetweenLinears(),
            (torch.randn(1, 10),),
            execution_mode=execution_mode,
            compressor=Quantizer,
        )

        # Only linear ops should appear in the tree; relu is filtered out
        tree_ops = inspector.summary.model.all_ops()
        tree_op_types = {op.op_type for op in tree_ops if op.op_type}
        assert "linear" in tree_op_types
        assert "relu" not in tree_op_types

        # Identify linear2 by its module_stack (names mirror named_modules in both modes)
        linear_ops = [op for op in tree_ops if op.op_type == "linear"]
        linear2_op = next(
            op
            for op in linear_ops
            if any(ctx.module_name.endswith("linear2") for ctx in op.module_stack)
        )

        # linear2's input should chain through relu back to a linear op
        relu_input = next(
            (inp for inp in linear2_op.inputs if inp.op_type == "relu"),
            None,
        )
        assert relu_input is not None, (
            f"Expected relu in linear2.inputs, got {[i.op_name for i in linear2_op.inputs]}"
        )
        linear1_upstream = next(
            (inp for inp in relu_input.inputs if inp.op_type == "linear"),
            None,
        )
        assert linear1_upstream is not None

    def test_boundary_ops_with_non_tree_ops(self, execution_mode: ExecutionMode) -> None:
        """Verify module boundary ops are correct when non-tree ops sit between tree ops."""

        class _Inner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear1 = nn.Linear(10, 10)
                self.linear2 = nn.Linear(10, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.linear1(x)
                x = torch.relu(x)
                x = self.linear2(x)
                return x

        class _Outer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner = _Inner()
                self.final = nn.Linear(10, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.inner(x)
                x = self.final(x)
                return x

        inspector = ModelInspector(
            _Outer(),
            (torch.randn(1, 10),),
            execution_mode=execution_mode,
            compressor=Quantizer,
        )
        root = inspector.summary.model
        inner = root.child_modules["inner"]

        inner_tree_ops = inner.all_ops()
        inner_op_types = {op.op_type for op in inner_tree_ops if op.op_type}
        assert "linear" in inner_op_types
        assert "relu" not in inner_op_types
        assert len(inner_tree_ops) == 2  # linear1 and linear2, relu filtered out

        linears_by_module = {}
        for op in inner_tree_ops:
            if op.op_type != "linear":
                continue
            for ctx in op.module_stack:
                if ctx.module_name == "inner.linear1":
                    linears_by_module["linear1"] = op
                elif ctx.module_name == "inner.linear2":
                    linears_by_module["linear2"] = op

        assert set(linears_by_module) == {"linear1", "linear2"}

        # linear1 is an input boundary (data comes from outside inner).
        assert any(
            e.op == linears_by_module["linear1"]
            for edges in inner.input_ops.values()
            for e in edges
        )
        # linear2 is an output boundary (data goes to outside inner).
        assert any(e.op == linears_by_module["linear2"] for e in inner.output_ops.values())
        # linear2 is NOT an input boundary (its data flows from within inner via relu).
        assert all(
            e.op != linears_by_module["linear2"]
            for edges in inner.input_ops.values()
            for e in edges
        )

    def test_module_level_input_ops(self, execution_mode: ExecutionMode) -> None:
        """Verify model-level inputs appear as placeholder-like OpInfos at the root boundary.

        Graph mode emits ``placeholder`` nodes; eager mode emits synthetic ``input_i``
        ops. Both should behave identically: empty module_stack, not is_state, no
        further inputs, present in the consuming op's ``inputs`` tuple, but absent
        from any module's tree ops or boundary lists.
        """
        inspector = ModelInspector(
            _SimpleConvModel(),
            (torch.randn(1, 3, 8, 8),),
            execution_mode=execution_mode,
        )
        root = inspector.summary.model

        # The first real op (conv) consumes the user input, so it must be in root.input_ops.
        conv_op = next(op for op in root.all_ops() if op.op_type == "conv2d")
        assert any(e.op == conv_op for edges in root.input_ops.values() for e in edges)

        # conv's inputs should include at least one placeholder-like OpInfo.
        placeholders = [inp for inp in conv_op.inputs if not inp.module_stack and not inp.is_state]
        assert len(placeholders) >= 1
        for ph in placeholders:
            assert ph.inputs == ()

        # Placeholder OpInfos must not appear in any module's tree or boundary lists.
        all_tree_ops = root.all_ops()
        for module in root.modules():
            for ph in placeholders:
                assert ph not in all_tree_ops
                assert all(e.op != ph for edges in module.input_ops.values() for e in edges)
                assert all(e.op != ph for e in module.output_ops.values())

    def test_module_level_output_ops(self, execution_mode: ExecutionMode) -> None:
        """Verify model-level outputs appear as output-like OpInfos at the root boundary.

        Graph mode emits a single ``output`` node; eager mode emits one ``output_i``
        per output tensor. Both should behave identically: empty module_stack, not
        is_state, no further outputs, present in the producing op's ``outputs`` tuple,
        but absent from any module's tree ops or boundary lists.
        """
        inspector = ModelInspector(
            _SimpleConvModel(),
            (torch.randn(1, 3, 8, 8),),
            execution_mode=execution_mode,
        )
        root = inspector.summary.model

        # The last real op (fc linear) produces the model output, so it's in root.output_ops.
        linear_op = next(op for op in root.all_ops() if op.op_type == "linear")
        assert any(e.op == linear_op for e in root.output_ops.values())

        output_consumers = [
            out
            for consumers in linear_op.outputs.values()
            for out in consumers
            if not out.module_stack and not out.is_state
        ]
        assert len(output_consumers) >= 1
        for out in output_consumers:
            assert not any(out.outputs.values())

        all_tree_ops = root.all_ops()
        for module in root.modules():
            for out in output_consumers:
                assert out not in all_tree_ops
                assert all(e.op != out for edges in module.input_ops.values() for e in edges)
                assert all(e.op != out for e in module.output_ops.values())

    def test_state_ops_for_parameters(self, execution_mode: ExecutionMode) -> None:
        """Verify parameters consumed by ops appear as is_state=True OpInfos.

        Graph mode emits ``get_attr`` nodes; eager mode emits synthetic state
        OpInfos on first reference. Both should share identical semantic
        behavior: is_state=True, empty module_stack, no inputs, and excluded
        from tree/boundary lists.
        """
        inspector = ModelInspector(
            _SimpleConvModel(),
            (torch.randn(1, 3, 8, 8),),
            execution_mode=execution_mode,
        )
        root = inspector.summary.model
        linear_op = next(op for op in root.all_ops() if op.op_type == "linear")

        state_inputs = [inp for inp in linear_op.inputs if inp.is_state]
        state_names = {inp.op_name for inp in state_inputs}
        assert any("weight" in n for n in state_names), state_names
        assert any("bias" in n for n in state_names), state_names

        for s in state_inputs:
            assert s.module_stack == ()
            assert s.inputs == ()

        all_tree_ops = root.all_ops()
        for module in root.modules():
            for s in state_inputs:
                assert s not in all_tree_ops
                assert all(e.op != s for edges in module.input_ops.values() for e in edges)
                assert all(e.op != s for e in module.output_ops.values())

        # State ops: _display_name drops any dotted prefix (e.g., "fc.weight" → "weight")
        # to match the suffix-only matching supported by state configs today.
        for s in state_inputs:
            assert s._display_name == s.op_name.rsplit(".", 1)[-1]
            assert "." not in s._display_name
            assert s._display_name != s.op_name

        # Non-state ops: _display_name equals op_name.
        assert linear_op._display_name == linear_op.op_name

    def test_state_ops_for_buffers(self, execution_mode: ExecutionMode) -> None:
        """Verify registered buffers consumed by ops appear as is_state=True OpInfos."""

        class _WithBuffer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer("scale", torch.tensor(2.0))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * self.scale

        inspector = ModelInspector(
            _WithBuffer(),
            (torch.randn(1, 10),),
            execution_mode=execution_mode,
        )
        mul_op = next(op for op in inspector.summary.model.all_ops() if op.op_type == "mul")
        state_inputs = [inp for inp in mul_op.inputs if inp.is_state]
        assert len(state_inputs) == 1
        assert "scale" in state_inputs[0].op_name

    def test_shared_state_is_not_duplicated(self, execution_mode: ExecutionMode) -> None:
        """Verify a parameter referenced by multiple ops yields a single shared state OpInfo.

        A parameter used more than once in ``forward`` should resolve to the same
        ``_OpInfo`` instance each time, not distinct duplicates that merely compare
        equal via ``op_name``.
        """

        class _SharedWeightModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.randn(10, 10))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                y = torch.matmul(x, self.weight)
                return torch.matmul(y, self.weight)

        inspector = ModelInspector(
            _SharedWeightModel(),
            (torch.randn(1, 10),),
            execution_mode=execution_mode,
        )

        state_inputs = [
            next(inp for inp in op.inputs if inp.is_state)
            for op in inspector.summary.model.all_ops()
            if any(inp.is_state for inp in op.inputs)
        ]
        assert len(state_inputs) == 2, "expected two ops to reference the weight"
        assert state_inputs[0].op is state_inputs[1].op, (
            "both references should resolve to the same state OpInfo instance"
        )

    def test_boundary_ops_topological_order(self, execution_mode: ExecutionMode) -> None:
        """Verify input_ops and output_ops of a module are in topological order.

        Uses a model where the root has two parallel child modules feeding into a
        combine op, so naive DFS ordering would differ from topological order.
        """

        class _ParallelModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.branch_a = nn.Linear(4, 4)
                self.branch_b = nn.Linear(4, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                u = self.branch_a(x)
                v = self.branch_b(x)
                return u + v

        inspector = ModelInspector(
            _ParallelModel(),
            (torch.randn(1, 4),),
            execution_mode=execution_mode,
        )
        root = inspector.summary.model
        branch_a = root.child_modules["branch_a"]
        branch_b = root.child_modules["branch_b"]

        # Each branch has exactly one input spec index (key) with one edge (the linear).
        assert len(branch_a.input_ops) == 1
        assert len(branch_a.output_ops) == 1
        assert len(branch_b.input_ops) == 1
        assert len(branch_b.output_ops) == 1

        # Both branches consume x from outside root. In graph mode each branch gets its
        # own spec_idx in root.input_ops; in eager mode both share spec_idx 0 (x appears
        # once in root's forward args and fans out to both branches).
        all_root_edges = [e for edges in root.input_ops.values() for e in edges]
        root_op_names = [e.op.op_name for e in all_root_edges]
        assert any(
            "branch_a" in n or "branch_a" in str(e.op.module_stack)
            for e, n in zip(all_root_edges, root_op_names, strict=True)
        )
        assert any(
            "branch_b" in n or "branch_b" in str(e.op.module_stack)
            for e, n in zip(all_root_edges, root_op_names, strict=True)
        )

        # branch_a's ops appear before branch_b's ops in the all_ops list (execution order).
        all_ops = inspector.summary.model.all_ops()
        all_op_names = [op.op_name for op in all_ops]
        branch_a_edge = branch_a.input_ops[0][0]
        branch_b_edge = branch_b.input_ops[0][0]
        assert all_op_names.index(branch_a_edge.op.op_name) < all_op_names.index(
            branch_b_edge.op.op_name
        )

        # Root input_ops preserves topological order. Find each edge's position as
        # (spec_idx, list_idx) and compare — works whether branches share a spec_idx
        # (eager: both at 0) or each has its own (graph: 0 and 1).
        def _edge_pos(ops: dict, edge: BoundaryEdge) -> tuple[int, int]:
            for spec_idx, edges in sorted(ops.items()):
                for list_idx, e in enumerate(edges):
                    if e == edge:
                        return (spec_idx, list_idx)
            raise AssertionError(f"edge not found: {edge}")

        assert _edge_pos(root.input_ops, branch_a_edge) < _edge_pos(root.input_ops, branch_b_edge)

    def test_op_inputs_index_with_state_before_activation(
        self, execution_mode: ExecutionMode
    ) -> None:
        """Verify op input dict key reflects full arg position when a state precedes an activation.

        In both graph and eager mode, when a model parameter appears at an earlier argument
        position than an activation (e.g., torch.mm(self.weight, x)), the state occupies
        arg index 0 and the activation occupies arg index 1. The op inputs dict must show
        key 1 for the activation, not key 0, because the key is the full arg position
        (= op_input_spec index), not the position within non-state inputs only.
        """

        class _WeightFirstModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 4))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.mm(self.weight, x)

        inspector = ModelInspector(
            _WeightFirstModel(),
            (torch.randn(4, 4),),
            execution_mode=execution_mode,
        )
        # Find the op: state at arg 0, activation at arg 1.
        mm_op = next(
            op
            for op in inspector.summary.model.all_ops()
            if len(op.inputs) >= 2 and op.inputs[0].is_state and not op.inputs[1].is_state
        )

        assert mm_op.inputs[0].is_state
        assert not mm_op.inputs[1].is_state

        # Formatted output must show the activation at dict key 1, not 0.
        formatted = inspector.format_summary(colorize=False)
        assert "op inputs:  {1:" in formatted

    def test_raw_attribute_untracked_in_eager_state_in_graph(
        self, execution_mode: ExecutionMode
    ) -> None:
        """Verify a raw tensor attribute appears as untracked in eager, as a state in graph.

        ``self.mask = torch.ones(8)`` is not registered as a parameter or buffer.
        In eager mode the inspector cannot trace its origin, so it appears as
        ``untracked_N`` at arg index 1 of the consuming op.  In graph mode
        ``torch.export`` captures it as a ``get_attr`` node, so it shows up as a
        state input.
        """

        class _ModelWithRawAttribute(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(8, 8)
                self.mask = torch.ones(8)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x) * self.mask

        inspector = ModelInspector(
            _ModelWithRawAttribute(),
            (torch.randn(2, 8),),
            execution_mode=execution_mode,
        )
        ops = inspector.summary.model.all_ops()
        mul_op = next(op for op in ops if op.op_type == "mul")
        mul_non_state_inputs = [inp for inp in mul_op.inputs if not inp.is_state]
        mul_state_inputs = [inp for inp in mul_op.inputs if inp.is_state]

        if execution_mode == ExecutionMode.EAGER:
            # mask is not registered — shows as untracked at arg index 1
            assert len(mul_non_state_inputs) == 2
            assert any(inp.op_name.startswith("untracked_") for inp in mul_non_state_inputs)
            assert not any("mask" in inp.op_name for inp in mul_state_inputs)
        else:
            # graph mode: FX captures self.mask as get_attr → state
            assert len(mul_non_state_inputs) == 1
            assert any("mask" in inp.op_name for inp in mul_state_inputs)

    def test_global_tensor_untracked_in_eager_state_in_graph(
        self, execution_mode: ExecutionMode
    ) -> None:
        """Verify a global tensor appears as untracked in eager, as a lifted state in graph.

        A module-level Python global (``_BIAS = torch.zeros(8)``) has no registered
        name in the model.  Eager mode cannot trace it and marks it ``untracked_N``
        at arg index 1.  Graph mode lifts it as a ``lifted_tensor_N`` placeholder
        which the inspector treats as a state.
        """
        _bias = torch.zeros(8)

        class _ModelWithGlobalTensor(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(8, 8)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.linear(x) + _bias

        inspector = ModelInspector(
            _ModelWithGlobalTensor(),
            (torch.randn(2, 8),),
            execution_mode=execution_mode,
        )
        ops = inspector.summary.model.all_ops()
        add_op = next(op for op in ops if op.op_type == "add")
        add_non_state_inputs = [inp for inp in add_op.inputs if not inp.is_state]
        add_state_inputs = [inp for inp in add_op.inputs if inp.is_state]

        if execution_mode == ExecutionMode.EAGER:
            # global tensor is untracked — appears at arg index 1
            assert len(add_non_state_inputs) == 2
            assert any(inp.op_name.startswith("untracked_") for inp in add_non_state_inputs)
        else:
            # graph mode: torch.export lifts the global as a lifted_tensor placeholder
            assert len(add_non_state_inputs) == 1
            assert len(add_state_inputs) >= 1

    def test_shared_module_op_count_and_boundaries(self, execution_mode: ExecutionMode) -> None:
        """Verify shared module handling for eager and graph modes.

        Eager mode only captures ops from the *first* traversal of a shared module
        (subsequent calls are blocked by ``traversed_modules``), but re-registers
        the second traversal's output tensor so downstream modules resolve it.
        Graph mode sees both calls as separate nodes.
        """

        class _ModelWithSharedModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.shared = nn.Linear(8, 8)
                self.tail = nn.Linear(8, 8)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.shared(x)
                x = self.shared(x)
                return self.tail(x)

        inspector = ModelInspector(
            _ModelWithSharedModule(),
            (torch.randn(2, 8),),
            execution_mode=execution_mode,
        )
        root = inspector.summary.model
        shared = root.child_modules.get("shared")
        tail = root.child_modules.get("tail")
        assert shared is not None
        assert tail is not None

        if execution_mode == ExecutionMode.EAGER:
            # Only the first traversal is recorded; second is skipped.
            assert len(shared.ops) == 1
            shared_linear = shared.ops[0]
            assert shared_linear.op_name == "shared.linear"

            # shared.linear's output goes to tail.linear (via the re-registered
            # second-traversal output tensor pointing back to the first traversal).
            assert shared_linear.outputs[0] == (
                next(op for op in tail.ops if op.op_name == "tail.linear"),
            )

            # tail.linear's first input comes from shared.linear.
            tail_linear = next(op for op in tail.ops if op.op_name == "tail.linear")
            assert tail_linear.inputs[0].op.op_name == "shared.linear"
        else:
            # graph mode unfolds both calls as distinct nodes.
            # Graph mode unfolds both calls as distinct nodes named linear and linear_1.
            assert len(shared.ops) == 2
            linear_1 = next(op for op in shared.ops if op.op_name == "linear_1")

            # linear feeds into linear_1 (second call reuses the shared weights).
            assert linear_1.inputs[0].op.op_name == "linear"

            # linear_1's output goes to tail's linear_2.
            tail_linear = next(op for op in tail.ops if op.op_name == "linear_2")
            assert tail_linear.inputs[0].op.op_name == "linear_1"


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

    def test_palettizer_in_graph_mode_raises(self) -> None:
        """Verify ValueError when using KMeansPalettizer with graph mode."""
        model = nn.Linear(10, 5)
        with pytest.raises(ValueError, match="not supported in graph mode"):
            ModelInspector(
                model,
                (torch.randn(1, 10),),
                execution_mode="graph",
                compressor=KMeansPalettizer,
            )


class TestEagerModeSpecific:
    """Tests specific to eager mode behavior."""

    def test_dynamic_control_flow(self) -> None:
        """Verify eager mode handles dynamic control flow (if/else)."""

        class _DynamicModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.branch_a = nn.Linear(10, 10)
                self.branch_b = nn.Linear(10, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if x.sum() > 0:
                    return self.branch_a(x)
                else:
                    return self.branch_b(x)

        model = _DynamicModel()
        # Positive input takes branch_a
        inspector = ModelInspector(
            model, (torch.ones(1, 10),), execution_mode="eager", compressor=Quantizer
        )
        op_names = [op.op_name for op in inspector.summary.model.all_ops()]
        assert "branch_a.linear" in op_names
        assert "branch_b.linear" not in op_names

        # Negative input takes branch_b
        inspector = ModelInspector(
            model, (-1.0 * torch.ones(1, 10),), execution_mode="eager", compressor=Quantizer
        )
        op_names = [op.op_name for op in inspector.summary.model.all_ops()]
        assert "branch_b.linear" in op_names
        assert "branch_a.linear" not in op_names

    def test_shared_module_only_captured_once(self) -> None:
        """Verify shared module instances only produce ops on first traversal."""

        class _SharedModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.shared = nn.Linear(10, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.shared(x)
                x = self.shared(x)
                return x

        inspector = ModelInspector(
            _SharedModel(),
            (torch.randn(1, 10),),
            execution_mode="eager",
            compressor=Quantizer,
        )
        ops = inspector.summary.model.all_ops()
        linear_ops = [op for op in ops if op.op_type == "linear"]
        assert len(linear_ops) == 1

    def test_passthrough_submodule_has_empty_boundary(self) -> None:
        """Verify a submodule that returns its input unchanged has empty output_ops.

        When a module's output tensor is the same as its input (not produced by any
        op in the subtree), the corresponding ``_module_output_producers`` entry has
        ``output_idx=None``. The guard in ``_populate_boundary_ops_eager`` should
        skip it, leaving ``output_ops`` empty.
        """

        class _SideEffectModule(nn.Module):
            """Has an internal op but returns its input unchanged."""

            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(4, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                _ = self.linear(x)  # runs an op internally, but output is discarded
                return x  # returns the original input — not produced by any subtree op

        class _Wrapper(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner = _SideEffectModule()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.inner(x)

        inspector = ModelInspector(
            _Wrapper(),
            (torch.randn(1, 4),),
            execution_mode="eager",
        )
        inner = inspector.summary.model.child_modules.get("inner")
        assert inner is not None
        assert inner.output_ops == {}

    def test_palettizer_compressor(self) -> None:
        """Verify KMeansPalettizer filters to only palettization-supported ops."""
        inspector = ModelInspector(
            _SimpleConvModel(),
            (torch.randn(1, 3, 8, 8),),
            execution_mode="eager",
            compressor=KMeansPalettizer,
        )
        ops = inspector.summary.model.all_ops()
        op_types = {op.op_type for op in ops}
        # KMeansPalettizer supports conv and linear but not add/mul/relu
        assert "conv2d" in op_types
        assert "linear" in op_types

    def test_input_output_op_names(self) -> None:
        """Verify eager mode names module-level input/output ops as ``input_i``/``output_i``."""
        inspector = ModelInspector(
            _SimpleConvModel(),
            (torch.randn(1, 3, 8, 8),),
            execution_mode="eager",
        )
        root = inspector.summary.model
        conv_op = next(op for op in root.all_ops() if op.op_type == "conv2d")
        linear_op = next(op for op in root.all_ops() if op.op_type == "linear")

        placeholder_names = {
            inp.op_name for inp in conv_op.inputs if not inp.module_stack and not inp.is_state
        }
        assert "input_0" in placeholder_names

        output_names = {
            out.op_name
            for consumers in linear_op.outputs.values()
            for out in consumers
            if not out.module_stack and not out.is_state
        }
        assert "output_0" in output_names

    def test_input_ops_one_per_input_tensor(self) -> None:
        """Verify each forward-argument tensor gets its own ``input_i`` OpInfo."""

        class _TwoInputs(nn.Module):
            def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return a + b

        inspector = ModelInspector(
            _TwoInputs(),
            (torch.randn(1, 10), torch.randn(1, 10)),
            execution_mode="eager",
        )
        add_op = next(op for op in inspector.summary.model.all_ops() if op.op_type == "add")
        placeholder_names = {
            inp.op_name for inp in add_op.inputs if not inp.module_stack and not inp.is_state
        }
        assert placeholder_names == {"input_0", "input_1"}

    def test_source_frames_include_nested_forward_calls(self) -> None:
        """Verify eager mode captures source frames from all forward() methods on the call stack.

        Graph mode exercises source-frame extraction in
        :py:meth:`TestModelInspector.test_source_frames`; this test is eager-specific
        because it asserts the multi-frame stack ordering unique to runtime interception.
        """
        inspector = ModelInspector(
            _NestedModel(),
            (torch.randn(1, 3, 8, 8),),
            execution_mode="eager",
        )
        # conv1 lives inside _Encoder.forward() which is called by _NestedModel.forward():
        # the source stack should contain at least two forward frames, outermost first.
        conv_op = next(
            op for op in inspector.summary.model.all_ops() if op.op_name == "encoder.conv1.conv2d"
        )
        assert len(conv_op.source_frames) >= 2
        assert all(f.function_name == "forward" for f in conv_op.source_frames)

    def test_inplace_op_connectivity(self) -> None:
        """Verify connectivity is correct when in-place ops mutate tensors."""

        class _InPlaceModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.linear(x)
                x.relu_()
                x = x + 1
                return x

        inspector = ModelInspector(
            _InPlaceModel(),
            (torch.randn(1, 10),),
            execution_mode="eager",
        )
        ops = inspector.summary.model.all_ops()
        ops_by_name = {op.op_name: op for op in ops}

        # relu_ is in-place: it consumes linear's output tensor (same id, different version)
        relu_op = ops_by_name["relu_"]
        assert len(relu_op.inputs) >= 1
        assert any("linear" in inp.op_name for inp in relu_op.inputs)

        # add consumes relu_'s output (the mutated tensor)
        add_op = ops_by_name["add"]
        assert len(add_op.inputs) >= 1
        assert any("relu_" in inp.op_name for inp in add_op.inputs)

    def test_weakref_removes_producer_on_dealloc(self) -> None:
        """Verify ``_tensor_producers`` entries are cleaned up when tensors die.

        ``_record_outputs`` registers a ``weakref.finalize`` callback that
        removes the producer entry when the tensor is deallocated. In CPython
        this fires deterministically when the tensor's refcount hits zero
        (no cyclic references involved here), so we can assert directly
        without relying on ``gc.collect`` heuristics.
        """
        mode = _EagerOpDiscoveryMode(nn.Linear(3, 3))
        tensor = torch.randn(3)
        op = OpInfo(
            op_name="fake",
            op_type="fake",
            module_stack=(),
            source_frames=(),
            inputs=(),
            outputs={},
            is_state=False,
        )
        mode._record_outputs(tensor, op)
        key = TensorIdVersion(id(tensor), tensor._version)
        assert mode._tensor_producers[key] == InputEdge(op=op, output_idx=0)
        del tensor
        assert key not in mode._tensor_producers

    def test_input_ops_disambiguates_multi_output_external_producer(self) -> None:
        """Verify input_ops correctly separates consumers of distinct outputs from the same op.

        When a multi-output external producer (e.g. chunk) feeds two of its outputs into
        a module as separate forward args, input_ops[0] should only list ops that consume
        output slot 0 and input_ops[1] should only list ops that consume output slot 1.
        The bug: keying external_to_consumers by op_name alone merges both consumer sets
        under the same key, so both spec positions get the wrong combined list.
        """

        class _Inner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear_a = nn.Linear(4, 4)
                self.linear_b = nn.Linear(4, 4)

            def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return self.linear_a(a) + self.linear_b(b)

        class _ChunkModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inner = _Inner()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                a, b = torch.chunk(x, 2, dim=-1)
                return self.inner(a, b)

        inspector = ModelInspector(
            _ChunkModel(),
            (torch.randn(1, 8),),
            execution_mode="eager",
        )
        inner = inspector.summary.model.child_modules["inner"]

        assert set(inner.input_ops.keys()) == {0, 1}
        ops_at_0 = {e.op.op_name for e in inner.input_ops[0]}
        ops_at_1 = {e.op.op_name for e in inner.input_ops[1]}
        assert ops_at_0 == {"inner.linear_a.linear"}
        assert ops_at_1 == {"inner.linear_b.linear"}

    def test_state_detection_after_inplace_mutation_via_view(self) -> None:
        """Verify state ops are still detected after their _version advances through a view.

        When a buffer is mutated in-place via a view (e.g., ``self.counter.view(-1).add_(1.0)``),
        the mutation increments ``self.counter._version`` without changing ``id(self.counter)``.
        The inspector must still recognise ``counter`` as a state input to any op that later
        consumes it.

        Before the fix, ``_states_to_names`` was keyed by
        ``TensorIdVersion(id(state), state._version)`` captured at ``__init__``. After the
        in-place mutation the version had advanced, so the lookup missed and counter was silently
        dropped from the consuming op's ``inputs``. Graph mode was unaffected because it identifies
        states via ``node.op == "get_attr"`` rather than tensor identity. The fix keys
        ``_states_to_names`` by ``id(state)`` only, which is stable for the model's lifetime.
        """

        class _CounterModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(4, 4)
                self.register_buffer("counter", torch.zeros(1))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Mutate counter through a view — advances counter._version without
                # changing id(counter).
                self.counter.view(-1).add_(1.0)
                x = self.linear(x)
                return x + self.counter

        inspector = ModelInspector(
            _CounterModel(),
            (torch.randn(1, 4),),
            execution_mode="eager",
        )
        add_op = next(op for op in inspector.summary.model.all_ops() if op.op_type == "add")
        state_names = {inp.op_name for inp in add_op.inputs if inp.is_state}
        assert any("counter" in name for name in state_names), (
            f"counter not found in state inputs of add op; states found: {state_names}"
        )
