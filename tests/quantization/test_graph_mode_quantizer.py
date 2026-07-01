# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Tests for GraphQuantizer and related classes.
"""

import logging
from copy import deepcopy
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from torch.export.dynamic_shapes import Dim
from torch.ops import coreai

from coreai_opt import ExportBackend
from coreai_opt._utils.metadata_utils import (
    STATE_DICT_METADATA_BUFFER_PREFIX as _COREML_BUFFER_PREFIX,
)
from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.quantization import (
    ModuleQuantizerConfig,
    QuantizationSpec,
    Quantizer,
    QuantizerConfig,
)
from coreai_opt.quantization._graph._annotation_pattern_registry import (
    BaseAnnotationPattern,
    _AnnotationPatternRegistry,
)
from coreai_opt.quantization._graph._annotation_utils import _is_fx_node_floating_point
from coreai_opt.quantization._graph._prepare_for_export import (
    MIL_ACT_QUANT_MODULE_PREFIX as _MIL_ACT_QUANT_MODULE_PREFIX,
)
from coreai_opt.quantization._graph.quantizer import _AnnotationHandler
from coreai_opt.quantization.spec import (
    PerBlockGranularity,
    PerTensorGranularity,
    QuantizationScheme,
    default_activation_quantization_spec,
    default_weight_quantization_spec,
)
from coreai_opt.quantization.spec.fake_quantize import FakeQuantizeImplBase
from coreai_opt.quantization.spec.qparams_calculator import (
    MovingAverageQParamsCalculator,
    StaticQParamsCalculator,
)
from tests.models.simple import SimpleModel


@pytest.fixture
def basic_config():
    """Fixture providing basic quantization config."""
    return QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={
                "weight": default_weight_quantization_spec(),
            },
            op_input_spec={
                "*": default_activation_quantization_spec(),
            },
            op_output_spec=None,
        )
    )


@pytest.fixture
def weight_input_act_output_act_config():
    """Fixture providing Weight + Activation Quantization config"""
    return QuantizerConfig()


@pytest.fixture
def weight_only_config():
    """Fixture providing weight-only quantization config."""
    return QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={
                "weight": default_weight_quantization_spec(),
            },
            op_input_spec=None,
            op_output_spec=None,
        )
    )


def get_fake_quant_nodes(model: torch.fx.GraphModule) -> list[torch.fx.Node]:
    """
    Returns list of fake quant nodes present in the input model
    """
    fake_quant_nodes = [
        node for node in model.graph.nodes if "activation_post_process" in node.name
    ]
    return fake_quant_nodes


class TestGraphModeQuantizer:
    """Test cases for GraphQuantizer class."""

    def test_init_with_config(self, simple_conv_linear_model, basic_config):
        """Test GraphQuantizer initialization with config."""
        quantizer = Quantizer(simple_conv_linear_model, basic_config)

        assert quantizer._model is simple_conv_linear_model
        assert quantizer._config == basic_config

    def test_init_without_config(self, simple_conv_linear_model):
        """Test GraphQuantizer initialization without config."""
        quantizer = Quantizer(simple_conv_linear_model)

        assert quantizer._model is simple_conv_linear_model
        assert quantizer._config == QuantizerConfig()

    def test_prepare_basic(self, simple_conv_linear_model, basic_config, simple_model_input):
        """Test basic prepare functionality."""
        quantizer = Quantizer(simple_conv_linear_model, basic_config)

        # Check that model is not prepared initially
        assert not quantizer._is_model_prepared(quantizer._model)

        prepared_model = quantizer.prepare((simple_model_input,))

        # Verify expected qparams calculators: 1 static (weight) + 1 moving
        # average (activation) per quantizable op (Conv2d, Linear).
        num_static = num_moving_avg = 0
        for module in prepared_model.modules():
            if isinstance(module, StaticQParamsCalculator):
                num_static += 1
            elif isinstance(module, MovingAverageQParamsCalculator):
                num_moving_avg += 1
        assert num_static == 2
        assert num_moving_avg == 2

        # Check that the model is now a GraphModule
        assert isinstance(prepared_model, torch.fx.GraphModule)

        # Check that the quantizer's internal model was updated
        assert quantizer._model is prepared_model

        # Check that the model is now marked as prepared
        assert quantizer._is_model_prepared(prepared_model)

        # Test that the prepared model can be called
        output = prepared_model(simple_model_input)
        assert output.shape == (1, 10)

    def test_prepare_with_dynamic_shapes(
        self, simple_conv_linear_model, basic_config, simple_model_input
    ):
        """Test prepare functionality with dynamic batch size."""

        # Create inputs with different batch sizes to test dynamic shapes
        larger_batch_input = torch.rand(10, 1, 28, 28)

        quantizer = Quantizer(simple_conv_linear_model, basic_config)

        # Check that model is not prepared initially
        assert not quantizer._is_model_prepared(quantizer._model)

        # Define dynamic shapes
        prepared_model = quantizer.prepare(
            example_inputs=(larger_batch_input,),
            dynamic_shapes={"x": (Dim.AUTO, Dim.STATIC, Dim.STATIC, Dim.STATIC)},
        )

        # Check that the model is now a GraphModule
        assert isinstance(prepared_model, torch.fx.GraphModule)

        # Check that the quantizer's internal model was updated
        assert quantizer._model is prepared_model

        # Check that the model is now marked as prepared
        assert quantizer._is_model_prepared(prepared_model)

        # Test that the prepared model can be called with larger batch size
        output_large = prepared_model(larger_batch_input)
        assert output_large.shape == (10, 10)

        # Test that the prepared model can be called with original batch size
        output = prepared_model(simple_model_input)
        assert output.shape == (1, 10)

    def test_finalize_with_none_model_arg(
        self, simple_conv_linear_model, basic_config, simple_model_input
    ):
        """Test finalize with None model (uses internal model)."""
        quantizer = Quantizer(simple_conv_linear_model, basic_config)
        quantizer.prepare((simple_model_input,))

        finalized_model = quantizer.finalize()

        assert isinstance(finalized_model, torch.fx.GraphModule)

    def test_prepared_model_supports_torch_inference(
        self, simple_conv_linear_model, basic_config, simple_model_input
    ):
        """Verify torch-based evaluation uses the prepared model directly."""
        quantizer = Quantizer(simple_conv_linear_model, basic_config)
        prepared_model = quantizer.prepare((simple_model_input,))

        assert isinstance(prepared_model, torch.fx.GraphModule)

        # Prepared model retains fake quantize layers for torch-based evaluation
        prepared_fake_quant_nodes = get_fake_quant_nodes(prepared_model)
        assert len(prepared_fake_quant_nodes) > 0

        # Prepared model supports torch-based inference directly
        output = prepared_model(simple_model_input)
        assert output.shape == (1, 10)

    def test_finalize_torch_backend(
        self, simple_conv_linear_model, basic_config, simple_model_input
    ):
        """Test finalize with _TORCH backend."""
        quantizer = Quantizer(simple_conv_linear_model, basic_config)
        prepared_model = quantizer.prepare((simple_model_input,))

        finalized_model = quantizer.finalize(prepared_model, ExportBackend._TORCH)

        assert isinstance(finalized_model, torch.fx.GraphModule)

        # Fake quantize layers should remain the same post finalize
        prepared_fake_quant_nodes = get_fake_quant_nodes(prepared_model)
        finalized_fake_quant_nodes = get_fake_quant_nodes(finalized_model)

        for prepared_node in prepared_fake_quant_nodes:
            finalized_node = [
                n for n in finalized_fake_quant_nodes if prepared_node.target == n.target
            ][0]

            assert prepared_node == finalized_node

        # Test inference on finalized model
        output = finalized_model(simple_model_input)
        assert output.shape == (1, 10)

    def test_finalize_after_deepcopy(
        self, simple_conv_linear_model, basic_config, simple_model_input
    ):
        """Finalizing a deepcopy of a prepared GraphModule must succeed."""

        quantizer = Quantizer(simple_conv_linear_model, basic_config)
        prepared_model = quantizer.prepare((simple_model_input,))
        copied_prepared_model = deepcopy(prepared_model)

        assert quantizer._is_model_prepared(copied_prepared_model)

        finalized_model = quantizer.finalize(copied_prepared_model, ExportBackend._TORCH)

        assert isinstance(finalized_model, torch.fx.GraphModule)
        output = finalized_model(simple_model_input)
        assert output.shape == (1, 10)

    def test_finalize_after_state_dict_roundtrip(
        self, simple_conv_linear_model, basic_config, simple_model_input
    ):
        """Save the prepared state_dict, reload into a fresh prepared model, then finalize.

        Calibration uses an input with a different scale than the eval input so
        the calibrated activation qparams encode information that must survive
        the state_dict round-trip — without ``load_state_dict``, the freshly
        prepared model produces a meaningfully different output.
        """
        calibration_input = simple_model_input * 5.0

        quantizer_a = Quantizer(simple_conv_linear_model, basic_config)
        prepared_a = quantizer_a.prepare((simple_model_input,))
        with quantizer_a.calibration_mode():
            prepared_a(calibration_input)
        expected_output = prepared_a(simple_model_input)
        saved_state_dict = prepared_a.state_dict()

        quantizer_b = Quantizer(SimpleModel(), basic_config)
        prepared_b = quantizer_b.prepare((simple_model_input,))
        prepared_b.load_state_dict(saved_state_dict)

        finalized_b = quantizer_b.finalize(prepared_b, ExportBackend._TORCH)
        actual_output = finalized_b(simple_model_input)

        assert torch.allclose(expected_output, actual_output)

    def test_finalize_mil_backend(
        self,
        simple_conv_linear_model: torch.nn.Module,
        basic_config: QuantizerConfig,
        simple_model_input: torch.Tensor,
    ) -> None:
        """Test CoreML backend export with weight and activation quantization."""
        quantizer = Quantizer(simple_conv_linear_model, basic_config)
        prepared_model = quantizer.prepare((simple_model_input,))

        fake_quant_nodes = get_fake_quant_nodes(prepared_model)

        finalized_model = quantizer.finalize(backend=ExportBackend.CoreML)

        # Count activation quantization modules (Sequential containing
        # quantize + dequantize)
        # Only count parent modules, not their children (quantize/dequantize submodules)
        activation_quant_modules = [
            name
            for name, module in finalized_model.named_modules()
            if name.startswith(_MIL_ACT_QUANT_MODULE_PREFIX) and "." not in name
        ]

        # Count weight quantization metadata buffers
        weight_metadata_buffers = [
            name
            for name, _ in finalized_model.named_buffers()
            if _COREML_BUFFER_PREFIX in name and "weight" in name
        ]

        # CoreML export transforms fake quants into different representations:
        # - Weight fake quants -> CoreML metadata buffers (4 properties each:
        #   compression_type, n_bits, scale, zero_point)
        # - Activation fake quants -> Sequential modules with quantize/dequantize
        # Total fake quants = (weight buffers // 4) + activation modules
        assert len(fake_quant_nodes) == (
            len(weight_metadata_buffers) // 4 + len(activation_quant_modules)
        )

    def test_finalize_mlir_backend(
        self, simple_conv_linear_model, basic_config, simple_model_input
    ):
        """Test that CoreAI backend inserts coreai ops"""
        quantizer = Quantizer(simple_conv_linear_model, basic_config)
        prepared_model = quantizer.prepare((simple_model_input,))

        fake_quant_nodes = get_fake_quant_nodes(prepared_model)
        assert len(fake_quant_nodes) == 4

        finalized_model = quantizer.finalize(backend=ExportBackend.CoreAI)

        blockwise_shift_scale_nodes = finalized_model.graph.find_nodes(
            op="call_function", target=coreai.constexpr_blockwise_shift_scale
        )
        quantize_nodes = finalized_model.graph.find_nodes(
            op="call_function", target=coreai.quantize
        )
        dequantize_nodes = finalized_model.graph.find_nodes(
            op="call_function", target=coreai.dequantize
        )

        assert len(fake_quant_nodes) == len(blockwise_shift_scale_nodes) + len(quantize_nodes)
        assert len(quantize_nodes) == len(dequantize_nodes)

    def test_finalize_mmap_dir_raises_on_graph_mode(
        self, simple_conv_linear_model, basic_config, simple_model_input, tmp_path
    ):
        """``mmap_dir`` is not supported in graph execution mode; finalize raises ``ValueError``."""
        quantizer = Quantizer(simple_conv_linear_model, basic_config)
        quantizer.prepare((simple_model_input,))

        with pytest.raises(ValueError, match="mmap_dir is only supported in eager execution mode"):
            quantizer.finalize(backend=ExportBackend.CoreAI, mmap_dir=str(tmp_path))

    def test_observer_fake_quant_status(
        self,
        simple_conv_linear_model,
        basic_config,
        simple_model_input,
    ):
        """Test observer and fake quant status"""
        quantizer = Quantizer(simple_conv_linear_model, basic_config)
        prepared_model = quantizer.prepare((simple_model_input,))

        # Verify that observers are disabled and fake quant is enabled after prepare
        for _name, module in prepared_model.named_modules():
            if isinstance(module, FakeQuantizeImplBase):
                assert module.observer_enabled.item() == 0
                assert module.fake_quant_enabled.item() == 1

        # Test entering calibration mode
        with quantizer.calibration_mode():
            # Verify that observers are enabled. Weight FQ stays on so
            # activation observers see the effect of quantized weights;
            # activation FQ is disabled so observers collect raw stats.
            for _name, module in prepared_model.named_modules():
                if isinstance(module, FakeQuantizeImplBase):
                    assert module.observer_enabled.item() == 1
                    expected_fq = (
                        1 if module.quantization_target == CompressionTargetTensor.WEIGHT else 0
                    )
                    assert module.fake_quant_enabled.item() == expected_fq

        # # Verify that observers are disabled and fake quant is enabled on exit
        for _name, module in prepared_model.named_modules():
            if isinstance(module, FakeQuantizeImplBase):
                assert module.observer_enabled.item() == 0
                assert module.fake_quant_enabled.item() == 1

    def test_calibration_mode_exception_handling(
        self,
        simple_conv_linear_model,
        basic_config,
        simple_model_input,
    ):
        """Test calibration_mode properly handles exceptions."""
        quantizer = Quantizer(simple_conv_linear_model, basic_config)
        prepared_model = quantizer.prepare((simple_model_input,))

        # Test that cleanup happens even when exception is raised
        with pytest.raises(ValueError):
            with quantizer.calibration_mode():
                raise ValueError("Test exception")

        # Verify cleanup still occurred
        for _name, module in prepared_model.named_modules():
            if isinstance(module, FakeQuantizeImplBase):
                assert module.observer_enabled.item() == 0
                assert module.fake_quant_enabled.item() == 1

    def test_calibration_mode_with_external_model(
        self, simple_conv_linear_model, basic_config, simple_model_input
    ):
        """Test calibration_mode with external model parameter."""
        quantizer = Quantizer(simple_conv_linear_model, basic_config)
        prepared_model = quantizer.prepare((simple_model_input,))

        # Create another prepared model
        other_model = SimpleModel()
        other_quantizer = Quantizer(other_model, basic_config)
        other_prepared_model = other_quantizer.prepare((simple_model_input,))

        # Before calibration, quantizer's internal model should point to prepared model
        assert quantizer._model is prepared_model

        # Test calibration_mode with external prepared model
        with quantizer.calibration_mode(other_prepared_model):
            # Verify that observers are enabled on the external model. Weight FQ
            # stays on; activation FQ is off.
            for _name, module in other_prepared_model.named_modules():
                if isinstance(module, FakeQuantizeImplBase):
                    assert module.observer_enabled.item() == 1
                    expected_fq = (
                        1 if module.quantization_target == CompressionTargetTensor.WEIGHT else 0
                    )
                    assert module.fake_quant_enabled.item() == expected_fq

            # The quantizer's internal model should be updated to the provided model
            assert quantizer._model is other_prepared_model

        # Verify cleanup occurred on the external model
        for _name, module in other_prepared_model.named_modules():
            if isinstance(module, FakeQuantizeImplBase):
                assert module.observer_enabled.item() == 0
                assert module.fake_quant_enabled.item() == 1

    def test_end_to_end_workflow(self, simple_conv_linear_model, basic_config, simple_model_input):
        """Test complete quantization workflow."""
        quantizer = Quantizer(simple_conv_linear_model, basic_config)
        simple_model_input_1 = simple_model_input
        simple_model_input_2 = torch.rand(1, 1, 28, 28)
        simple_model_input_3 = torch.rand(1, 1, 28, 28)

        # Step 1: Prepare
        prepared_model = quantizer.prepare((simple_model_input_1,))
        assert isinstance(prepared_model, torch.fx.GraphModule)
        # prepared model output should not match base model, since fake quant is enabled
        assert not torch.equal(
            prepared_model(simple_model_input_1),
            simple_conv_linear_model(simple_model_input_1),
        )

        # Step 2: Calibration (simulate)
        with quantizer.calibration_mode():
            prepared_out = prepared_model(simple_model_input_2)
            original_out = simple_conv_linear_model(simple_model_input_2)
            # prepared model output should NOT match base model: weight fake
            # quant stays on during calibration so activation observers see the
            # effect of quantized weights. Only activation FQ is disabled.
            assert not torch.equal(prepared_out, original_out)

        pre_finalize_out = prepared_model(simple_model_input_3)

        try:
            _p = deepcopy(prepared_model)
        except Exception as e:
            assert f"Deepcopy failed for prepared model with error: {e}"

        # Step 3: Finalize
        finalized_model = quantizer.finalize(backend=ExportBackend._TORCH)
        assert isinstance(finalized_model, torch.fx.GraphModule)
        # prepare model output before finalize should match output after finalize
        final_output = finalized_model(simple_model_input_3)
        assert torch.equal(pre_finalize_out, final_output)

        # Test output shape
        assert final_output.shape == (1, 10)

    def test_prepare_export_with_no_grad_flag(self, basic_config):
        """Test prepare with export_with_no_grad flag."""

        class CustomModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(10, 10)
                self.linear2 = torch.nn.Linear(10, 20)

            def forward(self, x):
                if torch.is_grad_enabled():
                    return self.linear1(x)
                else:
                    return self.linear2(x)

        model = CustomModule()
        example_input = torch.rand(1, 10)
        # Test with export_with_no_grad=True (default)
        quantizer = Quantizer(model, basic_config)
        prepared_model_true = quantizer.prepare(example_inputs=(example_input,))

        # Test prepared model output shape
        output_true = prepared_model_true(example_input)
        assert output_true.shape == (1, 20)

        # Test with export_with_no_grad=False
        quantizer2 = Quantizer(model, basic_config)
        prepared_model_false = quantizer2.prepare(
            example_inputs=(example_input,), export_with_no_grad=False
        )

        # Test prepared model output shape
        output_false = prepared_model_false(example_input)
        assert output_false.shape == (1, 10)

    def test_prepare_with_symint_mul_partition_collision(self):
        """Verify prepare() handles a SourcePartition with multiple call_function nodes.

        torch.export's ``insert_deferred_runtime_asserts`` synthesizes one SymInt
        ``mul`` per shape-runtime assertion, all sharing one ``torch_fn`` tag, so
        they collapse into a single ``SourcePartition``. The annotator picks the
        first node and the downstream ``_is_fx_node_floating_point`` filter no-ops
        on SymInt inputs.
        """

        H, W, B, embed_dim = 4, 4, 1, 8
        num_iters = 2  # >=2 to force the synthesized muls to collide.

        class SymIntMulModel(nn.Module):
            def __init__(self, num_iters: int, embed_dim: int) -> None:
                super().__init__()
                self.num_iters = num_iters
                self.linear = nn.Linear(embed_dim, embed_dim)

            def forward(self, x: torch.Tensor, spatial_shapes: torch.Tensor) -> torch.Tensor:
                for i in range(self.num_iters):
                    h = spatial_shapes[i, 0].item()
                    w = spatial_shapes[i, 1].item()
                    # The h * w == x.size(1) runtime assertion is synthesized as a
                    # SymInt mul node by ``insert_deferred_runtime_asserts``; one per
                    # iteration, all sharing one ``torch_fn`` tag — the partition
                    # collision this test guards against. The view itself uses static
                    # H, W instead of the SymInts to avoid the data-dependent reshape
                    # guards that fail to discharge on torch 2.8.
                    torch._check(h >= 0)
                    torch._check(w >= 0)
                    torch._check(h * w == x.size(1))
                    x_view = x.view(x.size(0), H, W, x.size(-1))
                    x_view = x_view.flatten(1, 2)
                    x = self.linear(x_view)
                return x

        model = SymIntMulModel(num_iters=num_iters, embed_dim=embed_dim).eval()
        example_inputs = (
            torch.randn(B, H * W, embed_dim),
            torch.tensor([[H, W]] * num_iters, dtype=torch.long),
        )

        act_only_config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec=None,
                op_input_spec={"*": default_activation_quantization_spec()},
                op_output_spec={"*": default_activation_quantization_spec()},
            )
        )

        quantizer = Quantizer(model, act_only_config)
        prepared_model = quantizer.prepare(example_inputs=example_inputs)
        assert isinstance(prepared_model, torch.fx.GraphModule)

        # The collision precondition: torch.export synthesized multiple SymInt muls.
        sym_mul_nodes = [
            n
            for n in prepared_model.graph.nodes
            if n.op == "call_function"
            and "mul" in n.name
            and isinstance(n.meta.get("val"), torch.SymInt)
        ]
        assert len(sym_mul_nodes) >= 2, (
            f"expected multiple SymInt mul nodes, got {[n.name for n in sym_mul_nodes]}"
        )

        # Per iteration the fp chain is `view -> flatten -> linear`, with one observer
        # on each of the 3 edges (view->flatten, flatten->linear, linear->next). If
        # observers were placed on SymInt ops too, the count would be higher.
        fake_quant_nodes = get_fake_quant_nodes(prepared_model)
        assert len(fake_quant_nodes) == 3 * num_iters


class TestAnnotationHandler:
    """Test cases for AnnotationHandler class."""

    @pytest.fixture
    def temp_pattern(self):
        """Temporarily register an annotation pattern for testing."""
        pattern_name = "_test_pattern"
        assert pattern_name not in _AnnotationPatternRegistry.REGISTRY

        @_AnnotationPatternRegistry.register(pattern_name)
        class TestAnnotator(BaseAnnotationPattern):
            generate_patterns_call_count = 0
            match_single_pattern_call_count = 0
            num_patterns = 0

            @classmethod
            def generate_patterns(cls):
                cls.generate_patterns_call_count += 1
                return [MagicMock() for _ in range(cls.num_patterns)]

            @classmethod
            def match_single_pattern(cls, _model, _pattern):
                cls.match_single_pattern_call_count += 1
                return {}

            @classmethod
            def get_annotator_func(cls):
                pass

        yield TestAnnotator

        del _AnnotationPatternRegistry.REGISTRY[pattern_name]

    def test_init(self, basic_config):
        """Test AnnotationHandler initialization."""
        module_configs = {"conv": ModuleQuantizerConfig()}
        handler = _AnnotationHandler(module_configs, {}, {})
        assert handler._module_configs is module_configs

    def test_annotate_all_patterns_generate_patterns_called_once(self, basic_config, temp_pattern):
        """Test that generate_patterns is called only once per annotation class"""

        NUM_PATTERNS = 3
        temp_pattern.num_patterns = NUM_PATTERNS

        handler = _AnnotationHandler({}, {}, {})

        # Create a simple GraphModule for testing
        model = torch.fx.symbolic_trace(SimpleModel())

        # Call _annotate_all_patterns
        handler._match_all_annotators(model)

        # Verify generate_patterns is called only once per annotation class
        # due to the caching mechanism in get_patterns()
        assert temp_pattern.generate_patterns_call_count == 1, (
            f"Expected generate_patterns to be called once for TestAnnotator, "
            f"but was called {temp_pattern.generate_patterns_call_count} times"
        )
        # Verify annotate_single_pattern is called only once per pattern
        assert temp_pattern.match_single_pattern_call_count == NUM_PATTERNS, (
            f"Expected annotate_single_pattern to be called {NUM_PATTERNS} times, but "
            f"was called {temp_pattern.match_single_pattern_call_count} times"
        )


class TestAPIOrdering:
    def test_multiple_prepare_calls(
        self, simple_conv_linear_model, basic_config, simple_model_input
    ):
        """Test calling prepare multiple times should raise error."""
        quantizer = Quantizer(simple_conv_linear_model, basic_config)

        # First prepare
        prepared_model1 = quantizer.prepare((simple_model_input,))
        assert isinstance(prepared_model1, torch.fx.GraphModule)
        assert quantizer._is_model_prepared(prepared_model1)

        # Second prepare should raise RuntimeError
        with pytest.raises(RuntimeError, match="Model has already been prepared"):
            quantizer.prepare((simple_model_input,))

    def test_finalize_before_prepare(self, simple_conv_linear_model, basic_config):
        """Test calling finalize before prepare should raise error."""
        quantizer = Quantizer(simple_conv_linear_model, basic_config)

        # Verify model is not prepared
        assert not quantizer._is_model_prepared(quantizer._model)

        # Should raise RuntimeError when trying to finalize unprepared model
        with pytest.raises(RuntimeError, match="Model must be prepared before finalization"):
            quantizer.finalize(None, ExportBackend._TORCH)

    def test_calibration_mode_before_prepare(self, simple_conv_linear_model, basic_config):
        """Test calling calibration_mode before prepare should raise error."""
        quantizer = Quantizer(simple_conv_linear_model, basic_config)

        # Verify model is not prepared
        assert not quantizer._is_model_prepared(quantizer._model)

        # Should raise RuntimeError when trying to enter calibration mode
        # with unprepared model
        with pytest.raises(
            RuntimeError,
            match="Model must be prepared before entering calibration mode",
        ):
            with quantizer.calibration_mode():
                pass

    def test_prepared_attribute_persistence(
        self, simple_conv_linear_model, basic_config, simple_model_input
    ):
        """Test that prepared attribute persists on the model."""
        quantizer = Quantizer(simple_conv_linear_model, basic_config)

        # Initially not prepared
        assert not quantizer._is_model_prepared(quantizer._model)

        # Prepare the model
        prepared_model = quantizer.prepare((simple_model_input,))

        # Check prepared attribute is set and accessible
        assert quantizer._is_model_prepared(prepared_model)

        # Check that the attribute persists through operations
        with quantizer.calibration_mode():
            assert quantizer._is_model_prepared(quantizer._model)

        # Still prepared after calibration
        assert quantizer._is_model_prepared(quantizer._model)

        # Check that the attribute is removed after finalize
        finalized_model = quantizer.finalize()
        assert not quantizer._is_model_prepared(finalized_model)


class TestGraphModeQuantizerTrainingMode:
    def _get_fake_quant_modules(self, model):
        fake_quants = []
        for module in model.modules():
            if isinstance(module, FakeQuantizeImplBase):
                fake_quants.append(module)
        return fake_quants

    def test_training_mode_state_validation(
        self, simple_conv_linear_model, simple_model_input, weight_only_config
    ):
        """
        Checks that the observer state and fake quant state is
        as expected before training mode, while in training mode,
        and after exiting training mode
        """
        model = simple_conv_linear_model

        quantizer = Quantizer(model, weight_only_config)
        prepared_model = quantizer.prepare((simple_model_input,))

        fake_quants = self._get_fake_quant_modules(prepared_model)
        assert len(fake_quants) == 2

        for fq in fake_quants:
            assert fq.fake_quant_enabled[0] == 1
            assert fq.observer_enabled[0] == 0

        with quantizer.training_mode():
            for fq in fake_quants:
                assert fq.fake_quant_enabled[0] == 1
                assert fq.observer_enabled[0] == 1

        for fq in fake_quants:
            assert fq.fake_quant_enabled[0] == 1
            assert fq.observer_enabled[0] == 0

    def test_training_mode_error_handling(self, simple_conv_linear_model, simple_model_input):
        """
        Test that the model raises the right exceptions
        """
        model = simple_conv_linear_model
        config = QuantizerConfig()
        quantizer = Quantizer(model, config)

        with pytest.raises(RuntimeError, match="Model must be prepared"):
            with quantizer.training_mode():
                pass

        quantizer.prepare((simple_model_input,))

        with pytest.raises(TypeError, match="must be a torch.fx.GraphModule"):
            with quantizer.training_mode(model="invalid"):
                pass

    def test_qat_scale_updates(
        self, simple_conv_linear_model, simple_model_input, weight_only_config
    ):
        """Test that quantization scales update during QAT training iterations."""
        model = simple_conv_linear_model

        quantizer = Quantizer(model, weight_only_config)
        prepared_model = quantizer.prepare((simple_model_input,))

        fake_quants = self._get_fake_quant_modules(prepared_model)
        assert len(fake_quants) > 0

        initial_scales = []
        for fq in fake_quants:
            scale, _, _ = fq.calculate_qparams()
            initial_scales.append(scale.clone())

        prev_weights = dict(
            [(name, param.data.clone()) for name, param in prepared_model.named_parameters()]
        )

        optimizer = torch.optim.SGD(prepared_model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()

        with quantizer.training_mode():
            for fq in fake_quants:
                assert fq.observer_enabled[0] == 1, "Observers should be enabled in training mode"
                assert fq.fake_quant_enabled[0] == 1, (
                    "Fake quant should be enabled in training mode"
                )

            for iteration in range(10):
                # Use different input/target combinations to force weight updates
                train_input = torch.rand_like(simple_model_input) * (2.0 + iteration)
                target = torch.randn(simple_model_input.size(0), 10) * (2.0 + iteration)

                output = prepared_model(train_input)
                loss = loss_fn(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Check that the weights are getting updated
                for prev_weight, param in zip(
                    prev_weights.values(), prepared_model.parameters(), strict=True
                ):
                    assert not torch.equal(prev_weight, param.data), (
                        f"Weights are not getting updated in training iteration {iteration}"
                    )

                prev_weights = dict(
                    [
                        (name, param.data.clone())
                        for name, param in prepared_model.named_parameters()
                    ]
                )

        final_scales = []
        for fq in fake_quants:
            scale, _, _ = fq.calculate_qparams()
            final_scales.append(scale.clone())

        for initial_scale, final_scale in zip(initial_scales, final_scales, strict=True):
            assert not torch.allclose(initial_scale, final_scale, atol=1e-6), (
                "Some of the scales aren't getting updated"
            )

        for fq in fake_quants:
            scale, _, _ = fq.calculate_qparams()
            assert scale.numel() > 0, "Scale should be computed"
            assert torch.all(scale > 0), "Scale should be positive"


class TestIntegerActivations:
    """
    Test cases for quantizing integer activations
    """

    @pytest.mark.parametrize(
        "val, op, target, expected",
        [
            (
                torch.randn(2, 3, dtype=torch.float32),
                "call_function",
                torch.ops.aten.relu.default,
                True,
            ),
            (
                torch.tensor([0, 1, 2], dtype=torch.int64),
                "call_function",
                torch.ops.aten.argmax.default,
                False,
            ),
            (
                torch.tensor([True, False, True], dtype=torch.bool),
                "call_function",
                torch.ops.aten.gt.Scalar,
                False,
            ),
            (None, "call_module", "embedding", False),
        ],
    )
    def test_is_fx_node_floating_point(self, val, op, target, expected):
        """
        Test _is_fx_node_floating_point helper function with various node types.
        """
        node = torch.fx.Node(
            graph=torch.fx.Graph(), name="test_node", op=op, target=target, args=(), kwargs={}
        )
        node.meta = {"val": val}

        assert _is_fx_node_floating_point(node) is expected

    def test_integer_activation_model_quantization(self, weight_input_act_output_act_config):
        class IntegerActivationModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, 3, padding=1)
                self.embedding = torch.nn.Embedding(16, 32)

            def forward(self, x):
                conv_out = self.conv(x)
                indices = torch.argmax(conv_out, dim=1)
                embedded = self.embedding(indices)

                return conv_out, indices, embedded

        model = IntegerActivationModel()
        input_tensor = torch.randn(2, 3, 28, 28)

        quantizer = Quantizer(model, weight_input_act_output_act_config)
        prepared_model = quantizer.prepare((input_tensor,))
        prepared_model(input_tensor)

    def test_dynamic_shapes_model_quantization(self, weight_input_act_output_act_config):
        class DynamicShapeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(1000, 10)
                self.linear = torch.nn.Linear(10, 5)

            def forward(self, input_ids, seq_lens):

                seq_len_input = input_ids.size(1)
                embedding_out = self.embedding(input_ids)
                offset = seq_len_input - seq_lens

                return self.linear(embedding_out), offset

        model = DynamicShapeModel()
        max_context_length = 128
        dynamic_shapes = {
            "input_ids": {1: torch.export.Dim("seq_ids", max=max_context_length - 1)},
            "seq_lens": None,
        }

        example_input_ids = torch.randint(0, 1000, (1, 10))
        model(example_input_ids, 2)

        quantizer = Quantizer(model, weight_input_act_output_act_config)
        prepared_model = quantizer.prepare(
            (
                example_input_ids,
                2,
            ),
            dynamic_shapes=dynamic_shapes,
        )
        _ = prepared_model(example_input_ids, 2)


@pytest.mark.parametrize("model_dtype", [torch.float32, torch.float16, torch.bfloat16])
class TestGraphModeHalfPrecisionSupport:
    """Test graph-mode quantization workflow with half precision models"""

    @pytest.mark.parametrize(
        "backend",
        [
            pytest.param(ExportBackend._TORCH, id="torch"),
            pytest.param(ExportBackend.CoreAI, id="coreai"),
            pytest.param(ExportBackend.CoreML, id="coreml"),
        ],
    )
    def test_prepare_calibrate_finalize_with_half_precision(
        self,
        simple_conv_linear_model,
        weight_input_act_output_act_config,
        simple_model_input,
        model_dtype,
        backend,
    ):
        """Test complete graph-mode workflow with half precision models"""
        model = simple_conv_linear_model.to(dtype=model_dtype)
        example_input = simple_model_input.to(dtype=model_dtype)

        quantizer = Quantizer(model, weight_input_act_output_act_config)
        prepared_model = quantizer.prepare((example_input,))

        with quantizer.calibration_mode():
            for _ in range(3):
                calib_input = torch.rand_like(example_input) * 5.0
                _ = prepared_model(calib_input)

        finalized_model = quantizer.finalize(backend=backend)
        if model_dtype == torch.float32 or backend in [
            ExportBackend._TORCH,
            ExportBackend.CoreAI,
        ]:
            output = finalized_model(example_input)
            assert output.shape == (1, 10)
            assert torch.all(torch.isfinite(output)), f"Backend {backend}: output contains inf/nan"
        else:
            # CoreML backend uses torch.quantize_per_tensor C++ bindings
            # which only work on Float tensor
            with pytest.raises(RuntimeError, match="Quantize only works on Float Tensor, got .*"):
                _ = finalized_model(example_input)

        # Verify scale dtypes are correct after finalize for CoreAI backend
        if backend == ExportBackend.CoreAI:
            for key, value in finalized_model.state_dict().items():
                if "scale" in key:
                    assert value.dtype == model_dtype, (
                        f"After finalize with {backend}, key {key}: "
                        f"expected dtype {model_dtype}, got {value.dtype}"
                    )


class TestPreservedAttributes:
    """Test cases for preserved_attributes functionality in graph-mode quantization."""

    def test_preserved_attributes_on_compressed_model(
        self, simple_conv_linear_model, basic_config, simple_model_input
    ):
        """Test that preserved attributes are available on the prepared model."""
        # Add custom attributes to the model
        simple_conv_linear_model.custom_attr = "test_value"
        simple_conv_linear_model.custom_list = [1, 2, 3]
        simple_conv_linear_model.custom_dict = {"key": "value", "num": 42}

        # Create config with preserved_attributes
        config = QuantizerConfig(
            preserved_attributes=["custom_attr", "custom_list", "custom_dict"],
        )

        quantizer = Quantizer(simple_conv_linear_model, config)
        prepared_model = quantizer.prepare((simple_model_input,))

        # Verify attributes are preserved on the prepared model
        assert hasattr(prepared_model, "custom_attr")
        assert prepared_model.custom_attr == "test_value"
        assert hasattr(prepared_model, "custom_list")
        assert prepared_model.custom_list == [1, 2, 3]
        assert hasattr(prepared_model, "custom_dict")
        assert prepared_model.custom_dict == {"key": "value", "num": 42}

        # Finalize the model
        finalized_model = quantizer.finalize()

        # Verify attributes are preserved on the finalized model
        assert hasattr(finalized_model, "custom_attr")
        assert finalized_model.custom_attr == "test_value"
        assert hasattr(finalized_model, "custom_list")
        assert finalized_model.custom_list == [1, 2, 3]
        assert hasattr(finalized_model, "custom_dict")
        assert finalized_model.custom_dict == {"key": "value", "num": 42}

    def test_preserved_attributes_survive_deepcopy(
        self, simple_conv_linear_model, basic_config, simple_model_input
    ):
        """Test that preserved attributes survive deepcopy of prepared model."""
        simple_conv_linear_model.my_config = {"key": "value"}

        config = QuantizerConfig(preserved_attributes=["my_config"])

        quantizer = Quantizer(simple_conv_linear_model, config)
        prepared_model = quantizer.prepare((simple_model_input,))

        # Deepcopy the prepared model
        copied_model = deepcopy(prepared_model)

        # Verify attribute survives deepcopy
        assert hasattr(copied_model, "my_config")
        assert copied_model.my_config == {"key": "value"}

        finalized_model = quantizer.finalize()

        # Deepcopy the finalized model
        copied_model = deepcopy(finalized_model)

        # Verify attribute survives deepcopy
        assert hasattr(copied_model, "my_config")
        assert copied_model.my_config == {"key": "value"}

    def test_preserved_attributes_missing_attribute_warning(
        self, simple_conv_linear_model, basic_config, simple_model_input, caplog
    ):
        """Test that a warning is logged for missing attributes."""
        config = QuantizerConfig(
            preserved_attributes=["nonexistent_attr"],
        )

        quantizer = Quantizer(simple_conv_linear_model, config)

        with caplog.at_level("WARNING"):
            prepared_model = quantizer.prepare((simple_model_input,))

        # Verify warning was logged
        assert any(
            "nonexistent_attr" in record.message and "will be skipped" in record.message
            for record in caplog.records
        )

        # Verify the missing attribute is not on the model
        assert not hasattr(prepared_model, "nonexistent_attr")

    @pytest.mark.parametrize(
        "backend",
        [
            pytest.param(ExportBackend._TORCH, id="torch"),
            pytest.param(ExportBackend.CoreAI, id="coreai"),
            pytest.param(ExportBackend.CoreML, id="coreml"),
        ],
    )
    def test_preserved_attributes_with_different_backends(
        self, simple_conv_linear_model, basic_config, simple_model_input, backend
    ):
        """Test that preserved attributes work with different export backends."""
        # Add custom attributes to the model
        simple_conv_linear_model.backend_specific_attr = f"value_for_{backend.name}"

        # Create config with preserved_attributes
        config = QuantizerConfig(
            preserved_attributes=["backend_specific_attr"],
        )

        quantizer = Quantizer(simple_conv_linear_model, config)
        prepared_model = quantizer.prepare((simple_model_input,))

        # Verify attribute is on prepared model
        assert hasattr(prepared_model, "backend_specific_attr")
        assert prepared_model.backend_specific_attr == f"value_for_{backend.name}"

        # Finalize with specific backend
        finalized_model = quantizer.finalize(backend=backend)

        # Verify attribute is preserved on finalized model
        assert hasattr(finalized_model, "backend_specific_attr")
        assert finalized_model.backend_specific_attr == f"value_for_{backend.name}"


class TestGraphModeQuantizerTrainEval:
    """Test that train/eval mode works on prepared and finalized GraphModules."""

    @staticmethod
    def _is_training(model: torch.fx.GraphModule) -> bool:
        """Check exported model training state via _exported_training attr."""
        return getattr(model, "_exported_training", True)

    def test_prepared_model_train_eval(
        self, simple_conv_linear_model, simple_model_input, weight_only_config
    ):
        """Verify .train() and .eval() don't raise on the prepared model."""
        quantizer = Quantizer(simple_conv_linear_model, weight_only_config)
        prepared_model = quantizer.prepare((simple_model_input,))

        # Should not raise NotImplementedError
        prepared_model.eval()
        assert not self._is_training(prepared_model)

        prepared_model.train()
        assert self._is_training(prepared_model)

        # Idempotent: calling the same mode twice should not error
        prepared_model.eval()
        prepared_model.eval()
        assert not self._is_training(prepared_model)

    def test_finalized_model_train_eval(
        self, simple_conv_linear_model, simple_model_input, weight_only_config
    ):
        """Verify .train() and .eval() don't raise on the finalized model."""
        quantizer = Quantizer(simple_conv_linear_model, weight_only_config)
        quantizer.prepare((simple_model_input,))
        finalized_model = quantizer.finalize()

        finalized_model.eval()
        assert not self._is_training(finalized_model)

        finalized_model.train()
        assert self._is_training(finalized_model)

    def test_train_eval_compatible_with_context_managers(
        self, simple_conv_linear_model, simple_model_input, weight_only_config
    ):
        """Verify .eval()/.train() and context managers interact correctly."""
        quantizer = Quantizer(simple_conv_linear_model, weight_only_config)
        prepared_model = quantizer.prepare((simple_model_input,))

        # Put model in eval mode via .eval()
        prepared_model.eval()
        assert not self._is_training(prepared_model)

        # training_mode context manager should switch to train and restore
        with quantizer.training_mode():
            assert self._is_training(prepared_model)
        assert not self._is_training(prepared_model)

        # Put model in train mode via .train()
        prepared_model.train()
        assert self._is_training(prepared_model)

        # calibration_mode switches to eval and restores
        with quantizer.calibration_mode():
            assert not self._is_training(prepared_model)
        assert self._is_training(prepared_model)
        assert prepared_model.training


class TestCompositeOpQuantization:
    """
    Tests that models with COREAI CompositeOps can be quantized
    """

    class SDPAModule(torch.nn.Module):
        def forward(self, query, key, value):
            from coreai_torch.composite_ops._sdpa import (  # noqa: PLC0415
                scaled_dot_product_attention as _scaled_dot_product_attention,
            )

            return _scaled_dot_product_attention(query, key, value, is_causal=True)

    class SimpleSDPAModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(16, 48)
            self.sdpa = TestCompositeOpQuantization.SDPAModule()

        def forward(self, x):
            qkv = self.proj(x)
            q, k, v = qkv.chunk(3, dim=-1)
            b, s, _ = q.shape
            q = q.reshape(b, 1, s, 16)
            k = k.reshape(b, 1, s, 16)
            v = v.reshape(b, 1, s, 16)
            return self.sdpa(q, k, v)

    @pytest.fixture
    def model(self):
        return self.SimpleSDPAModel()

    @pytest.fixture
    def example_input(self):
        return torch.randn(1, 4, 16)

    @pytest.mark.xfail(reason="tracked by coreai-torch issue #309")
    def test_composite_op_io_quantization(self, model, example_input):
        """
        Verify CompositeOps boundaries can be quantized
        """
        qspec_dict = {
            "dtype": "int8",
            "qscheme": "symmetric",
            "granularity": {"type": "per_tensor"},
        }
        config = QuantizerConfig.from_dict(
            {
                "quantization_config": {
                    "global_config": {"op_state_spec": {"weight": qspec_dict}},
                    "module_name_configs": {
                        "sdpa": {
                            "op_input_spec": None,
                            "op_output_spec": None,
                            "op_state_spec": None,
                            "module_input_spec": {
                                0: qspec_dict,
                                1: qspec_dict,
                                2: qspec_dict,
                            },
                            "module_output_spec": {
                                "*": qspec_dict,
                            },
                        }
                    },
                }
            }
        )

        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare((example_input,))
        assert isinstance(prepared_model, torch.fx.GraphModule)

        def _is_composite_op(node):
            return (
                node.op == "call_function"
                and isinstance(node.target, torch._ops.OpOverload)
                and node.target.namespace == "CompositeOps"
            )

        # Find the SDPA node
        sdpa_nodes = [
            n
            for n in prepared_model.graph.nodes
            if n.op == "call_function"
            and isinstance(n.target, torch._ops.OpOverload)
            and n.target._opname == "scaled_dot_product_attention"
        ]
        assert len(sdpa_nodes) == 1, f"Expected 1 SDPA node, got {len(sdpa_nodes)}"
        sdpa_node = sdpa_nodes[0]

        composite_op_input_nodes = [arg for arg in sdpa_node.args if _is_composite_op(arg)]
        composite_op_output_nodes = [user for user in sdpa_node.users if _is_composite_op(user)]

        assert len(composite_op_input_nodes) == 3, (
            f"Expected 3 custom input nodes (q/k/v), got {len(composite_op_input_nodes)}"
        )
        assert len(composite_op_output_nodes) == 1, "Expected 1 composite op output node"

        for node in composite_op_input_nodes + composite_op_output_nodes:
            assert "name" in node.kwargs, f"kwargs not restored for {node.name}"
            assert "op_name" in node.kwargs, f"kwargs not restored for {node.name}"
            assert node.kwargs["op_name"] == "scaled_dot_product_attention"

        for node in composite_op_input_nodes:
            input_node = node.args[0]
            assert "activation_post_process" in input_node.name, (
                f"Expected fake quant before {node.name} "
                f"(input={node.kwargs['name']}), "
                f"got {input_node.name} instead"
            )

        for node in composite_op_output_nodes:
            users = list(node.users.keys())
            assert any("activation_post_process" in u.name for u in users), (
                f"Expected fake quant after {node.name}, got users: {[u.name for u in users]}"
            )


class TestFP4MLIRExportValidation:
    """Test that FP4 export validation rejects unsupported configurations."""

    @pytest.mark.parametrize(
        "model, input_data, weight_granularity, with_fp4_activation, error_match",
        [
            pytest.param(
                nn.Linear(32, 64),
                torch.randn(1, 32),
                PerTensorGranularity(),
                False,
                "FP4 quantization requires PerBlockGranularity",
                id="per_tensor_granularity_rejected",
            ),
            pytest.param(
                nn.Linear(32, 64),
                torch.randn(1, 32),
                PerBlockGranularity(axis=1, block_size=16),
                False,
                r"FP4 export requires per-axis block sizes \(1, 32\) for a 2D weight",
                id="wrong_block_size_rejected",
            ),
            pytest.param(
                nn.Linear(32, 64),
                torch.randn(1, 32),
                PerBlockGranularity(axis=1, block_size=32),
                True,
                "FP4 activation quantization is not supported for MLIR export",
                id="fp4_activation_rejected",
            ),
            pytest.param(
                nn.Conv2d(32, 64, (3, 3)),
                torch.randn(1, 32, 5, 5),
                PerBlockGranularity(axis=1, block_size=32),
                False,
                r"FP4 export requires per-axis block sizes \(1, 1, 1, 32\) for a 4D weight",
                id="conv_layer_rejected",
            ),
        ],
    )
    def test_fp4_invalid_config_rejected(
        self,
        model,
        input_data,
        weight_granularity,
        with_fp4_activation,
        error_match,
    ):

        activation_spec = None
        if with_fp4_activation:
            activation_spec = QuantizationSpec(
                dtype="float4_e2m1fn",
                qscheme=QuantizationScheme.SYMMETRIC,
                granularity=PerBlockGranularity(axis=1, block_size=32),
            )

        config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec={
                    "weight": QuantizationSpec(
                        dtype="float4_e2m1fn",
                        qscheme=QuantizationScheme.SYMMETRIC,
                        granularity=weight_granularity,
                    ),
                },
                op_input_spec={"*": activation_spec} if activation_spec else None,
                op_output_spec={"*": activation_spec} if activation_spec else None,
            ),
            execution_mode="graph",
        )
        quantizer = Quantizer(model, config)
        quantizer.prepare((input_data,))

        with pytest.raises(RuntimeError, match=error_match):
            quantizer.finalize(backend=ExportBackend.CoreAI)


class TestBlockSizeMismatchSkipGraphMode:
    """Test that non-divisible block sizes produce a warning and skip quantization in graph mode."""

    def test_non_divisible_block_size_warns_and_skips(self, caplog):
        """Graph mode: non-divisible layer gets FQ disabled and removed, divisible stays."""
        # out_features=1000: 1000 % 32 != 0 → FQ disabled and removed
        # out_features=1024: 1024 % 32 == 0 → FQ stays enabled
        model = torch.nn.Sequential(
            torch.nn.Linear(768, 1000),
            torch.nn.Linear(1000, 1024),
        )
        example_inputs = (torch.randn(1, 768),)

        weight_spec = QuantizationSpec(
            dtype="int8",
            qscheme="symmetric",
            granularity=PerBlockGranularity(axis=0, block_size=32),
        )
        config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec={"weight": weight_spec},
                op_input_spec=None,
                op_output_spec=None,
            ),
            execution_mode="graph",
        )

        quantizer = Quantizer(model, config)

        with caplog.at_level(logging.WARNING):
            prepared_model = quantizer.prepare(example_inputs)

        assert prepared_model is not None
        assert any("Skipping quantization" in msg for msg in caplog.messages)

        # Only the second Linear (out_features=1024) is compatible with block_size=32
        fq_modules = [m for m in prepared_model.modules() if isinstance(m, FakeQuantizeImplBase)]
        assert all(not m.is_disabled() for m in fq_modules), (
            "Disabled FQ modules should be removed during prepare()"
        )
        assert len(fq_modules) == 1, (
            f"Expected 1 enabled FQ module (for divisible layer), got {len(fq_modules)}"
        )

    @pytest.mark.parametrize(
        "backend",
        [
            pytest.param(ExportBackend.CoreML, id="coreml"),
            pytest.param(ExportBackend.CoreAI, id="coreai"),
        ],
    )
    def test_all_nodes_disabled_finalize_raises_for_no_fq_nodes(self, caplog, backend):
        """Finalize should raise ValueError when all FQ nodes were disabled and removed."""
        # Both linears have out_features=1000, not divisible by block_size=32
        model = torch.nn.Sequential(
            torch.nn.Linear(768, 1000),
            torch.nn.Linear(1000, 1000),
        )
        example_inputs = (torch.randn(1, 768),)

        weight_spec = QuantizationSpec(
            dtype="int8",
            qscheme="symmetric",
            granularity=PerBlockGranularity(axis=0, block_size=32),
        )
        config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec={"weight": weight_spec},
                op_input_spec=None,
                op_output_spec=None,
            ),
            execution_mode="graph",
        )

        quantizer = Quantizer(model, config)

        with caplog.at_level(logging.WARNING):
            prepared_model = quantizer.prepare(example_inputs)

        assert prepared_model is not None
        assert any("Skipping quantization" in msg for msg in caplog.messages)

        # All FQ modules should have been removed during prepare
        fq_modules = [m for m in prepared_model.modules() if isinstance(m, FakeQuantizeImplBase)]
        assert len(fq_modules) == 0, f"Expected 0 FQ modules after prepare(), got {len(fq_modules)}"

        # Finalize should raise since there are no FQ nodes to export
        with pytest.raises(ValueError, match="no fake quantization nodes"):
            quantizer.finalize(backend=backend)


class TestFixedQParamsActivations:
    """
    Tests that activations with analytically known output ranges receive correct fixed qparams.
    """

    @staticmethod
    def _assert_fq_properties(
        fq: FakeQuantizeImplBase,
        expected_qscheme: QuantizationScheme,
        expected_float_range: tuple,
        expected_scale: float,
    ) -> None:
        assert fq.qscheme == expected_qscheme
        assert fq.qparams_calculator.qscheme == expected_qscheme
        assert fq.qscheme == fq.qparams_calculator.qscheme
        assert fq.qparams_calculator.float_range == expected_float_range
        scale, _, _ = fq.calculate_qparams()
        torch.testing.assert_close(scale, torch.full_like(scale, expected_scale))

    @pytest.mark.parametrize(
        "activation_fn, expected_qscheme, expected_float_range, expected_scale",
        [
            pytest.param(
                torch.sigmoid,
                QuantizationScheme.ASYMMETRIC,
                (0.0, 1.0),
                # Asymmetric [0, 1] over 255 int8 steps: scale = 1/255.
                1.0 / 255.0,
                id="sigmoid",
            ),
            pytest.param(
                torch.tanh,
                QuantizationScheme.SYMMETRIC,
                (-1.0, 1.0),
                # Symmetric [-1, 1] over 255 int8 steps: scale = 2/255.
                2.0 / 255.0,
                id="tanh",
            ),
            pytest.param(
                nn.Hardtanh(-3.0, 3.0),
                QuantizationScheme.SYMMETRIC,
                (-3.0, 3.0),
                # Symmetric [-3, 3] over 255 int8 steps: scale = 6/255.
                6.0 / 255.0,
                id="hardtanh_symmetric",
            ),
            pytest.param(
                nn.Hardtanh(0.0, 6.0),
                QuantizationScheme.ASYMMETRIC,
                (0.0, 6.0),
                # Asymmetric [0, 6] over 255 int8 steps: scale = 6/255.
                6.0 / 255.0,
                id="hardtanh_asymmetric",
            ),
        ],
    )
    def test_fixed_activation_qparams_stable_through_prepare_calibration_and_qat(
        self,
        activation_fn,
        expected_qscheme,
        expected_float_range,
        expected_scale,
    ):
        """Fixed-range activation ops should get the correct qscheme and float_range,
        with a scale that remains unchanged through calibration and QAT."""

        class ActivationLinearActivation(nn.Module):
            def __init__(self, fn):
                super().__init__()
                self.linear = nn.Linear(2, 2, bias=False)
                self._fn = fn

            def forward(self, x):
                x = self.linear(x)
                x = self._fn(x)
                return x

        torch.manual_seed(42)
        model = ActivationLinearActivation(activation_fn).eval()
        example_input = torch.randn(1, 2)

        quantizer = Quantizer(model, QuantizerConfig())
        prepared_model = quantizer.prepare((example_input,))

        # Run one forward to initialize the qparams calculators.
        prepared_model(example_input)

        fq = prepared_model.activation_post_process_2

        # After preparation.
        self._assert_fq_properties(fq, expected_qscheme, expected_float_range, expected_scale)

        # After calibration — qparams should be unchanged.
        with quantizer.calibration_mode():
            for _ in range(5):
                prepared_model(torch.randn(1, 2))

        self._assert_fq_properties(fq, expected_qscheme, expected_float_range, expected_scale)

        # After QAT — qparams should still be unchanged.
        optimizer = torch.optim.SGD(prepared_model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        with quantizer.training_mode():
            for _ in range(5):
                x = torch.randn(1, 2)
                optimizer.zero_grad()
                loss = criterion(prepared_model(x), torch.zeros(1, 2))
                loss.backward()
                optimizer.step()

        self._assert_fq_properties(fq, expected_qscheme, expected_float_range, expected_scale)

    def test_relu_output_qparams(self):
        """ReLU output fake quant should have ASYMMETRIC qscheme with float_range=(0, None).

        The min is analytically pinned at 0 (zero_point stays at -128 across all phases),
        while the max remains data-driven (float_range[1] is None).
        """

        class LinearRelu(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 2, bias=False)

            def forward(self, x):
                return torch.relu(self.linear(x))

        torch.manual_seed(42)
        model = LinearRelu().eval()
        example_input = torch.randn(1, 2)

        quantizer = Quantizer(model, QuantizerConfig())
        prepared_model = quantizer.prepare((example_input,))
        prepared_model(example_input)

        fq = prepared_model.activation_post_process_2

        assert fq.qscheme == QuantizationScheme.ASYMMETRIC
        assert fq.qparams_calculator.qscheme == QuantizationScheme.ASYMMETRIC
        assert fq.qscheme == fq.qparams_calculator.qscheme
        assert fq.qparams_calculator.float_range == (0.0, None)

        # Min is pinned to 0 — zero_point should stay at -128 across all phases.
        _, zp, _ = fq.calculate_qparams()
        assert torch.all(zp == -128)

        with quantizer.calibration_mode():
            for _ in range(5):
                prepared_model(torch.randn(1, 2))

        _, zp, _ = fq.calculate_qparams()
        assert torch.all(zp == -128)

        optimizer = torch.optim.SGD(prepared_model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        with quantizer.training_mode():
            for _ in range(5):
                x = torch.randn(1, 2)
                optimizer.zero_grad()
                loss = criterion(prepared_model(x), torch.zeros(1, 2))
                loss.backward()
                optimizer.step()

        _, zp, _ = fq.calculate_qparams()
        assert torch.all(zp == -128)

    def test_relu_qparams_propagated_through_passthrough_ops(self):
        """Relu's ASYMMETRIC qscheme and pinned-min float_range propagate through
        passthrough ops (view, squeeze, unsqueeze) to the following quantizable op's input FQ."""

        class LinearReluPassthroughLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(2, 2, bias=False)
                self.linear2 = nn.Linear(2, 2, bias=False)

            def forward(self, x):
                x = torch.relu(self.linear1(x))
                x = x.view(1, 2)
                x = x.squeeze(0)
                x = x.unsqueeze(0)
                return self.linear2(x)

        torch.manual_seed(42)
        model = LinearReluPassthroughLinear().eval()
        example_input = torch.randn(1, 2)

        quantizer = Quantizer(model, QuantizerConfig())
        prepared_model = quantizer.prepare((example_input,))
        prepared_model(example_input)

        # Both relu's output FQ and linear2's input FQ (reached through the passthrough chain)
        # should carry the relu-adjusted spec.
        fqs_with_relu_range = [
            (name, m)
            for name, m in prepared_model.named_modules()
            if isinstance(m, FakeQuantizeImplBase)
            and getattr(m.qparams_calculator, "float_range", None) == (0.0, None)
        ]
        assert len(fqs_with_relu_range) == 2, (
            f"Expected 2 FQ modules with float_range=(0.0, None) "
            f"(relu output + linear2 input via passthrough propagation), "
            f"got {len(fqs_with_relu_range)}: {[n for n, _ in fqs_with_relu_range]}"
        )
        for name, fq in fqs_with_relu_range:
            assert fq.qscheme == QuantizationScheme.ASYMMETRIC, (
                f"{name}: expected ASYMMETRIC qscheme"
            )
            assert fq.qparams_calculator.qscheme == QuantizationScheme.ASYMMETRIC, (
                f"{name}: qparams_calculator qscheme mismatch"
            )
            _, zp, _ = fq.calculate_qparams()
            assert torch.all(zp == -128), f"{name}: expected zero_point=-128"
