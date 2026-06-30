# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import copy
import logging
import os

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrize import is_parametrized
from torch.overrides import _get_current_function_mode

from coreai_opt import ExportBackend
from coreai_opt._utils.insertion.torch_function.modes import RegisterEagerOptimizationMode
from coreai_opt._utils.insertion.torch_function.registered_optimizers_tracker import (
    FunctionRegisteredOptimizers,
    RegisteredOptimizersTracker,
)
from coreai_opt.quantization import (
    ModuleQuantizerConfig,
    QuantizationSpec,
    Quantizer,
    QuantizerConfig,
)
from coreai_opt.quantization._eager.supported_ops_registry import EagerQuantizerSupportedOpsRegistry
from coreai_opt.quantization.config import OpQuantizerConfig
from coreai_opt.quantization.spec import (
    PerBlockGranularity,
    PerTensorGranularity,
    QuantizationGranularity,
    QuantizationScheme,
    default_activation_quantization_spec,
    default_weight_quantization_spec,
)
from coreai_opt.quantization.spec.fake_quantize import FakeQuantizeImplBase


class InnerModule(nn.Module):
    """Inner module with a linear layer."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        x = self.linear(x) + x
        return x


class DoubleInnerModel(nn.Module):
    """Model with two separate InnerModule instances."""

    def __init__(self):
        super().__init__()
        self.inner1 = InnerModule()
        self.inner2 = InnerModule()

    def forward(self, x, y):
        a = self.inner1(x)
        b = self.inner2(y)
        return a + b


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(32 * 28 * 28, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class SharedWeightModel(nn.Module):
    """Model with shared weights between modules."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 4, bias=False)
        self.linear2 = nn.Linear(4, 4, bias=False)
        # Share weight between linear1 and linear2
        self.linear2.weight = self.linear1.weight

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class SimpleLinearModel(nn.Module):
    """Simple model with Linear operation."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def simple_model():
    return SimpleModel()


@pytest.fixture
def example_input():
    return torch.randn(1, 1, 28, 28)


@pytest.fixture
def basic_config():
    return QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={
                "weight": default_weight_quantization_spec(),
            },
            op_input_spec={"*": default_activation_quantization_spec()},
            op_output_spec=None,
        ),
        execution_mode="eager",
    )


@pytest.fixture
def weight_only_config():
    return QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={
                "weight": default_weight_quantization_spec(),
            },
            op_input_spec=None,
            op_output_spec=None,
        ),
        execution_mode="eager",
    )


@pytest.fixture
def input_activation_only_config():
    return QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec=None,
            op_input_spec={"*": default_activation_quantization_spec()},
            op_output_spec=None,
        ),
        execution_mode="eager",
    )


@pytest.fixture
def output_activation_only_config():
    return QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec=None,
            op_input_spec=None,
            op_output_spec={"*": default_activation_quantization_spec()},
        ),
        execution_mode="eager",
    )


@pytest.fixture
def full_activation_config():
    return QuantizerConfig(execution_mode="eager")


@pytest.fixture
def conv2d_unregistered(monkeypatch):
    """Temporarily remove conv2d from the supported ops registry."""
    conv2d_class = EagerQuantizerSupportedOpsRegistry.get_class("conv2d")
    monkeypatch.setattr(conv2d_class, "ops", [])


class TestEagerQuantizer:
    def test_init_with_config(self, simple_model, basic_config):
        """
        Test the Quantizer with a simple model and the basic config
        and check that the model and config are getting propagated. Also check
        if model is prepared.
        """
        quantizer = Quantizer(simple_model, basic_config)
        assert quantizer._model is simple_model
        assert quantizer._config == basic_config
        assert not quantizer._is_model_prepared(quantizer._model)

    def test_init_with_default_config(self, simple_model):
        """
        Test Quantizer with default config, but execution mode overwritten.
        """
        default_eager_config = QuantizerConfig(execution_mode="eager")
        quantizer = Quantizer(simple_model, default_eager_config)
        assert quantizer._model is simple_model
        assert isinstance(quantizer._config, QuantizerConfig)
        assert quantizer._config == default_eager_config
        assert not quantizer._is_model_prepared(quantizer._model)

    def test_prepare_basic(self, simple_model, basic_config, example_input):
        """
        Check that the model is getting prepared properly and we can pass an example
        input through the model and get the correct shape output.

        Also check that the conv and linear layers have weight quantizers
        (through parametrization) and input activation quantization
        """
        quantizer = Quantizer(simple_model, basic_config)

        prepared_model = quantizer.prepare((example_input,))

        assert quantizer._is_model_prepared(prepared_model)

        assert prepared_model is simple_model
        assert is_parametrized(prepared_model.conv, "weight")
        assert is_parametrized(prepared_model.linear, "weight")

        # Check weight quantizers are inserted
        assert isinstance(
            prepared_model.conv.parametrizations["weight"][0],
            FakeQuantizeImplBase,
        )

        assert isinstance(
            prepared_model.linear.parametrizations["weight"][0],
            FakeQuantizeImplBase,
        )

        # Check activation quantizers are inserted
        act_quants = 0
        for name, mod in prepared_model.named_modules():
            if name.endswith("quantize_input"):
                assert isinstance(mod, FakeQuantizeImplBase)
                act_quants += 1

        # one for linear and one for conv
        assert act_quants == 2

        output = prepared_model(example_input)
        # The act handler registered_optimizers_tracker should be cleared out after each
        # forward pass of the prepared model.
        registered_optimizers_tracker = (
            quantizer._quantizer._handler.act_handler.registered_optimizers_tracker
        )
        assert registered_optimizers_tracker.get_registry_dict() == {}
        assert output.shape == (1, 10)

        # Test deepcopy ability of prepared model
        copied_model = copy.deepcopy(prepared_model)
        output_2 = copied_model(example_input)
        assert torch.equal(output, output_2)

    def test_prepare_weight_only(self, simple_model, weight_only_config, example_input):
        """
        Test basic functionality of preparing a weight-only quant config
        """
        quantizer = Quantizer(simple_model, weight_only_config)
        prepared_model = quantizer.prepare((example_input,))

        assert is_parametrized(prepared_model.conv, "weight") and isinstance(
            prepared_model.conv.parametrizations["weight"][0], FakeQuantizeImplBase
        )
        assert is_parametrized(prepared_model.linear, "weight") and isinstance(
            prepared_model.linear.parametrizations["weight"][0], FakeQuantizeImplBase
        )

        # Check that there are no activation quantizers
        act_quants = list(
            filter(lambda i: i[0].endswith("quantize_input"), prepared_model.named_modules())
        )
        assert len(act_quants) == 0

        # There are no other fake quant modules other than the ones for weight quant
        fq_mods = list(
            filter(lambda m: isinstance(m, FakeQuantizeImplBase), prepared_model.modules())
        )
        assert len(fq_mods) == 2

        output = prepared_model(example_input)
        assert output.shape == (1, 10)

    def test_multiple_prepare_calls(self, simple_model, basic_config, example_input):
        """
        If we re-prepare the model, it should throw a warning
        """
        quantizer = Quantizer(simple_model, basic_config)

        first_prepare = quantizer.prepare((example_input,))
        with pytest.warns():
            second_prepare = quantizer.prepare((example_input,))

        assert first_prepare is second_prepare

    def test_finalize_before_prepare(self, simple_model, basic_config):
        """
        If we call finalize before prepare, we should raise an error
        """
        quantizer = Quantizer(simple_model, basic_config)

        with pytest.raises(RuntimeError, match="Model has not been prepared"):
            quantizer.finalize()

    def test_finalize_basic(self, simple_model, basic_config, example_input):
        """
        Test basic functionality of finalize API
        """
        quantizer = Quantizer(simple_model, basic_config)
        prepared_model = quantizer.prepare((example_input,))

        finalized_model = quantizer.finalize()

        assert finalized_model is prepared_model

        fake_quant_modules = [
            m for m in prepared_model.modules() if isinstance(m, FakeQuantizeImplBase)
        ]

        for module in fake_quant_modules:
            assert module.observer_enabled.item() == 0
            assert module.fake_quant_enabled.item() == 1

        output = finalized_model(example_input)
        assert output.shape == (1, 10)

    def test_finalize_after_deepcopy(self, simple_model, basic_config, example_input):
        """Finalizing a deepcopy of a prepared eager model must succeed."""
        quantizer = Quantizer(simple_model, basic_config)
        prepared_model = quantizer.prepare((example_input,))
        copied_prepared_model = copy.deepcopy(prepared_model)

        assert quantizer._is_model_prepared(copied_prepared_model)

        finalized_model = quantizer.finalize(copied_prepared_model, ExportBackend._TORCH)

        output = finalized_model(example_input)
        assert output.shape == (1, 10)

    def test_finalize_after_state_dict_roundtrip(self, simple_model, basic_config, example_input):
        """Save the prepared state_dict, reload into a fresh prepared model, then finalize.

        Calibration uses an input with a different scale than the eval input so
        the calibrated activation qparams encode information that must survive
        the state_dict round-trip — without ``load_state_dict``, the freshly
        prepared model produces a meaningfully different output.
        """
        calibration_input = example_input * 5.0

        quantizer_a = Quantizer(simple_model, basic_config)
        prepared_a = quantizer_a.prepare((example_input,))
        with quantizer_a.calibration_mode():
            prepared_a(calibration_input)
        expected_output = prepared_a(example_input)
        saved_state_dict = prepared_a.state_dict()

        quantizer_b = Quantizer(SimpleModel(), basic_config)
        prepared_b = quantizer_b.prepare((example_input,))
        prepared_b.load_state_dict(saved_state_dict)

        finalized_b = quantizer_b.finalize(prepared_b, ExportBackend._TORCH)
        actual_output = finalized_b(example_input)

        assert torch.allclose(expected_output, actual_output)

    @staticmethod
    def _quantize_model(model, quantization_config, example_input, mmap_path, backend):
        quantizer = Quantizer(model, quantization_config)
        quantizer.prepare((example_input,))

        finalized_model = quantizer.finalize(
            backend=backend,
            mmap_dir=mmap_path,
        )
        return finalized_model

    @pytest.mark.parametrize(
        "backend",
        [ExportBackend._TORCH, ExportBackend.CoreAI, ExportBackend.CoreML],
    )
    def test_finalize_weight_parametrization_state_per_backend(
        self, backend, simple_model, basic_config, example_input
    ):
        """Post-finalize parametrization state is backend-specific:

        - _TORCH: parametrization preserved with dense ``.original`` intact
        - CoreAI: parametrization replaced with dequantize module; ``.original``
          cleared to a zero-size placeholder
        - CoreML: parametrization removed entirely
        """
        finalized = self._quantize_model(simple_model, basic_config, example_input, None, backend)

        weight_modules = [m for m in finalized.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        assert weight_modules, "fixture regressed: no Conv2d/Linear to check"

        if backend is ExportBackend._TORCH:
            for m in weight_modules:
                assert is_parametrized(m, "weight")
                assert m.parametrizations["weight"].original.numel() > 0
        elif backend is ExportBackend.CoreAI:
            for m in weight_modules:
                assert is_parametrized(m, "weight")
                assert m.parametrizations["weight"].original.numel() == 0
        elif backend is ExportBackend.CoreML:
            for m in weight_modules:
                assert not is_parametrized(m, "weight")
                assert m.weight.numel() > 0

    @pytest.mark.parametrize(
        "backend",
        [ExportBackend._TORCH, ExportBackend.CoreML],
    )
    def test_finalize_mmap_value_error_unsupported_backend(self, basic_config, tmp_path, backend):
        """``mmap_dir`` raises ``ValueError`` for any backend other than CoreAI."""
        model = SimpleLinearModel()
        example_input = torch.rand(1, 10)

        with pytest.raises(
            ValueError,
            match="mmap_dir is only supported with backend=ExportBackend.CoreAI",
        ):
            self._quantize_model(model, basic_config, example_input, str(tmp_path), backend)

    @pytest.mark.parametrize("use_mmap", [False, True])
    def test_finalize_mmap_files_exist(self, use_mmap, basic_config, tmp_path):
        """Finalize with CoreAI backend writes one safetensors file per
        weight-quantized module when ``mmap_dir`` is set, and nothing otherwise."""
        model = SimpleLinearModel()
        example_input = torch.rand(1, 10)

        mmap_dir = str(tmp_path) if use_mmap else None
        self._quantize_model(model, basic_config, example_input, mmap_dir, ExportBackend.CoreAI)

        files = sorted(os.listdir(tmp_path))
        if use_mmap:
            assert files == ["linear.weight.safetensors"]
        else:
            assert files == []

    @pytest.mark.parametrize(
        "shared_weight_model",
        [True, False],
        ids=["shared_weights", "no_sharing"],
    )
    def test_finalize_mmap_matches_non_mmap_output(
        self, basic_config, tmp_path, shared_weight_model
    ):
        """Finalize with and without ``mmap_dir`` produces numerically identical
        outputs, including for models with shared (weight-tied) layers."""
        if shared_weight_model:
            model_cls = SharedWeightModel
            example_input_shape = (1, 4)
        else:
            model_cls = SimpleLinearModel
            example_input_shape = (1, 10)

        model_no_mmap = model_cls()
        model_with_mmap = copy.deepcopy(model_no_mmap)
        example_input = torch.rand(*example_input_shape)

        finalized_no_mmap = self._quantize_model(
            model_no_mmap, basic_config, example_input, None, ExportBackend.CoreAI
        )
        finalized_with_mmap = self._quantize_model(
            model_with_mmap, basic_config, example_input, str(tmp_path), ExportBackend.CoreAI
        )

        if shared_weight_model:
            # Tied-weight modules should share a single dequant parametrization
            # post-finalize, both with and without mmap.
            assert (
                finalized_no_mmap.linear1.parametrizations.weight[0]
                is finalized_no_mmap.linear2.parametrizations.weight[0]
            ), "no-mmap finalize did not preserve sharing for weight-tied modules"
            assert (
                finalized_with_mmap.linear1.parametrizations.weight[0]
                is finalized_with_mmap.linear2.parametrizations.weight[0]
            ), "mmap finalize did not preserve sharing for weight-tied modules"

        with torch.no_grad():
            out_no_mmap = finalized_no_mmap(example_input)
            out_with_mmap = finalized_with_mmap(example_input)

        assert torch.equal(out_no_mmap, out_with_mmap)

    def test_finalize_state_dict_safetensors_roundtrip(self, basic_config, tmp_path):
        """An mmap-finalized model survives a state_dict save → load_file →
        load_state_dict round-trip with identical forward outputs. The reloaded
        tensors come back mmap-backed by the bundled safetensors file (per
        safetensors' default behavior)."""
        from safetensors.torch import load_file, save_file  # noqa: PLC0415

        model = SimpleLinearModel()
        example_input = torch.rand(1, 10)

        finalized = self._quantize_model(
            model,
            basic_config,
            example_input,
            str(tmp_path / "per_layer_mmap"),
            ExportBackend.CoreAI,
        )

        with torch.no_grad():
            out_before_roundtrip = finalized(example_input)

        # Bundle the full state_dict into a single safetensors file.
        bundled = tmp_path / "full_state.safetensors"
        save_file(
            {
                k: v.contiguous()
                for k, v in finalized.state_dict().items()
                if isinstance(v, torch.Tensor)
            },
            str(bundled),
        )

        # Reload via mmap and reassign onto the existing model.
        reloaded_sd = load_file(str(bundled), device="cpu")
        finalized.load_state_dict(reloaded_sd, assign=True)

        with torch.no_grad():
            out_after_roundtrip = finalized(example_input)

        assert torch.equal(out_before_roundtrip, out_after_roundtrip)

    def test_calibration_mode(self, simple_model, input_activation_only_config, example_input):
        """
        Test that calibration mode works as expected, and scales are getting updated
        """
        quantizer = Quantizer(simple_model, input_activation_only_config)
        simple_model.eval()
        prepared_model = quantizer.prepare((example_input,))

        fake_quant_modules = [
            m for m in prepared_model.modules() if isinstance(m, FakeQuantizeImplBase)
        ]

        for module in fake_quant_modules:
            assert module.observer_enabled.item() == 0
            assert module.fake_quant_enabled.item() == 1

        pre_calibration_scales = [mod.calculate_qparams()[0].clone() for mod in fake_quant_modules]

        with quantizer.calibration_mode():
            simple_model.eval()
            with torch.no_grad():
                simple_model(torch.rand_like(example_input))

            for module in fake_quant_modules:
                assert module.observer_enabled.item() == 1
                assert module.fake_quant_enabled.item() == 0

        post_calibration_scales = [mod.calculate_qparams()[0].clone() for mod in fake_quant_modules]

        for pre_scale, post_scale in zip(
            pre_calibration_scales, post_calibration_scales, strict=True
        ):
            assert (pre_scale - post_scale).abs() > 1e-6

        for module in fake_quant_modules:
            assert module.observer_enabled.item() == 0
            assert module.fake_quant_enabled.item() == 1

    def test_calibration_mode_exception_handling(self, simple_model, basic_config, example_input):
        """
        Test that when we have an exception in calibration mode,
        the observers are disabled and fake quant is enabled
        """
        quantizer = Quantizer(simple_model, basic_config)
        prepared_model = quantizer.prepare((example_input,))

        fake_quant_modules = [
            m for m in prepared_model.modules() if isinstance(m, FakeQuantizeImplBase)
        ]

        with pytest.raises(ValueError):
            with quantizer.calibration_mode():
                raise ValueError("Test exception")

        for module in fake_quant_modules:
            assert module.observer_enabled.item() == 0
            assert module.fake_quant_enabled.item() == 1

    def test_module_name_configs(self, simple_model, example_input):
        """
        Test that we can configure the Eager quantizer with module names
        """
        config = QuantizerConfig(
            global_config=None,
            module_name_configs={
                "conv": ModuleQuantizerConfig(
                    op_state_spec={
                        "weight": default_weight_quantization_spec(),
                    },
                    op_input_spec={"*": default_activation_quantization_spec()},
                    op_output_spec=None,
                )
            },
            execution_mode="eager",
        )

        quantizer = Quantizer(simple_model, config)
        prepared_model = quantizer.prepare((example_input,))
        assert not is_parametrized(prepared_model.linear, "weight")
        assert is_parametrized(prepared_model.conv, "weight")

        fq_mods = [
            i for i in prepared_model.named_modules() if isinstance(i[1], FakeQuantizeImplBase)
        ]
        assert len(fq_mods) == 2
        assert [n for n, _ in fq_mods if n.startswith("conv") and n.endswith("quantize_input")]
        linear_fq_mods = [
            n for n, _ in fq_mods if n.startswith("linear") and n.endswith("quantize_input")
        ]
        assert len(linear_fq_mods) == 0

    def test_module_type_configs(self, simple_model, example_input):
        """
        Test that we can configure the Eager quantizer with module types
        """
        config = QuantizerConfig(
            global_config=None,
            module_type_configs={
                torch.nn.Linear: ModuleQuantizerConfig(
                    op_state_spec={
                        "weight": default_weight_quantization_spec(),
                    },
                    op_input_spec={"*": default_activation_quantization_spec()},
                    op_output_spec=None,
                )
            },
            execution_mode="eager",
        )

        quantizer = Quantizer(simple_model, config)
        prepared_model = quantizer.prepare((example_input,))
        assert is_parametrized(prepared_model.linear, "weight")
        assert not is_parametrized(prepared_model.conv, "weight")

        fq_mods = [
            i for i in prepared_model.named_modules() if isinstance(i[1], FakeQuantizeImplBase)
        ]
        assert len(fq_mods) == 2
        assert [n for n, _ in fq_mods if n.startswith("linear") and n.endswith("quantize_input")]
        with pytest.raises(AssertionError):
            assert [n for n, _ in fq_mods if n.startswith("conv") and n.endswith("quantize_input")]

    def test_weight_quant(self, simple_model, weight_only_config, example_input):
        """
        Test that weight quantization is taking place
        """
        simple_model.eval()
        conv_weight = simple_model.conv.weight
        linear_weight = simple_model.linear.weight
        quantizer = Quantizer(simple_model, weight_only_config)
        prepared_model = quantizer.prepare((example_input,))

        with quantizer.calibration_mode():
            with torch.no_grad():
                prepared_model(torch.rand_like(example_input))

        # Check that we have some quantization error
        assert (conv_weight - prepared_model.conv.weight).abs().sum() > 1e-4
        assert (linear_weight - prepared_model.linear.weight).abs().sum() > 1e-4

    def test_act_quant(self, simple_model, input_activation_only_config, example_input):
        """
        Test that activation quantization is taking place
        """
        # Store activations that will be captured by hooks
        pre_quant_activations = []
        post_quant_activations = []

        def capture_pre_quant(module, input):
            pre_quant_activations.append(input[0].detach().clone())

        # Register hooks on original model and get pre-quantization activations
        hooks = []
        hooks.append(simple_model.conv.register_forward_pre_hook(capture_pre_quant))
        hooks.append(simple_model.linear.register_forward_pre_hook(capture_pre_quant))

        simple_model.eval()
        with torch.no_grad():
            _ = simple_model(example_input)

        # Remove hooks from original model
        for hook in hooks:
            hook.remove()

        # Prepare the model for quantization
        quantizer = Quantizer(simple_model, input_activation_only_config)
        prepared_model = quantizer.prepare((example_input,))

        # The key insight: we need to hook into the quantization modules themselves
        # to capture the quantized activations. The quantization happens through
        # the fake quantize modules that are added during preparation.

        # Find the quantization modules and hook into them to capture quantized outputs
        quant_modules = []
        for name, module in prepared_model.named_modules():
            if name.endswith("quantize_input"):
                quant_modules.append(module)

        def capture_quantized_output(module, input, output):
            post_quant_activations.append(output.detach().clone())

        # Register hooks on quantization modules to capture their outputs
        quant_hooks = []
        for qmod in quant_modules:
            quant_hooks.append(qmod.register_forward_hook(capture_quantized_output))

        # Run the prepared model to get quantized activations
        with torch.no_grad():
            _ = prepared_model(example_input)

        # Remove quantization hooks
        for hook in quant_hooks:
            hook.remove()

        # Compare pre-quantization inputs with quantized outputs
        # There should be at least some quantization error
        assert len(pre_quant_activations) == len(post_quant_activations)

        total_diff = 0
        for pre_act, post_act in zip(pre_quant_activations, post_quant_activations, strict=False):
            # The shapes might be different due to quantization, so we need
            # to handle this
            if pre_act.shape == post_act.shape:
                total_diff += (pre_act - post_act).abs().sum().item()

        # Assert that there is some quantization error (activations are different)
        assert total_diff > 1e-4, f"Expected quantization error > 1e-4, got {total_diff}"

    def test_output_act_quant(self, simple_model, output_activation_only_config, example_input):
        """
        Test that output activation quantization is working
        """
        pre_quant_outputs = []
        post_quant_outputs = []

        def capture_pre_quant_output(module, input, output):
            pre_quant_outputs.append(output.detach().clone())

        # Capture outputs from original model
        hooks = []
        hooks.append(simple_model.conv.register_forward_hook(capture_pre_quant_output))
        hooks.append(simple_model.linear.register_forward_hook(capture_pre_quant_output))

        simple_model.eval()
        with torch.no_grad():
            _ = simple_model(example_input)

        for hook in hooks:
            hook.remove()

        # Prepare model with output quantization
        quantizer = Quantizer(simple_model, output_activation_only_config)
        prepared_model = quantizer.prepare((example_input,))

        def capture_post_quant_output(module, input, output):
            post_quant_outputs.append(output.detach().clone())

        # Capture outputs from quantized model
        hooks = []
        hooks.append(prepared_model.conv.register_forward_hook(capture_post_quant_output))
        hooks.append(prepared_model.linear.register_forward_hook(capture_post_quant_output))

        prepared_model.eval()
        with torch.no_grad():
            _ = prepared_model(example_input)

        for hook in hooks:
            hook.remove()

        assert len(pre_quant_outputs) == len(post_quant_outputs)

        total_diff = 0
        for pre_out, post_out in zip(pre_quant_outputs, post_quant_outputs, strict=False):
            if pre_out.shape == post_out.shape:
                total_diff += (pre_out - post_out).abs().sum().item()

        assert total_diff > 1e-4, f"Expected quantization error > 1e-4, got {total_diff}"

    def test_output_act_quant_modules(
        self, simple_model, output_activation_only_config, example_input
    ):
        """
        Test that output activation quantization modules are created correctly
        """
        quantizer = Quantizer(simple_model, output_activation_only_config)
        prepared_model = quantizer.prepare((example_input,))

        # Check that output quantization modules exist
        output_quants = [
            name for name, _ in prepared_model.named_modules() if name.endswith("quantize_output_0")
        ]
        assert len(output_quants) == 2

        # Check no input quantization or weight quantization
        input_quants = [
            name for name, _ in prepared_model.named_modules() if name.endswith("quantize_input")
        ]
        assert len(input_quants) == 0
        assert not is_parametrized(prepared_model.conv, "weight")
        assert not is_parametrized(prepared_model.linear, "weight")

        # Model should run without errors
        output = prepared_model(example_input)
        assert output.shape == (1, 10)

    def test_full_activation_quant(self, simple_model, full_activation_config, example_input):
        """
        Test both input and output activation quantization together
        """
        quantizer = Quantizer(simple_model, full_activation_config)
        prepared_model = quantizer.prepare((example_input,))

        input_quants = [
            name for name, _ in prepared_model.named_modules() if name.endswith("quantize_input")
        ]
        output_quants = [
            name for name, _ in prepared_model.named_modules() if name.endswith("quantize_output_0")
        ]

        assert len(input_quants) == 2
        assert len(output_quants) == 2

        output = prepared_model(example_input)
        assert output.shape == (1, 10)

    def test_output_only_quantization(
        self, simple_model, output_activation_only_config, example_input
    ):
        """
        Test output-only quantization without input & weight quantization
        """
        quantizer = Quantizer(simple_model, output_activation_only_config)
        prepared_model = quantizer.prepare((example_input,))

        assert not is_parametrized(prepared_model.conv, "weight")
        assert not is_parametrized(prepared_model.linear, "weight")

        input_quants = [
            name for name, _ in prepared_model.named_modules() if name.endswith("quantize_input")
        ]
        assert len(input_quants) == 0

        output_quants = [
            name for name, _ in prepared_model.named_modules() if name.endswith("quantize_output_0")
        ]
        assert len(output_quants) == 2

        fq_mods = [m for m in prepared_model.modules() if isinstance(m, FakeQuantizeImplBase)]
        assert len(fq_mods) == 2

        output = prepared_model(example_input)
        assert output.shape == (1, 10)

    def test_nested_module_quantization(self, simple_model, example_input):
        """
        Test that we can configure a parent module and have nested modules take the
        desired configuration.
        """
        # Configure whole model with int8/int8 weight/activation quantization and
        # disable conv quantization.
        # Expectation is that only Linear layer is quantized.
        config = QuantizerConfig(
            module_type_configs={
                SimpleModel: ModuleQuantizerConfig(
                    op_input_spec={"*": default_activation_quantization_spec()},
                    op_state_spec={"weight": default_weight_quantization_spec()},
                    op_output_spec=None,
                ),
                torch.nn.Conv2d: None,
            },
            execution_mode="eager",
        )

        quantizer = Quantizer(simple_model, config)
        prepared_model = quantizer.prepare((example_input,))
        assert is_parametrized(prepared_model.linear, "weight")
        assert not is_parametrized(prepared_model.conv, "weight")

        fq_mods = [
            (name, module)
            for name, module in prepared_model.named_modules()
            if isinstance(module, FakeQuantizeImplBase)
        ]
        assert len(fq_mods) == 2
        assert [
            name
            for name, _ in fq_mods
            if name.startswith("linear") and name.endswith("quantize_input")
        ]
        assert not [n for n, _ in fq_mods if n.startswith("conv") and n.endswith("quantize_input")]

    def test_custom_module(self, basic_config):
        """
        Test a custom module with some functional ops
        """

        class CustomModule(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.register_parameter(
                    "weight", nn.Parameter(torch.rand(20, 20), requires_grad=True)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = F.linear(x, self.weight)
                x = torch.add(x, torch.ones_like(x))
                return x

        module = CustomModule()
        quantizer = Quantizer(module, basic_config)
        prepared_model = quantizer.prepare((torch.rand(10, 20),))

        assert is_parametrized(prepared_model, "weight")
        fq_mods = [
            i for i in prepared_model.named_modules() if isinstance(i[1], FakeQuantizeImplBase)
        ]
        # weight, linear input, 2 inputs to add
        assert len(fq_mods) == 4

        fq_mod_names = {i[0] for i in fq_mods}
        assert fq_mod_names == {
            "linear_quantize_input",
            "add_quantize_input",
            "add_quantize_other",
            "parametrizations.weight.0",
        }

    def test_registered_optimizers_tracker(self):
        """
        Test that module tracker contents are populated as expected.
        """

        class AddModel(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                x = torch.add(x, torch.tensor([1.0, 1.0]))
                x = x + torch.tensor([2.0, 1.0])
                x += torch.tensor([3.0, 1.0])
                x += torch.tensor([4.0, 1.0])
                x = x + torch.tensor([5.0, 1.0])
                return x

        model = AddModel()
        inp = (torch.tensor([0.0, 0.0]),)

        quantizer = Quantizer(model, QuantizerConfig(execution_mode="eager"))
        _ = quantizer.prepare(inp)

        module_tracker: RegisteredOptimizersTracker = (
            quantizer._quantizer._handler.act_handler.reference_tracker
        )
        assert len(module_tracker.get_registry_dict()) == 1
        assert module_tracker.has_module("")
        assert module_tracker.has_function("", "add")
        assert module_tracker.has_function("", "add_")
        with pytest.raises(RuntimeError, match="RegisteredOptimizersTracker has no module"):
            module_tracker.get_function_call_count("dummy_module", "dummy_func")
        assert module_tracker.get_function_call_count("", "dummy_func") == 0
        assert not module_tracker.has_function("", "dummy_func")

        assert module_tracker.get_function_call_count("", "add") == 3
        assert module_tracker.get_function_call_count("", "add_") == 2

        assert module_tracker.get_module_registrations("") == {
            "add": [
                FunctionRegisteredOptimizers(
                    input_optimizer_names=["add_quantize_input", "add_quantize_other"],
                    output_optimizer_names=["add_quantize_output_0"],
                ),
                FunctionRegisteredOptimizers(
                    input_optimizer_names=["add_1_quantize_input", "add_1_quantize_other"],
                    output_optimizer_names=["add_1_quantize_output_0"],
                ),
                FunctionRegisteredOptimizers(
                    input_optimizer_names=["add_2_quantize_input", "add_2_quantize_other"],
                    output_optimizer_names=["add_2_quantize_output_0"],
                ),
            ],
            "add_": [
                FunctionRegisteredOptimizers(
                    input_optimizer_names=["add__quantize_input", "add__quantize_other"],
                    output_optimizer_names=["add__quantize_output_0"],
                ),
                FunctionRegisteredOptimizers(
                    input_optimizer_names=["add__1_quantize_input", "add__1_quantize_other"],
                    output_optimizer_names=["add__1_quantize_output_0"],
                ),
            ],
        }

    def test_module_input_spec(self):
        """Test module-level input quantization."""
        config = QuantizerConfig(
            global_config=None,
            module_type_configs={
                InnerModule: ModuleQuantizerConfig(
                    op_input_spec=None,
                    op_output_spec=None,
                    op_state_spec=None,
                    module_input_spec={
                        0: default_activation_quantization_spec(),  # Quantize first input only
                    },
                )
            },
            execution_mode="eager",
        )

        model = DoubleInnerModel()
        example_inputs = (torch.randn(2, 4), torch.randn(2, 4))
        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_inputs)

        # Check quantizers exist for module inputs
        quantize_modules = {
            name
            for name, module in prepared_model.named_modules()
            if isinstance(module, FakeQuantizeImplBase)
        }

        # Should have quantizers for:
        # - inner1's first input x is used in both linear(x) and add(_, x)
        # - inner2's first input y is used in both linear(y) and add(_, y)
        # Module-level spec applies wherever the module input is used
        assert quantize_modules == {
            "inner1.linear.linear_quantize_input",
            "inner2.linear.linear_quantize_input",
            "inner1.add_quantize_other",
            "inner2.add_quantize_other",
        }

    def test_module_input_spec_identical_input_to_add(self):
        """
        Test that a module inputs being used in multiple places results in module input spec
        settings applying to all users of the input.
        """

        class MultipleInputUserModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2, bias=False)

            def forward(self, inp):
                x = self.linear(inp)
                return x + inp

        config = QuantizerConfig(
            global_config=None,
            module_name_configs={
                "": ModuleQuantizerConfig(
                    op_input_spec=None,
                    op_output_spec=None,
                    op_state_spec=None,
                    module_input_spec={0: default_activation_quantization_spec()},
                )
            },
            execution_mode="eager",
        )
        model = MultipleInputUserModel()
        example_inputs = (torch.randn(1, 2),)
        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_inputs)

        # Check quantizers exist for module inputs
        quantize_modules = {
            name
            for name, module in prepared_model.named_modules()
            if isinstance(module, FakeQuantizeImplBase)
        }

        # Should have quantizers for both add inputs
        assert quantize_modules == {"linear.linear_quantize_input", "add_quantize_other"}

    @pytest.mark.parametrize(
        "config",
        [
            QuantizerConfig(
                global_config=None,
                module_type_configs={
                    DoubleInnerModel: ModuleQuantizerConfig(
                        op_input_spec=None,
                        op_output_spec=None,
                        op_state_spec=None,
                        module_input_spec={
                            0: QuantizationSpec(),
                        },
                    ),
                    InnerModule: ModuleQuantizerConfig(
                        op_input_spec=None,
                        op_output_spec=None,
                        op_state_spec=None,
                        module_input_spec={
                            0: QuantizationSpec(dtype=torch.int4),
                        },
                    ),
                },
                execution_mode="eager",
            ),
            QuantizerConfig(
                global_config=None,
                module_type_configs={
                    InnerModule: ModuleQuantizerConfig(
                        op_input_spec=None,
                        op_output_spec=None,
                        op_state_spec=None,
                        module_input_spec={
                            0: QuantizationSpec(dtype=torch.int4),
                        },
                    ),
                    DoubleInnerModel: ModuleQuantizerConfig(
                        op_input_spec=None,
                        op_output_spec=None,
                        op_state_spec=None,
                        module_input_spec={
                            0: QuantizationSpec(),
                        },
                    ),
                },
                execution_mode="eager",
            ),
        ],
    )
    def test_module_input_spec_precedence(self, config):
        """
        Test that module level settings follow config precedence rules.
        """
        model = DoubleInnerModel()
        example_inputs = (torch.randn(2, 4), torch.randn(2, 4))
        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_inputs)
        second_spec_dtype = list(config.module_type_configs.values())[-1].module_input_spec[0].dtype
        quantizer = prepared_model.inner1.linear.linear_quantize_input
        assert quantizer.qparams_calculator.dtype == second_spec_dtype

    @pytest.mark.parametrize(
        "config, expected",
        [
            (
                QuantizerConfig(
                    global_config=None,
                    module_name_configs={
                        "": ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec=None,
                            module_output_spec={"*": default_activation_quantization_spec()},
                        )
                    },
                    execution_mode="eager",
                ),
                {"add__quantize_output_0"},
            ),
            (
                QuantizerConfig(
                    global_config=None,
                    module_name_configs={
                        "linear": ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec=None,
                            module_output_spec={"*": default_activation_quantization_spec()},
                        )
                    },
                    execution_mode="eager",
                ),
                {"linear.linear_quantize_output_0"},
            ),
        ],
    )
    def test_module_output_spec_with_in_place_add(self, config, expected):
        """
        Test that module output settings are applied to the correct function when in-place functions
        are in play.
        """

        class InPlaceAddModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2, bias=False)

            def forward(self, inp):
                x = self.linear(inp)
                x += inp
                return x

        model = InPlaceAddModel()
        example_inputs = (torch.randn(1, 2),)
        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_inputs)

        quantize_modules = {
            name
            for name, module in prepared_model.named_modules()
            if isinstance(module, FakeQuantizeImplBase)
        }

        assert quantize_modules == expected

    @pytest.mark.parametrize(
        "config, expected_quantizers",
        [
            [
                QuantizerConfig(
                    global_config=None,
                    module_type_configs={
                        DoubleInnerModel: ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec=None,
                            module_output_spec={
                                0: QuantizationSpec(),
                            },
                        ),
                        InnerModule: ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec=None,
                            module_output_spec={
                                0: QuantizationSpec(dtype=torch.int4),
                            },
                        ),
                    },
                    execution_mode="eager",
                ),
                {
                    "inner1.add_quantize_output_0": torch.int4,
                    "inner2.add_quantize_output_0": torch.int4,
                    "add_quantize_output_0": torch.int8,
                },
            ],
            [
                QuantizerConfig(
                    global_config=None,
                    module_type_configs={
                        InnerModule: ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec=None,
                            module_output_spec={
                                0: QuantizationSpec(dtype=torch.int4),
                            },
                        ),
                        DoubleInnerModel: ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec=None,
                            module_output_spec={
                                0: QuantizationSpec(),
                            },
                        ),
                    },
                    execution_mode="eager",
                ),
                {"add_quantize_output_0": torch.int8},
            ],
        ],
    )
    def test_module_output_spec_precedence(self, config, expected_quantizers):
        """Test module-level output quantization."""
        model = DoubleInnerModel()
        example_inputs = (torch.randn(2, 4), torch.randn(2, 4))
        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_inputs)

        quantize_modules = {
            name
            for name, module in prepared_model.named_modules()
            if isinstance(module, FakeQuantizeImplBase)
        }

        assert len(quantize_modules) == len(expected_quantizers)
        for expected_quantizer, expected_dtype in expected_quantizers.items():
            assert (
                prepared_model.get_submodule(expected_quantizer).qparams_calculator.dtype
                == expected_dtype
            )

    def test_warn_on_multiple_specifications_for_same_module_input(self, caplog):
        class MyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, inp, inp2):
                return torch.add(inp, inp2)

        model = MyModel()

        config = QuantizerConfig(
            global_config=None,
            module_name_configs={
                "": ModuleQuantizerConfig(
                    module_input_spec={
                        0: default_activation_quantization_spec(),
                        1: default_activation_quantization_spec(),
                    },
                )
            },
            execution_mode="eager",
        )

        model = MyModel()
        example_input = torch.randn(1, 2)
        quantizer = Quantizer(model, config)
        quantizer.prepare(
            (example_input, example_input),
        )

        assert any(
            "Components dict contains multiple specifications" in record.message
            for record in caplog.records
        )

    def test_module_input_output_with_intermediate_tensors(self):
        """
        Test that module input and output spec settings are applied correctly to intermediate
        functions in a module which take module inputs or are module outputs.
        """

        class IntermediateInputOutputModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2, bias=False)
                self.relu = torch.nn.ReLU()

            def forward(self, inp1, inp2):
                x1 = self.linear(inp1)
                x = x1 + inp2
                x = self.relu(x)
                return x, x1

        config = QuantizerConfig(
            global_config=None,
            module_name_configs={
                "": ModuleQuantizerConfig(
                    op_input_spec=None,
                    op_output_spec=None,
                    op_state_spec=None,
                    module_input_spec={1: default_activation_quantization_spec()},
                    module_output_spec={1: default_activation_quantization_spec()},
                )
            },
            execution_mode="eager",
        )

        model = IntermediateInputOutputModel()
        example_inputs = (torch.randn(1, 2), torch.randn(1, 2))
        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_inputs)

        quantize_modules = {
            name
            for name, module in prepared_model.named_modules()
            if isinstance(module, FakeQuantizeImplBase)
        }

        # Should have quantizers for module outputs
        # The module output is the result of the add operation
        assert quantize_modules == {"linear.linear_quantize_output_0", "add_quantize_other"}

    def test_module_spec_overrides_op_spec(self):
        """Test that module-level specs override op-level specs."""
        config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                # Op-level: quantize all inputs
                op_state_spec=None,
                op_output_spec=None,
            ),
            module_name_configs={
                "": ModuleQuantizerConfig(
                    op_state_spec=None,
                    op_output_spec=None,
                    module_input_spec={
                        0: None,
                        1: default_activation_quantization_spec(),
                    },
                )
            },
            execution_mode="eager",
        )

        model = DoubleInnerModel()
        example_inputs = (torch.randn(2, 4), torch.randn(2, 4))
        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_inputs)

        quantize_modules = {
            name
            for name, module in prepared_model.named_modules()
            if isinstance(module, FakeQuantizeImplBase)
        }

        # inner1.linear's input and inner1.add's second input should be disabled
        assert quantize_modules == {
            "inner1.add_quantize_input",
            "inner2.linear.linear_quantize_input",
            "inner2.add_quantize_input",
            "inner2.add_quantize_other",
            "add_quantize_input",
            "add_quantize_other",
        }

    def test_module_state_spec_basic(self):
        """Test that module_state_spec overrides op_state_spec."""
        # Create a custom spec for module-level state
        int4_spec = QuantizationSpec(dtype=torch.int4)

        config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec={"weight": default_weight_quantization_spec()},
                op_input_spec=None,
                op_output_spec=None,
            ),
            module_type_configs={
                nn.Linear: ModuleQuantizerConfig(
                    op_state_spec=None,
                    op_input_spec=None,
                    op_output_spec=None,
                    module_state_spec={"weight": int4_spec},
                )
            },
            execution_mode="eager",
        )

        model = torch.nn.Linear(4, 4, bias=False)
        example_input = (torch.randn(2, 4),)

        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_input)

        # Check that linear layer's weight is parametrized
        assert is_parametrized(prepared_model, "weight")

        # Get the quantizer for the weight
        weight_quantizer = prepared_model.parametrizations.weight[0]

        # Verify it uses torch.int4 dtype (from module_state_spec)
        assert weight_quantizer.qparams_calculator.dtype == torch.int4

    @pytest.mark.parametrize(
        "config",
        [
            QuantizerConfig(
                global_config=None,
                module_name_configs={
                    "linear1": ModuleQuantizerConfig(
                        op_state_spec=None,
                        op_input_spec=None,
                        op_output_spec=None,
                        module_state_spec={"weight": default_weight_quantization_spec()},
                    ),
                    "linear2": ModuleQuantizerConfig(
                        op_state_spec=None,
                        op_input_spec=None,
                        op_output_spec=None,
                        module_state_spec={"weight": QuantizationSpec(dtype=torch.int4)},
                    ),
                },
                execution_mode="eager",
            ),
            QuantizerConfig(
                global_config=None,
                module_name_configs={
                    "linear2": ModuleQuantizerConfig(
                        op_state_spec=None,
                        op_input_spec=None,
                        op_output_spec=None,
                        module_state_spec={"weight": default_weight_quantization_spec()},
                    ),
                    "linear1": ModuleQuantizerConfig(
                        op_state_spec=None,
                        op_input_spec=None,
                        op_output_spec=None,
                        module_state_spec={"weight": QuantizationSpec(dtype=torch.int4)},
                    ),
                },
                execution_mode="eager",
            ),
            QuantizerConfig(
                global_config=None,
                module_type_configs={
                    nn.Linear: ModuleQuantizerConfig(
                        op_state_spec=None,
                        op_input_spec=None,
                        op_output_spec=None,
                        module_state_spec={"weight": default_weight_quantization_spec()},
                    )
                },
                module_name_configs={
                    "linear1": ModuleQuantizerConfig(
                        op_state_spec=None,
                        op_input_spec=None,
                        op_output_spec=None,
                        module_state_spec={"weight": QuantizationSpec(dtype=torch.int4)},
                    )
                },
                execution_mode="eager",
            ),
        ],
    )
    def test_module_state_spec_shared_weights_module_type_precedence(self, config):
        """
        Test that shared weights use the correct spec based on config precedence.

        When multiple modules share the same weight and have module_state_spec configured,
        the precedence rules apply: MODULE_NAME > MODULE_TYPE > GLOBAL.
        """
        model = SharedWeightModel()
        example_input = (torch.randn(2, 4),)

        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_input)

        # Both should use the same quantizer (shared state)
        assert is_parametrized(prepared_model.linear1, "weight")
        assert is_parametrized(prepared_model.linear2, "weight")

        # They should share the same parametrization
        assert (
            prepared_model.linear1.parametrizations.weight[0]
            is prepared_model.linear2.parametrizations.weight[0]
        )

        # Should use per_channel (from module_type config)
        weight_quantizer = prepared_model.linear1.parametrizations.weight[0]
        assert weight_quantizer.qparams_calculator.dtype == torch.int4

    def test_module_state_spec_no_quantization(self):
        """Test that setting module_state_spec to None disables weight quantization."""
        config = QuantizerConfig(
            global_config=None,
            module_type_configs={
                nn.Linear: ModuleQuantizerConfig(
                    op_state_spec=None,
                    op_input_spec=None,
                    op_output_spec=None,
                    module_state_spec={"weight": default_weight_quantization_spec()},
                )
            },
            module_name_configs={
                "linear1": ModuleQuantizerConfig(
                    op_state_spec=None,
                    op_input_spec=None,
                    op_output_spec=None,
                    module_state_spec={"weight": None},
                )
            },
            execution_mode="eager",
        )

        model = SharedWeightModel()
        example_input = (torch.randn(2, 4),)

        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_input)

        assert not is_parametrized(prepared_model.linear1, "weight")
        assert not is_parametrized(prepared_model.linear2, "weight")

    def test_module_state_spec_wildcard(self):
        """Test module_state_spec with wildcard '*' to quantize all states."""
        config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec=None,
                op_input_spec=None,
                op_output_spec=None,
            ),
            module_type_configs={
                nn.Linear: ModuleQuantizerConfig(
                    op_state_spec=None,
                    op_input_spec=None,
                    op_output_spec=None,
                    module_state_spec={"*": default_weight_quantization_spec()},
                )
            },
            execution_mode="eager",
        )

        model = torch.nn.Linear(4, 4, bias=True)
        example_input = (torch.randn(2, 4),)

        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_input)

        # Both weight and bias should be parametrized (wildcard matches all)
        assert is_parametrized(prepared_model, "weight")
        assert is_parametrized(prepared_model, "bias")

    @pytest.mark.parametrize(
        "config, expected_flags",
        [
            pytest.param(
                # No module-level specs anywhere: all modules should be False
                QuantizerConfig(
                    global_config=ModuleQuantizerConfig(
                        op_state_spec={"weight": default_weight_quantization_spec()},
                        op_input_spec=None,
                        op_output_spec=None,
                    ),
                    execution_mode="eager",
                ),
                {
                    "": False,
                    "inner1": False,
                    "inner2": False,
                    "inner1.linear": False,
                    "inner2.linear": False,
                },
                id="no_module_level_spec",
            ),
            pytest.param(
                # module_input_spec on InnerModule: InnerModule instances and their
                # children should be True, root should be False
                QuantizerConfig(
                    global_config=None,
                    module_type_configs={
                        InnerModule: ModuleQuantizerConfig(
                            op_state_spec=None,
                            op_input_spec=None,
                            op_output_spec=None,
                            module_input_spec={0: default_activation_quantization_spec()},
                        ),
                    },
                    execution_mode="eager",
                ),
                {
                    "": False,
                    "inner1": True,
                    "inner2": True,
                    "inner1.linear": True,
                    "inner2.linear": True,
                },
                id="module_input_spec_on_inner",
            ),
            pytest.param(
                # module_state_spec on root: all modules should inherit True
                QuantizerConfig(
                    global_config=None,
                    module_name_configs={
                        "": ModuleQuantizerConfig(
                            op_state_spec=None,
                            op_input_spec=None,
                            op_output_spec=None,
                            module_state_spec={"weight": default_weight_quantization_spec()},
                        ),
                    },
                    execution_mode="eager",
                ),
                {
                    "": True,
                    "inner1": True,
                    "inner2": True,
                    "inner1.linear": True,
                    "inner2.linear": True,
                },
                id="module_state_spec_on_root",
            ),
            pytest.param(
                # module_output_spec on one specific module by name: only that module
                # and its children should be True
                QuantizerConfig(
                    global_config=None,
                    module_name_configs={
                        "inner1": ModuleQuantizerConfig(
                            op_state_spec=None,
                            op_input_spec=None,
                            op_output_spec=None,
                            module_output_spec={0: default_activation_quantization_spec()},
                        ),
                    },
                    execution_mode="eager",
                ),
                {
                    "": False,
                    "inner1": True,
                    "inner2": False,
                    "inner1.linear": True,
                    "inner2.linear": False,
                },
                id="module_output_spec_on_inner1_only",
            ),
        ],
    )
    def test_fill_module_has_module_spec_dict(self, config, expected_flags):
        """Test that _fill_module_has_module_spec_dict correctly identifies which modules
        have module-level specs, including inheritance from parent modules."""

        model = DoubleInnerModel()
        quantizer = Quantizer(model, config)
        module_components_dict = quantizer._quantizer._handler.module_components_dict

        result = {}
        RegisterEagerOptimizationMode._fill_module_has_module_spec_dict(
            module_name="",
            module=model,
            parent_has_module_spec=False,
            module_components_dict=module_components_dict,
            module_has_module_spec_dict=result,
        )

        for module_name, expected in expected_flags.items():
            assert result[module_name] == expected, (
                f"Module '{module_name}': expected {expected}, got {result[module_name]}"
            )

    def test_e2e_workflow(self, basic_config, simple_model, example_input):
        """Test complete quantization workflow."""
        base_model = copy.deepcopy(simple_model)
        quantizer = Quantizer(simple_model, basic_config)
        example_input_1 = example_input
        example_input_2 = torch.rand(1, 1, 28, 28)
        example_input_3 = torch.rand(1, 1, 28, 28)

        # Step 1: Prepare
        prepared_model = quantizer.prepare((example_input_1,))

        # prepared model output should not match base model, since fake quant is enabled
        assert not torch.equal(prepared_model(example_input_1), base_model(example_input_1))

        # Step 2: Calibration (simulate)
        with quantizer.calibration_mode():
            prepared_out = prepared_model(example_input_2)
            original_out = base_model(example_input_2)
            # prepare model output should match base model, since fake quant is disabled
            assert torch.equal(prepared_out, original_out)

        pre_finalize_out = prepared_model(example_input_3)

        # Step 3: Finalize
        finalized_model = quantizer.finalize()
        # prepare model output before finalize should match output after finalize
        final_output = finalized_model(example_input_3)
        assert torch.equal(pre_finalize_out, final_output)

        # Test output shape
        assert final_output.shape == (1, 10)
        pass


@pytest.mark.parametrize(
    "module",
    [
        torch.nn.Linear(10, 10),
        torch.nn.Conv2d(10, 10, 5, 5),
    ],
)
@pytest.mark.parametrize("dtype", ["int4", "uint4", "int8", "uint8"])
@pytest.mark.parametrize("qscheme", ["symmetric", "symmetric_with_clipping", "asymmetric"])
@pytest.mark.parametrize(
    "granularity",
    [
        {"type": "per_tensor"},
        {"type": "per_channel", "axis": 1},
        {"type": "per_block", "axis": 1, "block_size": 2},
    ],
)
def test_eager_quantizer_all_config_options(module, dtype, qscheme, granularity):
    """
    Test eager quantizer works with different combinations of config options
    """
    granularity = copy.deepcopy(granularity)
    module = copy.deepcopy(module)
    granularity = QuantizationGranularity.maybe_build_from_dict(granularity)
    qspec_kwargs = {
        "dtype": dtype,
        "qscheme": qscheme,
        "granularity": granularity,
        "fake_quantize_cls": "default",
        "qparam_calculator_cls": "default",
        "range_calculator_cls": "minmax",
    }

    weight_qspec = QuantizationSpec(**qspec_kwargs)

    activation_qspec_kwargs = qspec_kwargs.copy()
    activation_qspec_kwargs["qparam_calculator_cls"] = "moving_average"
    activation_qspec_kwargs["granularity"] = {"type": "per_tensor"}
    activation_qspec = QuantizationSpec(**activation_qspec_kwargs)

    config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={"weight": weight_qspec},
            op_input_spec={"*": activation_qspec},
            op_output_spec=None,
        ),
        execution_mode="eager",
    )
    quantizer = Quantizer(module, config)
    model = quantizer.prepare((torch.rand(1, 10, 10, 10),))
    model(torch.rand(1, 10, 10, 10))


def test_eager_quantizer_with_registry_and_state_dict(temp_dir, basic_config):
    """
    Test that we can save and load an eager quantized model
    """

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3)
            self.linear = nn.Linear(16, 10)

        def forward(self, x):
            x = self.conv(x)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = x.flatten(1)
            x = self.linear(x)
            x = torch.add(x, torch.ones_like(x))
            return x

    model = TestModel()

    quantizer = Quantizer(model, basic_config)

    example_input = torch.rand(1, 3, 8, 8)
    prepared_model = quantizer.prepare((example_input,))

    output = prepared_model(example_input)
    assert output is not None

    state_dict = prepared_model.state_dict()
    state_dict_path = os.path.join(temp_dir, "eager_quant_model_state_dict.pt")
    torch.save(state_dict, state_dict_path)
    assert state_dict is not None
    assert len(state_dict) > 0

    new_model = TestModel()
    new_quantizer = Quantizer(new_model, basic_config)
    new_prepared_model = new_quantizer.prepare((example_input,))

    state_dict = torch.load(state_dict_path)
    new_prepared_model.load_state_dict(state_dict)

    new_output = new_prepared_model(example_input)
    torch.testing.assert_close(output, new_output)


def test_custom_param_quantization():
    """Test that custom parameter name works with weight quantization."""

    def custom_op(input, kernel):
        return F.linear(input, kernel)

    class TestModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.kernel = nn.Parameter(torch.randn(10, 5))

        def forward(self, x):
            return custom_op(x, self.kernel)

    model = TestModule()
    config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={"kernel": default_weight_quantization_spec()}
        ),
        execution_mode="eager",
    )
    quantizer = Quantizer(model, config)
    prepared_model = quantizer.prepare((torch.randn(2, 5),))

    assert is_parametrized(prepared_model, "kernel")


def test_multi_input_quantization():
    """Test specifying specific input indices to quantize."""

    class AddModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, inp1, inp2):
            return self.relu(inp2 + inp1)

    model = AddModel()
    example_input = (torch.tensor([0.3, 127.0]), torch.tensor([0.37, 10.4]))
    _ = model(*example_input)
    config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_input_spec={1: default_activation_quantization_spec()}, op_output_spec=None
        ),
        execution_mode="eager",
    )
    quantizer = Quantizer(model, config)
    prepared_model = quantizer.prepare(example_input)
    fq_modules = {
        name: module
        for name, module in prepared_model.named_modules()
        if isinstance(module, FakeQuantizeImplBase)
    }
    assert len(fq_modules) == 1
    out = prepared_model(*example_input)
    # Add's second input is the tensor with 127.0 as the max value, so the scale should
    # reflect that. With full int8 range [-128, 127], the denominator is computed as
    # (127 + 128) / 2 = 127.5, so scale = 127 / 127.5 ≈ 0.9961
    assert torch.isclose(
        fq_modules["add_quantize_other"].qparams_calculator.scale,
        torch.tensor(127.0 / 127.5),
        rtol=1e-3,
    )
    # Only add's second input should be quantized to [0.0, 127.0]. The first
    # input should remain unchanged. Output changes slightly due to quant scale.
    assert torch.allclose(out, torch.tensor([0.37, 137.4]), rtol=1e-2)


@pytest.mark.parametrize(
    "config, warning_msg",
    [
        (
            QuantizerConfig(
                global_config=ModuleQuantizerConfig(
                    op_input_spec=None,
                    op_state_spec={"int_weight": default_activation_quantization_spec()},
                    op_output_spec=None,
                ),
                execution_mode="eager",
            ),
            "Config is attempting to set state",
        ),
        (
            QuantizerConfig(
                global_config=ModuleQuantizerConfig(),
                execution_mode="eager",
            ),
            "",
        ),
        (
            QuantizerConfig(
                global_config=ModuleQuantizerConfig(
                    op_state_spec={"*": default_activation_quantization_spec()}
                ),
                execution_mode="eager",
            ),
            "",
        ),
    ],
)
def test_warn_on_nonquantizable_tensor(caplog, config, warning_msg):
    class ModelWithInts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.int_weight = torch.nn.Parameter(torch.randint(0, 10, (1, 2)), requires_grad=False)

        def forward(self, inp):
            return inp + self.int_weight

    model = ModelWithInts()
    example_input = (torch.tensor([[1.0, 2.0]]),)
    _ = model(*example_input)

    quantizer = Quantizer(model, config)
    _ = quantizer.prepare(example_input)
    if "*" in config.global_config.op_input_spec or "*" in config.global_config.op_state_spec:
        assert len(caplog.records) == 0
    else:
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "WARNING"
        assert warning_msg in caplog.records[0].message


@pytest.mark.parametrize(
    "config, expected_error_msg",
    [
        pytest.param(
            QuantizerConfig(
                global_config=ModuleQuantizerConfig(
                    op_input_spec={"some_name": default_activation_quantization_spec()}
                ),
                execution_mode="eager",
            ),
            "Only integer indices or '*'",
            id="op_input_check-global_config",
        ),
        pytest.param(
            QuantizerConfig(
                module_type_configs={
                    "some_module.type": ModuleQuantizerConfig(
                        op_input_spec={"some_name": default_activation_quantization_spec()}
                    ),
                },
                execution_mode="eager",
            ),
            "Only integer indices or '*'",
            id="op_input_check-module_type_configs",
        ),
        pytest.param(
            QuantizerConfig(
                module_type_configs={
                    "some_module.type": ModuleQuantizerConfig(
                        op_input_spec={"some_name": default_activation_quantization_spec()}
                    )
                },
                execution_mode="eager",
            ),
            "Only integer indices or '*'",
            id="op_input_check-module_type_configs-2",
        ),
        pytest.param(
            QuantizerConfig(
                module_name_configs={
                    "some_name": ModuleQuantizerConfig(
                        op_input_spec={"some_name": default_activation_quantization_spec()}
                    )
                },
                execution_mode="eager",
            ),
            "Only integer indices or '*'",
            id="op_input_check-module_name_configs",
        ),
    ],
)
def test_invalid_quantization_configs(config, expected_error_msg):
    """
    Test that validate config succeeds in raising NotImplementedError for currently
    unsupported configs.
    """
    model = SimpleModel()
    with pytest.raises(NotImplementedError, match=expected_error_msg):
        _ = Quantizer(model, config)


def test_preserved_attributes_warning_in_eager_mode(simple_model, example_input):
    """Test that a UserWarning is raised when preserved_attributes is set."""
    config = QuantizerConfig(
        preserved_attributes=["some_attr"],
        execution_mode="eager",
    )

    with pytest.warns(UserWarning, match="'preserved_attributes' is only supported in graph mode"):
        quantizer = Quantizer(simple_model, config)

        # Quantization should go through ignoring preserved_attributes
        assert quantizer is not None
        prepared_model = quantizer.prepare((example_input,))
        assert prepared_model is not None


def test_interleaved_modules_and_functions():
    """
    Test that a module calling into a nested module with functional calls before and
    after have all functionals registered properly.
    """

    class InnerModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2, bias=False)

        def forward(self, inp, inp2):
            x = self.linear(inp)
            x = torch.add(inp, inp2)
            return x

    class OuterModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2, bias=False)
            self.inner_model = InnerModel()

        def forward(self, inp, inp2):
            x = torch.add(inp, inp2)
            x = self.linear(inp)
            x = self.inner_model(x, x)
            x = x + inp2
            return x

    model = OuterModel()
    example_input = (torch.randn(1, 2), torch.randn(1, 2))

    _ = model(*example_input)
    config = QuantizerConfig(execution_mode="eager")
    quantizer = Quantizer(model, config)
    prepared_model = quantizer.prepare(example_input)

    tracker: RegisteredOptimizersTracker = (
        quantizer._quantizer._handler.act_handler.reference_tracker
    )

    # Check root module ("") traversal
    assert tracker.get_module_registrations("")["add"] == [
        FunctionRegisteredOptimizers(
            input_optimizer_names=["add_quantize_input", "add_quantize_other"],
            output_optimizer_names=["add_quantize_output_0"],
        ),
        FunctionRegisteredOptimizers(
            input_optimizer_names=["add_1_quantize_input", "add_1_quantize_other"],
            output_optimizer_names=["add_1_quantize_output_0"],
        ),
    ]
    assert hasattr(prepared_model, "add_quantize_input")
    assert hasattr(prepared_model, "add_quantize_other")
    assert hasattr(prepared_model, "add_quantize_output_0")
    assert hasattr(prepared_model, "add_1_quantize_input")
    assert hasattr(prepared_model, "add_1_quantize_other")
    assert hasattr(prepared_model, "add_1_quantize_output_0")

    # Check linear module traversal
    assert tracker.get_module_registrations("linear")["linear"] == [
        FunctionRegisteredOptimizers(
            input_optimizer_names=["linear_quantize_input"],
            output_optimizer_names=["linear_quantize_output_0"],
        )
    ]
    assert hasattr(prepared_model.linear, "linear_quantize_input")
    assert hasattr(prepared_model.linear, "linear_quantize_output_0")

    # Check inner_model traversal
    assert tracker.get_module_registrations("inner_model")["add"] == [
        FunctionRegisteredOptimizers(
            input_optimizer_names=["add_quantize_input", "add_quantize_other"],
            output_optimizer_names=["add_quantize_output_0"],
        )
    ]
    assert hasattr(prepared_model.inner_model, "add_quantize_input")
    assert hasattr(prepared_model.inner_model, "add_quantize_other")
    assert hasattr(prepared_model.inner_model, "add_quantize_output_0")

    # Check inner_model.linear first traversal
    assert tracker.get_module_registrations("inner_model.linear")["linear"] == [
        FunctionRegisteredOptimizers(
            input_optimizer_names=["linear_quantize_input"],
            output_optimizer_names=["linear_quantize_output_0"],
        )
    ]
    assert hasattr(prepared_model.inner_model.linear, "linear_quantize_input")
    assert hasattr(prepared_model.inner_model.linear, "linear_quantize_output_0")


@pytest.mark.parametrize(
    "config",
    [
        QuantizerConfig(
            global_config=None,
            module_name_configs={
                "l1": ModuleQuantizerConfig(
                    op_input_spec={
                        "*": QuantizationSpec(
                            dtype=torch.int2,
                            qscheme=QuantizationScheme.SYMMETRIC,
                            granularity=PerTensorGranularity(),
                            fake_quantize_cls="default",
                            qparam_calculator_cls="default",
                            range_calculator_cls="minmax",
                        )
                    },
                    op_state_spec={
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
                ),
                "l": ModuleQuantizerConfig(
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
                    op_state_spec={
                        "*": QuantizationSpec(
                            dtype=torch.uint8,
                            qscheme=QuantizationScheme.SYMMETRIC,
                            granularity=PerTensorGranularity(),
                            fake_quantize_cls="default",
                            qparam_calculator_cls="default",
                            range_calculator_cls="minmax",
                        )
                    },
                    op_output_spec=None,
                ),
            },
            execution_mode="eager",
        ),
        pytest.param(
            QuantizerConfig(
                global_config=None,
                module_name_configs={
                    "l": ModuleQuantizerConfig(
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
                        op_state_spec={
                            "*": QuantizationSpec(
                                dtype=torch.uint8,
                                qscheme=QuantizationScheme.SYMMETRIC,
                                granularity=PerTensorGranularity(),
                                fake_quantize_cls="default",
                                qparam_calculator_cls="default",
                                range_calculator_cls="minmax",
                            )
                        },
                        op_output_spec=None,
                    ),
                    "l1": ModuleQuantizerConfig(
                        op_input_spec={
                            "*": QuantizationSpec(
                                dtype=torch.int2,
                                qscheme=QuantizationScheme.SYMMETRIC,
                                granularity=PerTensorGranularity(),
                                fake_quantize_cls="default",
                                qparam_calculator_cls="default",
                                range_calculator_cls="minmax",
                            )
                        },
                        op_state_spec={
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
                    ),
                },
                execution_mode="eager",
            ),
        ),
    ],
)
def test_reused_modules(config):
    """
    Test that when reused modules are involved, we see expected behavior in whether or
    not activation quantizers are shared along with correct config application.
    """

    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.p1 = nn.Linear(20, 10)
            self.p2 = nn.Linear(20, 10)
            self.l = nn.Linear(10, 10)
            self.l1 = self.l

        def forward(self, x):
            x1 = self.l(self.p1(x))
            x2 = self.l1(self.p2(x))
            return torch.add(x1, x2)

    model = MyModule()
    model.eval()

    quantizer = Quantizer(model, config)
    prepared_model = quantizer.prepare((torch.rand(1, 20),))

    assert prepared_model.l == prepared_model.l1
    second_config_entry = list(config.module_name_configs.values())[1]
    assert (
        prepared_model.l.parametrizations["weight"][0].qparams_calculator.dtype
        == second_config_entry.op_state_spec["*"].dtype
    )
    assert (
        prepared_model.l.linear_quantize_input.qparams_calculator.dtype
        == second_config_entry.op_input_spec["*"].dtype
    )


@pytest.mark.parametrize(
    "config",
    [
        QuantizerConfig(
            global_config=None,
            module_name_configs={
                "inner_model.add": ModuleQuantizerConfig(
                    op_input_spec=None,
                    op_output_spec=None,
                    op_state_spec={"*": default_activation_quantization_spec()},
                )
            },
            execution_mode="eager",
        ),
        QuantizerConfig(
            global_config=None,
            module_name_configs={
                "inner_model.add": ModuleQuantizerConfig(
                    op_input_spec=None,
                    op_output_spec=None,
                    op_state_spec={"my_param": default_activation_quantization_spec()},
                )
            },
            execution_mode="eager",
        ),
        QuantizerConfig(
            global_config=None,
            module_name_configs={
                "inner_model.add": ModuleQuantizerConfig(
                    op_input_spec=None,
                    op_output_spec=None,
                    op_state_spec={"param1": default_activation_quantization_spec()},
                )
            },
            execution_mode="eager",
        ),
    ],
)
def test_shared_param_model(config):
    class Add(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, inp1, inp2):
            return inp1 + inp2

    class InnerModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param1 = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
            self.add = Add()

        def forward(self, inp):
            return self.add(self.param1, inp)

    class OuterModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.my_param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
            self.inner_model = InnerModel()

        def forward(self, inp):
            return self.inner_model(inp)

    model = OuterModel()
    model.inner_model.param1 = model.my_param
    inp = torch.randn(1, 2)

    quantizer = Quantizer(model, config)
    prepared_model = quantizer.prepare((inp,))

    assert (
        prepared_model.parametrizations["my_param"][0]
        is prepared_model.inner_model.parametrizations["param1"][0]
    )


def test_op_name_quantization():
    class AddModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, inp1, inp2, inp3):
            x = inp1 + inp2  # op type: add
            x += inp3  # op type: add_
            return x

    config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_input_spec=None,
            op_output_spec=None,
            op_name_config={
                "add_": OpQuantizerConfig(
                    op_input_spec={1: default_activation_quantization_spec()}, op_output_spec=None
                ),
                "add": OpQuantizerConfig(
                    op_input_spec={0: default_activation_quantization_spec()}, op_output_spec=None
                ),
            },
        ),
        execution_mode="eager",
    )
    # Expectation:
    # - First add's first input quantized (set by op_name add config)
    # - Second add's second input quantized (set by op_name add_ config)

    model = AddModel()
    example_inputs = (torch.randn(1, 2), torch.randn(1, 2), torch.randn(1, 2))
    quantizer = Quantizer(model, config)
    prepared_model = quantizer.prepare(example_inputs)

    quantize_modules = {
        name
        for name, module in prepared_model.named_modules()
        if isinstance(module, FakeQuantizeImplBase)
    }

    assert quantize_modules == {
        "add_quantize_input",
        "add__quantize_other",
    }


def test_op_type_op_name_quantization():
    """
    Test that op_type and op_name configs take effect.
    """

    class InnerAdd(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, inp1, inp2):
            return inp1 + inp2

    class AddModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner_add = InnerAdd()

        def forward(self, inp1, inp2, inp3):
            x = inp1 + inp2
            x = self.inner_add(inp3, x)
            return x

    config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_input_spec=None,
            op_output_spec=None,
            op_type_config={"add": OpQuantizerConfig(op_output_spec=None)},
        ),
        module_name_configs={
            "inner_add": ModuleQuantizerConfig(
                op_input_spec=None,
                op_output_spec=None,
                op_name_config={
                    # Both add names below match "add". Test that the latter
                    # config is the one that is applied.
                    ".*add.*": OpQuantizerConfig(op_input_spec=None, op_output_spec=None),
                    ".*add": OpQuantizerConfig(
                        op_input_spec={1: default_activation_quantization_spec()},
                    ),
                },
            )
        },
        execution_mode="eager",
    )
    # Expectation:
    # - Both of first add's inputs quantized (set by global op_type)
    # - Only second add's second input quantized (set by module_name "inner_add")
    # - Second add's output quantized (set by module_name "inner_add")

    model = AddModel()
    example_inputs = (torch.randn(1, 2), torch.randn(1, 2), torch.randn(1, 2))
    quantizer = Quantizer(model, config)
    prepared_model = quantizer.prepare(example_inputs)

    quantize_modules = {
        name
        for name, module in prepared_model.named_modules()
        if isinstance(module, FakeQuantizeImplBase)
    }

    assert quantize_modules == {
        "add_quantize_input",
        "add_quantize_other",
        "inner_add.add_quantize_other",
        "inner_add.add_quantize_output_0",
    }


def test_op_name_op_type_precedence():
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
            op_type_config={
                "add": OpQuantizerConfig(
                    op_input_spec={0: default_activation_quantization_spec()}, op_output_spec=None
                )
            },
            op_name_config={
                "add_1": OpQuantizerConfig(
                    op_input_spec={1: default_activation_quantization_spec()}, op_output_spec=None
                )
            },
        ),
        execution_mode="eager",
    )
    # Expectation:
    # - First add's first input quantized (set by op_type config)
    # - Second add's second input quantized (set by op_name config)

    model = AddModel()
    example_inputs = (torch.randn(1, 2), torch.randn(1, 2), torch.randn(1, 2))
    quantizer = Quantizer(model, config)
    prepared_model = quantizer.prepare(example_inputs)

    quantize_modules = {
        name
        for name, module in prepared_model.named_modules()
        if isinstance(module, FakeQuantizeImplBase)
    }

    assert quantize_modules == {
        "add_quantize_input",
        "add_1_quantize_other",
    }


def test_op_name_disambiguates_nested_ops():
    """
    Test that op_name_config can distinguish between identically-named ops in
    different submodules using the fully-qualified op name (e.g.,
    "submodule_a.add" vs "submodule_b.add").
    """

    class InnerModule(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    class OuterModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.submodule_a = InnerModule()
            self.submodule_b = InnerModule()

        def forward(self, x, y, z):
            a = self.submodule_a(x, y)
            b = self.submodule_b(a, z)
            return b

    config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_input_spec=None,
            op_output_spec=None,
            op_name_config={
                # Only quantize the add inside submodule_a
                "submodule_a.add": OpQuantizerConfig(
                    op_input_spec={0: default_activation_quantization_spec()},
                    op_output_spec=None,
                ),
                # Only quantize the add inside submodule_b (different spec)
                "submodule_b.add": OpQuantizerConfig(
                    op_input_spec=None,
                    op_output_spec={0: default_activation_quantization_spec()},
                ),
            },
        ),
        execution_mode="eager",
    )

    model = OuterModel()
    example_inputs = (torch.randn(1, 2), torch.randn(1, 2), torch.randn(1, 2))
    quantizer = Quantizer(model, config)
    prepared_model = quantizer.prepare(example_inputs)

    quantize_modules = {
        name
        for name, module in prepared_model.named_modules()
        if isinstance(module, FakeQuantizeImplBase)
    }

    # submodule_a.add: only first input quantized
    # submodule_b.add: only output quantized
    assert quantize_modules == {
        "submodule_a.add_quantize_input",
        "submodule_b.add_quantize_output_0",
    }


def test_op_name_disambiguates_ops_in_sequential():
    """
    Test that op_name_config can target ops inside nn.Sequential submodules
    using the fully-qualified path (e.g., "seq.0.add" vs "seq.1.add").
    """

    class AddModule(torch.nn.Module):
        def forward(self, x):
            return x + x

    class SeqModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = nn.Sequential(AddModule(), AddModule())

        def forward(self, x):
            return self.seq(x)

    config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_input_spec=None,
            op_output_spec=None,
            op_name_config={
                # Only quantize the add inside seq.0
                "seq.0.add": OpQuantizerConfig(
                    op_input_spec={0: default_activation_quantization_spec()},
                    op_output_spec=None,
                ),
                # Only quantize the add inside seq.1
                "seq.1.add": OpQuantizerConfig(
                    op_input_spec=None,
                    op_output_spec={0: default_activation_quantization_spec()},
                ),
            },
        ),
        execution_mode="eager",
    )

    model = SeqModel()
    example_inputs = (torch.randn(1, 2),)
    quantizer = Quantizer(model, config)
    prepared_model = quantizer.prepare(example_inputs)

    quantize_modules = {
        name
        for name, module in prepared_model.named_modules()
        if isinstance(module, FakeQuantizeImplBase)
    }

    # seq.0.add: only first input quantized
    # seq.1.add: only output quantized
    assert quantize_modules == {
        "seq.0.add_quantize_input",
        "seq.1.add_quantize_output_0",
    }


def test_op_name_local_counter_per_submodule():
    """
    Test that the local op counter resets per submodule. If submodule_a has two
    adds they become submodule_a.add and submodule_a.add_1; if submodule_b
    also has two adds they become submodule_b.add and submodule_b.add_1
    (not submodule_b.add_2/add_3).

    Each of the four adds is configured with a distinct dtype via op_name_config
    to verify that the correct spec is associated with the correct op.
    """

    class TwoAddModule(torch.nn.Module):
        def forward(self, x):
            a = x + x
            b = a + a
            return b

    class TwoSubmoduleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.submodule_a = TwoAddModule()
            self.submodule_b = TwoAddModule()

        def forward(self, x):
            x = self.submodule_a(x)
            x = self.submodule_b(x)
            return x

    # Use four distinct dtypes so each add gets a unique, verifiable spec.
    dtype_a0 = torch.int8
    dtype_a1 = torch.uint8
    dtype_b0 = torch.int4
    dtype_b1 = torch.uint4

    config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_input_spec=None,
            op_output_spec=None,
            op_name_config={
                "submodule_a.add": OpQuantizerConfig(
                    op_input_spec=None,
                    op_output_spec={0: QuantizationSpec(dtype=dtype_a0)},
                    op_state_spec=None,
                ),
                "submodule_a.add_1": OpQuantizerConfig(
                    op_input_spec=None,
                    op_output_spec={0: QuantizationSpec(dtype=dtype_a1)},
                    op_state_spec=None,
                ),
                "submodule_b.add": OpQuantizerConfig(
                    op_input_spec=None,
                    op_output_spec={0: QuantizationSpec(dtype=dtype_b0)},
                    op_state_spec=None,
                ),
                "submodule_b.add_1": OpQuantizerConfig(
                    op_input_spec=None,
                    op_output_spec={0: QuantizationSpec(dtype=dtype_b1)},
                    op_state_spec=None,
                ),
            },
        ),
        execution_mode="eager",
    )

    model = TwoSubmoduleModel()
    example_inputs = (torch.randn(1, 4),)
    quantizer = Quantizer(model, config)
    prepared_model = quantizer.prepare(example_inputs)

    # Collect all fake-quantize modules by name
    fq_modules = {
        name: module
        for name, module in prepared_model.named_modules()
        if isinstance(module, FakeQuantizeImplBase)
    }

    # Verify exactly four output quantizers exist with the expected names
    assert set(fq_modules.keys()) == {
        "submodule_a.add_quantize_output_0",
        "submodule_a.add_1_quantize_output_0",
        "submodule_b.add_quantize_output_0",
        "submodule_b.add_1_quantize_output_0",
    }

    # Verify each quantizer got the correct dtype from its op_name_config entry
    assert fq_modules["submodule_a.add_quantize_output_0"].dtype == dtype_a0
    assert fq_modules["submodule_a.add_1_quantize_output_0"].dtype == dtype_a1
    assert fq_modules["submodule_b.add_quantize_output_0"].dtype == dtype_b0
    assert fq_modules["submodule_b.add_1_quantize_output_0"].dtype == dtype_b1


def test_bn_train_eval_mode():
    """
    Test preparing a model with BN in train mode and evaluating in eval"""

    class ConvBNReluModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 2, 1, padding=1)
            self.bn = nn.BatchNorm2d(2)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(3, 3, 1, padding=1)
            self.bn2 = nn.BatchNorm2d(3)
            self.relu2 = nn.ReLU()
            self.flatten = torch.nn.Flatten()
            self.linear = torch.nn.Linear(300, 3, bias=False)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            return x

    model = ConvBNReluModel().train()
    example_input = (torch.randn(2, 1, 8, 8),)

    quantizer = Quantizer(model, QuantizerConfig(execution_mode="eager"))
    prepared_model = quantizer.prepare(example_input)

    prepared_model.eval()
    _ = prepared_model(*example_input)

    tracker: RegisteredOptimizersTracker = (
        quantizer._quantizer._handler.act_handler.reference_tracker
    )
    assert "add_" not in tracker.get_module_registrations("bn")


@pytest.mark.parametrize("execution_mode", ["eager", "graph"])
def test_bn_stats_unchanged_after_prepare_and_calibration(execution_mode):
    """Test that BatchNorm running stats are not modified during prepare() and
    calibration_mode(), but are modified during training_mode().
    """

    class ConvBNModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 4, 3, padding=1)
            self.bn = nn.BatchNorm2d(4)
            self.relu = nn.ReLU()
            self.flatten = nn.Flatten()
            self.linear = nn.Linear(4 * 8 * 8, 10)

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.flatten(x)
            x = self.linear(x)
            return x

    model = ConvBNModel().train()
    example_input = (torch.randn(2, 1, 8, 8),)

    # Run a forward pass in train mode to establish meaningful BN running stats
    model(torch.randn(2, 1, 8, 8))

    # Snapshot BN running stats before prepare
    bn_running_mean_before = model.bn.running_mean.clone()
    bn_running_var_before = model.bn.running_var.clone()

    quantizer = Quantizer(model, QuantizerConfig(execution_mode=execution_mode))
    prepared_model = quantizer.prepare(example_input)

    # BN running stats should be unchanged after prepare
    assert torch.equal(prepared_model.bn.running_mean, bn_running_mean_before), (
        "BN running_mean was modified during prepare()"
    )
    assert torch.equal(prepared_model.bn.running_var, bn_running_var_before), (
        "BN running_var was modified during prepare()"
    )

    # Snapshot again before calibration
    bn_running_mean_before_calib = prepared_model.bn.running_mean.clone()
    bn_running_var_before_calib = prepared_model.bn.running_var.clone()

    # Run calibration
    with quantizer.calibration_mode():
        prepared_model(torch.randn(2, 1, 8, 8))

    # BN running stats should be unchanged after calibration
    assert torch.equal(prepared_model.bn.running_mean, bn_running_mean_before_calib), (
        "BN running_mean was modified during calibration_mode()"
    )
    assert torch.equal(prepared_model.bn.running_var, bn_running_var_before_calib), (
        "BN running_var was modified during calibration_mode()"
    )

    # Snapshot before training mode
    bn_running_mean_before_train = prepared_model.bn.running_mean.clone()
    bn_running_var_before_train = prepared_model.bn.running_var.clone()

    # Run training mode - BN stats SHOULD change here
    with quantizer.training_mode():
        prepared_model(torch.randn(2, 1, 8, 8))

    # BN running stats should be updated during training mode
    assert not torch.equal(prepared_model.bn.running_mean, bn_running_mean_before_train), (
        "BN running_mean was NOT modified during training_mode()"
    )
    assert not torch.equal(prepared_model.bn.running_var, bn_running_var_before_train), (
        "BN running_var was NOT modified during training_mode()"
    )


def test_exit_torch_function_mode_on_error():
    class MyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, inp, inp2, raise_error=False):
            if raise_error:
                raise RuntimeError("Intentional error")
            x = inp + inp2
            return x

    model = MyModel()
    example_input = (torch.randn(1, 2), torch.randn(1, 2))
    quantizer = Quantizer(model, QuantizerConfig(execution_mode="eager"))
    prepared_model = quantizer.prepare(example_input)
    with pytest.raises(RuntimeError, match="Intentional error"):
        _ = prepared_model(*example_input, True)
    assert _get_current_function_mode() is None


@pytest.mark.parametrize(
    "config, expected_quantizers",
    [
        [
            QuantizerConfig(execution_mode="eager"),
            set(),
        ],
        [
            QuantizerConfig(
                global_config=None,
                module_name_configs={
                    "": ModuleQuantizerConfig(
                        op_input_spec=None,
                        op_output_spec=None,
                        op_state_spec=None,
                        module_input_spec={"*": default_activation_quantization_spec()},
                        module_state_spec={"weight": default_weight_quantization_spec()},
                    )
                },
                execution_mode="eager",
            ),
            {"conv2d_quantize_input", "parametrizations.weight.0"},
        ],
    ],
)
def test_module_vs_op_quantization_for_unsupported_function(
    config, expected_quantizers, conv2d_unregistered
):
    """
    Test that module level spec settings allow quantization for functions that are not in the
    supported ops registry.
    """
    assert EagerQuantizerSupportedOpsRegistry.get_class("conv2d").ops == []
    model = torch.nn.Conv2d(2, 2, (1, 1), bias=False)
    example_input = (torch.randn(1, 2, 8, 8),)
    quantizer = Quantizer(model, config)
    prepared_model = quantizer.prepare(example_input)
    quantizers = {
        name
        for name, module in prepared_model.named_modules()
        if isinstance(module, FakeQuantizeImplBase)
    }
    assert quantizers == expected_quantizers


class TestEagerHalfPrecisionSupport:
    """Test Eager quantization workflow with half precision models"""

    # TODO: if the model_dtype list has torch.floa16 followed by torch.bfloat16
    # or vice-versa, the prepare step fails with
    # "RuntimeError: Quantize only works on Float Tensor, got Half"
    # for some reason, there is _MILActivationQuantizeModule inside the model
    # I am suspecting there is some fixture corruption happening
    @pytest.mark.parametrize(
        "model_dtype",
        [
            torch.bfloat16,
            torch.float32,
            torch.float16,
        ],
    )
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
        simple_model,
        full_activation_config,
        example_input,
        model_dtype,
        backend,
    ):
        """Test complete Eager workflow with half precision models"""
        model = simple_model.to(dtype=model_dtype)
        input_tensor = example_input.to(dtype=model_dtype)

        quantizer = Quantizer(model, full_activation_config)
        prepared_model = quantizer.prepare((input_tensor,))

        with quantizer.calibration_mode():
            for _ in range(3):
                calib_input = torch.rand_like(input_tensor) * 5.0
                _ = prepared_model(calib_input)

        finalized_model = quantizer.finalize(backend=backend)
        if model_dtype == torch.float32 or backend in [
            ExportBackend._TORCH,
            ExportBackend.CoreAI,
        ]:
            output = finalized_model(input_tensor)
            assert output.shape == (1, 10)
            assert torch.all(torch.isfinite(output)), f"Backend {backend}: output contains inf/nan"
        else:
            # CoreML backend uses torch.quantize_per_tensor C++ bindings
            # which only work on Float tensor
            with pytest.raises(RuntimeError, match="Quantize only works on Float Tensor, got"):
                _ = finalized_model(input_tensor)

        # Verify scale dtypes are correct after finalize for MLIR backend
        if backend == ExportBackend.CoreAI:
            for key, value in finalized_model.state_dict().items():
                if "scale" in key:
                    assert value.dtype == model_dtype, (
                        f"After finalize with {backend}, key {key}: "
                        f"expected dtype {model_dtype}, got {value.dtype}"
                    )


# =========================================================================
# Model classes for testing QuantizerSupportedOpsRegistry ops
# =========================================================================


class SimpleConvModel(nn.Module):
    """Simple model with Conv operation."""

    def __init__(self, dim=1):
        super().__init__()
        if dim == 1:
            self.conv = nn.Conv1d(3, 16, 3, padding=1)
        elif dim == 2:
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
        elif dim == 3:
            self.conv = nn.Conv3d(3, 16, 3, padding=1)
        else:
            raise ValueError(f"Unsupported dimension {dim}")

    def forward(self, x):
        return self.conv(x)


class SimpleConvTransposeModel(nn.Module):
    """Simple model with ConvTranspose operation."""

    def __init__(self, dim=1):
        super().__init__()
        if dim == 1:
            self.conv = nn.ConvTranspose1d(3, 16, 3, padding=1)
        elif dim == 2:
            self.conv = nn.ConvTranspose2d(3, 16, 3, padding=1)
        elif dim == 3:
            self.conv = nn.ConvTranspose3d(3, 16, 3, padding=1)
        else:
            raise ValueError(f"Unsupported dimension {dim}")

    def forward(self, x):
        return self.conv(x)


class SimpleEmbeddingModel(nn.Module):
    """Simple model with Embedding operation."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 32)

    def forward(self, x):
        return self.embedding(x)


class SimpleBinaryModel(nn.Module):
    """Simple model with binary callable."""

    def __init__(self, binary_fn):
        super().__init__()
        self.binary_fn = binary_fn
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        y = self.linear(x)
        if self.binary_fn == torch.matmul:
            return self.binary_fn(x.T, y)
        return self.binary_fn(x, y)


class SimpleAddModel(nn.Module):
    """Simple model with add operation."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        y = self.linear(x)
        return x + y


class SimpleMatMulModel(nn.Module):
    """Simple model with matmul operation using @ operator."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        y = self.linear(x)
        return x.T @ y


class SimpleMulModel(nn.Module):
    """Simple model with mul operation."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        y = self.linear(x)
        return x * y


class SimpleSubModel(nn.Module):
    """Simple model with sub operation."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        y = self.linear(x)
        return x - y


# =========================================================================
# Test class for EagerQuantizerSupportedOpsRegistry quantization patterns
# =========================================================================


class TestEagerQuantizationPatterns:
    """
    Test suite for verifying models with ops in EagerQuantizerSupportedOpsRegistry
    are quantized as expected.
    """

    @pytest.fixture
    def weight_spec(self):
        """Default weight quantization spec."""
        return default_weight_quantization_spec()

    @pytest.fixture
    def activation_spec(self):
        """Default activation quantization spec."""
        return default_activation_quantization_spec()

    def _count_weight_quantizers(self, model):
        """Count the number of weight quantizers (parametrized weights)."""
        count = 0
        for module in model.modules():
            if is_parametrized(module, "weight"):
                if isinstance(module.parametrizations["weight"][0], FakeQuantizeImplBase):
                    count += 1
        return count

    def _count_input_act_quantizers(self, model):
        """Count the number of input activation quantizers."""
        count = 0
        for name, module in model.named_modules():
            if not isinstance(module, FakeQuantizeImplBase):
                continue
            # Check for input quantizers: quantize_input or quantize_self/quantize_other
            if name.endswith("quantize_input"):
                count += 1
            elif name.endswith("quantize_self") or name.endswith("quantize_other"):
                count += 1
        return count

    def _count_output_act_quantizers(self, model):
        """Count the number of output activation quantizers."""
        count = 0
        for name, module in model.named_modules():
            if not isinstance(module, FakeQuantizeImplBase):
                continue
            # Check for output quantizers: *quantize_output* pattern in the last segment
            last_segment = name.split(".")[-1]
            if "quantize_output" in last_segment:
                count += 1
        return count

    def _build_config(self, quantization_type, weight_spec, activation_spec):
        """Build QuantizerConfig based on quantization type."""
        if quantization_type == "weight_only":
            return QuantizerConfig(
                global_config=ModuleQuantizerConfig(
                    op_state_spec={
                        "weight": weight_spec,
                    },
                    op_input_spec=None,
                    op_output_spec=None,
                ),
                execution_mode="eager",
            )
        elif quantization_type == "input_activation_only":
            return QuantizerConfig(
                global_config=ModuleQuantizerConfig(
                    op_state_spec=None,
                    op_input_spec={"*": activation_spec},
                    op_output_spec=None,
                ),
                execution_mode="eager",
            )
        elif quantization_type == "output_activation_only":
            return QuantizerConfig(
                global_config=ModuleQuantizerConfig(
                    op_state_spec=None,
                    op_input_spec=None,
                    op_output_spec={"*": activation_spec},
                ),
                execution_mode="eager",
            )
        elif quantization_type == "weight_and_activation":
            return QuantizerConfig(
                global_config=ModuleQuantizerConfig(
                    op_state_spec={"weight": weight_spec},
                    op_input_spec={"*": activation_spec},
                    op_output_spec={"*": activation_spec},
                ),
                execution_mode="eager",
            )
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")

    # -------------------------------------------------------------------------
    # Standard quantization pattern: Conv, ConvTranspose, Linear, Embedding
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "model,example_input",
        [
            pytest.param(SimpleConvModel(dim=1), torch.randn(1, 3, 32), id="conv1d"),
            pytest.param(SimpleConvModel(dim=2), torch.randn(1, 3, 32, 32), id="conv2d"),
            pytest.param(SimpleConvModel(dim=3), torch.randn(1, 3, 8, 8, 8), id="conv3d"),
            pytest.param(
                SimpleConvTransposeModel(dim=1), torch.randn(1, 3, 32), id="conv_transpose1d"
            ),
            pytest.param(
                SimpleConvTransposeModel(dim=2), torch.randn(1, 3, 32, 32), id="conv_transpose2d"
            ),
            pytest.param(
                SimpleConvTransposeModel(dim=3), torch.randn(1, 3, 8, 8, 8), id="conv_transpose3d"
            ),
            pytest.param(SimpleLinearModel(), torch.randn(1, 10), id="linear"),
            pytest.param(SimpleEmbeddingModel(), torch.randint(0, 100, (1, 5)), id="embedding"),
        ],
    )
    @pytest.mark.parametrize(
        "quantization_type",
        ["weight_only", "input_activation_only", "output_activation_only", "weight_and_activation"],
    )
    def test_weighted_mod_quantization_pattern(
        self,
        model,
        example_input,
        quantization_type,
        weight_spec,
        activation_spec,
    ):
        """Test quantization for standard weighted modules."""
        model = copy.deepcopy(model)
        config = self._build_config(quantization_type, weight_spec, activation_spec)
        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare((example_input,))

        # Determine expected counts based on quantization type
        if quantization_type == "weight_only":
            expected_weight_fq, expected_input_fq, expected_output_fq = 1, 0, 0
        elif quantization_type == "input_activation_only":
            expected_weight_fq, expected_input_fq, expected_output_fq = 0, 1, 0
        elif quantization_type == "output_activation_only":
            expected_weight_fq, expected_input_fq, expected_output_fq = 0, 0, 1
        elif quantization_type == "weight_and_activation":
            expected_weight_fq, expected_input_fq, expected_output_fq = 1, 1, 1

        if isinstance(model, SimpleEmbeddingModel):
            # Embedding input is integer, so no input quantization should be applied
            expected_input_fq = 0

        # Verify quantizer counts
        assert self._count_weight_quantizers(prepared_model) == expected_weight_fq
        assert self._count_input_act_quantizers(prepared_model) == expected_input_fq
        assert self._count_output_act_quantizers(prepared_model) == expected_output_fq

    # -------------------------------------------------------------------------
    # Binary ops pattern: Add, Mul, Sub, MatMul
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "model",
        [
            pytest.param(SimpleBinaryModel(binary_fn=torch.add), id="add"),
            pytest.param(SimpleAddModel(), id="add_operator"),
            pytest.param(SimpleBinaryModel(binary_fn=torch.mul), id="mul"),
            pytest.param(SimpleMulModel(), id="mul_operator"),
            pytest.param(SimpleBinaryModel(binary_fn=torch.sub), id="sub"),
            pytest.param(SimpleSubModel(), id="sub_operator"),
            pytest.param(SimpleMatMulModel(), id="matmul_operator"),
        ],
    )
    @pytest.mark.parametrize(
        "quantization_type",
        ["weight_only", "input_activation_only", "output_activation_only", "weight_and_activation"],
    )
    def test_binary_ops_quantization_pattern(
        self,
        model,
        quantization_type,
        weight_spec,
        activation_spec,
    ):
        """Test quantization for binary operation models (Add, Mul, Sub, MatMul)."""
        model = copy.deepcopy(model)
        example_input = (torch.randn(1, 10),)

        config = self._build_config(quantization_type, weight_spec, activation_spec)
        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_input)

        # Determine expected counts based on quantization type
        if quantization_type == "weight_only":
            expected_weight_fq, expected_input_fq, expected_output_fq = 1, 0, 0
        elif quantization_type == "input_activation_only":
            expected_weight_fq, expected_input_fq, expected_output_fq = 0, 3, 0
        elif quantization_type == "output_activation_only":
            expected_weight_fq, expected_input_fq, expected_output_fq = 0, 0, 2
        elif quantization_type == "weight_and_activation":
            expected_weight_fq, expected_input_fq, expected_output_fq = 1, 3, 2

        # Verify quantizer counts
        assert self._count_weight_quantizers(prepared_model) == expected_weight_fq
        assert self._count_input_act_quantizers(prepared_model) == expected_input_fq
        assert self._count_output_act_quantizers(prepared_model) == expected_output_fq


class TestFP4MLIRExportValidation:
    """Test that FP4 eager export validation rejects unsupported configurations."""

    @pytest.mark.parametrize(
        "model, input_data, weight_granularity, with_fp4_activation, error_match",
        [
            pytest.param(
                nn.Linear(32, 64),
                torch.randn(1, 32),
                PerTensorGranularity(),
                False,
                "FP4 quantization requires PerBlockGranularity with block_size=32",
                id="per_tensor_granularity_rejected",
            ),
            pytest.param(
                nn.Linear(32, 64),
                torch.randn(1, 32),
                PerBlockGranularity(axis=1, block_size=16),
                False,
                "FP4 quantization requires PerBlockGranularity with block_size=32",
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
                "FP4 weight quantization export is only supported for 2D weight tensors",
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
            execution_mode="eager",
        )
        quantizer = Quantizer(model, config)
        quantizer.prepare((input_data,))

        with pytest.raises(ValueError, match=error_match):
            quantizer.finalize(backend=ExportBackend.CoreAI)


class TestBlockSizeMismatchSkip:
    """Test that non-divisible block sizes produce a warning and skip quantization."""

    def test_non_divisible_block_size_warns_and_skips(self, caplog):
        """Linear(768, 1000) with block_size=32 on axis=0: 1000 % 32 != 0."""
        model = nn.Sequential(nn.Linear(768, 1000))
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
            execution_mode="eager",
        )

        quantizer = Quantizer(model, config)

        with caplog.at_level(logging.WARNING):
            prepared_model = quantizer.prepare(example_inputs)

        # Should not crash
        assert prepared_model is not None

        # Warning should have been logged
        assert any("Skipping quantization" in msg for msg in caplog.messages)

        # The disabled FQ parametrization should be removed during prepare
        linear = prepared_model[0]
        assert not is_parametrized(linear, "weight")

        # No disabled FQ modules should remain
        fq_modules = [m for m in prepared_model.modules() if isinstance(m, FakeQuantizeImplBase)]
        assert len(fq_modules) == 0

        # Forward pass should return unquantized output (weight unchanged)
        ref_model = nn.Sequential(nn.Linear(768, 1000))
        ref_model[0].weight = nn.Parameter(linear.weight.detach().clone())
        ref_model[0].bias = nn.Parameter(linear.bias.detach().clone())

        test_input = torch.randn(1, 768)
        torch.testing.assert_close(prepared_model(test_input), ref_model(test_input))

    def test_divisible_block_size_not_disabled(self):
        """Linear(768, 1024) with block_size=32 on axis=0: 1024 % 32 == 0."""
        model = nn.Sequential(nn.Linear(768, 1024))
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
            execution_mode="eager",
        )

        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(example_inputs)

        linear = prepared_model[0]
        assert is_parametrized(linear, "weight")
        fake_quant_mod = linear.parametrizations.weight[0]
        assert isinstance(fake_quant_mod, FakeQuantizeImplBase)
        assert not fake_quant_mod.is_disabled()

    def test_disabled_activation_fq_removed_and_forward_pass_succeeds(self, caplog):
        """
        Activation FQ with non-divisible block size is removed from nested module,
        subsequent forward passes work.
        """

        class NestedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.block = nn.Sequential(nn.Linear(768, 1000), nn.ReLU())
                self.head = nn.Linear(1000, 10)

            def forward(self, x):
                x = self.block(x)
                return self.head(x)

        model = NestedModel()
        example_inputs = (torch.randn(1, 768),)

        act_spec = QuantizationSpec(
            dtype="int8",
            qscheme="symmetric",
            granularity=PerBlockGranularity(axis=0, block_size=32),
        )
        config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec=None,
                op_input_spec={"*": act_spec},
                op_output_spec=None,
            ),
            execution_mode="eager",
        )

        quantizer = Quantizer(model, config)
        with caplog.at_level(logging.WARNING):
            prepared_model = quantizer.prepare(example_inputs)

        assert prepared_model is not None
        assert any("Skipping quantization" in msg for msg in caplog.messages)

        # No disabled FQ modules should remain
        fq_modules = [m for m in prepared_model.modules() if isinstance(m, FakeQuantizeImplBase)]
        assert len(fq_modules) == 0

        # Subsequent forward pass through calibration should not raise
        with quantizer.calibration_mode():
            with torch.no_grad():
                prepared_model(torch.randn(1, 768))
