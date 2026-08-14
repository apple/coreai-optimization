# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for ``QuantizerConfig.presets`` and ``ModuleQuantizerConfig.presets``.

Covers preset factory defaults, composition with ``set_module_type`` /
``set_module_name`` / ``only_for`` / ``without``, and end-to-end demos
that run the full quantization workflow on a small model.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from coreai_opt._utils.config_utils import ALL_TENSORS
from coreai_opt._utils.python_utils import fqn
from coreai_opt.base_model_compressor import ExportBackend
from coreai_opt.quantization import (
    ModuleQuantizerConfig,
    Quantizer,
    QuantizerConfig,
)
from coreai_opt.quantization.config.quantization_config import ExecutionMode
from coreai_opt.quantization.spec import (
    PerBlockGranularity,
    PerChannelGranularity,
    QuantizationScheme,
)
from coreai_opt.quantization.spec.fake_quantize import FakeQuantizeImplBase

# Canonical fully-disabled override produced by only_for / without / set_global(None).
DISABLED_MODULE_CONFIG = ModuleQuantizerConfig(
    op_input_spec=None,
    op_output_spec=None,
    op_state_spec=None,
    module_input_spec=None,
    module_output_spec=None,
    module_state_spec=None,
)


class _DemoModel(nn.Module):
    """Multi-group model for testing wildcard module-name config patterns."""

    def __init__(self) -> None:
        super().__init__()
        self.text_encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.detr_decoder = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.head = nn.Linear(32, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.detr_decoder(self.text_encoder(x)))


class _BatchNormModel(nn.Module):
    """Exercises conv-bn folding: conv-bn-relu, conv-bn, pool, flatten, linear."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        self.fc = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = torch.flatten(nn.functional.adaptive_avg_pool2d(x, 1), 1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Preset factories
# ---------------------------------------------------------------------------


def _module_config(
    config: QuantizerConfig | ModuleQuantizerConfig,
) -> ModuleQuantizerConfig:
    """Return the inner ModuleQuantizerConfig from either preset return type."""
    return config.global_config if isinstance(config, QuantizerConfig) else config


_OWNER_CLASSES = [QuantizerConfig, ModuleQuantizerConfig]
_PRESET_NAMES = ["w8", "w4", "w4_per_block"]
_DEFAULT_SPECS = {
    "w8": (torch.int8, PerChannelGranularity, None),
    "w4": (torch.int4, PerChannelGranularity, None),
    "w4_per_block": (torch.int4, PerBlockGranularity, 32),
}


@pytest.mark.parametrize("owner_cls", _OWNER_CLASSES)
@pytest.mark.parametrize("preset_name", _PRESET_NAMES)
def test_preset_returns_correct_type(owner_cls, preset_name):
    """Each preset factory returns its owner config type."""
    config = getattr(owner_cls.presets, preset_name)()
    assert isinstance(config, owner_cls)


@pytest.mark.parametrize("owner_cls", _OWNER_CLASSES)
@pytest.mark.parametrize("preset_name", _PRESET_NAMES)
def test_preset_is_weight_only(owner_cls, preset_name):
    """All built-in presets are weight-only — activations explicitly disabled."""
    module = _module_config(getattr(owner_cls.presets, preset_name)())
    assert module.op_input_spec == {ALL_TENSORS: None}
    assert module.op_output_spec == {ALL_TENSORS: None}


@pytest.mark.parametrize("owner_cls", _OWNER_CLASSES)
@pytest.mark.parametrize("preset_name", _PRESET_NAMES)
def test_preset_default_spec(owner_cls, preset_name):
    """Each preset produces a symmetric weight spec with the expected dtype and granularity."""
    expected_dtype, expected_granularity_cls, expected_block_size = _DEFAULT_SPECS[preset_name]
    weight_spec = _module_config(getattr(owner_cls.presets, preset_name)()).op_state_spec["weight"]
    assert weight_spec
    assert weight_spec.dtype == expected_dtype
    assert weight_spec.qscheme == QuantizationScheme.SYMMETRIC
    assert isinstance(weight_spec.granularity, expected_granularity_cls)
    assert weight_spec.granularity.axis is None
    if expected_block_size is not None:
        assert weight_spec.granularity.block_size == expected_block_size


def test_w8_axis_override():
    """w8 honors a custom axis kwarg."""
    config = QuantizerConfig.presets.w8(axis=1)
    weight_spec = _module_config(config).op_state_spec["weight"]
    assert weight_spec.granularity.axis == 1


@pytest.mark.parametrize("owner_cls", _OWNER_CLASSES)
def test_w4_per_block_block_size_override(owner_cls):
    """w4_per_block honors a custom block_size kwarg."""
    config = owner_cls.presets.w4_per_block(block_size=64)
    weight_spec = _module_config(config).op_state_spec["weight"]
    assert weight_spec.granularity.block_size == 64


# ``w4_per_block`` is omitted: this model's channel counts aren't divisible by
# the default block size, so it quantizes nothing.
@pytest.mark.parametrize("preset_name", ["w8", "w4"])
def test_preset_does_not_quantize_batchnorm_affine_params(preset_name):
    """A conv-bn pattern quantizes the folded conv weight, not the bn's affine scale.

    The bn node is covered so that no observer lands on the internal conv->bn edge,
    but a pattern's state spec belongs to the op it is anchored on. Quantizing
    ``bn.weight`` also breaks the axis-default pass, since ``batch_norm`` has no
    per-channel axis default to resolve ``axis=None`` against.
    """
    config = getattr(QuantizerConfig.presets, preset_name)()
    prepared = Quantizer(_BatchNormModel().eval(), config).prepare(
        example_inputs=(torch.randn(2, 3, 8, 8),)
    )

    fq_names = {
        name
        for name, module in prepared.named_modules()
        if isinstance(module, FakeQuantizeImplBase)
    }
    # One per weight-bearing op: conv1, conv2, fc. Matches main_oss.
    assert len(fq_names) == 3

    for node in prepared.graph.nodes:
        if node.op == "call_function" and node.name.startswith("batch_norm"):
            weight_arg = node.all_input_nodes[1]
            assert weight_arg.name not in fq_names, (
                f"{node.name} weight {weight_arg.name} should not be fake-quantized"
            )


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def test_preset_composes_with_set_module_type_skip():
    config = QuantizerConfig.presets.w4().set_module_type(nn.LayerNorm, None)
    layer_norm_key = "torch.nn.modules.normalization.LayerNorm"
    assert layer_norm_key in config.module_type_configs


def test_module_preset_as_module_type_override_value():
    config = QuantizerConfig.presets.w4().set_module_type(
        nn.Embedding,
        ModuleQuantizerConfig.presets.w8(),
    )
    embedding_key = "torch.nn.modules.sparse.Embedding"
    embedding_config = config.module_type_configs[embedding_key]
    assert embedding_config
    assert embedding_config.op_state_spec
    assert embedding_config.op_state_spec["weight"]
    assert embedding_config.op_state_spec["weight"].dtype == torch.int8


def test_module_preset_as_module_name_override_value():
    config = QuantizerConfig.presets.w4().set_module_name(
        "decoder.lm_head",
        ModuleQuantizerConfig.presets.w8(),
    )
    name_config = config.module_name_configs["decoder.lm_head"]
    assert name_config
    assert name_config.op_state_spec
    assert name_config.op_state_spec["weight"]
    assert name_config.op_state_spec["weight"].dtype == torch.int8


def test_w8_global_with_module_w4_embedding_override():
    """Mixed bit-widths: w8 globally, w4 per module type."""
    config = QuantizerConfig.presets.w8().set_module_type(
        nn.Embedding,
        ModuleQuantizerConfig.presets.w4(),
    )
    embedding_key = "torch.nn.modules.sparse.Embedding"
    embedding_config = config.module_type_configs[embedding_key]
    assert embedding_config
    assert embedding_config.op_state_spec
    weight_spec = embedding_config.op_state_spec["weight"]
    assert weight_spec
    assert weight_spec.dtype == torch.int4
    assert isinstance(weight_spec.granularity, PerChannelGranularity)


# ---------------------------------------------------------------------------
# End-to-end demos
#
# These run the full prepare/finalize pipeline on a small model. They are
# regression guards — if a preset silently stops producing a valid
# configuration for ``Quantizer``, these tests fail.
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "preset_factory",
    [
        pytest.param(QuantizerConfig.presets.w8, id="w8"),
        pytest.param(QuantizerConfig.presets.w4, id="w4"),
        pytest.param(QuantizerConfig.presets.w4_per_block, id="w4_per_block"),
    ],
)
def test_demo_preset_runs_end_to_end(
    simple_linear_model, simple_linear_model_input, preset_factory
):
    """Verify each weight-only preset runs through prepare/finalize/forward."""
    config = preset_factory()
    quantizer = Quantizer(simple_linear_model, config)
    quantizer.prepare((simple_linear_model_input,))
    finalized_model = quantizer.finalize()
    finalized_model(simple_linear_model_input)


@pytest.mark.slow
def test_demo_w4_quantize_all_layers():
    """Apply w4 preset to every layer using eager mode."""
    model = _DemoModel()
    inp = torch.randn(4, 64)

    config = QuantizerConfig.presets.w4()
    config.set_execution_mode(ExecutionMode.EAGER)

    quantizer = Quantizer(model, config)
    prepared = quantizer.prepare(example_inputs=(inp,))
    model_int4 = quantizer.finalize(prepared, backend=ExportBackend.CoreAI)
    model_int4(inp)


@pytest.mark.slow
def test_demo_w4_excluding_detr_decoder():
    """Apply w4 preset to all layers except detr_decoder via set_module_name."""
    model = _DemoModel()
    inp = torch.randn(4, 64)

    config = QuantizerConfig.presets.w4()
    config.set_module_name("detr_decoder.*", None)
    config.set_execution_mode(ExecutionMode.EAGER)

    quantizer = Quantizer(model, config)
    prepared = quantizer.prepare(example_inputs=(inp,))
    model_opt = quantizer.finalize(prepared, backend=ExportBackend.CoreAI)
    model_opt(inp)


@pytest.mark.slow
def test_demo_mixed_presets(simple_linear_model, simple_linear_model_input):
    """w4 globally, w8 for a specific module, LayerNorm excluded."""
    config = (
        QuantizerConfig.presets.w4()
        .set_module_name("l2", ModuleQuantizerConfig.presets.w8())
        .without(nn.LayerNorm)
    )

    quantizer = Quantizer(simple_linear_model, config)
    quantizer.prepare((simple_linear_model_input,))
    finalized_model = quantizer.finalize()

    finalized_model(simple_linear_model_input)


@pytest.mark.slow
def test_demo_parameterized_preset(simple_linear_model, simple_linear_model_input):
    """Override a preset's defaults via kwargs."""
    config = QuantizerConfig.presets.w4_per_block(block_size=64)

    quantizer = Quantizer(simple_linear_model, config)
    quantizer.prepare((simple_linear_model_input,))
    finalized_model = quantizer.finalize()

    finalized_model(simple_linear_model_input)


@pytest.mark.slow
def test_demo_only_for(simple_linear_model, simple_linear_model_input):
    """only_for: apply preset only to Linear layers."""
    config = QuantizerConfig.presets.w8().only_for(nn.Linear)

    quantizer = Quantizer(simple_linear_model, config)
    quantizer.prepare((simple_linear_model_input,))
    finalized_model = quantizer.finalize()

    finalized_model(simple_linear_model_input)


# ---------------------------------------------------------------------------
# only_for — redistribute global config to a narrow set of targets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "targets",
    [
        pytest.param((nn.Linear,), id="single"),
        pytest.param((nn.Linear, nn.Conv2d), id="multiple"),
    ],
)
def test_only_for_module_types(targets):
    """only_for with one or more module types disables global and adds each as override."""
    config = QuantizerConfig.presets.w8()
    original_spec = config.global_config
    config.only_for(*targets)

    assert config.global_config == DISABLED_MODULE_CONFIG
    for target in targets:
        assert config.module_type_configs[fqn(target)] == original_spec


def test_only_for_module_name_string():
    """only_for also accepts module name strings."""
    config = QuantizerConfig.presets.w8()
    original_spec = config.global_config
    config.only_for("encoder.layer.0")

    assert config.global_config == DISABLED_MODULE_CONFIG
    assert config.module_name_configs["encoder.layer.0"] == original_spec


def test_only_for_mixed_types_and_names():
    """Types and names can be mixed in the same call."""
    config = QuantizerConfig.presets.w8()
    original_spec = config.global_config
    config.only_for(nn.Linear, "decoder.lm_head")

    assert config.global_config == DISABLED_MODULE_CONFIG
    assert config.module_type_configs["torch.nn.modules.linear.Linear"] == original_spec
    assert config.module_name_configs["decoder.lm_head"] == original_spec


def test_only_for_chains_with_without():
    """only_for result composes with subsequent without() calls."""
    config = QuantizerConfig.presets.w8()
    original_spec = config.global_config
    config.only_for(nn.Linear).without("decoder.lm_head")

    assert config.global_config == DISABLED_MODULE_CONFIG
    assert config.module_type_configs["torch.nn.modules.linear.Linear"] == original_spec
    assert config.module_name_configs["decoder.lm_head"] == DISABLED_MODULE_CONFIG


def test_only_for_deepcopies_spec_per_target():
    """Each target gets an independent spec — overrides don't share state."""
    config = QuantizerConfig.presets.w8().only_for(nn.Linear, nn.Conv2d)

    linear_key = "torch.nn.modules.linear.Linear"
    conv_key = "torch.nn.modules.conv.Conv2d"
    linear_spec = config.module_type_configs[linear_key]
    conv_spec = config.module_type_configs[conv_key]

    assert linear_spec == conv_spec
    assert linear_spec is not conv_spec


def test_only_for_overwrites_existing_module_type_override():
    """only_for overwrites a pre-existing set_module_type override with the former global."""
    config = QuantizerConfig.presets.w4()
    former_global = config.global_config

    config.set_module_type(nn.Linear, ModuleQuantizerConfig.presets.w8())
    linear_key = "torch.nn.modules.linear.Linear"
    assert config.module_type_configs[linear_key] != former_global

    config.only_for(nn.Linear)
    assert config.module_type_configs[linear_key] == former_global


@pytest.mark.parametrize(
    "targets",
    [
        pytest.param([nn.Linear, nn.Conv2d, "decoder.lm_head"], id="list"),
        pytest.param((nn.Linear, nn.Conv2d, "decoder.lm_head"), id="tuple"),
    ],
)
def test_only_for_accepts_sequence_argument(targets):
    """A single sequence (list or tuple) is equivalent to unpacked varargs."""
    config = QuantizerConfig.presets.w8()
    original_spec = config.global_config
    config.only_for(targets)

    assert config.global_config == DISABLED_MODULE_CONFIG
    assert config.module_type_configs["torch.nn.modules.linear.Linear"] == original_spec
    assert config.module_type_configs["torch.nn.modules.conv.Conv2d"] == original_spec
    assert config.module_name_configs["decoder.lm_head"] == original_spec


def test_only_for_string_arg_is_not_unpacked():
    """A bare string target is not iterated character-by-character."""
    config = QuantizerConfig.presets.w8()
    original_spec = config.global_config
    config.only_for("decoder.lm_head")

    assert config.module_name_configs["decoder.lm_head"] == original_spec
    # A character-unpacked string would produce single-letter name entries.
    assert len(config.module_name_configs) == 1


# ---------------------------------------------------------------------------
# without — mark targets as disabled
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "targets",
    [
        pytest.param((nn.LayerNorm,), id="single"),
        pytest.param((nn.LayerNorm, nn.Embedding), id="multiple"),
    ],
)
def test_without_module_types(targets):
    """without with one or more module types adds each as a disabled override."""
    config = QuantizerConfig.presets.w8()
    original_spec = config.global_config
    config.without(*targets)

    assert config.global_config == original_spec
    for target in targets:
        assert config.module_type_configs[fqn(target)] == DISABLED_MODULE_CONFIG


def test_without_module_name_string():
    """Without also accepts module name strings."""
    config = QuantizerConfig.presets.w8()
    original_spec = config.global_config
    config.without("decoder.lm_head")

    assert config.global_config == original_spec
    assert config.module_name_configs["decoder.lm_head"] == DISABLED_MODULE_CONFIG


def test_without_mixed_types_and_names():
    """Types and names can be mixed in a single without call."""
    config = QuantizerConfig.presets.w8().without(nn.LayerNorm, "decoder.lm_head")

    assert (
        config.module_type_configs["torch.nn.modules.normalization.LayerNorm"]
        == DISABLED_MODULE_CONFIG
    )
    assert config.module_name_configs["decoder.lm_head"] == DISABLED_MODULE_CONFIG


def test_without_no_targets_is_noop():
    """without() with no targets supports varargs unpacking of empty lists."""
    config = QuantizerConfig.presets.w8()
    original_spec = config.global_config
    config.without()

    assert config.global_config == original_spec
    assert config.module_type_configs == {}
    assert config.module_name_configs == {}


def test_without_accepts_list_argument():
    """A single list of targets is equivalent to unpacked varargs."""
    config = QuantizerConfig.presets.w8().without([nn.LayerNorm, "decoder.lm_head"])

    assert (
        config.module_type_configs["torch.nn.modules.normalization.LayerNorm"]
        == DISABLED_MODULE_CONFIG
    )
    assert config.module_name_configs["decoder.lm_head"] == DISABLED_MODULE_CONFIG


def test_without_empty_list_is_noop():
    """without([]) is a no-op — safe to use with dynamically-built exclusion lists."""
    config = QuantizerConfig.presets.w8()
    original_spec = config.global_config
    config.without([])

    assert config.global_config == original_spec
    assert config.module_type_configs == {}
    assert config.module_name_configs == {}


# ---------------------------------------------------------------------------
# only_for / without — error cases
# ---------------------------------------------------------------------------


def test_only_for_raises_when_no_targets():
    """only_for with no targets is a user error — would disable everything."""
    config = QuantizerConfig.presets.w8()
    with pytest.raises(ValueError, match="at least one target"):
        config.only_for()


def test_only_for_raises_when_called_twice():
    """Second only_for would silently disable the new targets — guard against it."""
    config = QuantizerConfig.presets.w8().only_for(nn.Linear)
    with pytest.raises(ValueError, match="non-disabled global_config"):
        config.only_for(nn.Conv2d)


def test_only_for_raises_after_set_global_none():
    """set_global(None) followed by only_for is the same footgun as double only_for."""
    config = QuantizerConfig.presets.w8()
    config.set_global(None)
    with pytest.raises(ValueError, match="non-disabled global_config"):
        config.only_for(nn.Linear)


@pytest.mark.parametrize("method_name", ["only_for", "without"])
def test_method_raises_on_invalid_target_type(method_name):
    """Non-class, non-string targets are rejected with a helpful message."""
    config = QuantizerConfig.presets.w8()
    method = getattr(config, method_name)
    with pytest.raises(TypeError, match="must be module types or name strings"):
        method(42)


@pytest.mark.parametrize("method_name", ["only_for", "without"])
def test_method_raises_on_non_module_class(method_name):
    """Classes that aren't nn.Module subclasses get a dedicated error."""
    config = QuantizerConfig.presets.w8()
    method = getattr(config, method_name)
    with pytest.raises(TypeError, match=r"nn.Module subclasses or name strings"):
        method(int)


@pytest.mark.parametrize(
    ("method_name", "valid_targets"),
    [
        pytest.param("only_for", (nn.Linear, nn.Conv2d), id="only_for"),
        pytest.param("without", (nn.LayerNorm, nn.Embedding), id="without"),
    ],
)
def test_method_is_atomic_on_invalid_target(method_name, valid_targets):
    """A bad target aborts the call before any mutation happens."""
    config = QuantizerConfig.presets.w8()
    original_spec = config.global_config
    method = getattr(config, method_name)
    first, second = valid_targets

    with pytest.raises(TypeError):
        method(first, 42, second)

    assert config.global_config == original_spec
    assert config.module_type_configs == {}
    assert config.module_name_configs == {}
