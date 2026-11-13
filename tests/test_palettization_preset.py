# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for ``KMeansPalettizerConfig.presets``.

Covers preset factory defaults, composition with ``set_module_type`` /
``only_for`` / ``without``, and end-to-end demos that run the full
palettization workflow on a small model.
"""

from __future__ import annotations

import pytest
import torch.nn as nn

from coreai_opt.palettization import (
    KMeansPalettizer,
    KMeansPalettizerConfig,
    ModuleKMeansPalettizerConfig,
)
from coreai_opt.palettization.spec import (
    PerGroupedChannelGranularity,
    PerTensorGranularity,
)

# Canonical fully-disabled override produced by only_for / without / set_global(None).
DISABLED_MODULE_CONFIG = ModuleKMeansPalettizerConfig(
    op_input_spec=None,
    op_output_spec=None,
    op_state_spec=None,
    module_input_spec=None,
    module_output_spec=None,
    module_state_spec=None,
)


# ---------------------------------------------------------------------------
# Built-in preset defaults
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("preset_name", ["w4", "w6"])
def test_grouped_preset_returns_kmeans_palettizer_config(preset_name):
    config = getattr(KMeansPalettizerConfig.presets, preset_name)()
    assert isinstance(config, KMeansPalettizerConfig)


@pytest.mark.parametrize(
    ("preset_name", "expected_n_bits"),
    [("w4", 4), ("w6", 6)],
)
def test_grouped_preset_default_spec(preset_name, expected_n_bits):
    config = getattr(KMeansPalettizerConfig.presets, preset_name)()
    weight_spec = config.global_config.op_state_spec["weight"]
    assert weight_spec.n_bits == expected_n_bits
    assert isinstance(weight_spec.granularity, PerGroupedChannelGranularity)
    assert weight_spec.granularity.axis == 0
    assert weight_spec.granularity.group_size == 16
    assert config.global_config.op_state_spec["in_proj_weight"] == weight_spec


@pytest.mark.parametrize("preset_name", ["w4", "w6"])
def test_grouped_preset_group_size_override(preset_name):
    config = getattr(KMeansPalettizerConfig.presets, preset_name)(group_size=32)
    assert config.global_config.op_state_spec["weight"].granularity.group_size == 32
    assert config.global_config.op_state_spec["in_proj_weight"].granularity.group_size == 32


def test_w8_default_spec():
    config = KMeansPalettizerConfig.presets.w8()
    weight_spec = config.global_config.op_state_spec["weight"]
    assert weight_spec.n_bits == 8
    assert isinstance(weight_spec.granularity, PerTensorGranularity)
    assert config.global_config.op_state_spec["in_proj_weight"] == weight_spec


@pytest.mark.parametrize("preset_name", ["w4", "w6"])
def test_module_grouped_preset_returns_module_kmeans_palettizer_config(preset_name):
    config = getattr(ModuleKMeansPalettizerConfig.presets, preset_name)()
    assert isinstance(config, ModuleKMeansPalettizerConfig)


@pytest.mark.parametrize(
    ("preset_name", "expected_n_bits"),
    [("w4", 4), ("w6", 6)],
)
def test_module_grouped_preset_default_spec(preset_name, expected_n_bits):
    config = getattr(ModuleKMeansPalettizerConfig.presets, preset_name)()
    weight_spec = config.op_state_spec["weight"]
    assert weight_spec.n_bits == expected_n_bits
    assert isinstance(weight_spec.granularity, PerGroupedChannelGranularity)
    assert weight_spec.granularity.axis == 0
    assert weight_spec.granularity.group_size == 16
    assert config.op_state_spec["in_proj_weight"] == weight_spec


@pytest.mark.parametrize("preset_name", ["w4", "w6"])
def test_module_grouped_preset_group_size_override(preset_name):
    config = getattr(ModuleKMeansPalettizerConfig.presets, preset_name)(group_size=32)
    assert config.op_state_spec["weight"].granularity.group_size == 32
    assert config.op_state_spec["in_proj_weight"].granularity.group_size == 32


def test_module_w8_default_spec():
    config = ModuleKMeansPalettizerConfig.presets.w8()
    weight_spec = config.op_state_spec["weight"]
    assert weight_spec.n_bits == 8
    assert isinstance(weight_spec.granularity, PerTensorGranularity)
    assert config.op_state_spec["in_proj_weight"] == weight_spec


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def test_preset_composes_with_set_module_type_skip():
    config = KMeansPalettizerConfig.presets.w4().set_module_type(nn.LayerNorm, None)
    layer_norm_key = "torch.nn.modules.normalization.LayerNorm"
    assert layer_norm_key in config.module_type_configs


def test_w4_global_with_module_w8_linear_override():
    config = KMeansPalettizerConfig.presets.w4().set_module_type(
        nn.Linear, ModuleKMeansPalettizerConfig.presets.w8()
    )
    assert config.global_config.op_state_spec["weight"].n_bits == 4
    assert config.global_config.op_state_spec["in_proj_weight"].n_bits == 4
    linear_cfg = config.module_type_configs["torch.nn.modules.linear.Linear"]
    assert linear_cfg.op_state_spec["weight"].n_bits == 8
    assert linear_cfg.op_state_spec["in_proj_weight"].n_bits == 8
    assert isinstance(linear_cfg.op_state_spec["weight"].granularity, PerTensorGranularity)


# ---------------------------------------------------------------------------
# End-to-end demos
#
# These run the full prepare/finalize pipeline on a small model. They are
# regression guards — if a preset silently stops producing a valid
# configuration for ``KMeansPalettizer``, these tests fail.
# ---------------------------------------------------------------------------


@pytest.mark.requires_coreai
@pytest.mark.slow
@pytest.mark.parametrize("preset_name", ["w4", "w6"])
def test_demo_grouped_preset(preset_name, simple_linear_model, simple_linear_model_input):
    """N-bit palettization, per-grouped-channel."""
    config = getattr(KMeansPalettizerConfig.presets, preset_name)()

    palettizer = KMeansPalettizer(simple_linear_model, config)
    palettizer.prepare((simple_linear_model_input,))
    finalized_model = palettizer.finalize()

    finalized_model(simple_linear_model_input)


@pytest.mark.requires_coreai
@pytest.mark.slow
def test_demo_w8(simple_linear_model, simple_linear_model_input):
    """w8: 8-bit palettization, per-tensor (lossless baseline)."""
    config = KMeansPalettizerConfig.presets.w8()

    palettizer = KMeansPalettizer(simple_linear_model, config)
    palettizer.prepare((simple_linear_model_input,))
    finalized_model = palettizer.finalize()

    finalized_model(simple_linear_model_input)


# ---------------------------------------------------------------------------
# only_for — redistribute global config to a narrow set of targets
# ---------------------------------------------------------------------------


def test_only_for_single_module_type():
    """only_for with a single module type disables global and adds the type."""
    config = KMeansPalettizerConfig.presets.w4()
    original_spec = config.global_config
    config.only_for(nn.Linear)

    assert config.global_config == DISABLED_MODULE_CONFIG
    assert config.module_type_configs["torch.nn.modules.linear.Linear"] == original_spec


def test_only_for_multiple_module_types():
    """only_for with multiple types redistributes global to each."""
    config = KMeansPalettizerConfig.presets.w4()
    original_spec = config.global_config
    config.only_for(nn.Linear, nn.Conv2d)

    assert config.global_config == DISABLED_MODULE_CONFIG
    assert config.module_type_configs["torch.nn.modules.linear.Linear"] == original_spec
    assert config.module_type_configs["torch.nn.modules.conv.Conv2d"] == original_spec


def test_only_for_module_name_string():
    """only_for also accepts module name strings."""
    config = KMeansPalettizerConfig.presets.w4()
    original_spec = config.global_config
    config.only_for("encoder.layer.0")

    assert config.global_config == DISABLED_MODULE_CONFIG
    assert config.module_name_configs["encoder.layer.0"] == original_spec


def test_only_for_mixed_types_and_names():
    """Types and names can be mixed in the same call."""
    config = KMeansPalettizerConfig.presets.w4()
    original_spec = config.global_config
    config.only_for(nn.Linear, "decoder.lm_head")

    assert config.global_config == DISABLED_MODULE_CONFIG
    assert config.module_type_configs["torch.nn.modules.linear.Linear"] == original_spec
    assert config.module_name_configs["decoder.lm_head"] == original_spec


def test_only_for_chains_with_without():
    """only_for result composes with subsequent without() calls."""
    config = KMeansPalettizerConfig.presets.w4()
    original_spec = config.global_config
    config.only_for(nn.Linear).without("decoder.lm_head")

    assert config.global_config == DISABLED_MODULE_CONFIG
    assert config.module_type_configs["torch.nn.modules.linear.Linear"] == original_spec
    assert config.module_name_configs["decoder.lm_head"] == DISABLED_MODULE_CONFIG


def test_only_for_deepcopies_spec_per_target():
    """Each target gets an independent spec — overrides don't share state."""
    config = KMeansPalettizerConfig.presets.w4().only_for(nn.Linear, nn.Conv2d)

    linear_key = "torch.nn.modules.linear.Linear"
    conv_key = "torch.nn.modules.conv.Conv2d"
    linear_spec = config.module_type_configs[linear_key]
    conv_spec = config.module_type_configs[conv_key]

    assert linear_spec == conv_spec
    assert linear_spec is not conv_spec


def test_only_for_accepts_list_argument():
    """A single list of targets is equivalent to unpacked varargs."""
    config = KMeansPalettizerConfig.presets.w4()
    original_spec = config.global_config
    config.only_for([nn.Linear, nn.Conv2d])

    assert config.global_config == DISABLED_MODULE_CONFIG
    assert config.module_type_configs["torch.nn.modules.linear.Linear"] == original_spec
    assert config.module_type_configs["torch.nn.modules.conv.Conv2d"] == original_spec


# ---------------------------------------------------------------------------
# without — mark targets as disabled
# ---------------------------------------------------------------------------


def test_without_single_module_type():
    """without with a single module type adds it as a disabled override."""
    config = KMeansPalettizerConfig.presets.w4()
    original_spec = config.global_config
    config.without(nn.LayerNorm)

    assert config.global_config == original_spec  # global unchanged
    assert (
        config.module_type_configs["torch.nn.modules.normalization.LayerNorm"]
        == DISABLED_MODULE_CONFIG
    )


def test_without_multiple_module_types():
    """without with multiple types adds each as a disabled override."""
    config = KMeansPalettizerConfig.presets.w4()
    original_spec = config.global_config
    config.without(nn.LayerNorm, nn.Embedding)

    assert config.global_config == original_spec
    assert (
        config.module_type_configs["torch.nn.modules.normalization.LayerNorm"]
        == DISABLED_MODULE_CONFIG
    )
    assert config.module_type_configs["torch.nn.modules.sparse.Embedding"] == DISABLED_MODULE_CONFIG


def test_without_module_name_string():
    """without also accepts module name strings."""
    config = KMeansPalettizerConfig.presets.w4()
    original_spec = config.global_config
    config.without("decoder.lm_head")

    assert config.global_config == original_spec
    assert config.module_name_configs["decoder.lm_head"] == DISABLED_MODULE_CONFIG


def test_without_mixed_types_and_names():
    """Types and names can be mixed in a single without call."""
    config = KMeansPalettizerConfig.presets.w4().without(nn.LayerNorm, "decoder.lm_head")

    assert (
        config.module_type_configs["torch.nn.modules.normalization.LayerNorm"]
        == DISABLED_MODULE_CONFIG
    )
    assert config.module_name_configs["decoder.lm_head"] == DISABLED_MODULE_CONFIG


def test_without_no_targets_is_noop():
    """without() with no targets supports varargs unpacking of empty lists."""
    config = KMeansPalettizerConfig.presets.w4()
    original_spec = config.global_config
    config.without()

    assert config.global_config == original_spec
    assert config.module_type_configs == {}
    assert config.module_name_configs == {}


def test_without_accepts_list_argument():
    """A single list of targets is equivalent to unpacked varargs."""
    config = KMeansPalettizerConfig.presets.w4().without([nn.LayerNorm, "decoder.lm_head"])

    assert (
        config.module_type_configs["torch.nn.modules.normalization.LayerNorm"]
        == DISABLED_MODULE_CONFIG
    )
    assert config.module_name_configs["decoder.lm_head"] == DISABLED_MODULE_CONFIG


# ---------------------------------------------------------------------------
# only_for / without — error cases
# ---------------------------------------------------------------------------


def test_only_for_raises_when_no_targets():
    """only_for with no targets is a user error — would disable everything."""
    config = KMeansPalettizerConfig.presets.w4()
    with pytest.raises(ValueError, match="at least one target"):
        config.only_for()


def test_only_for_raises_when_called_twice():
    """Second only_for would silently disable the new targets — guard against it."""
    config = KMeansPalettizerConfig.presets.w4().only_for(nn.Linear)
    with pytest.raises(ValueError, match="non-disabled global_config"):
        config.only_for(nn.Conv2d)


def test_only_for_raises_after_set_global_none():
    """set_global(None) followed by only_for is the same footgun as double only_for."""
    config = KMeansPalettizerConfig.presets.w4()
    config.set_global(None)
    with pytest.raises(ValueError, match="non-disabled global_config"):
        config.only_for(nn.Linear)


def test_only_for_raises_on_invalid_target_type():
    """Non-class, non-string targets are rejected with a helpful message."""
    config = KMeansPalettizerConfig.presets.w4()
    with pytest.raises(TypeError, match="must be module types or name strings"):
        config.only_for(42)  # type: ignore[arg-type]


def test_without_raises_on_invalid_target_type():
    """Non-class, non-string targets are rejected with a helpful message."""
    config = KMeansPalettizerConfig.presets.w4()
    with pytest.raises(TypeError, match="must be module types or name strings"):
        config.without(42)  # type: ignore[arg-type]


def test_only_for_raises_on_non_module_class():
    """Classes that aren't nn.Module subclasses get a dedicated error."""
    config = KMeansPalettizerConfig.presets.w4()
    with pytest.raises(TypeError, match="nn.Module subclasses or name strings"):
        config.only_for(int)  # type: ignore[arg-type]


def test_without_raises_on_non_module_class():
    """Classes that aren't nn.Module subclasses get a dedicated error."""
    config = KMeansPalettizerConfig.presets.w4()
    with pytest.raises(TypeError, match="nn.Module subclasses or name strings"):
        config.without(int)  # type: ignore[arg-type]


def test_only_for_is_atomic_on_invalid_target():
    """A bad target aborts only_for before any mutation happens."""
    config = KMeansPalettizerConfig.presets.w4()
    original_spec = config.global_config

    with pytest.raises(TypeError):
        config.only_for(nn.Linear, 42, nn.Conv2d)  # type: ignore[arg-type]

    assert config.global_config == original_spec
    assert config.module_type_configs == {}
    assert config.module_name_configs == {}


def test_without_is_atomic_on_invalid_target():
    """A bad target aborts without before any mutation happens."""
    config = KMeansPalettizerConfig.presets.w4()
    original_spec = config.global_config

    with pytest.raises(TypeError):
        config.without(nn.LayerNorm, 42, nn.Embedding)  # type: ignore[arg-type]

    assert config.global_config == original_spec
    assert config.module_type_configs == {}
    assert config.module_name_configs == {}
