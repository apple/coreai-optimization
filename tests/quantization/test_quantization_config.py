# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import io

import pytest
import torch
from pydantic import ValidationError

import tests.utils as utils
from coreai_opt._utils.config_utils import ALL_TENSORS as _ALL_TENSORS, spec_dict_is_active
from coreai_opt.quantization import QuantizationSpec, Quantizer
from coreai_opt.quantization.config.quantization_config import (
    _QUANTIZATION_CONFIG,
    ExecutionMode,
    ModuleQuantizerConfig,
    OpQuantizerConfig,
    QATSchedule,
    QuantizerConfig,
)
from coreai_opt.quantization.spec import QuantizationFormulation
from coreai_opt.quantization.spec.fake_quantize import FakeQuantizeImplBase
from coreai_opt.quantization.spec.spec import (
    default_activation_quantization_spec,
    default_weight_quantization_spec,
)
from tests.quantization.test_quantization_spec import (
    expanded_dtype_allowlist,  # noqa: F401
)

GLOBAL_CONFIG = "global_config"
MODULE_TYPE_CONFIGS = "module_type_configs"
MODULE_NAME_CONFIGS = "module_name_configs"
MODULE_INPUT_SPEC = "module_input_spec"
MODULE_OUTPUT_SPEC = "module_output_spec"
MODULE_STATE_SPEC = "module_state_spec"
OP_TYPE_CONFIG = "op_type_config"
OP_NAME_CONFIG = "op_name_config"
OP_INPUT_SPEC = "op_input_spec"
OP_OUTPUT_SPEC = "op_output_spec"
OP_STATE_SPEC = "op_state_spec"

DISABLED_MODULE_CONFIG = ModuleQuantizerConfig(
    op_input_spec=None,
    op_output_spec=None,
    op_state_spec=None,
    module_input_spec=None,
    module_output_spec=None,
    module_state_spec=None,
)


def test_init_with_defaults():
    """Test initialization with default values"""
    config = QuantizerConfig()
    # Check that default config has the expected structure
    assert config.global_config == ModuleQuantizerConfig()
    assert config.module_type_configs == {}
    assert config.module_name_configs == {}

    config = QuantizerConfig(module_name_configs={})
    # Check that default config has the expected structure
    assert config.global_config == ModuleQuantizerConfig()
    assert config.module_type_configs == {}
    assert config.module_name_configs == {}


def test_init_with_explicit_none():
    """Test initialization with explicit None"""
    config = QuantizerConfig(
        global_config=None,
        module_type_configs=None,
    )
    assert config.global_config == DISABLED_MODULE_CONFIG
    assert config.module_type_configs == {}
    assert config.module_name_configs == {}


def test_init_invalid_global():
    """Test initalization with module_input_spec provided for global config"""
    module_config = ModuleQuantizerConfig(
        module_input_spec={0: None}  # Non-empty dict to trigger validation error
    )
    with pytest.raises(ValidationError, match="global_config cannot have module_input_spec"):
        _ = QuantizerConfig(global_config=module_config)


def test_init_module_quantizer_config():
    """Test default ModuleQuantizerConfig init"""
    module_config = ModuleQuantizerConfig()
    assert module_config.op_input_spec == {_ALL_TENSORS: default_activation_quantization_spec()}
    assert module_config.op_output_spec == {_ALL_TENSORS: default_activation_quantization_spec()}
    assert module_config.op_state_spec == {"weight": default_weight_quantization_spec()}
    assert module_config.module_input_spec == {}
    assert module_config.module_output_spec == {}
    assert module_config.module_state_spec == {}
    assert module_config.qat_schedule is None


def test_init_module_quantizer_config_none_and_empty_dict_differ():
    """``None`` disables an op-level category; ``{}`` expresses no opinion.

    Module-level specs are unaffected — they keep normalizing None to ``{}``,
    since only op-level specs feed per-tensor annotation decisions.
    """
    empty = ModuleQuantizerConfig(
        op_input_spec={},
        op_output_spec={},
        op_state_spec={},
        module_input_spec={},
        module_output_spec={},
        module_state_spec={},
    )
    disabled = ModuleQuantizerConfig(
        op_input_spec=None,
        op_output_spec=None,
        op_state_spec=None,
        module_input_spec=None,
        module_output_spec=None,
        module_state_spec=None,
    )
    assert empty != disabled
    assert empty.op_input_spec == {}
    assert disabled.op_input_spec == {_ALL_TENSORS: None}
    # Module-level specs stay empty either way.
    assert empty.module_input_spec == disabled.module_input_spec == {}
    # Neither asks for compression.
    assert not spec_dict_is_active(empty.op_input_spec)
    assert not spec_dict_is_active(disabled.op_input_spec)


def test_init_op_quantizer_config():
    """Test default OpQuantizerConfig init"""
    op_config = OpQuantizerConfig()
    assert op_config.op_input_spec[_ALL_TENSORS] == default_activation_quantization_spec()
    assert op_config.op_output_spec[_ALL_TENSORS] == default_activation_quantization_spec()
    assert op_config.op_state_spec["weight"] == default_weight_quantization_spec()


def test_init_op_quantizer_config_none_and_empty_dict_differ():
    """``None`` disables a category; ``{}`` expresses no opinion about it.

    These were equivalent before, both normalizing to ``{}``. Keeping them
    distinct is what lets a consumer tell "the user excluded this tensor" from
    "no key addressed this tensor".
    """
    empty = OpQuantizerConfig(
        op_input_spec={},
        op_output_spec={},
        op_state_spec={},
    )
    disabled = OpQuantizerConfig(
        op_input_spec=None,
        op_output_spec=None,
        op_state_spec=None,
    )
    assert empty != disabled
    assert empty.op_input_spec == {}
    assert disabled.op_input_spec == {_ALL_TENSORS: None}
    # Neither asks for compression.
    assert not spec_dict_is_active(empty.op_input_spec)
    assert not spec_dict_is_active(disabled.op_input_spec)


def test_from_yaml(tmp_path, expanded_dtype_allowlist):  # noqa: F811
    """Test from_yaml with op-level and module-level configs"""
    # Write YAML as raw string to support anchors, aliases, and merge keys
    config_content = """
quantization_spec:
    weight_spec: &weight_spec
        dtype: int4
        qscheme: symmetric
        qformulation: zp
        granularity:
            type: per_channel
            axis: 1
        fake_quantize_cls: default
        qparam_calculator_cls: default
        range_calculator_cls: minmax

    activation_spec: &activation_spec
        dtype: int7
        qscheme: asymmetric
        qformulation: minval
        granularity:
            type: per_tensor
        fake_quantize_cls: default
        qparam_calculator_cls: moving_average
        range_calculator_cls: minmax

    module_spec: &module_spec
        dtype: int8
        qscheme: symmetric
        qformulation: zp
        granularity:
            type: per_tensor
        fake_quantize_cls: default
        qparam_calculator_cls: default
        range_calculator_cls: minmax

quantization_config:
    global_config:
        op_input_spec:
            input_0: *activation_spec
        op_state_spec:
            weight: *weight_spec
        qat_schedule:
            enable_observer: 0
            enable_fake_quant: 100
            disable_observer: 500
    module_type_configs:
        torch.nn.modules.linear.Linear:
            op_input_spec:
                input_0: *activation_spec
            op_state_spec:
                weight: *weight_spec
            module_input_spec:
                0:
                    <<: *activation_spec
                    dtype: int8
                    qformulation: zp
            qat_schedule:
                enable_observer: 0
                enable_fake_quant: 5
    module_name_configs:
        module_name:
            op_input_spec:
                input_0: *activation_spec
            op_state_spec:
                weight: *weight_spec
            module_output_spec:
                0: *module_spec
            qat_schedule:
                enable_observer: 0
                enable_fake_quant: 3
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)
    config = QuantizerConfig.from_yaml(config_file)

    # Test global config - op level
    assert config.global_config.op_input_spec["input_0"].dtype == torch.int7
    assert config.global_config.op_state_spec["weight"].dtype == torch.int4
    assert config.global_config.module_input_spec == {}
    assert config.global_config.module_output_spec == {}
    assert config.global_config.module_state_spec == {}

    # Test module type config - op level
    assert (
        config.module_type_configs["torch.nn.modules.linear.Linear"].op_input_spec["input_0"].dtype
        == torch.int7
    )
    assert (
        config.module_type_configs["torch.nn.modules.linear.Linear"].op_state_spec["weight"].dtype
        == torch.int4
    )

    # Test module type config - module level
    assert (
        config.module_type_configs["torch.nn.modules.linear.Linear"].module_input_spec[0].dtype
        == torch.int8
    )

    # Test module name config - op level
    assert config.module_name_configs["module_name"].op_input_spec["input_0"].dtype == torch.int7
    assert config.module_name_configs["module_name"].op_state_spec["weight"].dtype == torch.int4

    # Test module name config - module level
    assert config.module_name_configs["module_name"].module_output_spec[0].dtype == torch.int8

    # Test qat_schedule - global
    assert config.global_config.qat_schedule is not None
    assert config.global_config.qat_schedule.enable_fake_quant == 100
    assert config.global_config.qat_schedule.disable_observer == 500

    # Test qat_schedule - module type
    lin_type = "torch.nn.modules.linear.Linear"
    assert config.module_type_configs[lin_type].qat_schedule.enable_fake_quant == 5

    # Test qat_schedule - module name
    assert config.module_name_configs["module_name"].qat_schedule.enable_fake_quant == 3

    assert (
        config.global_config.op_input_spec["input_0"].qformulation == QuantizationFormulation.MINVAL
    )
    assert config.global_config.op_state_spec["weight"].qformulation == QuantizationFormulation.ZP
    assert (
        config.module_name_configs["module_name"].module_output_spec[0].qformulation
        == QuantizationFormulation.ZP
    )

    assert (
        config.module_type_configs[lin_type].module_input_spec[0].qformulation
        == QuantizationFormulation.ZP
    )


def test_from_yaml_no_config_or_spec(tmp_path):
    """Test from_yaml with empty config or spec raises error"""
    config_content = {}
    config_file = utils.create_yaml_file(tmp_path, "config.yaml", config_content)
    with pytest.raises(RuntimeError, match="Did not find 'quantization_config'"):
        _ = QuantizerConfig.from_yaml(config_file)


def test_from_yaml_unexpected_key(tmp_path):
    """Test from_yaml with unexpected key raises error"""
    config_content = {"spec1": None}
    config_file = utils.create_yaml_file(tmp_path, "config.yaml", config_content)
    with pytest.raises(RuntimeError, match="Found unexpected key 'spec1'"):
        _ = QuantizerConfig.from_yaml(config_file)


def test_from_yaml_digit_string_keys_e2e(simple_linear_model, simple_linear_model_input):
    """Config with both bare int and quoted digit-string keys quantizes a model end-to-end."""
    config_content = """\
quantization_config:
    module_type_configs:
        torch.nn.modules.linear.Linear:
            op_state_spec:
                weight:
                    dtype: int8
                    qscheme: symmetric
                    granularity:
                        type: per_channel
                        axis: 0
            op_input_spec:
                0:
                    dtype: int8
                    qscheme: symmetric
                    granularity:
                        type: per_tensor
            op_output_spec:
                "0":
                    dtype: int8
                    qscheme: symmetric
                    granularity:
                        type: per_tensor
"""
    config = QuantizerConfig.from_yaml(io.StringIO(config_content))

    # Both key forms should produce int keys
    linear_cfg = config.module_type_configs["torch.nn.modules.linear.Linear"]
    assert list(linear_cfg.op_input_spec.keys()) == [0]
    assert list(linear_cfg.op_output_spec.keys()) == [0]

    quantizer = Quantizer(simple_linear_model, config)
    prepared_model = quantizer.prepare((simple_linear_model_input,))

    with quantizer.calibration_mode():
        prepared_model(simple_linear_model_input)

    finalized = quantizer.finalize()
    output = finalized(simple_linear_model_input)
    assert output.shape == simple_linear_model_input.shape


def test_from_yaml_null_config(tmp_path):
    """Test from_yaml with op-level and module-level configs"""
    config_content = """
quantization_config:
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)
    with pytest.raises(TypeError):
        _ = QuantizerConfig.from_yaml(config_file)


def test_from_yaml_module_type_name_default(tmp_path):
    """
    Test that an empty module type or name quantization config leads to default
    quantization behavior
    """
    config_content = {
        _QUANTIZATION_CONFIG: {
            GLOBAL_CONFIG: None,
            MODULE_TYPE_CONFIGS: {
                "torch.nn.Linear": {}  # Empty dict to get default behavior
            },
            MODULE_NAME_CONFIGS: {"linear[0-3]": {}},
        }
    }
    config_file = utils.create_yaml_file(tmp_path, "config.yaml", config_content)
    config = QuantizerConfig.from_yaml(config_file)

    # Check that default config has the expected structure
    assert config.global_config == DISABLED_MODULE_CONFIG
    assert config.module_type_configs["torch.nn.Linear"].op_input_spec == {
        _ALL_TENSORS: default_activation_quantization_spec()
    }
    assert config.module_name_configs["linear[0-3]"].op_input_spec == {
        _ALL_TENSORS: default_activation_quantization_spec()
    }


def test_from_yaml_default_op_config(tmp_path, expanded_dtype_allowlist):  # noqa: F811
    config_content = """
quantization_spec:
    weight_spec: &weight_spec
        dtype: int4
        qscheme: symmetric
        granularity:
            type: per_channel
            axis: 1
        fake_quantize_cls: default
        qparam_calculator_cls: default
        range_calculator_cls: minmax

    activation_spec: &activation_spec
        dtype: int7
        qscheme: asymmetric
        granularity:
            type: per_tensor
        fake_quantize_cls: default
        qparam_calculator_cls: moving_average
        range_calculator_cls: minmax

quantization_config:
    global_config:
    module_type_configs:
        torch.nn.modules.linear.Linear:
        torch.nn.MultiheadAttention:
            op_type_config:
                aten.linear.default: null
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)
    config = QuantizerConfig.from_yaml(config_file)
    assert config.global_config == DISABLED_MODULE_CONFIG
    assert config.module_type_configs["torch.nn.modules.linear.Linear"] == DISABLED_MODULE_CONFIG
    assert config.module_type_configs["torch.nn.MultiheadAttention"].op_input_spec == {
        _ALL_TENSORS: default_activation_quantization_spec()
    }
    assert config.module_type_configs["torch.nn.MultiheadAttention"].op_type_config[
        "aten.linear.default"
    ] == OpQuantizerConfig(op_input_spec=None, op_output_spec=None, op_state_spec=None)


# The following unit tests test example scenarios in the quantization config user guide
def test_example_1_config_default_settings(tmp_path):
    """
    Test that a empty dict quantization config leads to default int8 weight/activation
    quantization
    """
    config_content = {_QUANTIZATION_CONFIG: {}}
    config_file = utils.create_yaml_file(tmp_path, "config.yaml", config_content)
    config = QuantizerConfig.from_yaml(config_file)

    # Check that default config has the expected structure
    assert (
        config.global_config.op_input_spec[_ALL_TENSORS] == default_activation_quantization_spec()
    )
    assert (
        config.global_config.op_output_spec[_ALL_TENSORS] == default_activation_quantization_spec()
    )
    assert config.global_config.op_state_spec["weight"] == default_weight_quantization_spec()
    assert config.module_type_configs == {}
    assert config.module_name_configs == {}
    assert config.global_config.module_input_spec == {}
    assert config.global_config.module_output_spec == {}
    assert config.global_config.module_state_spec == {}
    assert config.global_config.op_name_config == {}
    assert config.global_config.op_type_config == {}


def test_example_2_disabling_all_quantization(tmp_path):
    """Test Example 2: Disabling All Quantization"""
    yaml_content = """
quantization_config:
    global_config: null
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    # Load from YAML
    config_from_yaml = QuantizerConfig.from_yaml(config_file)

    # Create programmatically
    config_programmatic = QuantizerConfig(global_config=None)

    # Compare
    assert config_from_yaml == config_programmatic


def test_example_3_basic_custom_global_configuration(tmp_path):
    """Test Example 3: Basic Custom Global Configuration"""

    yaml_content = """
quantization_spec:
    spec1: &int4_activation
        dtype: int4
        qscheme: symmetric
        granularity:
            type: per_tensor
        fake_quantize_cls: default
        qparam_calculator_cls: moving_average
        range_calculator_cls: minmax

    spec2: &int4_weight
        dtype: int4
        qscheme: symmetric
        granularity:
            type: per_tensor
        fake_quantize_cls: default
        qparam_calculator_cls: default
        range_calculator_cls: minmax

quantization_config:
    global_config:
        op_input_spec:
            "*": *int4_activation
        op_output_spec:
            "*": *int4_activation
        op_state_spec:
            weight: *int4_weight
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    # Load from YAML
    config_from_yaml = QuantizerConfig.from_yaml(config_file)

    # Create programmatically
    int4_activation = QuantizationSpec(
        dtype="int4",
        qscheme="symmetric",
        granularity={"type": "per_tensor"},
        fake_quantize_cls="default",
        qparam_calculator_cls="moving_average",
        range_calculator_cls="minmax",
    )

    int4_weight = QuantizationSpec(
        dtype="int4",
        qscheme="symmetric",
        granularity={"type": "per_tensor"},
        fake_quantize_cls="default",
        qparam_calculator_cls="default",
        range_calculator_cls="minmax",
    )

    config_programmatic = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_input_spec={_ALL_TENSORS: int4_activation},
            op_output_spec={_ALL_TENSORS: int4_activation},
            op_state_spec={"weight": int4_weight},
        )
    )

    # Compare
    assert config_from_yaml == config_programmatic


def test_example_4_weight_only_quantization(tmp_path):
    """Test Example 4: Weight-Only Quantization"""

    yaml_content = """
quantization_spec:
    spec1: &int4_weight
        dtype: int4
        qscheme: symmetric
        granularity:
            type: per_channel
            axis: 1
        fake_quantize_cls: default
        qparam_calculator_cls: default
        range_calculator_cls: minmax

quantization_config:
    global_config:
        op_input_spec: null
        op_output_spec: null
        op_state_spec:
            weight: *int4_weight
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    # Load from YAML
    config_from_yaml = QuantizerConfig.from_yaml(config_file)

    # Create programmatically
    int4_weight = QuantizationSpec(
        dtype="int4",
        qscheme="symmetric",
        granularity={"type": "per_channel", "axis": 1},
        fake_quantize_cls="default",
        qparam_calculator_cls="default",
        range_calculator_cls="minmax",
    )

    config_programmatic = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_input_spec=None, op_output_spec=None, op_state_spec={"weight": int4_weight}
        )
    )

    # Compare
    assert config_from_yaml == config_programmatic


def test_example_5_module_type_specific_configuration(tmp_path):
    """Test Example 5: Module Type-Specific Configuration"""

    yaml_content = """
quantization_spec:
    spec1: &int4_weight
        dtype: int4
        qscheme: symmetric
        granularity:
            type: per_channel
            axis: 1
        fake_quantize_cls: default
        qparam_calculator_cls: default
        range_calculator_cls: minmax

quantization_config:
    module_type_configs:
        torch.nn.modules.linear.Linear:
            op_state_spec:
                weight: *int4_weight
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    # Load from YAML
    config_from_yaml = QuantizerConfig.from_yaml(config_file)

    int4_weight = QuantizationSpec(
        dtype="int4",
        qscheme="symmetric",
        granularity={"type": "per_channel", "axis": 1},
        fake_quantize_cls="default",
        qparam_calculator_cls="default",
        range_calculator_cls="minmax",
    )

    config_programmatic = QuantizerConfig(
        module_type_configs={
            "torch.nn.modules.linear.Linear": ModuleQuantizerConfig(
                op_state_spec={"weight": int4_weight},
            )
        },
    )

    # Compare
    assert config_from_yaml == config_programmatic


def test_example_6_disabling_quantization_for_specific_modules(tmp_path):
    """Test Example 6: Disabling Quantization for Specific Modules"""

    yaml_content = """
quantization_config:
    module_name_configs:
        model.encoder.layer.0: null
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    # Load from YAML
    config_from_yaml = QuantizerConfig.from_yaml(config_file)

    config_programmatic = QuantizerConfig(
        module_name_configs={"model.encoder.layer.0": None},
    )

    # Compare
    assert config_from_yaml == config_programmatic


def test_example_7_quantizing_only_specific_module_type(tmp_path):
    """Test Example 7: Quantizing Only a Specific Module Type"""

    yaml_content = """
quantization_spec:
    spec1: &int8_activation
        dtype: int8
        qscheme: symmetric
        granularity:
            type: per_tensor
        fake_quantize_cls: default
        qparam_calculator_cls: moving_average
        range_calculator_cls: minmax

    spec2: &int8_weight
        dtype: int8
        qscheme: symmetric
        granularity:
            type: per_channel
            axis: 1
        fake_quantize_cls: default
        qparam_calculator_cls: default
        range_calculator_cls: minmax

quantization_config:
    global_config: null
    module_type_configs:
        torch.nn.modules.linear.Linear:
            op_input_spec:
                "*": *int8_activation
            op_output_spec:
                "*": *int8_activation
            op_state_spec:
                weight: *int8_weight
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    # Load from YAML
    config_from_yaml = QuantizerConfig.from_yaml(config_file)

    # Create programmatically
    int8_activation = QuantizationSpec(
        dtype="int8",
        qscheme="symmetric",
        granularity={"type": "per_tensor"},
        fake_quantize_cls="default",
        qparam_calculator_cls="moving_average",
        range_calculator_cls="minmax",
    )

    int8_weight = QuantizationSpec(
        dtype="int8",
        qscheme="symmetric",
        granularity={"type": "per_channel", "axis": 1},
        fake_quantize_cls="default",
        qparam_calculator_cls="default",
        range_calculator_cls="minmax",
    )

    config_programmatic = QuantizerConfig(
        global_config=None,
        module_type_configs={
            "torch.nn.modules.linear.Linear": ModuleQuantizerConfig(
                op_input_spec={_ALL_TENSORS: int8_activation},
                op_output_spec={_ALL_TENSORS: int8_activation},
                op_state_spec={"weight": int8_weight},
            )
        },
    )

    # Compare
    assert config_from_yaml == config_programmatic


def test_example_8_module_level_configuration(tmp_path):
    """Test Example 8: Module-Level Configuration - Quantize Boundaries Only"""

    yaml_content = """
quantization_spec:
    spec1: &int8_activation
        dtype: int8
        qscheme: symmetric
        granularity:
            type: per_tensor
        fake_quantize_cls: default
        qparam_calculator_cls: moving_average
        range_calculator_cls: minmax

quantization_config:
    global_config:
        op_input_spec:
            "*": *int8_activation
        op_output_spec:
            "*": *int8_activation

    module_type_configs:
        torch.nn.modules.activation.MultiheadAttention:
            op_input_spec:
            op_output_spec:
            module_input_spec:
                0: *int8_activation
                1: *int8_activation
                2: *int8_activation
            module_output_spec:
                "*": *int8_activation
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    # Load from YAML
    config_from_yaml = QuantizerConfig.from_yaml(config_file)

    # Create programmatically
    int8_activation = QuantizationSpec(
        dtype="int8",
        qscheme="symmetric",
        granularity={"type": "per_tensor"},
        fake_quantize_cls="default",
        qparam_calculator_cls="moving_average",
        range_calculator_cls="minmax",
    )

    config_programmatic = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_input_spec={_ALL_TENSORS: int8_activation},
            op_output_spec={_ALL_TENSORS: int8_activation},
        ),
        module_type_configs={
            "torch.nn.modules.activation.MultiheadAttention": ModuleQuantizerConfig(
                op_input_spec=None,
                op_output_spec=None,
                module_input_spec={
                    0: int8_activation,
                    1: int8_activation,
                    2: int8_activation,
                },
                module_output_spec={
                    _ALL_TENSORS: int8_activation,
                },
            )
        },
    )

    # Compare
    assert config_from_yaml == config_programmatic


def test_example_9_op_type_specific_configuration(tmp_path):
    """Test Example 9: Op Type-Specific Configuration"""

    yaml_content = """
quantization_spec:
    spec1: &int4_activation
        dtype: int4
        qscheme: symmetric
        granularity:
            type: per_tensor
        fake_quantize_cls: default
        qparam_calculator_cls: moving_average
        range_calculator_cls: minmax

quantization_config:
    global_config:
        op_output_spec:
            "*": *int4_activation
        op_state_spec: null
        op_type_config:
            aten.linear.default:
                op_input_spec:
                    "*": *int4_activation

            aten.matmul.default: null
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    # Load from YAML
    config_from_yaml = QuantizerConfig.from_yaml(config_file)

    # Create programmatically
    int4_activation = QuantizationSpec(
        dtype="int4",
        qscheme="symmetric",
        granularity={"type": "per_tensor"},
        fake_quantize_cls="default",
        qparam_calculator_cls="moving_average",
        range_calculator_cls="minmax",
    )

    config_programmatic = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_output_spec={_ALL_TENSORS: int4_activation},
            op_state_spec=None,
            op_type_config={
                "aten.linear.default": OpQuantizerConfig(
                    op_input_spec={_ALL_TENSORS: int4_activation},
                ),
                "aten.matmul.default": None,
            },
        )
    )

    # Compare
    assert config_from_yaml == config_programmatic


def test_example_10_regex_pattern_matching(tmp_path):
    """Test Example 10: Regex Pattern Matching for Module Names"""

    # Test with simple regex patterns
    yaml_content = """
quantization_spec:
    spec1: &int8_activation
        dtype: int8
        qscheme: symmetric
        granularity:
            type: per_tensor
        fake_quantize_cls: default
        qparam_calculator_cls: moving_average
        range_calculator_cls: minmax

quantization_config:
    global_config:
        op_input_spec:
            "*": *int8_activation
        op_output_spec:
            "*": *int8_activation

    module_name_configs:
        model.decoder.layer.*: null
        model.encoder.layer.[0-3]:
            op_input_spec:
                "*": *int8_activation
            op_output_spec: null
            op_state_spec: null
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    # Load from YAML
    config_from_yaml = QuantizerConfig.from_yaml(config_file)

    # Create programmatically - regex is YAML-only feature
    # In programmatic API, regex patterns in keys are treated as literal strings
    int8_activation = QuantizationSpec(
        dtype="int8",
        qscheme="symmetric",
        granularity={"type": "per_tensor"},
        fake_quantize_cls="default",
        qparam_calculator_cls="moving_average",
        range_calculator_cls="minmax",
    )

    config_programmatic = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_input_spec={_ALL_TENSORS: int8_activation},
            op_output_spec={_ALL_TENSORS: int8_activation},
        ),
        module_name_configs={
            "model.decoder.layer.*": None,
            "model.encoder.layer.[0-3]": ModuleQuantizerConfig(
                op_input_spec={_ALL_TENSORS: int8_activation},
                op_output_spec=None,
                op_state_spec=None,
            ),
        },
    )

    # Compare
    assert config_from_yaml == config_programmatic


def test_example_11_complex_hierarchical_configuration(tmp_path):
    """Test Example 11: Complex Hierarchical Configuration"""

    yaml_content = """
quantization_spec:
    spec1: &int4_weight
        dtype: int4
        qscheme: symmetric
        granularity:
            type: per_channel
            axis: 1
        fake_quantize_cls: default
        qparam_calculator_cls: default
        range_calculator_cls: minmax

    spec2: &int8_weight
        dtype: int8
        qscheme: symmetric
        granularity:
            type: per_channel
            axis: 1
        fake_quantize_cls: default
        qparam_calculator_cls: default
        range_calculator_cls: minmax

quantization_config:
    module_type_configs:
        torch.nn.modules.linear.Linear:
            op_state_spec:
                weight: *int4_weight

    module_name_configs:
        model.embeddings:
            op_state_spec:
                weight: *int8_weight

        model.classifier:
            module_output_spec:
                "*": null
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    # Load from YAML
    config_from_yaml = QuantizerConfig.from_yaml(config_file)

    # Create programmatically
    int4_weight = QuantizationSpec(
        dtype="int4",
        qscheme="symmetric",
        granularity={"type": "per_channel", "axis": 1},
        fake_quantize_cls="default",
        qparam_calculator_cls="default",
        range_calculator_cls="minmax",
    )

    int8_weight = QuantizationSpec(
        dtype="int8",
        qscheme="symmetric",
        granularity={"type": "per_channel", "axis": 1},
        fake_quantize_cls="default",
        qparam_calculator_cls="default",
        range_calculator_cls="minmax",
    )

    config_programmatic = QuantizerConfig(
        module_type_configs={
            "torch.nn.modules.linear.Linear": ModuleQuantizerConfig(
                op_state_spec={"weight": int4_weight},
            )
        },
        module_name_configs={
            "model.embeddings": ModuleQuantizerConfig(
                op_state_spec={"weight": int8_weight},
            ),
            "model.classifier": ModuleQuantizerConfig(
                module_output_spec={_ALL_TENSORS: None},
            ),
        },
    )

    # Compare
    assert config_from_yaml == config_programmatic


@pytest.mark.parametrize("execution_mode", ["eager", "graph"])
def test_none_op_type_config_disables_quantization(execution_mode):
    """
    Test that setting an op_type_config value to None disables quantization
    for that op type while other ops remain quantized.
    """

    class AddLinearModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, x):
            return self.linear(x) + x

    config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_type_config={"add": None},
        ),
        execution_mode=execution_mode,
    )

    model = AddLinearModel()
    example_inputs = (torch.randn(1, 4),)
    quantizer = Quantizer(model, config)
    prepared_model = quantizer.prepare(example_inputs)

    if execution_mode == "eager":
        fq_names = {
            name
            for name, module in prepared_model.named_modules()
            if isinstance(module, FakeQuantizeImplBase)
        }
        assert not any("add" in name for name in fq_names), (
            f"Expected no fake quantize modules for 'add', but found: "
            f"{[n for n in fq_names if 'add' in n]}"
        )
        assert any("linear" in name for name in fq_names), (
            f"Expected fake quantize modules for 'linear', but found none in: {fq_names}"
        )
    else:
        # PT2E: verify add node has no output quantization and fewer FQ modules
        baseline_config = QuantizerConfig(global_config=ModuleQuantizerConfig())
        baseline_quantizer = Quantizer(AddLinearModel(), baseline_config)
        baseline_prepared = baseline_quantizer.prepare(example_inputs)
        baseline_fq_count = sum(
            1 for m in baseline_prepared.modules() if isinstance(m, FakeQuantizeImplBase)
        )
        disabled_fq_count = sum(
            1 for m in prepared_model.modules() if isinstance(m, FakeQuantizeImplBase)
        )
        assert disabled_fq_count < baseline_fq_count, (
            f"Expected fewer FQ modules with add disabled ({disabled_fq_count}) "
            f"than baseline ({baseline_fq_count})"
        )
        for node in prepared_model.graph.nodes:
            if node.op == "call_function" and "add" in str(node.target):
                annotation = node.meta.get("quantization_annotation")
                if annotation is not None:
                    assert annotation.output_qspec is None, (
                        "Expected no output quantization for disabled add node"
                    )


@pytest.mark.parametrize("execution_mode", ["eager", "graph"])
def test_none_op_name_config_disables_quantization(execution_mode):
    """
    Test that setting an op_name_config value to None disables quantization
    for the matching op while other ops of the same type remain quantized.
    """

    class TwoAddModule(torch.nn.Module):
        def forward(self, x):
            a = x + x
            b = a + a
            return b

    config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_name_config={"add": None},
        ),
        execution_mode=execution_mode,
    )

    model = TwoAddModule()
    example_inputs = (torch.randn(1, 4),)
    quantizer = Quantizer(model, config)
    prepared_model = quantizer.prepare(example_inputs)

    if execution_mode == "eager":
        fq_names = {
            name
            for name, module in prepared_model.named_modules()
            if isinstance(module, FakeQuantizeImplBase)
        }
        assert not any(name.startswith("add_quantize") for name in fq_names), (
            f"Expected no fake quantize for first 'add', but found: "
            f"{[n for n in fq_names if n.startswith('add_quantize')]}"
        )
        assert any("add_1" in name for name in fq_names), (
            f"Expected fake quantize modules for 'add_1', but found none in: {fq_names}"
        )
    else:
        # PT2E: verify first add has no annotation, second add is quantized
        for node in prepared_model.graph.nodes:
            if node.op == "call_function" and node.name == "add":
                annotation = node.meta.get("quantization_annotation")
                if annotation is not None:
                    assert annotation.output_qspec is None, (
                        "Expected no output quantization for disabled first 'add'"
                    )
            elif node.op == "call_function" and node.name == "add_1":
                annotation = node.meta.get("quantization_annotation")
                assert annotation is not None, "Expected quantization annotation for second 'add_1'"


# ---------------------------------------------------------------------------
# set_execution_mode
# ---------------------------------------------------------------------------


def test_set_execution_mode_updates_field():
    config = QuantizerConfig()
    assert config.execution_mode is ExecutionMode.GRAPH
    config.set_execution_mode(ExecutionMode.EAGER)
    assert config.execution_mode is ExecutionMode.EAGER


def test_set_execution_mode_returns_self_for_chaining():
    config = QuantizerConfig()
    assert config.set_execution_mode(ExecutionMode.EAGER) is config


def test_set_execution_mode_accepts_string_value():
    config = QuantizerConfig()
    config.set_execution_mode("eager")
    assert config.execution_mode is ExecutionMode.EAGER


def test_set_execution_mode_rejects_invalid_value():
    config = QuantizerConfig()
    with pytest.raises(ValueError, match="not a valid ExecutionMode"):
        config.set_execution_mode("not-a-mode")


def test_set_execution_mode_matches_preset_kwarg():
    """Post-construction setter must produce the same config as the kwarg form."""
    via_setter = QuantizerConfig.presets.w4().set_execution_mode(ExecutionMode.EAGER)
    via_kwarg = QuantizerConfig.presets.w4(execution_mode=ExecutionMode.EAGER)
    assert via_setter == via_kwarg


# ---------------------------------------------------------------------------
# QATSchedule config tests
# ---------------------------------------------------------------------------


class TestQATSchedule:
    """Tests for the QATSchedule pydantic model."""

    def test_default_values(self):
        schedule = QATSchedule()
        assert schedule.enable_observer == 0
        assert schedule.enable_fake_quant == 0
        assert schedule.disable_observer is None

    def test_custom_values(self):
        schedule = QATSchedule(
            enable_observer=0,
            enable_fake_quant=500,
            disable_observer=1500,
        )
        assert schedule.enable_observer == 0
        assert schedule.enable_fake_quant == 500
        assert schedule.disable_observer == 1500

    def test_disable_observer_none_valid(self):
        schedule = QATSchedule(
            enable_observer=0,
            enable_fake_quant=100,
            disable_observer=None,
        )
        assert schedule.disable_observer is None

    def test_validation_negative_enable_observer(self):
        with pytest.raises(ValidationError):
            QATSchedule(enable_observer=-1)

    def test_validation_fake_quant_lt_observer(self):
        with pytest.raises(ValidationError):
            QATSchedule(enable_observer=10, enable_fake_quant=5)

    @pytest.mark.parametrize(
        "enable_observer,disable_observer",
        [(5, 5), (10, 5)],
        ids=["equal", "less_than"],
    )
    def test_validation_disable_observer_le_enable_observer(
        self, enable_observer, disable_observer
    ):
        with pytest.raises(ValidationError):
            QATSchedule(
                enable_observer=enable_observer,
                enable_fake_quant=enable_observer,
                disable_observer=disable_observer,
            )

    def test_disable_observer_equal_to_fake_quant_valid(self):
        """disable_observer == enable_fake_quant is allowed."""
        schedule = QATSchedule(
            enable_observer=0,
            enable_fake_quant=100,
            disable_observer=100,
        )
        assert schedule.disable_observer == 100

    def test_validation_disable_observer_lt_fake_quant(self):
        with pytest.raises(ValidationError):
            QATSchedule(
                enable_observer=0,
                enable_fake_quant=10,
                disable_observer=5,
            )

    def test_from_dict(self):
        schedule = QATSchedule.model_validate(
            {
                "enable_observer": 0,
                "enable_fake_quant": 500,
                "disable_observer": 1500,
            }
        )
        assert schedule.enable_observer == 0
        assert schedule.enable_fake_quant == 500
        assert schedule.disable_observer == 1500


class TestQATScheduleInConfig:
    """Tests for QATSchedule integration in QuantizerConfig hierarchy."""

    def test_schedule_in_config_hierarchy(self):
        """QATSchedule can be set at global, module_type, and module_name levels."""
        global_schedule = QATSchedule(enable_observer=0, enable_fake_quant=10, disable_observer=20)
        lin_schedule = QATSchedule(enable_observer=0, enable_fake_quant=5)
        conv_schedule = QATSchedule(enable_observer=0, enable_fake_quant=2)
        name_schedule = QATSchedule(enable_observer=0, enable_fake_quant=3)

        lin_type = "torch.nn.modules.linear.Linear"
        conv_type = "torch.nn.modules.conv.Conv2d"

        config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(qat_schedule=global_schedule),
            module_type_configs={
                lin_type: ModuleQuantizerConfig(qat_schedule=lin_schedule),
                conv_type: ModuleQuantizerConfig(qat_schedule=conv_schedule),
            },
            module_name_configs={
                "conv": ModuleQuantizerConfig(qat_schedule=name_schedule),
            },
        )

        assert config.global_config.qat_schedule == global_schedule
        assert config.module_type_configs[lin_type].qat_schedule == lin_schedule
        assert config.module_type_configs[conv_type].qat_schedule == conv_schedule
        assert config.module_name_configs["conv"].qat_schedule == name_schedule
