# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import io
from typing import ClassVar

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError

from coreai_opt._utils.config_utils import ALL_TENSORS, ConfigLevel
from coreai_opt.common import CompressionType
from coreai_opt.config import (
    CompressionConfig,
    ModuleCompressionConfig,
    OpCompressionConfig,
)
from coreai_opt.config.spec import CompressionSpec

# Test fixtures: Concrete implementations of abstract classes


class DummySpec(CompressionSpec):
    """Dummy compression spec for testing."""

    _compression_type = CompressionType.QUANTIZATION
    value: int = 42


class OpFooCompressionConfig(OpCompressionConfig[DummySpec]):
    """Concrete OpCompressionConfig for testing."""

    @classmethod
    def get_default_input_spec(cls) -> dict[str | int, DummySpec | None]:
        return {"*": DummySpec(value=1)}

    @classmethod
    def get_default_output_spec(cls) -> dict[str | int, DummySpec | None]:
        return {0: DummySpec(value=2)}

    @classmethod
    def get_default_state_spec(cls) -> dict[str, DummySpec | None]:
        return {"weight": DummySpec(value=3)}


class ModuleFooCompressionConfig(ModuleCompressionConfig[OpFooCompressionConfig, DummySpec]):
    compression_ratio: float = 0.5
    method: str = "default"
    enabled: bool = True


class FooCompressionConfig(CompressionConfig[ModuleFooCompressionConfig]):
    _CONFIG_KEY: ClassVar[str] = "foo_config"
    _SPEC_KEY: ClassVar[str] = "foo_spec"


DEFAULT_MODULE_CONFIG = ModuleFooCompressionConfig()
DISABLED_MODULE_CONFIG = ModuleFooCompressionConfig(
    op_input_spec=None,
    op_output_spec=None,
    op_state_spec=None,
    module_input_spec=None,
    module_output_spec=None,
    module_state_spec=None,
)


# ===========================
# OpCompressionConfig Tests
# ===========================


def test_op_compression_config_defaults():
    """Test that default specs are applied when fields are omitted."""
    config = OpFooCompressionConfig()

    # Defaults should be applied from the class methods
    assert config.op_input_spec == {"*": DummySpec(value=1)}
    assert config.op_output_spec == {0: DummySpec(value=2)}
    assert config.op_state_spec == {"weight": DummySpec(value=3)}


def test_op_compression_config_explicit_none():
    """Explicit None disables the category, as ``{"*": None}``."""
    config = OpFooCompressionConfig(op_input_spec=None, op_output_spec=None, op_state_spec=None)

    # None is shorthand for "compress no tensor here", kept at the tensor level so
    # it stays distinguishable from {}, which names no tensor and says nothing.
    assert config.op_input_spec == {ALL_TENSORS: None}
    assert config.op_output_spec == {ALL_TENSORS: None}
    assert config.op_state_spec == {ALL_TENSORS: None}


def test_op_compression_config_with_specs():
    """Test creating OpCompressionConfig with actual specs."""
    custom_input_spec = {0: DummySpec(value=10), 1: None}
    custom_output_spec = {"output": DummySpec(value=20)}
    custom_state_spec = {"bias": DummySpec(value=30), "weight": None}

    config = OpFooCompressionConfig(
        op_input_spec=custom_input_spec,
        op_output_spec=custom_output_spec,
        op_state_spec=custom_state_spec,
    )

    assert config.op_input_spec == custom_input_spec
    assert config.op_output_spec == custom_output_spec
    assert config.op_state_spec == custom_state_spec


def test_op_compression_config_validation():
    """Test field validation (extra fields forbidden)."""
    with pytest.raises(ValidationError):
        OpFooCompressionConfig(
            op_input_spec={}, op_output_spec={}, op_state_spec={}, extra_field="not_allowed"
        )


# ===========================
# ModuleCompressionConfig Tests
# ===========================


def test_module_compression_config_defaults_from_op_config():
    """Test that defaults are inherited from OpConfigT."""
    config = ModuleFooCompressionConfig()

    # Should get defaults from OpFooCompressionConfig
    assert config.op_input_spec == {"*": DummySpec(value=1)}
    assert config.op_output_spec == {0: DummySpec(value=2)}
    assert config.op_state_spec == {"weight": DummySpec(value=3)}

    # Module-level specs should be empty dicts
    assert config.op_type_config == {}
    assert config.op_name_config == {}
    assert config.module_input_spec == {}
    assert config.module_output_spec == {}
    assert config.module_state_spec == {}


def test_module_compression_config_all_fields():
    """Test all fields can be set independently."""
    op_config = OpFooCompressionConfig(
        op_input_spec={0: DummySpec(value=5)}, op_output_spec={}, op_state_spec={}
    )

    config = ModuleFooCompressionConfig(
        compression_ratio=0.8,
        method="custom",
        enabled=False,
        op_input_spec={0: DummySpec(value=10)},
        op_output_spec={1: DummySpec(value=20)},
        op_state_spec={"bias": DummySpec(value=30)},
        op_type_config={"linear": op_config},
        op_name_config={"layer1": op_config},
        module_input_spec={0: DummySpec(value=40)},
        module_output_spec={0: DummySpec(value=50)},
        module_state_spec={"weight": DummySpec(value=60)},
    )

    assert config.compression_ratio == 0.8
    assert config.method == "custom"
    assert config.enabled is False
    assert config.op_input_spec == {0: DummySpec(value=10)}
    assert config.op_output_spec == {1: DummySpec(value=20)}
    assert config.op_state_spec == {"bias": DummySpec(value=30)}
    assert config.op_type_config == {"linear": op_config}
    assert config.op_name_config == {"layer1": op_config}
    assert config.module_input_spec == {0: DummySpec(value=40)}
    assert config.module_output_spec == {0: DummySpec(value=50)}
    assert config.module_state_spec == {"weight": DummySpec(value=60)}


def test_module_compression_config_explicit_none():
    """Test None conversion behavior."""
    config = ModuleFooCompressionConfig(
        op_input_spec=None,
        op_output_spec=None,
        op_state_spec=None,
        op_type_config=None,
        op_name_config=None,
        module_input_spec=None,
        module_output_spec=None,
        module_state_spec=None,
    )

    # Op-level specs disable at the tensor level; the rest are simply empty.
    assert config.op_input_spec == {ALL_TENSORS: None}
    assert config.op_output_spec == {ALL_TENSORS: None}
    assert config.op_state_spec == {ALL_TENSORS: None}
    assert config.op_type_config == {}
    assert config.op_name_config == {}
    assert config.module_input_spec == {}
    assert config.module_output_spec == {}
    assert config.module_state_spec == {}


def test_module_compression_config_none_op_type_config():
    """Test that None values in op_type_config are normalized to disabled OpConfigs."""
    config = ModuleFooCompressionConfig(
        op_type_config={"linear": None, "conv": OpFooCompressionConfig()}
    )

    # None should be normalized to an OpConfig with every category disabled
    assert isinstance(config.op_type_config["linear"], OpFooCompressionConfig)
    assert config.op_type_config["linear"].op_input_spec == {ALL_TENSORS: None}
    assert config.op_type_config["linear"].op_output_spec == {ALL_TENSORS: None}
    assert config.op_type_config["linear"].op_state_spec == {ALL_TENSORS: None}

    # Non-None should be unchanged (has defaults)
    assert config.op_type_config["conv"].op_input_spec == {"*": DummySpec(value=1)}


def test_module_compression_config_none_op_name_config():
    """Test that None values in op_name_config are normalized to disabled OpConfigs."""
    config = ModuleFooCompressionConfig(
        op_name_config={"layer1.weight": None, "layer2.weight": OpFooCompressionConfig()}
    )

    # None should be normalized to an OpConfig with every category disabled
    assert isinstance(config.op_name_config["layer1.weight"], OpFooCompressionConfig)
    assert config.op_name_config["layer1.weight"].op_input_spec == {ALL_TENSORS: None}
    assert config.op_name_config["layer1.weight"].op_output_spec == {ALL_TENSORS: None}
    assert config.op_name_config["layer1.weight"].op_state_spec == {ALL_TENSORS: None}

    # Non-None should be unchanged (has defaults)
    assert config.op_name_config["layer2.weight"].op_input_spec == {"*": DummySpec(value=1)}


# ===========================
# CompressionConfig Tests
# ===========================


def test_default_config_init():
    """Test that default global_config is automatically created."""
    config = FooCompressionConfig()
    # global_config should be automatically created from the generic type parameter
    assert config.global_config is not None
    assert isinstance(config.global_config, ModuleFooCompressionConfig)
    assert config.module_name_configs == {}
    assert config.module_type_configs == {}
    # The default module config should match DEFAULT_MODULE_CONFIG
    assert config.global_config == DEFAULT_MODULE_CONFIG


def test_default_config_with_explicit_none_global():
    """Test explicitly setting global_config to None normalizes to disabled config."""
    config = FooCompressionConfig(global_config=None)
    assert config.global_config == DISABLED_MODULE_CONFIG
    assert config.module_name_configs == {}
    assert config.module_type_configs == {}
    assert config.get_module_config("conv", None) == DISABLED_MODULE_CONFIG


def test_set_global_config():
    """Test setting global config."""
    config = FooCompressionConfig(global_config=None)
    assert config.global_config == DISABLED_MODULE_CONFIG

    custom_config = ModuleFooCompressionConfig(compression_ratio=0.9)
    config.set_global(custom_config)
    assert config.global_config == custom_config
    assert config.get_module_config("conv", None) == custom_config


def test_set_module_type_config():
    config = FooCompressionConfig()
    assert config.module_type_configs == {}
    config.set_module_type(nn.Conv2d, DEFAULT_MODULE_CONFIG)
    config.set_module_type("torch.nn.modules.linear.Linear", None)
    assert len(config.module_type_configs) == 2
    assert "torch.nn.modules.conv.Conv2d" in config.module_type_configs
    assert "torch.nn.modules.linear.Linear" in config.module_type_configs

    mod1 = nn.Conv2d(10, 10, 3)
    mod2 = nn.Linear(10, 10)
    assert config.get_module_config("conv", mod1) == DEFAULT_MODULE_CONFIG
    assert config.get_module_config("linear", mod2) == DISABLED_MODULE_CONFIG


def test_normalize_module_type():
    config = FooCompressionConfig()  # Create an instance to access the method

    # Test with valid type input
    assert config._normalize_module_type(nn.Linear) == "torch.nn.modules.linear.Linear"

    # Test with valid string input
    assert (
        config._normalize_module_type("torch.nn.modules.linear.Linear")
        == "torch.nn.modules.linear.Linear"
    )

    # Test invalid string input
    with pytest.raises(ValueError):
        config._normalize_module_type("Linear")

    # Test invalid type input
    with pytest.raises(TypeError):
        config._normalize_module_type(123)

    with pytest.raises(TypeError):
        config._normalize_module_type(object)


def test_set_module_type_config_invalid():
    # validation in init
    with pytest.raises(ValueError):
        config = FooCompressionConfig(module_type_configs={torch.cat: DEFAULT_MODULE_CONFIG})

    # validation in setter
    config = FooCompressionConfig()
    with pytest.raises(TypeError):
        config.set_module_type(nn.Parameter(), DEFAULT_MODULE_CONFIG)


def test_set_module_name_config():
    config = FooCompressionConfig()
    assert config.module_name_configs == {}
    config.set_module_name("conv1", DEFAULT_MODULE_CONFIG)
    config.set_module_name("linear1", None)
    assert len(config.module_name_configs) == 2
    assert config.get_module_config("conv1", None) == DEFAULT_MODULE_CONFIG
    assert config.get_module_config("linear1", None) == DISABLED_MODULE_CONFIG


def test_regex_module_name_matching():
    config = FooCompressionConfig(global_config=None)  # No global config
    config.set_module_name(r"layer\d+\.weight", DEFAULT_MODULE_CONFIG)

    # Test regex matching for module names
    module_config = config.get_module_config("layer1.weight", None)
    assert module_config == DEFAULT_MODULE_CONFIG

    module_config = config.get_module_config("layer42.weight", nn.Module())
    assert module_config == DEFAULT_MODULE_CONFIG

    # Non-matching name should fall back to disabled global config
    module_config = config.get_module_config("other_name", nn.Module())
    assert module_config == DISABLED_MODULE_CONFIG


def test_config_resolution():
    default_config = DEFAULT_MODULE_CONFIG
    conv_config = ModuleFooCompressionConfig(compression_ratio=0.75)
    conv1_config = ModuleFooCompressionConfig(compression_ratio=0.95)

    config = FooCompressionConfig(
        global_config=default_config,
        module_type_configs={nn.Conv2d: conv_config},
        module_name_configs={"conv1": conv1_config},
    )

    conv_mod = nn.Conv2d(10, 10, 3)
    linear_mod = nn.Linear(10, 10)
    assert config.get_module_config("conv1", conv_mod) == conv1_config
    assert config.get_module_config("conv2", conv_mod) == conv_config
    assert config.get_module_config("linear", linear_mod) == default_config


def test_global_config_cannot_have_module_specs():
    """Test that global_config cannot have module-level specs."""
    # Test with module_input_spec
    with pytest.raises(ValidationError, match="global_config cannot have"):
        FooCompressionConfig(
            global_config=ModuleFooCompressionConfig(module_input_spec={0: DummySpec(value=1)})
        )

    # Test with module_output_spec
    with pytest.raises(ValidationError, match="global_config cannot have"):
        FooCompressionConfig(
            global_config=ModuleFooCompressionConfig(module_output_spec={0: DummySpec(value=1)})
        )

    # Test with module_state_spec
    with pytest.raises(ValidationError, match="global_config cannot have"):
        FooCompressionConfig(
            global_config=ModuleFooCompressionConfig(
                module_state_spec={"weight": DummySpec(value=1)}
            )
        )


def test_dict_serialization():
    default_config = DEFAULT_MODULE_CONFIG
    conv_config = ModuleFooCompressionConfig(compression_ratio=0.75)
    conv1_config = ModuleFooCompressionConfig(compression_ratio=0.95)

    config = FooCompressionConfig(
        global_config=default_config,
        module_type_configs={nn.Conv2d: conv_config},
        module_name_configs={"conv1": conv1_config},
    )

    # Serialize to dict
    config_dict = config.to_dict()

    # Deserialize from dict
    new_config = FooCompressionConfig.from_dict(config_dict)

    # Ensure the two configs match
    assert new_config.global_config == config.global_config
    assert new_config.module_type_configs == config.module_type_configs
    assert new_config.module_name_configs == config.module_name_configs


def test_from_yaml():
    yaml_content = """
    foo_config:
        global_config:
            compression_ratio: 0.9
            method: yaml_global
            enabled: true
        module_type_configs:
            torch.nn.modules.linear.Linear:
                compression_ratio: 0.5
                method: yaml_linear
                enabled: false
        module_name_configs: {}
    """

    yaml_stream = io.StringIO(yaml_content)
    config = FooCompressionConfig.from_yaml(yaml_stream)

    assert config.global_config is not None
    global_config = config.global_config
    assert global_config.compression_ratio == 0.9
    assert global_config.method == "yaml_global"
    assert global_config.enabled is True

    assert "torch.nn.modules.linear.Linear" in config.module_type_configs
    linear_config = config.module_type_configs["torch.nn.modules.linear.Linear"]
    assert linear_config.compression_ratio == 0.5
    assert linear_config.method == "yaml_linear"
    assert linear_config.enabled is False

    assert len(config.module_name_configs) == 0


def test_generic_type_extraction():
    """Test that _get_module_config_class works correctly."""
    module_config_cls = FooCompressionConfig._get_module_config_class()
    assert module_config_cls is ModuleFooCompressionConfig

    # Test with base CompressionConfig (no generic parameter)
    base_cls = CompressionConfig._get_module_config_class()
    assert base_cls is None


def test_get_fake_module_kwargs():
    """
    Test that _get_fake_module_kwargs returns only subclass-defined fields.
    """
    # Create a config with both base class fields and subclass fields
    config = ModuleFooCompressionConfig(
        # Base class fields
        op_input_spec={0: DummySpec(value=1)},
        op_output_spec={1: DummySpec(value=2)},
        op_state_spec={"weight": DummySpec(value=3)},
        module_input_spec={0: DummySpec(value=4)},
        module_output_spec={1: DummySpec(value=5)},
        module_state_spec={"bias": DummySpec(value=6)},
        # Subclass-specific fields
        compression_ratio=0.75,
        method="custom_method",
        enabled=False,
    )

    # Get compressor-specific settings
    settings = config._get_fake_module_kwargs()

    # Should only contain subclass-defined fields
    assert "compression_ratio" in settings
    assert "method" in settings
    assert "enabled" in settings
    assert settings["compression_ratio"] == 0.75
    assert settings["method"] == "custom_method"
    assert settings["enabled"] is False

    # Should NOT contain base class fields
    assert "op_input_spec" not in settings
    assert "op_output_spec" not in settings
    assert "op_state_spec" not in settings
    assert "op_type_config" not in settings
    assert "op_name_config" not in settings
    assert "module_input_spec" not in settings
    assert "module_output_spec" not in settings
    assert "module_state_spec" not in settings


def test_from_dict_with_config_key():
    """Test from_dict with _CONFIG_KEY defined (unwrapping behavior)."""

    # Create a test subclass with _CONFIG_KEY
    class TestCompressionConfig(CompressionConfig[ModuleFooCompressionConfig]):
        _CONFIG_KEY: ClassVar[str] = "test_config"
        _SPEC_KEY: ClassVar[str] = "test_spec"

    # Test with valid nested config
    config_dict = {
        "test_config": {"global_config": {"compression_ratio": 0.8, "method": "nested_method"}}
    }
    config = TestCompressionConfig.from_dict(config_dict)

    assert config.global_config is not None
    assert config.global_config.compression_ratio == 0.8
    assert config.global_config.method == "nested_method"


def test_from_dict_with_config_key_missing():
    """Test from_dict raises error when required _CONFIG_KEY is missing."""

    class TestCompressionConfig(CompressionConfig[ModuleFooCompressionConfig]):
        _CONFIG_KEY: ClassVar[str] = "test_config"
        _SPEC_KEY: ClassVar[str] = "test_spec"

    config_dict = {"test_spec": {}}

    with pytest.raises(RuntimeError, match="Did not find 'test_config'"):
        TestCompressionConfig.from_dict(config_dict)


def test_from_dict_with_unexpected_key():
    """Test from_dict raises error when unexpected keys are present."""

    class TestCompressionConfig(CompressionConfig[ModuleFooCompressionConfig]):
        _CONFIG_KEY: ClassVar[str] = "test_config"
        _SPEC_KEY: ClassVar[str] = "test_spec"

    config_dict = {"test_config": {}, "unexpected_key": {}}

    with pytest.raises(RuntimeError, match="Found unexpected key 'unexpected_key'"):
        TestCompressionConfig.from_dict(config_dict)


def test_from_dict_with_spec_key_allowed():
    """Test from_dict allows _SPEC_KEY if defined."""

    class TestCompressionConfig(CompressionConfig[ModuleFooCompressionConfig]):
        _CONFIG_KEY: ClassVar[str] = "test_config"
        _SPEC_KEY: ClassVar[str] = "test_spec"

    config_dict = {
        "test_config": {"global_config": {"compression_ratio": 0.9}},
        "test_spec": {"some_spec": "value"},
    }

    # Should not raise an error because test_spec is an allowed key
    config = TestCompressionConfig.from_dict(config_dict)
    assert config.global_config.compression_ratio == 0.9


def test_from_dict_with_undefined_config_key():
    """Test from_dict raises error when _CONFIG_KEY is not defined."""

    class TestCompressionConfig(CompressionConfig[ModuleFooCompressionConfig]):
        # _CONFIG_KEY intentionally not defined
        _SPEC_KEY: ClassVar[str] = "test_spec"

    config_dict = {"test_config": {}}

    with pytest.raises(RuntimeError, match="must define _CONFIG_KEY"):
        TestCompressionConfig.from_dict(config_dict)


def test_from_dict_with_undefined_spec_key():
    """Test from_dict raises error when _SPEC_KEY is not defined."""

    class TestCompressionConfig(CompressionConfig[ModuleFooCompressionConfig]):
        _CONFIG_KEY: ClassVar[str] = "test_config"
        # _SPEC_KEY intentionally not defined

    config_dict = {"test_config": {}}

    with pytest.raises(RuntimeError, match="must define _SPEC_KEY"):
        TestCompressionConfig.from_dict(config_dict)


def test_digit_str_keys_coerced_to_int():
    """Digit-string keys in int-capable spec fields are coerced to int."""
    config = ModuleFooCompressionConfig(
        op_input_spec={"0": DummySpec(value=1), "1": DummySpec(value=2)},
        module_input_spec={"0": DummySpec(value=3)},
        module_output_spec={"0": DummySpec(value=4), "*": DummySpec(value=5)},
    )
    assert list(config.op_input_spec.keys()) == [0, 1]
    assert list(config.module_input_spec.keys()) == [0]
    assert set(config.module_output_spec.keys()) == {0, "*"}

    # str-only fields are NOT affected
    config2 = ModuleFooCompressionConfig(
        op_state_spec={"0": DummySpec(value=1), "weight": DummySpec(value=2)},
    )
    assert "0" in config2.op_state_spec


class TestBuildModuleConfigMap:
    """Test suite for build_module_config_dict functionality."""

    def create_test_model(self):
        """Create a test model with nested modules for testing."""

        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3)
                self.conv2 = nn.Conv2d(16, 32, 3)
                self.linear1 = nn.Linear(32, 64)
                self.linear2 = nn.Linear(64, 10)

                # Nested module structure
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3),  # features.0
                    nn.ReLU(),  # features.1
                    nn.Conv2d(64, 128, 3),  # features.2
                )

                # Deeper nesting
                self.classifier = nn.Sequential(
                    nn.Linear(128, 256),  # classifier.0
                    nn.Dropout(0.5),  # classifier.1
                    nn.Linear(256, 10),  # classifier.2
                )

            def forward(self, x):
                # Forward function doesn't matter for testing purposes
                return x

        return TestModel()

    def test_global_config_only(self):
        """Test that global config is applied to all modules."""
        model = self.create_test_model()
        global_config = ModuleFooCompressionConfig(compression_ratio=0.7)

        config = FooCompressionConfig(global_config=global_config)
        config_map = config.build_module_config_dict(model)

        # Check structure
        assert set(config_map.keys()) == {
            ConfigLevel.MODULE_NAME,
            ConfigLevel.MODULE_TYPE,
            ConfigLevel.GLOBAL,
        }
        assert config_map[ConfigLevel.MODULE_NAME] == {}
        assert config_map[ConfigLevel.MODULE_TYPE] == {}

        # All modules should have global config
        global_configs = config_map[ConfigLevel.GLOBAL]
        expected_modules = {
            "",
            "conv1",
            "conv2",
            "linear1",
            "linear2",
            "features",
            "features.0",
            "features.1",
            "features.2",
            "classifier",
            "classifier.0",
            "classifier.1",
            "classifier.2",
        }
        assert set(global_configs.keys()) == expected_modules

        # All should have the same global config
        for module_config in global_configs.values():
            assert module_config == global_config

    def test_module_type_config(self):
        """Test module type configuration."""
        model = self.create_test_model()
        conv_config = ModuleFooCompressionConfig(compression_ratio=0.8)
        linear_config = ModuleFooCompressionConfig(compression_ratio=0.6)

        config = FooCompressionConfig(
            module_type_configs={nn.Conv2d: conv_config, nn.Linear: linear_config}
        )
        config_map = config.build_module_config_dict(model)

        # Check module type configs
        type_configs = config_map[ConfigLevel.MODULE_TYPE]

        # Conv2d modules should have conv_config
        conv_modules = ["conv1", "conv2", "features.0", "features.2"]
        for module_name in conv_modules:
            assert type_configs[module_name] == conv_config

        # Linear modules should have linear_config
        linear_modules = ["linear1", "linear2", "classifier.0", "classifier.2"]
        for module_name in linear_modules:
            assert type_configs[module_name] == linear_config

        # Other modules should not be in module type configs
        other_modules = ["", "features", "features.1", "classifier", "classifier.1"]
        for module_name in other_modules:
            assert module_name not in type_configs

    def test_module_name_config_exact_match(self):
        """Test exact module name matching."""
        model = self.create_test_model()
        conv1_config = ModuleFooCompressionConfig(compression_ratio=0.9)
        features_config = ModuleFooCompressionConfig(compression_ratio=0.3)

        config = FooCompressionConfig(
            module_name_configs={"conv1": conv1_config, "features": features_config}
        )
        config_map = config.build_module_config_dict(model)

        name_configs = config_map[ConfigLevel.MODULE_NAME]

        # conv1 and its children should not be affected by features config
        # since conv1 is at top level
        assert name_configs["conv1"] == conv1_config

        # features and all its children should have features_config due to recursion
        expected_features_modules = ["features", "features.0", "features.1", "features.2"]
        for module_name in expected_features_modules:
            assert name_configs[module_name] == features_config

    def test_module_name_config_regex_match(self):
        """Test regex patterns in module names."""
        model = self.create_test_model()
        regex_config = ModuleFooCompressionConfig(compression_ratio=0.4)

        config = FooCompressionConfig(
            module_name_configs={
                r"features\.\d+": regex_config,  # Match modules in features
                r".*linear\d+": regex_config,  # Match linear1, linear2
            }
        )
        config_map = config.build_module_config_dict(model)

        name_configs = config_map[ConfigLevel.MODULE_NAME]

        # Should match features.0, features.1, features.2
        features_children = ["features.0", "features.1", "features.2"]
        for module_name in features_children:
            assert name_configs[module_name] == regex_config

        # Should match linear1, linear2
        linear_modules = ["linear1", "linear2"]
        for module_name in linear_modules:
            assert name_configs[module_name] == regex_config

        # Should not match parent features module or other linears in classifier
        assert "features" not in name_configs
        assert "classifier.0" not in name_configs
        assert "classifier.2" not in name_configs

    def test_priority_precedence(self):
        """Test that module_name > module_type > global precedence is respected."""
        model = self.create_test_model()

        global_config = ModuleFooCompressionConfig(compression_ratio=0.1)
        conv_type_config = ModuleFooCompressionConfig(compression_ratio=0.2)
        conv1_name_config = ModuleFooCompressionConfig(compression_ratio=0.3)

        config = FooCompressionConfig(
            global_config=global_config,
            module_type_configs={nn.Conv2d: conv_type_config},
            module_name_configs={"conv1": conv1_name_config},
        )
        config_map = config.build_module_config_dict(model)

        # conv1 should only appear in MODULE_NAME (highest priority)
        assert config_map[ConfigLevel.MODULE_NAME]["conv1"] == conv1_name_config
        assert "conv1" not in config_map[ConfigLevel.MODULE_TYPE]
        assert "conv1" not in config_map[ConfigLevel.GLOBAL]

        # conv2 should only appear in MODULE_TYPE (middle priority)
        assert "conv2" not in config_map[ConfigLevel.MODULE_NAME]
        assert config_map[ConfigLevel.MODULE_TYPE]["conv2"] == conv_type_config
        assert "conv2" not in config_map[ConfigLevel.GLOBAL]

        # linear1 should only appear in GLOBAL (lowest priority)
        assert "linear1" not in config_map[ConfigLevel.MODULE_NAME]
        assert "linear1" not in config_map[ConfigLevel.MODULE_TYPE]
        assert config_map[ConfigLevel.GLOBAL]["linear1"] == global_config

    def test_multiple_regex_matches_same_level(self):
        """Test that later configs overwrite earlier ones within the same level."""
        model = self.create_test_model()

        first_config = ModuleFooCompressionConfig(compression_ratio=0.5)
        second_config = ModuleFooCompressionConfig(compression_ratio=0.7)

        config = FooCompressionConfig(
            module_name_configs={
                r"conv.*": first_config,  # Matches conv1, conv2
                r"conv1": second_config,  # Also matches conv1 - should win
            }
        )
        config_map = config.build_module_config_dict(model)

        name_configs = config_map[ConfigLevel.MODULE_NAME]

        # conv1 should have second_config (later match wins)
        assert name_configs["conv1"] == second_config

        # conv2 should have first_config
        assert name_configs["conv2"] == first_config

    def test_recursive_child_module_application(self):
        """Test that configs are applied recursively to child modules."""
        model = self.create_test_model()

        features_config = ModuleFooCompressionConfig(compression_ratio=0.6)
        classifier_config = ModuleFooCompressionConfig(compression_ratio=0.8)

        config = FooCompressionConfig(
            module_name_configs={"features": features_config, "classifier": classifier_config}
        )
        config_map = config.build_module_config_dict(model)

        name_configs = config_map[ConfigLevel.MODULE_NAME]

        # features and all its children should have features_config
        features_modules = ["features", "features.0", "features.1", "features.2"]
        for module_name in features_modules:
            assert name_configs[module_name] == features_config

        # classifier and all its children should have classifier_config
        classifier_modules = ["classifier", "classifier.0", "classifier.1", "classifier.2"]
        for module_name in classifier_modules:
            assert name_configs[module_name] == classifier_config

    def test_child_module_priority_override(self):
        """Test that child modules respect priority rules even with recursion."""
        model = self.create_test_model()

        # Parent module gets module_name config (highest priority)
        features_config = ModuleFooCompressionConfig(compression_ratio=0.6)
        # Child module gets more specific module_name config (should override)
        features_0_config = ModuleFooCompressionConfig(compression_ratio=0.9)
        # Type config for Conv2d (lower priority)
        conv_type_config = ModuleFooCompressionConfig(compression_ratio=0.3)

        config = FooCompressionConfig(
            module_type_configs={nn.Conv2d: conv_type_config},
            module_name_configs={
                "features": features_config,
                "features.0": features_0_config,  # More specific, should override
            },
        )
        config_map = config.build_module_config_dict(model)

        name_configs = config_map[ConfigLevel.MODULE_NAME]

        # features.0 should have its specific config, not the parent's
        assert name_configs["features.0"] == features_0_config

        # Other features children should have parent config
        assert name_configs["features.1"] == features_config
        assert name_configs["features.2"] == features_config
        assert name_configs["features"] == features_config

    def test_none_configs(self):
        """Test handling of None configurations (normalized to disabled configs)."""
        model = self.create_test_model()

        config = FooCompressionConfig(
            global_config=None,
            module_name_configs={"conv1": None},
            module_type_configs={nn.Linear: None},
        )
        config_map = config.build_module_config_dict(model)

        # conv1 should have disabled config in module name configs
        assert config_map[ConfigLevel.MODULE_NAME]["conv1"] == DISABLED_MODULE_CONFIG

        # Linear modules should have disabled config in module type configs
        assert config_map[ConfigLevel.MODULE_TYPE]["linear1"] == DISABLED_MODULE_CONFIG
        assert config_map[ConfigLevel.MODULE_TYPE]["linear2"] == DISABLED_MODULE_CONFIG

        # Other modules should have disabled config in global configs
        assert config_map[ConfigLevel.GLOBAL]["conv2"] == DISABLED_MODULE_CONFIG

    def test_empty_model(self):
        """Test with a model that has no named modules."""

        class EmptyModel(nn.Module):
            def forward(self, x):
                return x

        model = EmptyModel()
        global_config = ModuleFooCompressionConfig()

        config = FooCompressionConfig(global_config=global_config)
        config_map = config.build_module_config_dict(model)

        # Should only have the root module ""
        assert config_map[ConfigLevel.GLOBAL] == {"": global_config}
        assert config_map[ConfigLevel.MODULE_NAME] == {}
        assert config_map[ConfigLevel.MODULE_TYPE] == {}

    def test_complex_nested_regex_interaction(self):
        """Test complex interactions between regex patterns and nested modules."""
        model = self.create_test_model()

        # Multiple overlapping patterns
        pattern1_config = ModuleFooCompressionConfig(compression_ratio=0.1)
        pattern2_config = ModuleFooCompressionConfig(compression_ratio=0.2)
        pattern3_config = ModuleFooCompressionConfig(compression_ratio=0.3)

        config = FooCompressionConfig(
            module_name_configs={
                r".*\.0": pattern1_config,  # Matches features.0, classifier.0
                r"features\..*": pattern2_config,  # Matches all features children
                r"classifier\..*": pattern3_config,  # Matches all classifier children
            }
        )
        config_map = config.build_module_config_dict(model)

        name_configs = config_map[ConfigLevel.MODULE_NAME]

        # Later patterns should win for overlapping matches
        # pattern2 comes after pattern1
        assert name_configs["features.0"] == pattern2_config
        assert name_configs["features.1"] == pattern2_config
        assert name_configs["features.2"] == pattern2_config
        # pattern3 comes after pattern1
        assert name_configs["classifier.0"] == pattern3_config
        assert name_configs["classifier.1"] == pattern3_config
        assert name_configs["classifier.2"] == pattern3_config

    def test_config_propagation_to_children(self):
        """
        Test that compression-specific properties and op-level settings are
        propagated to child modules, but module-level settings are not.
        """
        model = self.create_test_model()

        # Create parent config with:
        # - Compression-specific properties (compression_ratio, method, enabled)
        # - Op-level settings
        # - Module-level settings
        parent_config = ModuleFooCompressionConfig(
            compression_ratio=0.85,
            method="custom_method",
            enabled=False,
            op_input_spec={0: DummySpec(value=10)},
            op_output_spec={1: DummySpec(value=20)},
            op_state_spec={"weight": DummySpec(value=30)},
            module_input_spec={0: DummySpec(value=40)},
            module_output_spec={0: DummySpec(value=50)},
            module_state_spec={"weight": DummySpec(value=60)},
        )

        config = FooCompressionConfig(module_name_configs={"features": parent_config})
        config_map = config.build_module_config_dict(model)

        name_configs = config_map[ConfigLevel.MODULE_NAME]

        # Parent module should have the full config
        assert name_configs["features"] == parent_config

        # Child modules should have:
        # 1. Same compression-specific properties
        # 2. Same op-level settings
        # 3. Empty module-level settings (not propagated)
        for child_name in ["features.0", "features.1", "features.2"]:
            child_config = name_configs[child_name]

            # Check compression-specific properties are propagated
            assert child_config.compression_ratio == 0.85
            assert child_config.method == "custom_method"
            assert child_config.enabled is False

            # Check op-level settings are propagated
            assert child_config.op_input_spec == {0: DummySpec(value=10)}
            assert child_config.op_output_spec == {1: DummySpec(value=20)}
            assert child_config.op_state_spec == {"weight": DummySpec(value=30)}

            # Check module-level settings are NOT propagated (should be empty)
            assert child_config.module_input_spec == {}
            assert child_config.module_output_spec == {}
            assert child_config.module_state_spec == {}
