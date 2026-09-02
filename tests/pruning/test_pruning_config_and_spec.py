# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for pruning config hierarchy and spec validation."""

import pytest
import torch
import torch.nn as nn

from coreai_opt._utils.config_utils import spec_dict_is_active
from coreai_opt.pruning.config import (
    MagnitudePrunerConfig,
    ModuleMagnitudePrunerConfig,
    OpMagnitudePrunerConfig,
)
from coreai_opt.pruning.spec import (
    ChannelStructured,
    PruneImplBase,
    PruningScheme,
    PruningSpec,
    Unstructured,
    _MagnitudePruneImpl,
    default_weight_pruning_spec,
)


class TestPruningSpec:
    """Tests for PruningSpec defaults, validation, serialization, and scheme."""

    def test_default_spec(self) -> None:
        """Default spec has 0.5 sparsity, Unstructured scheme, and default algo."""
        spec = PruningSpec()
        assert spec.target_sparsity == 0.5
        assert isinstance(spec.pruning_scheme, Unstructured)
        assert spec.pruning_algo is _MagnitudePruneImpl

    @pytest.mark.parametrize(
        "target_sparsity,pruning_scheme",
        [
            (0.25, Unstructured()),
            (0.75, Unstructured()),
            (0.5, ChannelStructured(axis=0)),
            (0.9, ChannelStructured(axis=1)),
        ],
        ids=["25%-unstructured", "75%-unstructured", "50%-channel-ax0", "90%-channel-ax1"],
    )
    def test_custom_spec(self, target_sparsity: float, pruning_scheme: PruningScheme) -> None:
        """Custom spec values are accepted and stored correctly."""
        spec = PruningSpec(target_sparsity=target_sparsity, pruning_scheme=pruning_scheme)
        assert spec.target_sparsity == target_sparsity
        assert type(spec.pruning_scheme) is type(pruning_scheme)
        if hasattr(pruning_scheme, "axis") and pruning_scheme.axis is not None:
            assert spec.pruning_scheme.axis == pruning_scheme.axis

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"target_sparsity": -0.1}, "greater than or equal to 0"),
            ({"target_sparsity": 1.5}, "less than or equal to 1"),
            ({"pruning_algo": "nonexistent"}, "No class is registered"),
        ],
        ids=["negative-sparsity", "over-1-sparsity", "invalid-algo"],
    )
    def test_invalid_spec(self, kwargs: dict, match: str) -> None:
        """Invalid spec values raise ValueError."""
        with pytest.raises(ValueError, match=match):
            PruningSpec(**kwargs)

    def test_save_and_reload_spec(self) -> None:
        """model_dump() -> PruningSpec(**dump) round-trips correctly."""
        original = PruningSpec(target_sparsity=0.75, pruning_scheme=ChannelStructured(axis=1))
        dump = original.model_dump()
        restored = PruningSpec(**dump)
        assert restored.target_sparsity == original.target_sparsity
        assert type(restored.pruning_scheme) is type(original.pruning_scheme)
        assert restored.pruning_scheme.axis == original.pruning_scheme.axis

    @pytest.mark.parametrize(
        "scheme_dict,expected_type,expected_axis",
        [
            ({"type": "unstructured"}, Unstructured, None),
            ({"type": "channel_structured", "axis": 0}, ChannelStructured, 0),
            ({"type": "channel_structured", "axis": 1}, ChannelStructured, 1),
        ],
        ids=["unstructured-dict", "channel-ax0-dict", "channel-ax1-dict"],
    )
    def test_pruning_scheme_round_trip(
        self, scheme_dict: dict, expected_type: type, expected_axis: int | None
    ) -> None:
        """PruningScheme constructed from dict resolves correctly."""
        spec = PruningSpec(pruning_scheme=scheme_dict)
        assert isinstance(spec.pruning_scheme, expected_type)
        assert spec.pruning_scheme.axis == expected_axis

    def test_pruning_algo_round_trip(self) -> None:
        """Default pruning_algo string resolves to the class object."""
        spec = PruningSpec(pruning_algo="default")
        assert spec.pruning_algo is _MagnitudePruneImpl

        spec_cls = PruningSpec(pruning_algo=_MagnitudePruneImpl)
        assert spec_cls.pruning_algo is _MagnitudePruneImpl

    def test_default_weight_pruning_spec(self) -> None:
        """Factory function returns expected defaults."""
        spec = default_weight_pruning_spec()
        assert spec.target_sparsity == 0.5
        assert isinstance(spec.pruning_scheme, Unstructured)
        assert spec.pruning_algo is _MagnitudePruneImpl

    def test_custom_pruning_algo(self) -> None:
        """Register a custom pruning algo and verify it can be set via string and class."""

        @PruneImplBase.register("random_test")
        class _RandomPruneImpl(PruneImplBase):
            @staticmethod
            def compute_mask(weight, sparsity, pruning_scheme):
                return (torch.rand_like(weight) >= sparsity).to(weight.dtype)

        spec_str = PruningSpec(pruning_algo="random_test")
        assert spec_str.pruning_algo is _RandomPruneImpl

        spec_cls = PruningSpec(pruning_algo=_RandomPruneImpl)
        assert spec_cls.pruning_algo is _RandomPruneImpl

        del PruneImplBase.REGISTRY["random_test"]


class TestPruningConfig:
    """Tests for pruning configuration hierarchy and validation.

    Uses a model with multiple submodule types to verify config resolution.
    """

    @pytest.fixture
    def multi_module_model(self) -> nn.Module:
        """Model with 5 submodules: 2 Linear, 2 Conv2d, 1 MultiheadAttention-like."""

        class MultiModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(20, 20, bias=False)
                self.linear2 = nn.Linear(20, 20, bias=False)
                self.conv1 = nn.Conv2d(3, 8, 3, bias=False)
                self.conv2 = nn.Conv2d(8, 16, 3, bias=False)
                self.proj = nn.Linear(20, 10, bias=False)

            def forward(self, x):
                return self.proj(self.linear2(self.linear1(x)))

        return MultiModel()

    def test_global_config_defaults(self, multi_module_model: nn.Module) -> None:
        """Global config applies to all modules."""
        config = MagnitudePrunerConfig()
        for name, module in multi_module_model.named_modules():
            if name == "":
                continue
            mc = config.get_module_config(name, module)
            assert mc.op_state_spec is not None
            assert "weight" in mc.op_state_spec
            assert mc.op_state_spec["weight"].target_sparsity == 0.5

    def test_disallow_op_activation_spec(self) -> None:
        """MagnitudePrunerConfig rejects activation specs at any level."""
        spec = default_weight_pruning_spec()

        with pytest.raises(ValueError, match="does not support op_input_spec"):
            MagnitudePrunerConfig(
                global_config=ModuleMagnitudePrunerConfig(
                    op_state_spec={"weight": spec},
                    op_type_config={
                        "linear": OpMagnitudePrunerConfig(op_input_spec={"x": spec}),
                    },
                )
            )

        with pytest.raises(ValueError, match="does not support op_output_spec"):
            MagnitudePrunerConfig(
                global_config=ModuleMagnitudePrunerConfig(
                    op_state_spec={"weight": spec},
                    op_type_config={
                        "linear": OpMagnitudePrunerConfig(op_output_spec={"y": spec}),
                    },
                )
            )

    def test_module_name_config(self, multi_module_model: nn.Module) -> None:
        """module_name_configs overrides global for a specific module only."""
        name_spec = PruningSpec(target_sparsity=0.9)
        config = MagnitudePrunerConfig(
            module_name_configs={
                "linear1": ModuleMagnitudePrunerConfig(op_state_spec={"weight": name_spec}),
            },
        )

        linear1_config = config.get_module_config("linear1", multi_module_model.linear1)
        assert linear1_config.op_state_spec["weight"].target_sparsity == 0.9

        linear2_config = config.get_module_config("linear2", multi_module_model.linear2)
        assert linear2_config.op_state_spec["weight"].target_sparsity == 0.5

    def test_module_type_config(self, multi_module_model: nn.Module) -> None:
        """module_type_configs overrides global for all modules of that type."""
        type_spec = PruningSpec(target_sparsity=0.3)
        config = MagnitudePrunerConfig(
            module_type_configs={
                "torch.nn.modules.conv.Conv2d": ModuleMagnitudePrunerConfig(
                    op_state_spec={"weight": type_spec}
                ),
            },
        )

        conv1_config = config.get_module_config("conv1", multi_module_model.conv1)
        assert conv1_config.op_state_spec["weight"].target_sparsity == 0.3

        conv2_config = config.get_module_config("conv2", multi_module_model.conv2)
        assert conv2_config.op_state_spec["weight"].target_sparsity == 0.3

        linear1_config = config.get_module_config("linear1", multi_module_model.linear1)
        assert linear1_config.op_state_spec["weight"].target_sparsity == 0.5

    def test_op_name_config(self, multi_module_model: nn.Module) -> None:
        """op_name_config within a module_name_config specifies per-op overrides."""
        op_spec = OpMagnitudePrunerConfig(
            op_state_spec={"weight": PruningSpec(target_sparsity=0.8)}
        )
        config = MagnitudePrunerConfig(
            module_name_configs={
                "linear1": ModuleMagnitudePrunerConfig(
                    op_state_spec={"weight": PruningSpec(target_sparsity=0.2)},
                    op_name_config={"custom_op": op_spec},
                ),
            },
        )

        mc = config.get_module_config("linear1", multi_module_model.linear1)
        assert mc.op_state_spec["weight"].target_sparsity == 0.2
        assert "custom_op" in mc.op_name_config
        assert mc.op_name_config["custom_op"].op_state_spec["weight"].target_sparsity == 0.8

    def test_op_type_config(self, multi_module_model: nn.Module) -> None:
        """op_type_config within a module_name_config specifies per-op-type overrides."""
        op_spec = OpMagnitudePrunerConfig(
            op_state_spec={"weight": PruningSpec(target_sparsity=0.7)}
        )
        config = MagnitudePrunerConfig(
            module_name_configs={
                "linear1": ModuleMagnitudePrunerConfig(
                    op_state_spec={"weight": PruningSpec(target_sparsity=0.1)},
                    op_type_config={"linear": op_spec},
                ),
            },
        )

        mc = config.get_module_config("linear1", multi_module_model.linear1)
        assert mc.op_state_spec["weight"].target_sparsity == 0.1
        assert "linear" in mc.op_type_config
        assert mc.op_type_config["linear"].op_state_spec["weight"].target_sparsity == 0.7

    def test_module_and_op_config_hierarchy(self, multi_module_model: nn.Module) -> None:
        """Hierarchy: module_name > module_type > global at config resolution."""
        global_spec = PruningSpec(target_sparsity=0.5)
        type_spec = PruningSpec(target_sparsity=0.3)
        name_spec = PruningSpec(target_sparsity=0.9)

        config = MagnitudePrunerConfig(
            global_config=ModuleMagnitudePrunerConfig(op_state_spec={"weight": global_spec}),
            module_type_configs={
                "torch.nn.modules.linear.Linear": ModuleMagnitudePrunerConfig(
                    op_state_spec={"weight": type_spec}
                ),
            },
            module_name_configs={
                "linear1": ModuleMagnitudePrunerConfig(op_state_spec={"weight": name_spec}),
            },
        )

        linear1_config = config.get_module_config("linear1", multi_module_model.linear1)
        assert linear1_config.op_state_spec["weight"].target_sparsity == 0.9

        linear2_config = config.get_module_config("linear2", multi_module_model.linear2)
        assert linear2_config.op_state_spec["weight"].target_sparsity == 0.3

        conv1_config = config.get_module_config("conv1", multi_module_model.conv1)
        assert conv1_config.op_state_spec["weight"].target_sparsity == 0.5

    def test_module_state_spec(self, multi_module_model: nn.Module) -> None:
        """module_state_spec defines state for all ops within that module."""
        state_spec = PruningSpec(target_sparsity=0.6)
        config = MagnitudePrunerConfig(
            module_name_configs={
                "linear1": ModuleMagnitudePrunerConfig(
                    module_state_spec={"weight": state_spec},
                ),
            },
        )

        mc = config.get_module_config("linear1", multi_module_model.linear1)
        assert mc.module_state_spec is not None
        assert "weight" in mc.module_state_spec
        assert mc.module_state_spec["weight"].target_sparsity == 0.6

    def test_config_from_dict_round_trip(self) -> None:
        """from_dict / to_dict round-trips successfully."""
        config = MagnitudePrunerConfig()
        config_dict = config.to_dict()
        assert "magnitude_pruning_config" in config_dict
        restored = MagnitudePrunerConfig.from_dict(config_dict)
        assert restored.global_config.op_state_spec["weight"].target_sparsity == 0.5

    def test_config_disable_global(self) -> None:
        """Setting global_config to None disables pruning globally."""
        config = MagnitudePrunerConfig(global_config=None)
        assert config.global_config is not None
        assert not spec_dict_is_active(config.global_config.op_state_spec)
