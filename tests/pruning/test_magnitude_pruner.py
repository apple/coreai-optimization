# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for MagnitudePruner: prepare, forward, finalize, and config overrides."""

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_opt.pruning import (
    MagnitudePruner,
    MagnitudePrunerConfig,
    ModuleMagnitudePrunerConfig,
    PruningSpec,
)
from coreai_opt.pruning.config import (
    ConstantSparsitySchedule,
    OpMagnitudePrunerConfig,
    PolynomialDecaySchedule,
)
from coreai_opt.pruning.spec import ChannelStructured, PruneImplBase, Unstructured


@pytest.fixture
def linear_10x20_unique() -> nn.Linear:
    """Linear(20, 10) with unique magnitudes 1..200 for deterministic pruning."""
    model = nn.Linear(20, 10, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.arange(1, 201, dtype=torch.float32).reshape(10, 20))
    return model


@pytest.fixture
def linear_100x100_unique() -> nn.Linear:
    """Linear(100, 100) with unique magnitudes 1..10000 for deterministic pruning."""
    model = nn.Linear(100, 100, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.arange(1, 10001, dtype=torch.float32).reshape(100, 100))
    return model


class TestMagnitudePruner:
    """Tests for MagnitudePruner end-to-end behavior."""

    @pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["fp32", "fp16", "bf16"]
    )
    def test_default_magnitude_pruner(self, dtype: torch.dtype) -> None:
        """MagnitudePruner() with no config uses default 50% sparsity and keeps the weight dtype."""
        model = nn.Linear(100, 50, bias=False).to(dtype)
        with torch.no_grad():
            model.weight.copy_(torch.arange(1, 5001, dtype=torch.float32).reshape(50, 100))

        pruner = MagnitudePruner(model)
        pruner.prepare((torch.randn(1, 100, dtype=dtype),))

        assert model.weight.dtype == dtype
        sparsity = (model.weight == 0).float().mean().item()
        assert sparsity == 0.5

    def test_basic_prepare(self, linear_10x20_unique: nn.Linear) -> None:
        """Prepare registers parametrization on the model."""
        pruner = MagnitudePruner(linear_10x20_unique)
        pruner.prepare((torch.randn(1, 20),))

        assert hasattr(linear_10x20_unique, "parametrizations")
        assert "weight" in linear_10x20_unique.parametrizations
        assert isinstance(linear_10x20_unique.parametrizations.weight[0], PruneImplBase)

    def test_torch_finalize(self) -> None:
        """Finalize with Torch backend is a no-op (parametrizations stay active)."""
        model = nn.Linear(100, 50, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.arange(1, 5001, dtype=torch.float32).reshape(50, 100))

        pruner = MagnitudePruner(model)
        pruner.prepare((torch.randn(1, 100),))

        prepared_weight = model.weight.detach().clone()

        finalized = pruner.finalize()
        assert hasattr(finalized, "parametrizations")
        assert "weight" in finalized.parametrizations

        assert torch.equal(finalized.weight, prepared_weight)
        sparsity = (finalized.weight == 0).float().mean().item()
        assert sparsity == 0.5

    def test_invalid_state(self) -> None:
        """Double prepare raises; finalize without prepare raises."""
        model = nn.Linear(10, 10)
        pruner = MagnitudePruner(model)

        with pytest.raises(RuntimeError, match="must be prepared"):
            pruner.finalize()

        pruner.prepare((torch.randn(1, 10),))

        with pytest.raises(RuntimeError, match="already been prepared"):
            pruner.prepare((torch.randn(1, 10),))

    @pytest.mark.parametrize(
        "target,expected",
        [(0.0, 0.0), (0.25, 0.25), (0.75, 0.75), (1.0, 1.0)],
        ids=["0%", "25%", "75%", "100%"],
    )
    def test_pruner_different_sparsity_levels(
        self, linear_100x100_unique: nn.Linear, target: float, expected: float
    ) -> None:
        """Exact sparsity with unique magnitudes."""
        config = MagnitudePrunerConfig(
            global_config=ModuleMagnitudePrunerConfig(
                op_state_spec={"weight": PruningSpec(target_sparsity=target)}
            )
        )
        pruner = MagnitudePruner(linear_100x100_unique, config)
        pruner.prepare((torch.randn(1, 100),))

        sparsity = (linear_100x100_unique.weight == 0).float().mean().item()
        assert sparsity == expected, f"Expected {expected}, got {sparsity}"

    def test_magnitude_logic(self) -> None:
        """Tensor with values 0-99: prune to 50% keeps values 50-99."""
        model = nn.Linear(100, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.arange(0, 100, dtype=torch.float32).reshape(1, 100))

        config = MagnitudePrunerConfig(
            global_config=ModuleMagnitudePrunerConfig(
                op_state_spec={"weight": PruningSpec(target_sparsity=0.5)}
            )
        )
        pruner = MagnitudePruner(model, config)
        pruner.prepare((torch.randn(1, 100),))

        weight = model.weight.detach().flatten()
        surviving = weight[weight != 0]
        assert surviving.min().item() >= 50
        assert surviving.max().item() == 99

    def test_magnitude_pruner_hand_examples(self) -> None:
        """Hand-written 3x3 tensor pruned to ~33% sparsity."""
        model = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[9.0, 1.0, 5.0], [2.0, 8.0, 3.0], [7.0, 4.0, 6.0]]))

        config = MagnitudePrunerConfig(
            global_config=ModuleMagnitudePrunerConfig(
                op_state_spec={"weight": PruningSpec(target_sparsity=1.0 / 3.0)}
            )
        )
        pruner = MagnitudePruner(model, config)
        pruner.prepare((torch.randn(1, 3),))

        expected = torch.tensor([[9.0, 0.0, 5.0], [0.0, 8.0, 0.0], [7.0, 4.0, 6.0]])
        assert torch.equal(model.weight.detach(), expected)

    def test_duplicate_magnitudes(self) -> None:
        """Tensor of all-ones pruned to 50%: exactly half the values are pruned."""
        model = nn.Linear(10, 10, bias=False)
        nn.init.ones_(model.weight)

        config = MagnitudePrunerConfig(
            global_config=ModuleMagnitudePrunerConfig(
                op_state_spec={"weight": PruningSpec(target_sparsity=0.5)}
            )
        )
        pruner = MagnitudePruner(model, config)
        pruner.prepare((torch.randn(1, 10),))

        sparsity = (model.weight == 0).float().mean().item()
        assert sparsity == 0.5, (
            f"Even with duplicate magnitudes, exactly 50% should be pruned, got {sparsity:.2%}"
        )

    def test_magnitude_pruner_module_name(self) -> None:
        """module_name_configs prunes only the named module."""
        model = nn.Sequential(
            nn.Linear(20, 20, bias=False),
            nn.Linear(20, 20, bias=False),
        )
        with torch.no_grad():
            model[0].weight.copy_(torch.arange(1, 401, dtype=torch.float32).reshape(20, 20))
            model[1].weight.copy_(torch.arange(1, 401, dtype=torch.float32).reshape(20, 20))

        original_weight_0 = model[0].weight.data.clone()

        config = MagnitudePrunerConfig(
            global_config=None,
            module_name_configs={
                "1": ModuleMagnitudePrunerConfig(
                    op_state_spec={"weight": PruningSpec(target_sparsity=0.5)}
                ),
            },
        )
        pruner = MagnitudePruner(model, config)
        pruner.prepare((torch.randn(1, 20),))

        assert torch.equal(model[0].weight, original_weight_0)
        sparsity_1 = (model[1].weight == 0).float().mean().item()
        assert sparsity_1 == 0.5

    def test_magnitude_pruner_module_type(self) -> None:
        """module_type_configs applies to all modules of that type."""

        class MixedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(20, 20, bias=False)
                self.conv = nn.Conv2d(3, 8, 3, bias=False)

            def forward(self, x):
                return self.linear(x)

        model = MixedModel()
        with torch.no_grad():
            model.linear.weight.copy_(torch.arange(1, 401, dtype=torch.float32).reshape(20, 20))

        config = MagnitudePrunerConfig(
            global_config=None,
            module_type_configs={
                "torch.nn.modules.linear.Linear": ModuleMagnitudePrunerConfig(
                    op_state_spec={"weight": PruningSpec(target_sparsity=0.5)}
                ),
            },
        )
        pruner = MagnitudePruner(model, config)
        pruner.prepare((torch.randn(1, 20),))

        linear_sparsity = (model.linear.weight == 0).float().mean().item()
        assert linear_sparsity == 0.5

        conv_sparsity = (model.conv.weight == 0).float().mean().item()
        assert conv_sparsity == 0.0

    def test_magnitude_pruner_op_name(self) -> None:
        """op_name_config targets a specific op by name, leaving others at global."""

        class FlatModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_weight = nn.Parameter(
                    torch.arange(1, 37, dtype=torch.float32).reshape(4, 1, 3, 3)
                )
                self.linear_weight = nn.Parameter(
                    torch.arange(1, 17, dtype=torch.float32).reshape(4, 4)
                )

            def forward(self, x):
                x = F.conv2d(x, self.conv_weight, padding=1)
                x = x.mean(dim=(2, 3))
                return F.linear(x, self.linear_weight)

        model = FlatModel()

        op_config = OpMagnitudePrunerConfig(
            op_state_spec={"linear_weight": PruningSpec(target_sparsity=0.75)}
        )
        config = MagnitudePrunerConfig(
            global_config=ModuleMagnitudePrunerConfig(
                op_state_spec={"conv_weight": PruningSpec(target_sparsity=0.25)},
                op_name_config={"linear": op_config},
            )
        )
        pruner = MagnitudePruner(model, config)
        pruner.prepare((torch.randn(1, 1, 4, 4),))

        linear_sparsity = (model.linear_weight == 0).float().mean().item()
        conv_sparsity = (model.conv_weight == 0).float().mean().item()

        assert linear_sparsity == 0.75, (
            f"op_name_config 'linear' should give 75% sparsity, got {linear_sparsity}"
        )
        assert conv_sparsity == 0.25, f"Conv should use global 25% sparsity, got {conv_sparsity}"

    def test_magnitude_pruner_op_type(self, linear_10x20_unique: nn.Linear) -> None:
        """op_type_config overrides op_state_spec for matching op types."""
        op_config = OpMagnitudePrunerConfig(
            op_state_spec={"weight": PruningSpec(target_sparsity=0.75)}
        )
        config = MagnitudePrunerConfig(
            global_config=ModuleMagnitudePrunerConfig(
                op_state_spec={"weight": PruningSpec(target_sparsity=0.25)},
                op_type_config={"linear": op_config},
            )
        )
        pruner = MagnitudePruner(linear_10x20_unique, config)
        pruner.prepare((torch.randn(1, 20),))

        sparsity = (linear_10x20_unique.weight == 0).float().mean().item()
        assert sparsity == 0.75

    def test_magnitude_pruner_module_state(self) -> None:
        """module_state_spec prunes supported ops and skips unsupported ones."""

        class MultiOpModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 8, kernel_size=3, bias=False)
                self.bn = nn.BatchNorm2d(8)
                self.linear = nn.Linear(8, 4, bias=False)
                self.ln = nn.LayerNorm(4)

            def forward(self, x):
                x = self.bn(self.conv(x))
                x = x.mean(dim=(2, 3))
                return self.ln(self.linear(x))

        model = MultiOpModule()
        with torch.no_grad():
            model.conv.weight.copy_(torch.arange(1, 217, dtype=torch.float32).reshape(8, 3, 3, 3))
            model.linear.weight.copy_(torch.arange(1, 33, dtype=torch.float32).reshape(4, 8))

        config = MagnitudePrunerConfig(
            module_name_configs={
                "": ModuleMagnitudePrunerConfig(
                    module_state_spec={"weight": PruningSpec(target_sparsity=0.5)},
                ),
            },
        )
        pruner = MagnitudePruner(model, config)
        pruner.prepare((torch.randn(1, 3, 8, 8),))

        conv_sparsity = (model.conv.weight == 0).float().mean().item()
        linear_sparsity = (model.linear.weight == 0).float().mean().item()
        assert conv_sparsity == 0.5
        assert linear_sparsity == 0.5

    def test_channel_structured_pruning_hand(self) -> None:
        """Hand-written 4x4 tensor with channel pruning (axis=0) at 50%."""
        model = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            model.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 1.0, 1.0, 1.0],
                        [5.0, 5.0, 5.0, 5.0],
                        [2.0, 2.0, 2.0, 2.0],
                        [4.0, 4.0, 4.0, 4.0],
                    ]
                )
            )

        config = MagnitudePrunerConfig(
            global_config=ModuleMagnitudePrunerConfig(
                op_state_spec={
                    "weight": PruningSpec(
                        target_sparsity=0.5,
                        pruning_scheme=ChannelStructured(axis=0),
                    )
                }
            )
        )
        pruner = MagnitudePruner(model, config)
        pruner.prepare((torch.randn(1, 4),))

        expected = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [5.0, 5.0, 5.0, 5.0],
                [0.0, 0.0, 0.0, 0.0],
                [4.0, 4.0, 4.0, 4.0],
            ]
        )
        assert torch.equal(model.weight.detach(), expected)

    def test_channel_structured_conv2d(self) -> None:
        """Channel-structured pruning on Conv2d zeros entire output filters."""
        torch.manual_seed(42)
        model = nn.Conv2d(3, 8, kernel_size=3, bias=False)

        config = MagnitudePrunerConfig(
            global_config=ModuleMagnitudePrunerConfig(
                op_state_spec={
                    "weight": PruningSpec(
                        target_sparsity=0.5,
                        pruning_scheme=ChannelStructured(axis=0),
                    )
                }
            )
        )
        pruner = MagnitudePruner(model, config)
        pruner.prepare((torch.randn(1, 3, 8, 8),))

        weight = model.weight.detach()
        num_pruned = sum(1 for i in range(8) if weight[i].eq(0).all())
        assert num_pruned == 4, f"Expected 4/8 filters pruned, got {num_pruned}"

        for i in range(8):
            filt = weight[i]
            assert filt.eq(0).all() or filt.ne(0).all(), f"Filter {i} is partially pruned"

    def test_linear_unstructured_conv2d_channel_structured(self) -> None:
        """Apply unstructured to Linear and channel-structured to Conv2d in same model."""

        class MixedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 4, kernel_size=3, bias=False)
                self.linear = nn.Linear(4, 4, bias=False)

            def forward(self, x):
                x = self.conv(x)
                x = x.mean(dim=(2, 3))
                return self.linear(x)

        model = MixedModel()
        with torch.no_grad():
            model.linear.weight.copy_(torch.arange(1, 17, dtype=torch.float32).reshape(4, 4))

        config = MagnitudePrunerConfig(
            module_name_configs={
                "conv": ModuleMagnitudePrunerConfig(
                    op_state_spec={
                        "weight": PruningSpec(
                            target_sparsity=0.5,
                            pruning_scheme=ChannelStructured(axis=0),
                        )
                    }
                ),
                "linear": ModuleMagnitudePrunerConfig(
                    op_state_spec={
                        "weight": PruningSpec(
                            target_sparsity=0.5,
                            pruning_scheme=Unstructured(),
                        )
                    }
                ),
            },
        )
        pruner = MagnitudePruner(model, config)
        pruner.prepare((torch.randn(1, 3, 8, 8),))

        conv_weight = model.conv.weight.detach()
        num_pruned_filters = sum(1 for i in range(4) if conv_weight[i].eq(0).all())
        assert num_pruned_filters == 2, (
            f"Expected 2/4 conv filters pruned, got {num_pruned_filters}"
        )

        linear_weight = model.linear.weight.detach()
        linear_sparsity = (linear_weight == 0).float().mean().item()
        assert linear_sparsity == 0.5

    def test_channel_structured_duplicate_norms(self) -> None:
        """Channels with tied L1 norms are still pruned to exact target count.

        With norms [4, 8, 8, 8, 20] and 60% sparsity (prune 3 of 5 channels),
        the old kthvalue approach would keep all channels with norm >= 8 (4 channels),
        pruning only 1. topk correctly prunes exactly 3.
        """
        model = nn.Linear(4, 5, bias=False)
        with torch.no_grad():
            model.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0, 2.0],
                        [2.0, 2.0, 2.0, 2.0],
                        [2.0, 2.0, 2.0, 2.0],
                        [5.0, 5.0, 5.0, 5.0],
                    ]
                )
            )

        config = MagnitudePrunerConfig(
            global_config=ModuleMagnitudePrunerConfig(
                op_state_spec={
                    "weight": PruningSpec(
                        target_sparsity=0.6,
                        pruning_scheme=ChannelStructured(axis=0),
                    )
                }
            )
        )
        pruner = MagnitudePruner(model, config)
        pruner.prepare((torch.randn(1, 4),))

        weight = model.weight.detach()
        num_pruned = sum(1 for i in range(5) if weight[i].eq(0).all())
        assert num_pruned == 3, (
            f"Expected exactly 3/5 channels pruned despite tied norms, got {num_pruned}"
        )

    def test_backprop_through_pruned_weights(self, linear_100x100_unique: nn.Linear) -> None:
        """Gradients flow through unpruned entries and are zero where the mask is zero."""
        pruner = MagnitudePruner(
            linear_100x100_unique,
            MagnitudePrunerConfig(
                global_config=ModuleMagnitudePrunerConfig(
                    op_state_spec={"weight": PruningSpec(target_sparsity=0.5)},
                )
            ),
        )
        pruner.prepare((torch.randn(1, 100),))

        impl = linear_100x100_unique.parametrizations.weight[0]
        original = linear_100x100_unique.parametrizations.weight.original
        mask = impl.mask.detach().clone()

        x = torch.randn(4, 100)
        y_target = torch.randn(4, 100)
        F.mse_loss(linear_100x100_unique(x), y_target).backward()

        assert original.grad is not None
        assert (original.grad.ne(0).to(mask.dtype) == mask).all()


class TestSparsitySchedule:
    """Tests for pruner.step() and the sparsity-schedule wiring."""

    @staticmethod
    def _impl(module: nn.Module, param_name: str = "weight") -> PruneImplBase:
        return module.parametrizations[param_name][0]

    def test_step_with_no_schedule_is_noop(self, linear_100x100_unique: nn.Linear) -> None:
        """Without a sparsity_schedule, step() does not change impl.sparsity or the mask."""
        pruner = MagnitudePruner(linear_100x100_unique)
        pruner.prepare((torch.randn(1, 100),))

        impl = self._impl(linear_100x100_unique)
        sparsity_before = impl.sparsity
        mask_before = impl.mask.clone()

        for _ in range(10):
            pruner.step()

        assert pruner._step_count == 10
        assert impl.sparsity == sparsity_before
        assert torch.equal(impl.mask, mask_before)

    @pytest.mark.parametrize(
        "begin_step,steps_to_run,expected_sparsity,expected_zero_fraction",
        [
            (5, 4, 0.0, 0.0),
            (5, 5, 0.75, 0.75),
            (5, 100, 0.75, 0.75),
        ],
        ids=["before-begin", "at-begin", "after-begin"],
    )
    def test_constant_schedule(
        self,
        linear_100x100_unique: nn.Linear,
        begin_step: int,
        steps_to_run: int,
        expected_sparsity: float,
        expected_zero_fraction: float,
    ) -> None:
        """Constant schedule: 0 before begin_step, target_sparsity at/after."""
        target = 0.75
        config = MagnitudePrunerConfig(
            global_config=ModuleMagnitudePrunerConfig(
                op_state_spec={"weight": PruningSpec(target_sparsity=target)},
                sparsity_schedule=ConstantSparsitySchedule(begin_step=begin_step),
            )
        )
        pruner = MagnitudePruner(linear_100x100_unique, config)
        pruner.prepare((torch.randn(1, 100),))

        for _ in range(steps_to_run):
            pruner.step()

        impl = self._impl(linear_100x100_unique)
        assert impl.sparsity == expected_sparsity
        assert (linear_100x100_unique.weight == 0).float().mean().item() == pytest.approx(
            expected_zero_fraction
        )

    def test_polynomial_schedule_progression(self, linear_100x100_unique: nn.Linear) -> None:
        """Linear (power=1) polynomial schedule hits expected sparsities along the schedule."""
        target = 0.5
        total_iters = 10
        config = MagnitudePrunerConfig(
            global_config=ModuleMagnitudePrunerConfig(
                op_state_spec={"weight": PruningSpec(target_sparsity=target)},
                sparsity_schedule=PolynomialDecaySchedule(
                    begin_step=0, total_iters=total_iters, power=1.0
                ),
            )
        )
        pruner = MagnitudePruner(linear_100x100_unique, config)
        pruner.prepare((torch.randn(1, 100),))

        impl = self._impl(linear_100x100_unique)

        # Step 0 (no step() called yet) — schedule's step-0 value is applied in prepare.
        assert impl.sparsity == pytest.approx(0.0)

        # Walk to the midpoint and assert the linear-formula value.
        for _ in range(5):
            pruner.step()
        assert impl.sparsity == pytest.approx(target * 5 / 9)

        # Reach the end of the schedule.
        for _ in range(5):
            pruner.step()
        assert impl.sparsity == target
        assert (linear_100x100_unique.weight == 0).float().mean().item() == pytest.approx(target)

        # Stepping past the schedule's end stays at the target.
        for _ in range(20):
            pruner.step()
        assert impl.sparsity == target

    def test_per_module_schedule_resolution(self) -> None:
        """module_name_configs overrides global_config; different layers get different schedules."""
        model = nn.Sequential(
            nn.Linear(20, 20, bias=False),
            nn.Linear(20, 10, bias=False),
        )
        with torch.no_grad():
            model[0].weight.copy_(torch.arange(1, 401, dtype=torch.float32).reshape(20, 20))
            model[1].weight.copy_(torch.arange(1, 201, dtype=torch.float32).reshape(10, 20))

        config = MagnitudePrunerConfig(
            global_config=ModuleMagnitudePrunerConfig(
                op_state_spec={"weight": PruningSpec(target_sparsity=0.5)},
                sparsity_schedule=ConstantSparsitySchedule(begin_step=100),
            ),
            module_name_configs={
                "1": ModuleMagnitudePrunerConfig(
                    op_state_spec={"weight": PruningSpec(target_sparsity=0.25)},
                    sparsity_schedule=ConstantSparsitySchedule(begin_step=0),
                ),
            },
        )
        pruner = MagnitudePruner(model, config)
        pruner.prepare((torch.randn(1, 20),))

        # "1" overrides global: applied immediately at sparsity 0.25.
        # "0" follows global: still at 0 since begin_step=100 hasn't been reached.
        assert self._impl(model[0]).sparsity == 0.0
        assert self._impl(model[1]).sparsity == 0.25

        for _ in range(100):
            pruner.step()

        assert self._impl(model[0]).sparsity == 0.5
        assert self._impl(model[1]).sparsity == 0.25

    @pytest.mark.xfail(
        strict=True,
        reason="Pruner step count is not persisted in model.state_dict — resumes from 0.",
    )
    def test_save_resume_midschedule(self, linear_100x100_unique: nn.Linear) -> None:
        """Saving the model mid-schedule and resuming should continue from the saved step."""
        target = 0.5
        total_iters = 10

        def _build(model: nn.Linear) -> MagnitudePruner:
            config = MagnitudePrunerConfig(
                global_config=ModuleMagnitudePrunerConfig(
                    op_state_spec={"weight": PruningSpec(target_sparsity=target)},
                    sparsity_schedule=PolynomialDecaySchedule(
                        begin_step=0, total_iters=total_iters, power=1.0
                    ),
                )
            )
            p = MagnitudePruner(model, config)
            p.prepare((torch.randn(1, 100),))
            return p

        model_orig = copy.deepcopy(linear_100x100_unique)
        pruner_orig = _build(model_orig)
        halfway = total_iters // 2
        for _ in range(halfway):
            pruner_orig.step()

        state = model_orig.state_dict()
        model_resumed = copy.deepcopy(linear_100x100_unique)
        pruner_resumed = _build(model_resumed)
        model_resumed.load_state_dict(state)

        assert pruner_resumed._step_count == halfway
