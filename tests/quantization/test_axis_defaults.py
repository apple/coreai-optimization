# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for weight axis defaults and activation axis validation."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.quantization import (
    ModuleQuantizerConfig,
    QuantizationSpec,
    Quantizer,
    QuantizerConfig,
)
from coreai_opt.quantization._axis_defaults import (
    _WEIGHT_AXIS_SPECS,
    _apply_defaults,
    _WeightAxisSpec,
    _WeightFQMap,
)
from coreai_opt.quantization.spec import default_activation_quantization_spec
from coreai_opt.quantization.spec.fake_quantize import FakeQuantizeImplBase
from coreai_opt.quantization.spec.granularity import (
    PerBlockGranularity,
    PerChannelGranularity,
    PerTensorGranularity,
    QuantizationGranularity,
)

# Block size for PerBlockGranularity tests.  All ops in _SINGLE_OP_PARAMS have
# weight dims divisible by this value on both axes, and so does _FullPipelineModel.
_TEST_BLOCK_SIZE = 2


def _make_config(
    granularity: QuantizationGranularity,
    execution_mode: str = "graph",
    include_activation: bool = False,
) -> QuantizerConfig:
    """Build a QuantizerConfig with the given weight granularity."""
    act_spec = {"*": default_activation_quantization_spec()} if include_activation else None
    return QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={
                "weight": QuantizationSpec(
                    dtype=torch.int8,
                    qscheme="symmetric",
                    granularity=granularity,
                    fake_quantize_cls="default",
                    qparam_calculator_cls="static",
                    range_calculator_cls="minmax",
                ),
            },
            op_input_spec=act_spec,
            op_output_spec=act_spec,
        ),
        execution_mode=execution_mode,
    )


def _get_weight_fqs(
    model: nn.Module,
) -> list[FakeQuantizeImplBase]:
    """Return all weight FakeQuantize modules from a prepared model."""
    return [
        m
        for m in model.modules()
        if isinstance(m, FakeQuantizeImplBase)
        and m.quantization_target == CompressionTargetTensor.WEIGHT
    ]


# Test for every op currently in the defaults table.
# format: op, input, name/id
_SINGLE_OP_PARAMS = [
    pytest.param(
        lambda: nn.Conv1d(4, 16, 3, padding=1),
        lambda: torch.randn(1, 4, 16),
        id="conv1d",
    ),
    pytest.param(
        lambda: nn.Conv2d(4, 16, 3, padding=1),
        lambda: torch.randn(1, 4, 8, 8),
        id="conv2d",
    ),
    pytest.param(
        lambda: nn.Conv3d(4, 16, 3, padding=1),
        lambda: torch.randn(1, 4, 4, 4, 4),
        id="conv3d",
    ),
    pytest.param(
        lambda: nn.ConvTranspose1d(16, 8, 3, padding=1),
        lambda: torch.randn(1, 16, 16),
        id="conv_transpose1d",
    ),
    pytest.param(
        lambda: nn.ConvTranspose2d(16, 8, 3, padding=1),
        lambda: torch.randn(1, 16, 8, 8),
        id="conv_transpose2d",
    ),
    pytest.param(
        lambda: nn.ConvTranspose3d(16, 8, 3, padding=1),
        lambda: torch.randn(1, 16, 4, 4, 4),
        id="conv_transpose3d",
    ),
    pytest.param(
        lambda: nn.Linear(32, 16),
        lambda: torch.randn(1, 32),
        id="linear",
    ),
    pytest.param(
        lambda: nn.Embedding(10, 4),
        lambda: torch.tensor([[1, 0, 2]]),
        id="embedding",
    ),
]

# Granularities with axis=None, to be resolved by the defaults pass.
_GRANULARITY_NONE_PARAMS = [
    pytest.param(PerChannelGranularity(axis=None), id="per_channel"),
    pytest.param(PerBlockGranularity(axis=None, block_size=_TEST_BLOCK_SIZE), id="per_block"),
]

# Granularities with explicit non-default axes (should be preserved as-is).
# Per-channel default for Conv2d is 0, so explicit axis=1 tests preservation.
# Per-block default for Conv2d is 1, so explicit axis=0 tests preservation.
_GRANULARITY_EXPLICIT_PARAMS = [
    pytest.param(PerChannelGranularity(axis=1), id="per_channel"),
    pytest.param(PerBlockGranularity(axis=0, block_size=_TEST_BLOCK_SIZE), id="per_block"),
]

_EXECUTION_MODE_PARAMS = [
    pytest.param("graph", id="graph"),
    pytest.param("eager", id="eager"),
]

_INCLUDE_ACTIVATION_PARAMS = [
    pytest.param(False, id="weight_only"),
    pytest.param(True, id="with_activation"),
]


class _FullPipelineModel(nn.Module):
    """Model for full pipeline tests with **dims divisible by _TEST_BLOCK_SIZE** on both axes."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(32 * 28 * 28, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)


class TestWeightAxisDefaults:
    """Verify per-channel and per-block weight axis resolution for PT2E and eager modes."""

    @pytest.mark.parametrize("include_activation", _INCLUDE_ACTIVATION_PARAMS)
    @pytest.mark.parametrize("execution_mode", _EXECUTION_MODE_PARAMS)
    @pytest.mark.parametrize("make_model,make_input", _SINGLE_OP_PARAMS)
    @pytest.mark.parametrize("granularity", _GRANULARITY_NONE_PARAMS)
    def test_axis_none_resolved(
        self,
        make_model,
        make_input,
        granularity,
        execution_mode,
        include_activation,
    ):
        """axis=None resolves to the correct default for per-channel or per-block."""
        model = make_model()
        module_type = type(model)
        config = _make_config(granularity, execution_mode, include_activation)
        prepared = Quantizer(model, config).prepare((make_input(),))

        if isinstance(granularity, PerChannelGranularity):
            expected_axis = _WEIGHT_AXIS_SPECS[module_type].per_channel_axis
        else:
            expected_axis = _WEIGHT_AXIS_SPECS[module_type].per_block_axis

        weight_fqs = _get_weight_fqs(prepared)
        assert len(weight_fqs) == 1
        assert weight_fqs[0].granularity.axis == expected_axis

    @pytest.mark.parametrize("execution_mode", _EXECUTION_MODE_PARAMS)
    @pytest.mark.parametrize("granularity", _GRANULARITY_EXPLICIT_PARAMS)
    def test_explicit_axis_preserved(self, granularity, execution_mode):
        """Explicit axis value is not overridden by the defaults pass."""
        # Use in_channels=4 so dim 1 is divisible by _TEST_BLOCK_SIZE for PerBlockGranularity.
        config = _make_config(granularity, execution_mode)
        prepared = Quantizer(nn.Conv2d(4, 16, 3, padding=1), config).prepare(
            (torch.randn(1, 4, 8, 8),)
        )
        assert _get_weight_fqs(prepared)[0].granularity.axis == granularity.axis

    @pytest.mark.parametrize("execution_mode", _EXECUTION_MODE_PARAMS)
    def test_per_tensor_unaffected(self, execution_mode):
        """PerTensorGranularity passes through unchanged."""
        config = _make_config(PerTensorGranularity(), execution_mode)
        prepared = Quantizer(nn.Conv2d(3, 16, 3, padding=1), config).prepare(
            (torch.randn(1, 3, 8, 8),)
        )
        assert isinstance(_get_weight_fqs(prepared)[0].granularity, PerTensorGranularity)

    @pytest.mark.parametrize("include_activation", _INCLUDE_ACTIVATION_PARAMS)
    @pytest.mark.parametrize("execution_mode", _EXECUTION_MODE_PARAMS)
    @pytest.mark.parametrize("granularity", _GRANULARITY_NONE_PARAMS)
    def test_calibrate_finalize_succeeds(
        self,
        granularity,
        execution_mode,
        include_activation,
    ):
        """Run Full workflow (prepare -> calibrate -> finalize -> forward) with axis=None."""
        model = _FullPipelineModel()
        config = _make_config(granularity, execution_mode, include_activation)
        quantizer = Quantizer(model, config)
        example_input = torch.randn(1, 4, 28, 28)

        prepared = quantizer.prepare((example_input,))
        with quantizer.calibration_mode():
            prepared(example_input)
        quantizer.finalize()

        out = prepared(torch.randn(1, 4, 28, 28))
        assert out.shape == (1, 10)
        for fq in _get_weight_fqs(prepared):
            assert fq.granularity.axis is not None

    @pytest.mark.parametrize("granularity", _GRANULARITY_NONE_PARAMS)
    def test_error_for_op_without_axis_default(self, granularity):
        """
        If an op for which we don't support defaults isn't specified in the
        config, ensure ValueError is raised.

        Note: This test is eager-only. In PT2E, all the ops that support weight
        quantization are covered in our defaults table.

        In eager mode the handler parametrizes any module whose
        weight is consumed by a registered op (here ``torch.matmul``),
        regardless of whether the module type is in the axis defaults table.

        PT2E annotation patterns only match known module types (Conv, Linear,
        etc.), so an unknown module never receives a weight FQ in PT2E.
        """

        class _UnsupportedWeightModule(nn.Module):
            """Module with a weight parameter but no entry in the axis defaults table."""

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                return torch.matmul(x, self.weight)

        config = _make_config(granularity, execution_mode="eager")
        with pytest.raises(
            ValueError, match="Weight fake-quantize modules with unresolved axis=None remain"
        ):
            Quantizer(_UnsupportedWeightModule(), config).prepare((torch.randn(1, 4),))


def _activation_only_config(
    granularity: QuantizationGranularity,
    execution_mode: str = "graph",
) -> QuantizerConfig:
    """Build a QuantizerConfig with activation input quantization only."""
    return QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_input_spec={
                "*": QuantizationSpec(
                    dtype=torch.int8,
                    qscheme="symmetric",
                    granularity=granularity,
                    fake_quantize_cls="default",
                    qparam_calculator_cls="moving_average",
                    range_calculator_cls="minmax",
                ),
            },
        ),
        execution_mode=execution_mode,
    )


class TestWeightAxisSpec:
    """Verify _WeightAxisSpec.default_axis_for maps granularity types to axes."""

    def test_per_channel_returns_per_channel_axis(self):
        """PerChannelGranularity resolves to the spec's per_channel_axis."""
        spec = _WeightAxisSpec(per_channel_axis=0, per_block_axis=1)
        assert spec.default_axis_for(PerChannelGranularity(axis=None)) == 0

    def test_per_block_returns_per_block_axis(self):
        """PerBlockGranularity resolves to the spec's per_block_axis."""
        spec = _WeightAxisSpec(per_channel_axis=0, per_block_axis=1)
        assert spec.default_axis_for(PerBlockGranularity(axis=None, block_size=2)) == 1

    def test_other_granularity_returns_none(self):
        """A granularity that is neither per-channel nor per-block returns None."""
        spec = _WeightAxisSpec(per_channel_axis=0, per_block_axis=1)
        assert spec.default_axis_for(PerTensorGranularity()) is None


class TestActivationAxisValidation:
    """Verify that activation FQs with unresolved axis=None are caught at prepare time."""

    @pytest.mark.parametrize("execution_mode", _EXECUTION_MODE_PARAMS)
    def test_per_channel_axis_none_raises(self, execution_mode):
        """PerChannelGranularity(axis=None) on activations raises ValueError."""
        config = _activation_only_config(
            PerChannelGranularity(axis=None),
            execution_mode,
        )
        with pytest.raises(ValueError, match="Activation fake-quantize modules with unresolved"):
            Quantizer(nn.Linear(32, 16), config).prepare((torch.randn(1, 32),))

    @pytest.mark.parametrize("execution_mode", _EXECUTION_MODE_PARAMS)
    def test_per_block_scalar_axis_none_raises(self, execution_mode):
        """PerBlockGranularity(block_size=int, axis=None) on activations raises ValueError."""
        config = _activation_only_config(
            PerBlockGranularity(axis=None, block_size=_TEST_BLOCK_SIZE),
            execution_mode,
        )
        with pytest.raises(ValueError, match="Activation fake-quantize modules with unresolved"):
            Quantizer(nn.Linear(32, 16), config).prepare((torch.randn(1, 32),))

    @pytest.mark.parametrize("execution_mode", _EXECUTION_MODE_PARAMS)
    def test_per_block_tuple_axis_none_passes(self, execution_mode):
        """PerBlockGranularity with tuple block_size (multi-axis) does not raise."""
        config = _activation_only_config(
            PerBlockGranularity(axis=None, block_size=(1, 16)),
            execution_mode,
        )
        Quantizer(nn.Linear(32, 16), config).prepare((torch.randn(1, 32),))

    @pytest.mark.parametrize("execution_mode", _EXECUTION_MODE_PARAMS)
    def test_per_tensor_passes(self, execution_mode):
        """PerTensorGranularity on activations does not raise."""
        config = _activation_only_config(PerTensorGranularity(), execution_mode)
        Quantizer(nn.Linear(32, 16), config).prepare((torch.randn(1, 32),))

    @pytest.mark.parametrize("execution_mode", _EXECUTION_MODE_PARAMS)
    def test_error_message_includes_granularity_type(self, execution_mode):
        """Error message contains the granularity class name."""
        config = _activation_only_config(
            PerChannelGranularity(axis=None),
            execution_mode,
        )
        with pytest.raises(
            ValueError, match="Activation fake-quantize modules with unresolved axis=None:"
        ):
            Quantizer(nn.Linear(32, 16), config).prepare((torch.randn(1, 32),))


class _StubFakeQuantize:
    """Lightweight stand-in for FakeQuantizeImplBase in unit tests.

    Only provides the attributes that ``_apply_defaults`` and
    ``_resolve_axis_on_fake_quantize`` access: ``granularity`` and
    ``quantization_target``.
    """

    def __init__(self, granularity: QuantizationGranularity) -> None:
        self.granularity = granularity
        self.quantization_target = CompressionTargetTensor.WEIGHT


class TestApplyDefaultsGrouping:
    """Verify FQ grouping, partial coverage, and conflict detection in _apply_defaults."""

    def test_same_fq_multiple_consumers_same_default(self):
        """One FQ shared by two nn.Linear consumers resolves to axis 0."""
        fq = _StubFakeQuantize(PerChannelGranularity(axis=None))
        fq_map: _WeightFQMap = {fq: [(nn.Linear, "layer_a"), (nn.Linear, "layer_b")]}
        _apply_defaults(fq_map)
        assert fq.granularity.axis == 0

    def test_same_fq_one_default_one_none(self):
        """One FQ with one known consumer and one None consumer resolves without error."""
        fq = _StubFakeQuantize(PerChannelGranularity(axis=None))
        fq_map: _WeightFQMap = {fq: [(nn.Linear, "linear"), (None, "unknown")]}
        _apply_defaults(fq_map)
        assert fq.granularity.axis == 0

    def test_same_fq_conflicting_defaults(self):
        """One FQ consumed by Conv2d (axis 0) and ConvTranspose2d (axis 1) raises."""
        fq = _StubFakeQuantize(PerChannelGranularity(axis=None))
        fq_map: _WeightFQMap = {
            fq: [(nn.Conv2d, "conv"), (nn.ConvTranspose2d, "conv_t")],
        }
        with pytest.raises(ValueError, match="Conflicting default axes"):
            _apply_defaults(fq_map)

    def test_same_fq_all_unresolved(self):
        """One FQ with only None consumers raises unresolved error."""
        fq = _StubFakeQuantize(PerChannelGranularity(axis=None))
        fq_map: _WeightFQMap = {fq: [(None, "a"), (None, "b")]}
        with pytest.raises(
            ValueError, match="Weight fake-quantize modules with unresolved axis=None remain"
        ):
            _apply_defaults(fq_map)

    def test_different_fqs_independent(self):
        """Two independent FQs each resolve to their own correct axis."""
        fq_linear = _StubFakeQuantize(PerChannelGranularity(axis=None))
        fq_conv_t = _StubFakeQuantize(PerBlockGranularity(axis=None, block_size=2))
        fq_map: _WeightFQMap = {
            fq_linear: [(nn.Linear, "linear")],
            fq_conv_t: [(nn.ConvTranspose2d, "conv_t")],
        }
        _apply_defaults(fq_map)
        assert fq_linear.granularity.axis == 0
        assert fq_conv_t.granularity.axis == 0
