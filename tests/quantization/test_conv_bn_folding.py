# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for conv+bn folding in PT2E quantization.

- After finalize(), batch_norm nodes should be folded into conv weights
- Folding must preserve numerical accuracy
"""

from typing import NamedTuple

import pytest
import torch
from torch import nn

from coreai_opt import ExportBackend
from coreai_opt.quantization import ModuleQuantizerConfig, Quantizer, QuantizerConfig
from coreai_opt.quantization._graph._conv_bn_utils import _compute_fused_conv_bn_params
from coreai_opt.quantization.spec.fake_quantize import FakeQuantizeImplBase
from tests.export import export_utils

# ============================================================================
# Test Model Definitions
# ============================================================================


class SimpleConvBN(nn.Module):
    """Simple model with conv2d → batch_norm pattern."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(16)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiConvBN(nn.Module):
    """Larger model with multiple conv+bn blocks."""

    def __init__(self):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        # Block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Block 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Block 4
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.functional.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x


class ConvBNWithActivation(nn.Module):
    """Model with Conv→BN→ReLU pattern."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(16)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = nn.functional.relu(x)
        return x


class Conv1dBN(nn.Module):
    """Model with Conv1d→BatchNorm1d pattern."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(3, 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(16)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ConvTranspose2dBN(nn.Module):
    """Model with ConvTranspose2d→BatchNorm2d pattern."""

    def __init__(self):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(3, 16, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(16)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.bn(x)
        return x


class StandaloneBN(nn.Module):
    """Model with standalone BatchNorm (no preceding conv)."""

    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.bn(x)
        return x


class LargeConvBN(nn.Module):
    """Large model with very wide and deep convolutions."""

    def __init__(self):
        super().__init__()
        # Extremely wide convolutions (2048 channels)
        self.conv1 = nn.Conv2d(3, 512, kernel_size=7, padding=3, stride=2)
        self.bn1 = nn.BatchNorm2d(512)

        self.conv2 = nn.Conv2d(512, 1024, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(1024)

        self.conv3 = nn.Conv2d(1024, 2048, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(2048)

        self.conv4 = nn.Conv2d(2048, 1024, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(1024)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.functional.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x


class LargeLinearBN(nn.Module):
    """Large model with very wide Linear→BatchNorm1d."""

    def __init__(self):
        super().__init__()
        # Very large linear layers
        self.linear1 = nn.Linear(2048, 4096)
        self.bn1 = nn.BatchNorm1d(4096)

        self.linear2 = nn.Linear(4096, 2048)
        self.bn2 = nn.BatchNorm1d(2048)

        self.linear3 = nn.Linear(2048, 1024)
        self.bn3 = nn.BatchNorm1d(1024)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)

        x = self.linear3(x)
        x = self.bn3(x)
        return x


class ResidualBlock(nn.Module):
    """ResNet-style residual block with conv+bn."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = nn.functional.relu(out)
        return out


# ============================================================================
# Test Configuration
# ============================================================================


class ModelTestConfig(NamedTuple):
    """Configuration for a model test case."""

    name: str
    model_cls: type[nn.Module]
    model_kwargs: dict
    example_input: torch.Tensor
    expected_bn_count_before: int
    expected_bn_count_after: int
    expected_conv_weights: int
    description: str


# Test model configurations
TEST_MODELS = [
    ModelTestConfig(
        name="simple_conv_bn",
        model_cls=SimpleConvBN,
        model_kwargs={},
        example_input=torch.randn(1, 3, 32, 32),
        expected_bn_count_before=1,
        expected_bn_count_after=0,
        expected_conv_weights=1,
        description="Simple Conv2d + BatchNorm2d",
    ),
    ModelTestConfig(
        name="multi_conv_bn",
        model_cls=MultiConvBN,
        model_kwargs={},
        example_input=torch.randn(1, 3, 32, 32),
        expected_bn_count_before=4,
        expected_bn_count_after=0,
        expected_conv_weights=4,
        description="Multiple Conv2d + BatchNorm2d blocks",
    ),
    ModelTestConfig(
        name="conv_bn_activation",
        model_cls=ConvBNWithActivation,
        model_kwargs={},
        example_input=torch.randn(1, 3, 32, 32),
        expected_bn_count_before=1,
        expected_bn_count_after=0,
        expected_conv_weights=1,
        description="Conv2d + BatchNorm2d + ReLU",
    ),
    ModelTestConfig(
        name="conv1d_bn",
        model_cls=Conv1dBN,
        model_kwargs={},
        example_input=torch.randn(1, 3, 32),
        expected_bn_count_before=1,
        expected_bn_count_after=0,
        expected_conv_weights=1,
        description="Conv1d + BatchNorm1d",
    ),
    ModelTestConfig(
        name="conv_transpose2d_bn",
        model_cls=ConvTranspose2dBN,
        model_kwargs={},
        example_input=torch.randn(1, 3, 32, 32),
        expected_bn_count_before=1,
        expected_bn_count_after=0,
        expected_conv_weights=1,
        description="ConvTranspose2d + BatchNorm2d",
    ),
    ModelTestConfig(
        name="standalone_bn",
        model_cls=StandaloneBN,
        model_kwargs={},
        example_input=torch.randn(1, 3, 32, 32),
        expected_bn_count_before=1,
        expected_bn_count_after=1,  # Should NOT be folded
        expected_conv_weights=0,
        description="Standalone BatchNorm2d (no preceding conv)",
    ),
    ModelTestConfig(
        name="large_conv_bn",
        model_cls=LargeConvBN,
        model_kwargs={},
        example_input=torch.randn(1, 3, 224, 224),
        expected_bn_count_before=4,
        expected_bn_count_after=0,
        expected_conv_weights=4,
        description="Large Conv2d with 512/1024/2048 channels",
    ),
    ModelTestConfig(
        name="linear_bn",
        model_cls=LargeLinearBN,
        model_kwargs={},
        example_input=torch.randn(4, 2048),
        expected_bn_count_before=3,
        expected_bn_count_after=3,  # Should NOT be folded (Linear+BN not supported)
        expected_conv_weights=0,  # Linear weights, not conv
        description="Linear + BatchNorm1d (intentionally not folded)",
    ),
    ModelTestConfig(
        name="residual_block",
        model_cls=ResidualBlock,
        model_kwargs={"channels": 128},
        example_input=torch.randn(1, 128, 32, 32),
        expected_bn_count_before=2,
        expected_bn_count_after=0,
        expected_conv_weights=2,
        description="ResNet-style residual block",
    ),
]

# Test models for export verification (excludes models with no conv weights)
TEST_MODELS_FOR_EXPORT = [config for config in TEST_MODELS if config.expected_conv_weights > 0]


# ============================================================================
# Helper Functions
# ============================================================================


def count_batch_norm_nodes(model: torch.fx.GraphModule) -> int:
    """Count batch_norm nodes in the FX graph."""
    bn_count = 0
    for node in model.graph.nodes:
        if node.op == "call_function" and "batch_norm" in str(node.target):
            bn_count += 1
    return bn_count


_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


def _initialize_bn_with_nontrivial_params(model: nn.Module, seed: int = 0) -> None:
    """Set BN gamma/beta and running stats to non-trivial values.

    Default-initialized BNs have ``gamma=1, beta=0, mean=0, var=1, eps=1e-5``,
    which makes the conv+bn fold a near no-op (scale ≈ 1, bias ≈ 0). Tests
    against default BNs therefore won't catch arithmetic-order regressions in
    the fold. Re-initializing here ensures the fold has a real scale to bake in.
    """
    rng = torch.Generator().manual_seed(seed)
    for m in model.modules():
        if isinstance(m, _BN_TYPES):
            c = m.num_features
            m.running_mean.copy_(torch.randn(c, generator=rng) * 0.5)
            m.running_var.copy_(torch.rand(c, generator=rng) * 2 + 0.5)
            m.weight.data.copy_(torch.randn(c, generator=rng) * 0.7 + 1.0)
            m.bias.data.copy_(torch.randn(c, generator=rng) * 0.3)


def _capture_fq_io(
    model: torch.fx.GraphModule,
    example_input: torch.Tensor,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Run ``model(example_input)`` and capture (input, output) for every FQ module.

    Keyed by FQ module name, which is preserved across fold (the fold rewires
    graph args but does not rename or replace FQ modules).
    """
    captured: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    handles = []

    def make_hook(fq_name: str):
        def hook(_module, inputs, output):
            captured[fq_name] = (inputs[0].detach().clone(), output.detach().clone())

        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, FakeQuantizeImplBase):
            handles.append(mod.register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            model(example_input)
    finally:
        for h in handles:
            h.remove()

    return captured


# ============================================================================
# Parameterized Tests
# ============================================================================


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_config",
    TEST_MODELS,
    ids=[config.name for config in TEST_MODELS],
)
def test_conv_bn_folding_numerical_correctness(
    model_config: ModelTestConfig,
) -> None:
    """Verify conv+bn folding preserves numerical accuracy.

    This test checks that:
    1. BN nodes are folded (or not) as expected
    2. Each weight fake-quantize module sees bit-identical input and produces
       bit-identical output in the prepared and finalized graphs. This catches
       regressions in the fold's fp32 arithmetic order.
    3. Model outputs remain numerically identical before/after folding

    Backend-agnostic processing is tested with Torch backend only.
    Export backend logic is verified separately in
    test_conv_bn_folding_export_verification.
    """
    backend = ExportBackend.Torch
    # Create model and re-initialize BNs with non-trivial params so the fold
    # actually bakes a non-identity scale into conv weights.
    model = model_config.model_cls(**model_config.model_kwargs)
    _initialize_bn_with_nontrivial_params(model)
    model.eval()

    config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_input_spec=None,
            op_output_spec=None,
        ),
    )

    quantizer = Quantizer(model, config=config)
    prepared_model = quantizer.prepare((model_config.example_input,))

    # Verify BN count before finalize
    bn_count_before = count_batch_norm_nodes(prepared_model)
    assert bn_count_before == model_config.expected_bn_count_before, (
        f"{model_config.description}: Expected {model_config.expected_bn_count_before} "
        f"BN nodes before finalize, got {bn_count_before}"
    )

    # Capture prepared FQ I/O and overall output
    prepared_fq_io = _capture_fq_io(prepared_model, model_config.example_input)
    with torch.no_grad():
        output_before = prepared_model(model_config.example_input)

    # Finalize with backend (this triggers conv+bn folding)
    finalized_model = quantizer.finalize(backend=backend)

    # Verify BN count after finalize
    bn_count_after = count_batch_norm_nodes(finalized_model)
    assert bn_count_after == model_config.expected_bn_count_after, (
        f"{model_config.description}: Expected {model_config.expected_bn_count_after} "
        f"BN nodes after finalize, got {bn_count_after}"
    )

    # Capture finalized FQ I/O and overall output
    finalized_fq_io = _capture_fq_io(finalized_model, model_config.example_input)
    with torch.no_grad():
        output_after = finalized_model(model_config.example_input)

    # Each FQ module preserved across fold must see bit-identical input and
    # produce bit-identical output. Names match because fold rewires graph args
    # but never renames or recreates FQ modules.
    common_fq_names = sorted(set(prepared_fq_io) & set(finalized_fq_io))
    for fq_name in common_fq_names:
        prepared_in, prepared_out = prepared_fq_io[fq_name]
        finalized_in, finalized_out = finalized_fq_io[fq_name]
        assert torch.equal(prepared_in, finalized_in), (
            f"{model_config.description}: FQ '{fq_name}' input differs between "
            f"prepared and finalized.\n"
            f"  max |diff| = {(prepared_in - finalized_in).abs().max().item():.3e}"
        )
        assert torch.equal(prepared_out, finalized_out), (
            f"{model_config.description}: FQ '{fq_name}' output differs between "
            f"prepared and finalized.\n"
            f"  max |diff| = {(prepared_out - finalized_out).abs().max().item():.3e}"
        )

    # Sanity check that overall outputs stay close. With non-trivial BN params,
    # the prepared graph computes BN via F.batch_norm at runtime while the
    # folded graph computes the equivalent as conv + fused_bias. These are
    # algebraically equal but differ by ~1 ulp per layer in fp32, so we use a
    # loose tolerance here. The bit-exact FQ I/O check above is the strong
    # correctness invariant.
    assert torch.allclose(output_before, output_after, rtol=1e-4, atol=1e-4), (
        f"{model_config.description}: Outputs differ after folding.\n"
        f"Max abs difference: {(output_before - output_after).abs().max().item()}\n"
        f"Mean abs difference: {(output_before - output_after).abs().mean().item()}"
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("model_config", "backend"),
    [
        pytest.param(
            config,
            backend,
            id=f"model:{config.name}-backend:{backend.value}",
            # TODO: investigate hang during conv+BN folding export verification with MLIR path.
            marks=pytest.mark.xfail(
                reason="MLIR export hangs on the large conv+BN model.",
                run=False,
            )
            if config.name == "large_conv_bn" and backend == ExportBackend.CoreAI
            else (),
        )
        for config in TEST_MODELS_FOR_EXPORT
        for backend in [ExportBackend.CoreML, ExportBackend.CoreAI]
    ],
)
def test_conv_bn_folding_export_verification(
    model_config: ModelTestConfig,
    backend: ExportBackend,
) -> None:
    """Verify folded models export correctly to CoreML/CoreAI.

    This test checks that:
    1. The finalized model exports successfully
    2. Exported model outputs match the prepared model
    """
    # Create and prepare model
    model = model_config.model_cls(**model_config.model_kwargs)
    model.eval()

    config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_input_spec=None,
            op_name_config=None,
        ),
    )

    quantizer = Quantizer(model, config=config)
    prepared_model = quantizer.prepare((model_config.example_input,))

    with torch.no_grad():
        prepared_model_output = prepared_model(model_config.example_input)

    # Finalize with backend
    finalized_model = quantizer.finalize(backend=backend)

    # Determine expected ops for export verification
    # Only models with conv weights should have constexpr_blockwise_shift_scale ops
    expected_ops = {}
    if model_config.expected_conv_weights > 0:
        expected_ops["constexpr_blockwise_shift_scale"] = model_config.expected_conv_weights

    # Verify export works correctly and outputs match
    export_utils.convert_and_verify(
        finalized_model=finalized_model,
        input_data=model_config.example_input,
        expected_ops=expected_ops,
        export_backend=backend,
        prepared_model_output=prepared_model_output,
    )


@pytest.mark.parametrize("conv_dtype", [torch.float16, torch.bfloat16])
def test_compute_fused_conv_bn_params_preserves_conv_dtype(
    conv_dtype: torch.dtype,
) -> None:
    """Fused weight/bias must keep the original conv dtype.

    BN running stats and gamma/beta are commonly fp32 even when the conv weight
    is fp16/bf16
    """
    torch.manual_seed(0)
    out_channels = 8
    conv_w = torch.randn(out_channels, 4, 3, 3, dtype=conv_dtype)
    conv_b = torch.randn(out_channels, dtype=conv_dtype)
    # BN params at fp32 — the typical mixed-precision configuration
    bn_running_mean = torch.randn(out_channels, dtype=torch.float32) * 0.5
    bn_running_var = torch.rand(out_channels, dtype=torch.float32) * 2 + 0.5
    bn_weight = torch.randn(out_channels, dtype=torch.float32) * 0.7 + 1.0
    bn_bias = torch.randn(out_channels, dtype=torch.float32) * 0.3

    fused_w, fused_b = _compute_fused_conv_bn_params(
        conv_w,
        conv_b,
        bn_running_mean,
        bn_running_var,
        bn_eps=1e-5,
        bn_weight=bn_weight,
        bn_bias=bn_bias,
    )

    # Sanity: without the cast, conv_w * scale would promote to fp32.
    assert (
        conv_w * (bn_weight / torch.sqrt(bn_running_var + 1e-5)).reshape(-1, 1, 1, 1)
    ).dtype == torch.float32  # noqa: E501

    assert fused_w.dtype == conv_dtype, (
        f"fused_weight dtype was promoted from {conv_dtype} to {fused_w.dtype}"
    )
    assert fused_b.dtype == conv_dtype, (
        f"fused_bias dtype was promoted from {conv_dtype} to {fused_b.dtype}"
    )
