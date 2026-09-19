# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""End-to-end tests for the casting module's public API.

Tests cast_fp32_to_fp16, cast_int32_to_int16, and cast_to_16_bit_precision
on exported programs, verifying output fidelity, dtype distribution changes,
and weight size reduction.
"""

from collections import Counter

import pytest
import torch
import torch.nn as nn

from coreai_opt import ExportBackend
from coreai_opt.casting import (
    cast_fp32_to_fp16,
    cast_int32_to_int16,
    cast_to_16_bit_precision,
)
from coreai_opt.quantization import (
    Quantizer,
    QuantizerConfig,
)
from tests.models.simple import GatedMLPModel, SimpleModel

try:
    from torchvision.models import ResNet18_Weights, resnet18

    _HAS_TORCHVISION = True
except ImportError:
    _HAS_TORCHVISION = False


# =============================================================================
# Test models
# =============================================================================
class _IntDataModel(nn.Module):
    """Model with int data ops (add tensor+tensor) and unsafe embedding indices.

    Input: randint(0, 100, (1, 5)).
    """

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 16)
        self.linear = nn.Linear(16, 10)
        self.register_buffer("ones", torch.ones(1, dtype=torch.int64))

    def forward(self, indices):
        x = self.linear(self.embedding(indices))
        top_idx = torch.argmax(x, dim=-1)
        return x, top_idx + self.ones


class _ConstantFoldableIntModel(nn.Module):
    """Model with a constant-foldable integer subgraph (arange + arithmetic).

    Simulates the EfficientSAM pattern: integer creation ops feed into
    arithmetic with no user input dependency. These should NOT be cast to
    int16 because the downstream compiler should constant-fold them.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 10)
        self.register_buffer("offset", torch.tensor([1, 2, 3, 4, 5]))

    def forward(self, x):
        out = self.linear(x)
        indices = torch.arange(5) + self.offset
        return out, indices


class _PassthroughIntModel(nn.Module):
    """Model with an integer input that passes through untouched to the output.

    Simulates the emformer pattern: an si32 scalar passes through the graph
    without being used by any arithmetic op.
    """

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 8)

    def forward(self, x, length):
        return self.linear(x), length


# =============================================================================
# Helpers
# =============================================================================
def _export(model: nn.Module, example_input: tuple | torch.Tensor) -> torch.export.ExportedProgram:
    """Export a model to an ExportedProgram with decompositions applied."""
    if isinstance(example_input, torch.Tensor):
        example_input = (example_input,)
    return torch.export.export(model, example_input, strict=False).run_decompositions()


def _quantize_and_export(
    model: nn.Module, example_input: torch.Tensor
) -> torch.export.ExportedProgram:
    """Apply int8 quantization, finalize for CoreAI, then export."""
    config = QuantizerConfig(execution_mode="eager")
    quantizer = Quantizer(model, config)
    quantizer.prepare((example_input,))
    finalized = quantizer.finalize(backend=ExportBackend.CoreAI)
    return _export(finalized, example_input)


def _snr_db(ref: torch.Tensor, test: torch.Tensor) -> float:
    """Compute signal-to-noise ratio in dB."""
    ref_f = ref.float().flatten()
    noise = (ref_f - test.float().flatten()).norm()
    if noise == 0:
        return float("inf")
    return (20 * torch.log10(ref_f.norm() / noise)).item()


def _count_op_dtypes(ep: torch.export.ExportedProgram) -> Counter:
    """Count output dtypes across all call_function nodes."""
    counts: Counter = Counter()
    for node in ep.graph.nodes:
        if node.op != "call_function":
            continue
        val = node.meta.get("val")
        if hasattr(val, "dtype"):
            counts[val.dtype] += 1
        elif isinstance(val, (tuple, list)):
            for v in val:
                if hasattr(v, "dtype"):
                    counts[v.dtype] += 1
    return counts


def _total_param_bytes(ep: torch.export.ExportedProgram) -> int:
    """Sum nbytes of all tensors in state_dict."""
    return sum(t.nbytes for t in ep.state_dict.values() if isinstance(t, torch.Tensor))


def _run_ep(ep: torch.export.ExportedProgram, example_input: torch.Tensor) -> torch.Tensor:
    """Run an exported program and return the first tensor output."""
    if isinstance(example_input, torch.Tensor):
        example_input = (example_input,)
    out = ep.module()(*example_input)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def _fp16_ratio(counts: Counter) -> float:
    """Return the fraction of float ops that are fp16 (vs fp32+fp16 total)."""
    fp32 = counts.get(torch.float32, 0)
    fp16 = counts.get(torch.float16, 0)
    total = fp32 + fp16
    return fp16 / total if total > 0 else 0.0


# =============================================================================
# Fixtures
# =============================================================================
_FP_TOY_MODELS = [
    pytest.param(
        lambda: (SimpleModel().eval(), torch.randn(1, 1, 28, 28)),
        id="simple_conv_linear",
    ),
    pytest.param(
        lambda: (GatedMLPModel().eval(), torch.randn(1, 4, 32)),
        id="gated_mlp",
    ),
]

_FP_FULL_MODELS = [
    pytest.param(
        lambda: (resnet18(weights=ResNet18_Weights.DEFAULT).eval(), torch.randn(1, 3, 224, 224)),
        id="resnet18",
        marks=pytest.mark.skipif(not _HAS_TORCHVISION, reason="torchvision not installed"),
    ),
]


@pytest.fixture(params=_FP_TOY_MODELS)
def fp_toy_model_and_input(request):
    """Yield (model, example_input) for lightweight FP32 models."""
    return request.param()


@pytest.fixture(params=_FP_FULL_MODELS)
def fp_full_model_and_input(request):
    """Yield (model, example_input) for full-size pretrained FP32 models."""
    return request.param()


# =============================================================================
# FP32 -> FP16 tests
# =============================================================================
class TestCastFp32ToFp16:
    """Test cast_fp32_to_fp16 on lightweight models."""

    def test_cast(self, fp_toy_model_and_input):
        """Verify SNR, dtype shift, and weight reduction in a single cast run."""
        model, example_input = fp_toy_model_and_input
        ep = _export(model, example_input)
        ref_out = _run_ep(ep, example_input)
        before_bytes = _total_param_bytes(ep)
        assert _fp16_ratio(_count_op_dtypes(ep)) == 0.0, "Should start with no FP16 ops"

        cast_fp32_to_fp16(ep)
        cast_out = _run_ep(ep, example_input.half())

        # Output fidelity
        snr = _snr_db(ref_out, cast_out)
        assert snr > 30, f"SNR too low: {snr:.1f} dB"

        # Nearly all float ops converted to fp16
        ratio = _fp16_ratio(_count_op_dtypes(ep))
        assert ratio >= 0.95, f"Expected >= 95% FP16 ops, got {ratio:.0%}"

        # Weight size roughly halved
        reduction = 1 - _total_param_bytes(ep) / before_bytes
        assert reduction >= 0.45, f"Weight reduction only {reduction:.0%}, expected >= 45%"


# =============================================================================
# INT32 -> INT16 tests
# =============================================================================
class TestCastInt32ToInt16:
    """Test cast_int32_to_int16."""

    def test_int_data_model(self):
        """Int64 data ops narrowed to int16, but embedding indices stay wide."""
        model = _IntDataModel().eval()
        example_input = torch.randint(0, 100, (1, 5))
        ep = _export(model, example_input)
        before = _count_op_dtypes(ep)

        cast_int32_to_int16(ep)
        after = _count_op_dtypes(ep)

        # Int16 ops should appear
        assert after.get(torch.int16, 0) > before.get(torch.int16, 0), "Int16 count should increase"

        # Embedding's index input must not be narrowed
        for node in ep.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.embedding.default:
                index_node = node.args[1]
                val = index_node.meta.get("val")
                assert val.dtype != torch.int16, (
                    f"Embedding index should not be int16, got {val.dtype}"
                )
                break

    def test_constant_foldable_not_cast(self):
        """Integer ops in constant-foldable subgraphs should not be cast to int16."""
        model = _ConstantFoldableIntModel().eval()
        example_input = torch.randn(1, 5)
        ep = _export(model, example_input)

        cast_int32_to_int16(ep)

        # No node in the graph should have int16 dtype — the only integer ops
        # are in the constant-foldable path which should be left alone
        for node in ep.graph.nodes:
            if node.op != "call_function":
                continue
            val = node.meta.get("val")
            if hasattr(val, "dtype"):
                assert val.dtype != torch.int16, (
                    f"Node {node.name} ({node.target}) should not be int16 "
                    f"in a constant-foldable-only integer graph"
                )

    def test_passthrough_int_input_not_narrowed(self):
        """Integer inputs that pass through untouched should not be narrowed to int16."""
        model = _PassthroughIntModel().eval()
        example_inputs = (torch.randn(1, 4), torch.tensor([42]))
        ep = _export(model, example_inputs)

        # Find the integer user input placeholder before casting
        length_placeholder = None
        for node in ep.graph.nodes:
            if node.op == "placeholder":
                val = node.meta.get("val")
                if hasattr(val, "dtype") and val.dtype in {torch.int32, torch.int64}:
                    length_placeholder = node
                    break
        assert length_placeholder is not None, "Should find an integer placeholder"
        original_dtype = length_placeholder.meta["val"].dtype

        cast_int32_to_int16(ep)

        assert length_placeholder.meta["val"].dtype == original_dtype, (
            f"Passthrough int input should stay {original_dtype}, "
            f"got {length_placeholder.meta['val'].dtype}"
        )


# =============================================================================
# Combined cast tests
# =============================================================================
class TestCastTo16BitPrecision:
    """Test cast_to_16_bit_precision (FP + INT combined)."""

    def test_cast(self, fp_toy_model_and_input):
        """Verify SNR and dtype shift in a single combined cast run."""
        model, example_input = fp_toy_model_and_input
        ep = _export(model, example_input)
        ref_out = _run_ep(ep, example_input)
        assert _fp16_ratio(_count_op_dtypes(ep)) == 0.0

        cast_to_16_bit_precision(ep)
        cast_out = _run_ep(ep, example_input.half())

        # Output fidelity
        snr = _snr_db(ref_out, cast_out)
        assert snr > 30, f"SNR too low: {snr:.1f} dB"

        # Nearly all float ops converted to fp16
        ratio = _fp16_ratio(_count_op_dtypes(ep))
        assert ratio >= 0.95, f"Expected >= 95% FP16 ops, got {ratio:.0%}"


# =============================================================================
# Quantized model + casting tests
# =============================================================================
@pytest.fixture(params=_FP_TOY_MODELS)
def quantized_ep_and_input(request):
    """Yield (exported_program, example_input) for quantized FP32 models."""
    model, example_input = request.param()
    ep = _quantize_and_export(model, example_input)
    return ep, example_input


class TestCastAfterQuantization:
    """Test that casting works correctly on quantized (weight-only int8) models."""

    def test_fp16_cast(self):
        """FP16 cast converts remaining fp32 params to fp16, leaves int8 quantized buffers intact.

        Uses a partially-quantized SimpleModel (conv quantized, linear skipped) to verify
        both behaviors explicitly: unquantized fp32 weights move to fp16, int8 quantized_data
        buffers are untouched.
        """
        model = SimpleModel().eval()
        example_input = torch.randn(1, 1, 28, 28)

        # Quantize only conv; leave linear fp32 so we have a clear unquantized weight to track
        config = QuantizerConfig(
            execution_mode="eager",
            module_type_configs={nn.Linear: None},
        )
        quantizer = Quantizer(model, config)
        quantizer.prepare((example_input,))
        ep = _export(quantizer.finalize(backend=ExportBackend.CoreAI), example_input)
        ref_out = _run_ep(ep, example_input)

        int8_params = {n for n, t in ep.state_dict.items() if t.dtype == torch.int8}
        fp32_params = {
            n for n, t in ep.state_dict.items() if t.dtype == torch.float32 and t.numel() > 0
        }
        assert "conv.parametrizations.weight.0.quantized_data" in int8_params, (
            "Expected int8 params from conv quantization"
        )
        assert "linear.weight" in fp32_params, "Expected fp32 params from unquantized linear"

        cast_fp32_to_fp16(ep)
        cast_out = _run_ep(ep, example_input.half())

        snr = _snr_db(ref_out, cast_out)
        assert snr > 30, f"Quantized model FP16 SNR too low: {snr:.1f} dB"

        ratio = _fp16_ratio(_count_op_dtypes(ep))
        assert ratio >= 0.9, f"Expected >= 90% FP16 ops, got {ratio:.0%}"

        for name in fp32_params:
            assert ep.state_dict[name].dtype == torch.float16, f"{name}: expected fp16 after cast"
        for name in int8_params:
            assert ep.state_dict[name].dtype == torch.int8, f"{name}: expected int8 to be unchanged"

    def test_combined_cast(self, quantized_ep_and_input):
        """Quantized model: verify SNR and dtype shift after combined 16-bit cast."""
        ep, example_input = quantized_ep_and_input
        ref_out = _run_ep(ep, example_input)
        assert _fp16_ratio(_count_op_dtypes(ep)) == 0.0

        cast_to_16_bit_precision(ep)
        cast_out = _run_ep(ep, example_input.half())

        snr = _snr_db(ref_out, cast_out)
        assert snr > 30, f"Quantized model combined cast SNR too low: {snr:.1f} dB"

        ratio = _fp16_ratio(_count_op_dtypes(ep))
        assert ratio >= 0.95, f"Expected >= 95% FP16 ops, got {ratio:.0%}"


# =============================================================================
# Full model tests (pretrained, slower)
# =============================================================================
@pytest.mark.slow
class TestCastFullModels:
    """End-to-end cast tests on full pretrained models."""

    def test_cast_to_16_bit(self, fp_full_model_and_input):
        """Verify SNR, dtype shift, and weight reduction on a full model."""
        model, example_input = fp_full_model_and_input
        ep = _export(model, example_input)
        ref_out = _run_ep(ep, example_input)
        before_bytes = _total_param_bytes(ep)

        cast_to_16_bit_precision(ep)
        cast_out = _run_ep(ep, example_input.half())

        snr = _snr_db(ref_out, cast_out)
        assert snr > 30, f"SNR too low: {snr:.1f} dB"

        ratio = _fp16_ratio(_count_op_dtypes(ep))
        assert ratio >= 0.95, f"Expected >= 95% FP16 ops, got {ratio:.0%}"

        reduction = 1 - _total_param_bytes(ep) / before_bytes
        assert reduction >= 0.45, f"Weight reduction only {reduction:.0%}"


# =============================================================================
# Selective Op Skipping tests (Issue #7)
# =============================================================================
class _ActivationOverflowModel(nn.Module):
    """Model that triggers FP16 activation overflow in intermediate ops.

    At x = 15.0:
    exp(15.0) ~ 3.27e6 > 65504 (overflows to inf in FP16).
    log1p(exp(15.0)) ~ 15.0 (fits comfortably in FP16).
    """

    def forward(self, x):
        return torch.log1p(torch.exp(x))


class _TwoExpModel(nn.Module):
    """Model with two separate exp operations."""

    def forward(self, x):
        a = torch.exp(x)
        b = torch.exp(a * 0.001)
        return b


class _CreationModel(nn.Module):
    """Model with tensor creation op."""

    def forward(self, x):
        c = torch.ones_like(x)
        return x + c


class TestSelectiveOpSkipping:
    """Tests for selective op skipping in FP16 casting (Issue #7)."""

    def test_ignored_op_prevents_activation_overflow(self):
        """Ignoring exp and log1p keeps intermediate ops in FP32, preventing inf."""
        model = _ActivationOverflowModel().eval()
        x = torch.tensor([[15.0]])

        # Baseline: standard casting overflows to inf
        ep_std = _export(model, x)
        cast_fp32_to_fp16(ep_std)
        out_std = _run_ep(ep_std, x.half())
        assert torch.isinf(out_std).any(), "Expected inf without ignored_ops"

        # With ignored_ops: stays finite and matches reference
        ep_skipped = _export(model, x)
        cast_fp32_to_fp16(ep_skipped, ignored_ops={torch.ops.aten.exp, torch.ops.aten.log1p})
        out_skipped = _run_ep(ep_skipped, x.half())
        assert not torch.isinf(out_skipped).any(), "Expected finite output with ignored_ops"
        assert torch.allclose(out_skipped.float(), x, atol=1e-3)

    @pytest.mark.parametrize(
        "spec",
        [
            pytest.param({torch.ops.aten.exp.default}, id="op_overload_set"),
            pytest.param([torch.ops.aten.exp.default], id="op_overload_list"),
            pytest.param({torch.ops.aten.exp}, id="op_overload_packet_set"),
            pytest.param([torch.ops.aten.exp], id="op_overload_packet_list"),
            pytest.param(torch.ops.aten.exp, id="single_op_packet"),
            pytest.param(torch.ops.aten.exp.default, id="single_op_overload"),
            pytest.param(
                lambda node: node.target == torch.ops.aten.exp.default, id="predicate_target"
            ),
            pytest.param(lambda node: node.name.startswith("exp"), id="predicate_name"),
        ],
    )
    def test_ignored_op_matching_types(self, spec):
        """All supported op identifier formats correctly preserve op in FP32."""
        model = _ActivationOverflowModel().eval()
        x = torch.tensor([[1.0]])
        ep = _export(model, x)

        cast_fp32_to_fp16(ep, ignored_ops=spec)

        # exp node must have float32 output dtype
        exp_node = next(
            n
            for n in ep.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.exp.default
        )
        assert exp_node.meta["val"].dtype == torch.float32

    def test_target_specific_node_name(self):
        """Targeting a specific node name via predicate preserves only that instance in FP32."""
        model = _TwoExpModel().eval()
        x = torch.tensor([[1.0]])
        ep = _export(model, x)

        exp_nodes = [
            n
            for n in ep.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.exp.default
        ]
        assert len(exp_nodes) == 2
        second_node_name = exp_nodes[1].name

        # Ignore only the second exp node via predicate function
        cast_fp32_to_fp16(ep, ignored_ops=lambda node: node.name == second_node_name)

        assert exp_nodes[0].meta["val"].dtype == torch.float16
        assert exp_nodes[1].meta["val"].dtype == torch.float32

    def test_consecutive_ignored_ops_eliminate_redundant_casts(self):
        """Consecutive ignored ops do not have redundant intermediate casts between them."""
        model = _ActivationOverflowModel().eval()
        x = torch.tensor([[1.0]])
        ep = _export(model, x)

        cast_fp32_to_fp16(ep, ignored_ops={torch.ops.aten.exp, torch.ops.aten.log1p})

        # There should be NO cast between exp and log1p
        exp_node = next(
            n
            for n in ep.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.exp.default
        )
        log1p_node = next(
            n
            for n in ep.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.log1p.default
        )

        # log1p's direct argument should be exp_node directly, not an intermediate cast
        assert log1p_node.args[0] is exp_node

    def test_ignored_creation_op(self):
        """Ignored creation op remains FP32."""
        model = _CreationModel().eval()
        x = torch.randn(2, 2)
        ep = _export(model, x)

        cast_fp32_to_fp16(ep, ignored_ops={torch.ops.aten.full_like})

        full_node = next(
            n for n in ep.graph.nodes if n.op == "call_function" and "full_like" in str(n.target)
        )
        assert full_node.meta["val"].dtype == torch.float32

    def test_cast_to_16_bit_precision_forwards_ignored_ops(self):
        """cast_to_16_bit_precision forwards ignored_ops to FP16 casting."""
        model = _ActivationOverflowModel().eval()
        x = torch.tensor([[15.0]])
        ep = _export(model, x)

        cast_to_16_bit_precision(ep, ignored_ops={torch.ops.aten.exp, torch.ops.aten.log1p})
        out = _run_ep(ep, x.half())
        assert not torch.isinf(out).any()
        assert torch.allclose(out.float(), x, atol=1e-3)
