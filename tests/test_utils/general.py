# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""General test utilities."""

import importlib.util

import torch
from torch.ops import coreai

COREAI_AVAILABLE = importlib.util.find_spec("coreai") is not None


class SNRBelowThresholdError(AssertionError):
    """Raised when SNR or PSNR is below the required threshold."""

    def __init__(
        self,
        snr: float,
        psnr: float,
        snr_thresh: float,
        psnr_thresh: float,
        prefix: str = "",
    ) -> None:
        if snr <= snr_thresh:
            msg = f"{prefix}SNR {snr:.2f} below threshold {snr_thresh} (PSNR: {psnr:.2f})"
        else:
            msg = f"{prefix}PSNR {psnr:.2f} below threshold {psnr_thresh} (SNR: {snr:.2f})"
        super().__init__(msg)


def compute_snr_psnr(
    data: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[float, float]:
    """Compute Signal-to-Noise Ratio and Peak Signal-to-Noise Ratio.

    Compares a data tensor against a reference tensor, treating their difference
    as noise for SNR/PSNR calculation.

    Args:
        data: Data tensor to compare
        reference: Reference tensor

    Returns:
        Tuple of (SNR, PSNR) values

    """
    assert len(data) == len(reference), f"Tensor length mismatch: {len(data)} vs {len(reference)}"

    eps = 1e-5
    eps2 = 1e-10
    noise = data - reference
    noise_var = torch.sum(noise**2) / len(noise)
    signal_energy = torch.sum(reference**2) / len(reference)
    max_signal_energy = torch.amax(reference**2)
    snr = 10 * torch.log10((signal_energy + eps) / (noise_var + eps2))
    psnr = 10 * torch.log10((max_signal_energy + eps) / (noise_var + eps2))
    return snr.item(), psnr.item()


def verify_snr_psnr(
    data: torch.Tensor,
    reference: torch.Tensor,
    snr_thresh: float,
    psnr_thresh: float,
    label: str = "",
) -> None:
    """Verify SNR and PSNR meet thresholds.

    Args:
        data: Data tensor to compare (will be flattened)
        reference: Reference tensor (will be flattened)
        snr_thresh: Minimum acceptable SNR value
        psnr_thresh: Minimum acceptable PSNR value
        label: Optional label for error messages

    Raises:
        SNRBelowThresholdError: If SNR or PSNR is below the threshold
    """
    data_flat = data.float().flatten()
    reference_flat = reference.float().flatten()

    snr, psnr = compute_snr_psnr(data_flat, reference_flat)

    prefix = f"{label}: " if label else ""

    if snr <= snr_thresh or psnr <= psnr_thresh:
        raise SNRBelowThresholdError(snr, psnr, snr_thresh, psnr_thresh, prefix)


def get_fake_quant_nodes(model: torch.fx.GraphModule) -> list[torch.fx.Node]:
    """Return the activation_post_process observer nodes in a prepared graph."""
    return [node for node in model.graph.nodes if "activation_post_process" in node.name]


def assert_single_call_function_node(
    gm: torch.fx.GraphModule, target_substr: str, *, stage: str = ""
) -> torch.fx.Node:
    """Assert exactly one call_function node's target contains a substring.

    Current usage: externalized composite ops get a process-unique op name (sanitized module
    path plus a uuid4 suffix), so there is no stable target object to compare
    against and a substring match on ``str(node.target)`` is required.

    Args:
        gm: The graph module to search
        target_substr: Substring to match against ``str(node.target)``
        stage: Optional pipeline stage name, used only in the error message

    Returns:
        The single matching node

    """
    matches = [
        n for n in gm.graph.nodes if n.op == "call_function" and target_substr in str(n.target)
    ]
    where = f" in the {stage.upper()} graph" if stage else ""
    assert len(matches) == 1, (
        f"Expected exactly one call_function matching '{target_substr}'"
        f"{where}; got {[n.name for n in matches]}"
    )
    return matches[0]


def is_coreai_quantize(target: object) -> bool:
    """Whether an FX node target is the ``coreai::quantize`` op.

    Compares by object identity against ``torch.ops.coreai``. Both forms are
    matched because they are distinct objects and which one appears depends
    on the pipeline stage: ``Quantizer.finalize`` emits the
    ``OpOverloadPacket`` (``coreai.quantize``) while a decomposed
    ExportedProgram carries the ``OpOverload`` (``coreai.quantize.default``).

    The op resolves lazily on first call, so this raises AttributeError if
    the coreai op namespace was never registered (see ``COREAI_AVAILABLE``).
    """
    return target is coreai.quantize or target is coreai.quantize.default


def is_coreai_dequantize(target: object) -> bool:
    """Whether an FX node target is the ``coreai::dequantize`` op."""
    return target is coreai.dequantize or target is coreai.dequantize.default


def get_quantize_dtype(node: torch.fx.Node) -> torch.dtype | None:
    """Return the quantized dtype carried in an FX node's args, else None.

    ``coreai::quantize`` takes ``(input, scale, dtype)``, so its dtype is the
    third positional arg. ``coreai::dequantize`` takes ``(input, scale)`` and
    carries no dtype, so this returns None for a dequantize node. To read the
    dtype at a dequantize boundary, pass the quantize node feeding it
    (``dequantize_node.args[0]``).

    Args:
        node: The FX node to inspect

    Returns:
        The first ``torch.dtype`` positional arg, or None if there is none

    """
    return next((a for a in node.args if isinstance(a, torch.dtype)), None)
