# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""General test utilities."""

import importlib.util
from decimal import ROUND_FLOOR, Decimal

import torch

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
        # Floor (not round) the displayed values so they match the raw-value
        # comparison in verify_snr_psnr. Rounding could nudge a sub-threshold
        # value up past the threshold (e.g. 34.9999 -> "35.00"), producing a
        # self-contradictory "PSNR 35.00 below threshold 35.0" message.
        snr_display = floor_to_decimals(str(snr), 2)
        psnr_display = floor_to_decimals(str(psnr), 2)
        if snr <= snr_thresh:
            msg = f"{prefix}SNR {snr_display} below threshold {snr_thresh} (PSNR: {psnr_display})"
        else:
            msg = f"{prefix}PSNR {psnr_display} below threshold {psnr_thresh} (SNR: {snr_display})"
        super().__init__(msg)


def floor_to_decimals(value: str, decimals: int) -> Decimal:
    """Floor a numeric value to a fixed number of decimal places.

    Unlike ``round`` or ``f"{x:.2f}"`` (which round to nearest), this rounds
    toward negative infinity, so the result never exceeds ``value``.

    Args:
        value (str): Numeric value to floor, passed as a string for exact
            decimal parsing (avoids binary float representation error).
        decimals (int): Number of decimal places to keep.

    Returns:
        Decimal: ``value`` floored to ``decimals`` decimal places.
    """
    step = Decimal("1").scaleb(-decimals)
    return Decimal(value).quantize(step, rounding=ROUND_FLOOR)


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
