# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from decimal import Decimal

from tests.test_utils.general import SNRBelowThresholdError, floor_to_decimals


def test_floor_to_decimals():
    assert floor_to_decimals("3.149", 2) == Decimal("3.14")
    assert floor_to_decimals("3.140", 2) == Decimal("3.14")
    assert floor_to_decimals("-3.141", 2) == Decimal("-3.15")
    # A value just below a threshold must floor down, never round up.
    assert floor_to_decimals("34.9999", 2) == Decimal("34.99")


def test_snr_below_threshold_error_floors_displayed_value():
    # PSNR is below threshold; the displayed value must read "34.99", not the
    # rounded-up "35.00" that would contradict "below threshold 35.0".
    error = SNRBelowThresholdError(snr=100.0, psnr=34.9999, snr_thresh=80.0, psnr_thresh=35.0)
    message = str(error)
    assert "PSNR 34.99 below threshold 35.0" in message
    assert "35.00" not in message
