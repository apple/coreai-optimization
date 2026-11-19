# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Shared pytest skip markers for optional dependencies and Python version limits."""

import importlib.util
import sys

import pytest

COREMLTOOLS_AVAILABLE = importlib.util.find_spec("coremltools") is not None
COREMLTOOLS_SKIP_REASON = "coremltools is not installed"

requires_coremltools = pytest.mark.skipif(not COREMLTOOLS_AVAILABLE, reason=COREMLTOOLS_SKIP_REASON)
requires_torch_compile = pytest.mark.skipif(
    sys.version_info >= (3, 15), reason="torch.compile is not supported on Python 3.15+"
)
