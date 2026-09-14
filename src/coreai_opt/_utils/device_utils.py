# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Device and build-toolchain availability probes."""

import os
import shutil

import torch
from torch.utils.cpp_extension import CUDA_HOME


def cuda_available() -> bool:
    """Return True if a CUDA device is visible to torch."""
    return bool(torch.cuda.is_available())


def nvcc_available() -> bool:
    """Return True if ``nvcc`` is locatable the way ``cpp_extension`` builds.

    Looks on ``PATH`` first, then under ``CUDA_HOME`` (its env vars / default
    root). Independent of whether a CUDA device is present.
    """
    if shutil.which("nvcc") is not None:
        return True
    return CUDA_HOME is not None and os.path.isfile(os.path.join(CUDA_HOME, "bin", "nvcc"))


def triton_available() -> bool:
    """Return True if triton can be imported.

    Only ``ImportError`` counts as unavailable; other import-time errors (e.g. a
    broken native install) propagate.
    """
    try:
        import triton  # noqa: F401, PLC0415
    except ImportError:
        return False
    return True
