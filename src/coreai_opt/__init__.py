# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""coreai_opt - A library for PyTorch model compression and optimizations.

For deployment via Core AI on Apple Silicon.
"""

import importlib.metadata
import warnings

import torch

from coreai_opt._utils.version_utils import (
    torchao_torch_incompatibility as _torchao_torch_incompatibility,
)

_incompatibility = _torchao_torch_incompatibility(
    importlib.metadata.version("torchao"), torch.__version__
)
if _incompatibility:
    warnings.warn(_incompatibility, UserWarning, stacklevel=2)

from . import palettization, pruning, quantization  # noqa: E402
from ._about import __version__  # noqa: E402
from .common import CoreMLExportError, ExportBackend  # noqa: E402

__all__ = [
    "CoreMLExportError",
    "ExportBackend",
    "__version__",
]
