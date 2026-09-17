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
    untested_torch_version as _untested_torch_version,
)
from coreai_opt.common import TorchaoTorchIncompatibilityWarning, UntestedTorchVersionWarning

_incompatibility = _torchao_torch_incompatibility(
    importlib.metadata.version("torchao"), torch.__version__
)
if _incompatibility:
    warnings.warn(_incompatibility, TorchaoTorchIncompatibilityWarning, stacklevel=2)

_untested = _untested_torch_version(torch.__version__)
if _untested:
    warnings.warn(_untested, UntestedTorchVersionWarning, stacklevel=2)

from . import palettization, pruning, quantization  # noqa: E402
from ._about import __version__  # noqa: E402
from ._plugins import load_plugins as _load_plugins  # noqa: E402
from .common import CoreMLExportError, DependencyVersionWarning, ExportBackend  # noqa: E402

__all__ = [
    "CoreMLExportError",
    "DependencyVersionWarning",
    "ExportBackend",
    "TorchaoTorchIncompatibilityWarning",
    "UntestedTorchVersionWarning",
    "__version__",
]

# Last statement in the module: a plugin's registration code runs while this import is
# still in flight, so everything it may reach for has to be bound before this call.
_load_plugins()
