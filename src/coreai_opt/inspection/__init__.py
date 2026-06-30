# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Utilities for inspecting model operations and compression configuration.

This module provides tools for discovering operation names, types, and module
structure in PyTorch models, enabling users to write precise quantization
and compression configurations.

Example::

    import torch
    import torch.nn as nn
    from coreai_opt.inspection import ModelInspector

    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    inspector = ModelInspector(model, (torch.randn(1, 10),), execution_mode="graph")
    print(inspector.format_summary())
"""

from .model_inspector import ModelInspector
from .types import (
    BoundaryEdge,
    InputEdge,
    ModelSummary,
    ModuleContext,
    ModuleInfo,
    OpInfo,
    SourceFrame,
)

__all__ = [
    "BoundaryEdge",
    "InputEdge",
    "ModelInspector",
    "ModelSummary",
    "ModuleContext",
    "ModuleInfo",
    "OpInfo",
    "SourceFrame",
]
