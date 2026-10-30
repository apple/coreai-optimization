# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""DKM (Differentiable K-Means) training-time palettization."""

from .training_strategy import (
    SUPPORTED_CLUSTER_DIMS,
    DKMTrainingSpec,
    default_dkm_weight_palettization_spec,
)

__all__ = [
    "SUPPORTED_CLUSTER_DIMS",
    "DKMTrainingSpec",
    "default_dkm_weight_palettization_spec",
]
