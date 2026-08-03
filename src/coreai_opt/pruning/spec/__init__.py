# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Pruning spec components: specs, schemes, and parametrizations."""

from .prune import PruneImplBase, _MagnitudePruneImpl
from .scheme import BlockStructured, ChannelStructured, NMStructured, PruningScheme, Unstructured
from .spec import PruningSpec, default_weight_pruning_spec

__all__ = [
    "BlockStructured",
    "ChannelStructured",
    "NMStructured",
    "PruneImplBase",
    "PruningScheme",
    "PruningSpec",
    "Unstructured",
    "default_weight_pruning_spec",
]
