# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Pruning parametrization config and the fixture that provides it."""

from dataclasses import dataclass

import pytest

from coreai_opt import ExportBackend
from coreai_opt.pruning import MagnitudePrunerConfig, ModuleMagnitudePrunerConfig, PruningSpec
from coreai_opt.pruning.spec import ChannelStructured, PruningScheme, Unstructured


@dataclass
class ParametrizedPruneConfigs:
    """Container for parametrized pruning configs.

    Attributes:
        config: MagnitudePrunerConfig instance.
        target_sparsity: Target sparsity fraction.
        pruning_scheme: PruningScheme instance (Unstructured or ChannelStructured).
        backend: Export backend (CoreML or CoreAI).
    """

    config: MagnitudePrunerConfig
    target_sparsity: float
    pruning_scheme: PruningScheme | str
    backend: ExportBackend

    @classmethod
    def from_prune_params(
        cls,
        target_sparsity: float,
        pruning_scheme: PruningScheme | str,
        backend: ExportBackend,
    ) -> "ParametrizedPruneConfigs":
        spec = PruningSpec(target_sparsity=target_sparsity, pruning_scheme=pruning_scheme)
        config = MagnitudePrunerConfig(
            global_config=ModuleMagnitudePrunerConfig(op_state_spec={"weight": spec})
        )
        return cls(
            config=config,
            target_sparsity=target_sparsity,
            pruning_scheme=pruning_scheme,
            backend=backend,
        )


@pytest.fixture(
    params=[
        (target_sparsity, pruning_scheme, backend)
        for target_sparsity in [0.25, 0.5, 0.75]
        for pruning_scheme in [Unstructured(), ChannelStructured(axis=0)]
        for backend in [ExportBackend.CoreML, ExportBackend.CoreAI]
    ],
    ids=lambda p: f"sparsity:{p[0]}-scheme:{p[1].__class__.__name__}-backend:{p[2].value}",
)
def parametrized_prune_config(
    request: pytest.FixtureRequest,
) -> ParametrizedPruneConfigs:
    """Fixture for pruning configs parametrized across sparsity, scheme, and backend."""
    target_sparsity, pruning_scheme, backend = request.param
    return ParametrizedPruneConfigs.from_prune_params(target_sparsity, pruning_scheme, backend)
