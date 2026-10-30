# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""CPU tests for DKMTrainingConfig: defaults, validation, registration, and build_strategy()."""

import subprocess
import sys

import pytest
from pydantic import ValidationError

from coreai_opt.palettization.kmeans.dkm.training_strategy import (
    DKMTrainingSpec,
    _DKMTrainingStrategy,
    default_dkm_weight_palettization_spec,
)
from coreai_opt.palettization.spec import PerTensorGranularity
from coreai_opt.palettization.spec.training_strategy import TrainingStrategySpec


class TestDKMTrainingConfig:
    def test_defaults(self):
        c = DKMTrainingSpec()
        assert c.tau == 1e-4
        assert c.kmeans_max_iter == 3
        assert c.enforce_zero is False
        assert c.zero_threshold == 1e-7

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"tau": 0.0},
            {"tau": -1.0},
            {"zero_threshold": 0.0},
            {"zero_threshold": -1e-7},
            {"kmeans_max_iter": 0},
            {"kmeans_max_iter": -1},
        ],
    )
    def test_invalid_values_rejected(self, kwargs):
        with pytest.raises(ValidationError):
            DKMTrainingSpec(**kwargs)

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError):
            DKMTrainingSpec(epsilon=1e-4)

    def test_frozen(self):
        c = DKMTrainingSpec()
        with pytest.raises(ValidationError):
            c.tau = 1e-3

    def test_registered_under_dkm(self):
        assert "dkm" in TrainingStrategySpec.REGISTRY
        rebuilt = TrainingStrategySpec.maybe_build_from_dict(
            {"type": "dkm", "tau": 2e-4, "kmeans_max_iter": 5}
        )
        assert isinstance(rebuilt, DKMTrainingSpec)
        assert rebuilt.tau == 2e-4
        assert rebuilt.kmeans_max_iter == 5

    def test_registered_on_palettization_import_alone(self):
        """A fresh process that imports only ``coreai_opt.palettization`` can rebuild a DKM spec."""
        code = (
            "import coreai_opt.palettization\n"
            "from coreai_opt.palettization.spec.training_strategy import TrainingStrategySpec\n"
            "spec = TrainingStrategySpec.maybe_build_from_dict({'type': 'dkm', 'tau': 2e-4})\n"
            "assert type(spec).__name__ == 'DKMTrainingSpec' and spec.tau == 2e-4\n"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_serialization_round_trip(self):
        c = DKMTrainingSpec(tau=3e-4, enforce_zero=True, zero_threshold=1e-6)
        data = c.model_dump()
        assert data["type"] == "dkm"
        assert TrainingStrategySpec.maybe_build_from_dict(data) == c

    def test_build_strategy_threads_fields(self):
        c = DKMTrainingSpec(tau=2e-4, kmeans_max_iter=7, enforce_zero=True, zero_threshold=1e-6)
        s = c.build_strategy()
        assert isinstance(s, _DKMTrainingStrategy)
        assert s.tau == 2e-4
        assert s.kmeans_max_iter == 7
        assert s.enforce_zero is True
        assert s.zero_threshold == 1e-6

    def test_default_dkm_spec(self):
        spec = default_dkm_weight_palettization_spec()
        assert isinstance(spec.training_strategy_spec, DKMTrainingSpec)
        assert spec.n_bits == 4
        assert isinstance(spec.granularity, PerTensorGranularity)
        assert spec.cluster_dim == 1
