# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""CPU tests for the DKM training strategy and its export path.

The palettizer-level invariant tests are parametrized over ``_SCENARIOS``, a
curated list of ``PalettizationSpec`` combinations covering n_bits, granularity,
cluster_dim, lut_qspec, and enable_per_channel_scale. It is a hand-picked
representative set, not a full Cartesian product: many field combinations are
invalid (e.g. ``enable_fast_kmeans_mode`` forbids ``cluster_dim > 1``), so the
list crosses the settings pairwise around the interesting corners.
"""

import pytest
import torch
import torch.nn as nn

from coreai_opt import ExportBackend
from coreai_opt.palettization import (
    KMeansPalettizer,
    KMeansPalettizerConfig,
    ModuleKMeansPalettizerConfig,
)
from coreai_opt.palettization.kmeans.dkm.training_strategy import DKMTrainingSpec
from coreai_opt.palettization.kmeans.kmeans_fake_palettize import _KMeansFakePalettize
from coreai_opt.palettization.spec import (
    PalettizationSpec,
    PerGroupedChannelGranularity,
    PerTensorGranularity,
)
from coreai_opt.quantization.spec import QuantizationScheme, QuantizationSpec

# Weight shape shared by every scenario: nn.Linear(_IN, _OUT) -> weight (_OUT, _IN).
# Sized so even n_bits=8 (256 centroids) has enough vectors to seed at the
# finest granularity/cluster_dim used below.
_IN, _OUT = 64, 32

_INT8_LUT = QuantizationSpec(dtype=torch.int8, qscheme=QuantizationScheme.SYMMETRIC)

# Each scenario is the set of PalettizationSpec kwargs that differ from the
# defaults (n_bits=4, per-tensor, cluster_dim=1, no LUT quant, no per-channel scale).
_SCENARIOS = [
    pytest.param({}, id="n4-pt"),
    pytest.param({"n_bits": 2, "lut_qspec": _INT8_LUT}, id="n2-pt-int8lut"),
    pytest.param({"n_bits": 8}, id="n8-pt"),
    pytest.param({"n_bits": 3, "enable_per_channel_scale": True}, id="n3-pt-pcs"),
    pytest.param({"n_bits": 6, "cluster_dim": 2}, id="n6-pt-cd2"),
    pytest.param(
        {"granularity": PerGroupedChannelGranularity(axis=0, group_size=8)},
        id="n4-g8-cd1",
    ),
    pytest.param(
        {
            "granularity": PerGroupedChannelGranularity(axis=0, group_size=8),
            "cluster_dim": 2,
            "lut_qspec": _INT8_LUT,
        },
        id="n4-g8-cd2-int8lut",
    ),
    pytest.param(
        {"granularity": PerGroupedChannelGranularity(axis=0, group_size=16), "cluster_dim": 4},
        id="n4-g16-cd4",
    ),
    pytest.param(
        {
            "n_bits": 2,
            "granularity": PerGroupedChannelGranularity(axis=0, group_size=8),
            "enable_per_channel_scale": True,
        },
        id="n2-g8-pcs",
    ),
    pytest.param({"cluster_dim": 2, "enable_per_channel_scale": True}, id="n4-pt-cd2-pcs"),
]


def _dkm_spec(
    *,
    n_bits=4,
    granularity=None,
    cluster_dim=1,
    lut_qspec=None,
    enable_per_channel_scale=False,
    **dkm_kwargs,
):
    return PalettizationSpec(
        n_bits=n_bits,
        granularity=granularity or PerTensorGranularity(),
        cluster_dim=cluster_dim,
        enable_per_channel_scale=enable_per_channel_scale,
        lut_qspec=lut_qspec,
        training_strategy_spec=DKMTrainingSpec(**dkm_kwargs),
    )


def _fp_module(spec):
    return _KMeansFakePalettize(**spec.__dict__)


def _build_and_train(spec_kwargs, *, steps=3):
    """Prepare a single-Linear model for DKM and run ``steps`` training forwards.

    Returns ``(palettizer, prepared, x)`` with the model left in eval mode.
    """
    spec = _dkm_spec(**spec_kwargs)
    model = nn.Sequential(nn.Linear(_IN, _OUT, bias=False))
    config = KMeansPalettizerConfig(
        global_config=ModuleKMeansPalettizerConfig(
            enable_fast_kmeans_mode=(spec.cluster_dim == 1),
            op_state_spec={"weight": spec},
        )
    )
    palettizer = KMeansPalettizer(model, config)
    prepared = palettizer.prepare((torch.randn(2, _IN),))
    x = torch.randn(2, _IN)
    if steps:
        with palettizer.training_mode():
            for _ in range(steps):
                prepared(x)
    prepared.eval()
    return palettizer, prepared, x


class TestDKMSpecMatrix:
    """Invariants that hold for every valid spec combination, at the palettizer level."""

    @pytest.mark.parametrize("spec_kwargs", _SCENARIOS)
    def test_training_forward_reconstructs_and_flows_grad(self, spec_kwargs):
        spec = _dkm_spec(**spec_kwargs)
        model = nn.Sequential(nn.Linear(_IN, _OUT, bias=False))
        config = KMeansPalettizerConfig(
            global_config=ModuleKMeansPalettizerConfig(
                enable_fast_kmeans_mode=(spec.cluster_dim == 1),
                op_state_spec={"weight": spec},
            )
        )
        palettizer = KMeansPalettizer(model, config)
        prepared = palettizer.prepare((torch.randn(2, _IN),))
        x = torch.randn(2, _IN)
        with palettizer.training_mode():
            out = prepared(x)  # DKM soft k-means runs here
            out.sum().backward()

        assert out.shape == (2, _OUT)
        assert torch.isfinite(out).all()
        # Unlike the default (frozen) strategy, DKM's soft assignment is
        # differentiable, so the palettized weight receives a finite gradient.
        weight = prepared[0].parametrizations.weight.original
        assert weight.grad is not None and torch.isfinite(weight.grad).all()

    @pytest.mark.parametrize("spec_kwargs", _SCENARIOS)
    def test_train_eval_consistency(self, spec_kwargs):
        """Train- and eval-mode reconstructions of the same weight must agree at tiny ``tau``.

        A sparse tail can differ: train applies ``attn`` (assigned against the pre-update
        centroids) to the post-update quantized centroids, while eval hard-assigns against the
        final centroids. With in-loop LUT quant the last rounding can shift a centroid across
        the int8 grid and flip a few boundary weights by ~a centroid spacing (up to ~0.6% of
        weights in quant scenarios, ~0% otherwise). Bound the 99th percentile, not the max.
        """
        m = _fp_module(_dkm_spec(**spec_kwargs, tau=1e-6))
        w = torch.randn(_OUT, _IN)
        m._initialize(w)

        m.train()
        for _ in range(3):
            train_out = m.forward(w)

        m.eval()
        eval_out = m.forward(w)
        assert (train_out - eval_out).abs().quantile(0.99) < 1e-4

    @pytest.mark.parametrize("spec_kwargs", _SCENARIOS)
    def test_state_dict_roundtrip_preserves_eval_output(self, spec_kwargs):
        _, prepared, x = _build_and_train(spec_kwargs)
        ref = prepared(x)

        _, reloaded, _ = _build_and_train(spec_kwargs, steps=0)
        reloaded.load_state_dict(prepared.state_dict())
        reloaded.eval()
        assert torch.equal(reloaded(x), ref)

    @pytest.mark.parametrize("spec_kwargs", _SCENARIOS)
    def test_finalize_torch_roundtrip(self, spec_kwargs):
        palettizer, prepared, x = _build_and_train(spec_kwargs)
        eval_out = prepared(x)
        finalized = palettizer.finalize(backend=ExportBackend.Torch)
        assert torch.equal(eval_out, finalized(x))

    def test_finalize_coreai_backend_succeeds(self):
        palettizer, _, _ = _build_and_train({})
        finalized = palettizer.finalize(backend=ExportBackend.CoreAI)
        assert finalized is not None


class TestDKMStrategyInternals:
    """Module-level checks on behavior the palettizer workflow does not expose."""

    def test_guard_rejects_non_kmeans_module(self):
        strategy = DKMTrainingSpec().build_strategy()
        with pytest.raises(TypeError):
            strategy.train_forward(object(), torch.randn(4, 8))

    @pytest.mark.parametrize("cluster_dim", [3, 8])
    def test_guard_rejects_unsupported_cluster_dim(self, cluster_dim):
        strategy = DKMTrainingSpec().build_strategy()
        module = _fp_module(_dkm_spec(cluster_dim=cluster_dim))
        with pytest.raises(ValueError, match="cluster_dim"):
            strategy.train_forward(module, torch.randn(_OUT, _IN))

    def test_train_forward_mutates_centroids_and_marks_indices_stale(self):
        m = _fp_module(_dkm_spec())
        w = torch.randn(_OUT, _IN)
        m._initialize(w)  # seed centroids without a DKM step
        before = m.centroids.clone()

        m.train()
        out = m.forward(w)

        assert out.shape == w.shape
        assert not torch.equal(m.centroids, before)  # DKM mutated the centroids
        assert m._indices_stale is True

    def test_enforce_zero_reserves_exact_zero_code(self):
        m = _fp_module(_dkm_spec(enforce_zero=True))
        w = torch.randn(_OUT, _IN)
        m._initialize(w)
        m.train()
        m.forward(w)
        # the nearest-origin centroid is pinned, so an exact-zero code exists
        assert torch.all(m.centroids.norm(dim=-1).amin(dim=1) == 0)
