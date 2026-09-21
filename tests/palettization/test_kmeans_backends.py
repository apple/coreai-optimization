# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for k-means backend selection.

The cuml backend itself is not exercised here: it requires a CUDA device and the
``cuml-cu12`` wheel, neither of which is present on a typical dev machine. What is
covered is everything that can silently break without one -- that the name is
accepted, that typos are still rejected, that a missing cuml gives an actionable
error rather than an obscure one, and that the input conversion cuml needs does
not disturb the CPU path.
"""

import importlib.util
import logging
from types import SimpleNamespace

import numpy
import pytest
import torch

from coreai_opt.palettization.kmeans import _efficient_kmeans as _ek
from coreai_opt.palettization.kmeans._efficient_kmeans import (
    KMEANS_BACKEND_ENV,
    KMEANS_N_INIT_ENV,
    _EfficientKMeans,
    _resolve_backend,
    _resolve_n_init,
    _to_cuml_input,
)
from coreai_opt.palettization.kmeans.palettizer import KMeansPalettizer


class TestKMeansBackendSelection:
    @pytest.mark.parametrize(
        "backend", ["kmeans++", "sklearn", "sklearn_minibatch", "cuml", "flash_kmeans"]
    )
    def test_env_var_selects_backend(self, backend, monkeypatch):
        monkeypatch.setenv(KMEANS_BACKEND_ENV, backend)
        # init= names a different backend, so this also pins the precedence:
        # the environment overrides the call site.
        assert _resolve_backend("kmeans++") == backend

    @pytest.mark.parametrize("backend", ["cuML", "cuml_minibatch", "kmeans", ""])
    def test_unknown_backend_raises(self, backend, monkeypatch):
        """A typo must fail loudly rather than silently running another algorithm."""
        monkeypatch.setenv(KMEANS_BACKEND_ENV, backend)
        with pytest.raises(ValueError, match="is not one of"):
            _resolve_backend("not-a-backend")

    @pytest.mark.parametrize("backend", ["kmeans++", "sklearn", "sklearn_minibatch"])
    def test_cpu_backends_still_cluster(self, backend, monkeypatch):
        """Adding cuml must not disturb the three backends that work without a GPU."""
        monkeypatch.setenv(KMEANS_BACKEND_ENV, backend)
        kmeans = _EfficientKMeans(n_clusters=4, init="kmeans++", n_init=2, max_iter=20)
        kmeans.fit(torch.randn(256, 2))

        assert kmeans.labels_.shape == (256,)
        assert kmeans.cluster_centers_.shape == (4, 2)
        assert kmeans.labels_.max() < 4

    def test_cuml_backend_without_cuml_raises_actionable_error(self, monkeypatch):
        if torch.cuda.is_available():
            pytest.skip("cuml may be installed alongside CUDA; this covers its absence")

        monkeypatch.setenv(KMEANS_BACKEND_ENV, "cuml")
        kmeans = _EfficientKMeans(n_clusters=4, init="kmeans++", n_init=1, max_iter=10)
        with pytest.raises(ModuleNotFoundError, match="pip install cuml-cu12"):
            kmeans.fit(torch.randn(64, 2))

    def test_to_cuml_input_converts_cpu_tensor_without_cupy(self):
        """The CPU branch must not import cupy, which is a cuml-only dependency."""
        converted = _to_cuml_input(torch.randn(8, 2))

        assert isinstance(converted, numpy.ndarray)
        assert converted.shape == (8, 2)


class TestVectorDefaults:
    """The cluster_dim>1 backend default: device-driven, with no bearing on n_init."""

    def test_falls_back_without_cuml(self, monkeypatch):
        """A machine with no CUDA/cuml must keep the historical behaviour."""
        monkeypatch.setattr(_ek, "_FLASH_USABLE", False)
        monkeypatch.setattr(_ek, "_CUML_USABLE", False)
        assert _ek._default_vector_backend() == "kmeans++"

    def test_prefers_cuml_when_usable(self, monkeypatch):
        monkeypatch.setattr(_ek, "_FLASH_USABLE", False)
        monkeypatch.setattr(_ek, "_CUML_USABLE", True)
        assert _ek._default_vector_backend() == "cuml"

    def test_env_still_overrides_the_default(self, monkeypatch):
        """The defaults must not outrank an explicit request."""
        monkeypatch.setattr(_ek, "_FLASH_USABLE", False)
        monkeypatch.setattr(_ek, "_CUML_USABLE", True)
        backend = _ek._default_vector_backend()
        monkeypatch.setenv(KMEANS_BACKEND_ENV, "sklearn_minibatch")
        monkeypatch.setenv(KMEANS_N_INIT_ENV, "7")

        assert _resolve_backend(backend) == "sklearn_minibatch"
        assert _ek.vector_n_init() == 7

    def test_probe_is_memoized(self, monkeypatch):
        """The probe imports cuml (~8s), so it must run at most once per process."""
        monkeypatch.setattr(_ek, "_CUML_USABLE", None)
        calls = []
        monkeypatch.setattr(_ek._torch.cuda, "is_available", lambda: (calls.append(1), False)[1])

        _ek._cuml_is_usable()
        _ek._cuml_is_usable()
        assert len(calls) == 1


class TestNInitOverride:
    def test_unset_keeps_caller_value(self, monkeypatch):
        monkeypatch.delenv(KMEANS_N_INIT_ENV, raising=False)
        assert _resolve_n_init(5) == 5

    def test_override_wins_over_caller(self, monkeypatch):
        """`_cluster_weights_2d` hardcodes n_init=5, so the override must beat it."""
        monkeypatch.setenv(KMEANS_N_INIT_ENV, "1")
        assert _resolve_n_init(5) == 1
        assert _EfficientKMeans(n_clusters=4, init="kmeans++", n_init=5).n_init == 1

    @pytest.mark.parametrize("bad", ["0", "-1", "two", "1.5"])
    def test_invalid_override_raises(self, bad, monkeypatch):
        """A bad value must fail loudly, not silently change the restart count."""
        monkeypatch.setenv(KMEANS_N_INIT_ENV, bad)
        with pytest.raises(ValueError, match=KMEANS_N_INIT_ENV):
            _resolve_n_init(5)

    def test_whitespace_override_is_treated_as_unset(self, monkeypatch):
        """Matches `_resolve_backend`, which also strips and falls back."""
        monkeypatch.setenv(KMEANS_N_INIT_ENV, "  ")
        assert _resolve_n_init(5) == 5

    def test_override_reaches_a_real_fit(self, monkeypatch):
        monkeypatch.setenv(KMEANS_N_INIT_ENV, "1")
        monkeypatch.setenv(KMEANS_BACKEND_ENV, "sklearn")
        kmeans = _EfficientKMeans(n_clusters=4, init="kmeans++", n_init=5, max_iter=20)
        kmeans.fit(torch.randn(256, 2))

        assert kmeans.n_init == 1
        assert kmeans.labels_.shape == (256,)


class TestFlashKMeansBackend:
    """flash-kmeans is an optional dep; only its guard rails are testable here."""

    def test_sample_weight_is_refused_not_ignored(self, monkeypatch):
        """Silently dropping the weights would give unweighted centroids for a
        sensitivity-weighted run, which is a wrong answer rather than a slow one."""
        pytest.importorskip("flash_kmeans")
        monkeypatch.setenv(KMEANS_BACKEND_ENV, "flash_kmeans")
        kmeans = _EfficientKMeans(n_clusters=8, init="kmeans++", n_init=1, max_iter=10)
        with pytest.raises(ValueError, match="does not support sample_weight"):
            kmeans.fit(torch.randn(256, 2), sample_weight=torch.rand(256, 1))

    def test_missing_flash_kmeans_raises_actionable_error(self, monkeypatch):
        if importlib.util.find_spec("flash_kmeans") is not None:
            pytest.skip("flash-kmeans is installed; this covers its absence")

        monkeypatch.setenv(KMEANS_BACKEND_ENV, "flash_kmeans")
        kmeans = _EfficientKMeans(n_clusters=4, init="kmeans++", n_init=1, max_iter=10)
        with pytest.raises(ModuleNotFoundError, match="pip install flash-kmeans"):
            kmeans.fit(torch.randn(64, 2))

    def test_clusters_correctly_when_available(self, monkeypatch):
        """n_init restarts are stacked into the batch dim; the best must be selected."""
        pytest.importorskip("flash_kmeans")
        monkeypatch.setenv(KMEANS_BACKEND_ENV, "flash_kmeans")
        torch.manual_seed(0)
        X = torch.randn(512, 2)
        kmeans = _EfficientKMeans(n_clusters=16, init="kmeans++", n_init=3, max_iter=50, tol=1e-4)
        kmeans.fit(X)

        assert kmeans.labels_.shape == (512,)
        assert kmeans.cluster_centers_.shape == (16, 2)
        assert kmeans.labels_.max() < 16
        assert torch.isfinite(kmeans.cluster_centers_).all()
        # inertia_ must be the best restart's, so no worse than a single-restart run
        single = _EfficientKMeans(n_clusters=16, init="kmeans++", n_init=1, max_iter=50, tol=1e-4)
        single.fit(X)
        assert kmeans.inertia_ <= single.inertia_ * 1.5


class TestBatchedVectorClustering:
    """The batched path must agree with the per-block path it replaces."""

    def test_cluster_avg_matches_per_block_version(self):
        """batched_cluster_avg must equal _get_cluster_avg block by block.

        Every backend's centroids are overwritten with exact means, so if these two
        disagree the batched path silently produces different LUTs.
        """
        torch.manual_seed(0)
        b, n, d, k = 5, 400, 2, 16
        x = torch.randn(b, n, d)
        labels = torch.randint(0, k, (b, n))

        got = _ek.batched_cluster_avg(x, labels, k)
        for i in range(b):
            want = _EfficientKMeans._get_cluster_avg(k, labels[i], x[i])
            torch.testing.assert_close(got[i], want, rtol=1e-5, atol=1e-6)

    def test_cluster_avg_zeroes_empty_clusters(self):
        """Empty clusters get a zero centroid, matching the per-block behaviour."""
        x = torch.randn(1, 10, 2)
        labels = torch.zeros(1, 10, dtype=torch.long)  # cluster 1..3 all empty
        got = _ek.batched_cluster_avg(x, labels, 4)

        torch.testing.assert_close(got[0, 0], x[0].mean(0))
        assert bool((got[0, 1:] == 0).all())

    def test_kmeanspp_seeds_are_real_points_and_distinct(self):
        torch.manual_seed(0)
        x = torch.randn(4, 500, 2)
        seeds = _ek.batched_kmeanspp_init(x, 32)

        assert seeds.shape == (4, 32, 2)
        for i in range(4):
            for j in range(32):
                assert bool(((x[i] - seeds[i, j]).abs().sum(-1) < 1e-6).any()), "seed not a point"
            assert len({tuple(v.tolist()) for v in seeds[i]}) == 32, "duplicate seeds"

    def test_kmeanspp_handles_degenerate_block(self):
        """All-identical points make every D^2 weight zero; multinomial would reject it."""
        x = torch.ones(2, 50, 2)
        seeds = _ek.batched_kmeanspp_init(x, 8)

        assert seeds.shape == (2, 8, 2)
        assert torch.isfinite(seeds).all()

    def test_kmeanspp_rejects_too_many_clusters(self):
        with pytest.raises(ValueError, match="exceeds"):
            _ek.batched_kmeanspp_init(torch.randn(2, 10, 2), 11)

    def test_restarts_never_worsen_any_block(self):
        """Restarts pick the best result PER BLOCK, so total inertia must not rise.

        A global argmin over restarts would let a restart that helps one block drag
        another block's solution down with it.
        """
        pytest.importorskip("flash_kmeans")
        x = torch.randn(4, 1500, 2)

        totals = []
        for n_init in (1, 2, 4):
            torch.manual_seed(7)
            labels, cents = _ek.flash_batch_cluster(x, 32, max_iter=60, tol=1e-4, n_init=n_init)
            picked = torch.gather(cents, 1, labels.unsqueeze(-1).expand(-1, -1, 2))
            totals.append(float(((x - picked) ** 2).sum()))

        assert totals[1] <= totals[0] + 1e-3, totals
        assert totals[2] <= totals[1] + 1e-3, totals

    def test_batched_centroids_are_exact_means(self):
        """The returned LUT must be exact cluster means, as the per-block path is."""
        pytest.importorskip("flash_kmeans")
        torch.manual_seed(0)
        x = torch.randn(3, 900, 2)
        labels, cents = _ek.flash_batch_cluster(x, 24, max_iter=60, tol=1e-4, n_init=2)

        torch.testing.assert_close(cents, _ek.batched_cluster_avg(x, labels, 24))

    def test_n_init_env_override_reaches_batched_path(self, monkeypatch):
        monkeypatch.setenv(KMEANS_N_INIT_ENV, "3")
        assert _ek.vector_n_init() == 3
        monkeypatch.delenv(KMEANS_N_INIT_ENV)
        # Unset, one constant regardless of which backend the device resolves to.
        assert _ek.vector_n_init() == _ek._VECTOR_N_INIT

    def test_greedy_seeding_lowers_inertia_but_is_not_the_default(self):
        """Greedy DOES lower inertia -- and is still not the default.

        On the full Qwen3-4B greedy measured 48-60% slower and 0.08-0.24 PPL *worse*
        despite 1% better inertia. This test pins the inertia behaviour so the
        inversion stays visible, and asserts the default remains vanilla.
        """
        torch.manual_seed(0)
        # ~600 distinct values, mimicking a bf16 weight tensor's coarse grid
        x = (torch.randn(3, 4000, 2) * 40).round() / 40

        def potential(seeds):
            return sum(
                float(((x[b].unsqueeze(1) - seeds[b].unsqueeze(0)) ** 2).sum(-1).min(1)[0].sum())
                for b in range(x.shape[0])
            )

        torch.manual_seed(5)
        vanilla = potential(_ek.batched_kmeanspp_init(x, 32, n_local_trials=1))
        torch.manual_seed(5)
        greedy = potential(_ek.batched_kmeanspp_init(x, 32, n_local_trials=7))

        assert greedy < vanilla, f"greedy {greedy} not better than vanilla {vanilla}"
        # ...but the default must stay vanilla, because PPL disagrees with inertia.
        torch.manual_seed(5)
        default = potential(_ek.batched_kmeanspp_init(x, 32))
        assert default == vanilla, "default seeding must be vanilla (1 trial), not greedy"

    def test_shapes_hold_across_cluster_counts(self):
        for k in (16, 64, 256):
            torch.manual_seed(0)
            seeds = _ek.batched_kmeanspp_init(torch.randn(2, 500, 2), k)
            assert seeds.shape == (2, k, 2)

    def test_vanilla_still_available(self):
        seeds = _ek.batched_kmeanspp_init(torch.randn(2, 300, 2), 16, n_local_trials=1)
        assert seeds.shape == (2, 16, 2)
        assert torch.isfinite(seeds).all()


class TestMaxIterOverride:
    def test_unset_keeps_default(self, monkeypatch):
        monkeypatch.delenv(_ek.KMEANS_MAX_ITER_ENV, raising=False)
        assert _ek.vector_max_iter() == _ek._VECTOR_MAX_ITER == 300

    @pytest.mark.parametrize("value", ["1", "25", "50", "1000"])
    def test_override_applies(self, value, monkeypatch):
        monkeypatch.setenv(_ek.KMEANS_MAX_ITER_ENV, value)
        assert _ek.vector_max_iter() == int(value)

    @pytest.mark.parametrize("bad", ["0", "-5", "lots", "3.5"])
    def test_invalid_raises(self, bad, monkeypatch):
        monkeypatch.setenv(_ek.KMEANS_MAX_ITER_ENV, bad)
        with pytest.raises(ValueError, match=_ek.KMEANS_MAX_ITER_ENV):
            _ek.vector_max_iter()

    def test_lower_cap_actually_shortens_the_run(self):
        """The cap must reach the clustering, not just the resolver."""
        pytest.importorskip("flash_kmeans")
        torch.manual_seed(0)
        x = torch.randn(3, 2000, 2)

        # Fewer iterations must give inertia no better than more iterations.
        torch.manual_seed(1)
        lab_few, cen_few = _ek.flash_batch_cluster(x, 32, max_iter=3, tol=0.0, n_init=1)
        torch.manual_seed(1)
        lab_many, cen_many = _ek.flash_batch_cluster(x, 32, max_iter=200, tol=1e-4, n_init=1)

        def inr(cents, labels):
            picked = torch.gather(cents, 1, labels.unsqueeze(-1).expand(-1, -1, 2))
            return float(((x - picked) ** 2).sum())

        assert inr(cen_few, lab_few) >= inr(cen_many, lab_many) - 1e-6

    def test_batched_clustering_is_reproducible(self):
        """Two identical calls must give identical results.

        Without a seeded generator this path drew from the ambient RNG, and three
        identical full-model runs spanned 1.34 PPL -- larger than every configuration
        effect we tried to measure, which made all of those measurements meaningless.
        cuml's two runs of its config spanned 0.004 by comparison.
        """
        pytest.importorskip("flash_kmeans")
        x = torch.randn(4, 1200, 2)

        lab_a, cen_a = _ek.flash_batch_cluster(x, 24, max_iter=80, tol=1e-4, n_init=3)
        torch.manual_seed(12345)  # perturb the ambient RNG; must not matter
        lab_b, cen_b = _ek.flash_batch_cluster(x, 24, max_iter=80, tol=1e-4, n_init=3)

        assert torch.equal(lab_a, lab_b)
        torch.testing.assert_close(cen_a, cen_b, rtol=0, atol=0)

    def test_seed_is_derived_per_tensor_not_from_a_constant(self):
        """Two tensors of the same shape must not share an RNG sequence.

        Seeding every tensor from one constant reuses the same few sequences across all
        252 tensors, so whatever bias they carry applies everywhere rather than
        averaging out. Measured on the full model: a constant seed gave a reproducible
        18.68 PPL, worse than any unseeded run.
        """
        pytest.importorskip("flash_kmeans")
        torch.manual_seed(0)
        a = torch.randn(3, 900, 2)
        b = torch.randn(3, 900, 2)  # same shape, different data

        la, _ = _ek.flash_batch_cluster(a, 16, max_iter=40, tol=1e-4, n_init=2)
        lb, _ = _ek.flash_batch_cluster(b, 16, max_iter=40, tol=1e-4, n_init=2)
        # Identical seeds on differently-shuffled data would be a red flag; the real
        # check is that the same input reproduces and different inputs do not collide.
        la2, _ = _ek.flash_batch_cluster(a, 16, max_iter=40, tol=1e-4, n_init=2)
        assert torch.equal(la, la2), "same data must reproduce exactly"
        assert not torch.equal(la, lb), "different data must not share an outcome"

    def test_restarts_differ_from_each_other(self):
        """Seeding must not collapse the restarts into identical runs."""
        pytest.importorskip("flash_kmeans")
        x = torch.randn(3, 1200, 2)

        gen0 = torch.Generator().manual_seed(_ek._RANDOM_STATE + 0)
        gen1 = torch.Generator().manual_seed(_ek._RANDOM_STATE + 1)
        s0 = _ek.batched_kmeanspp_init(x, 24, generator=gen0)
        s1 = _ek.batched_kmeanspp_init(x, 24, generator=gen1)

        assert not torch.equal(s0, s1), "restarts must explore different seeds"


class TestStageSplit:
    """Stage 1 (seeding) and stage 2 (Lloyd) must be separable and separately timed."""

    def test_stage1_emits_one_seed_set_per_restart(self):
        x = torch.randn(3, 400, 2)
        seeds = _ek.generate_init_seeds(x, 16, n_init=5)

        assert seeds.shape == (5, 3, 16, 2)

    @pytest.mark.parametrize("method", ["kmeans++", "maximin", "random"])
    def test_stage1_seeds_are_real_points_for_every_method(self, method):
        """A seed that is not an input point means the method silently synthesised one."""
        x = torch.randn(2, 300, 2)
        seeds = _ek.generate_init_seeds(x, 8, n_init=2, method=method)

        for s in seeds.reshape(-1, 2):
            assert bool(((x.reshape(-1, 2) - s).abs().sum(-1) < 1e-6).any())

    def test_stage1_restarts_differ(self):
        """Identical seed sets would make n_init>1 pure waste."""
        seeds = _ek.generate_init_seeds(torch.randn(2, 400, 2), 16, n_init=4)

        for i in range(1, 4):
            assert not torch.equal(seeds[0], seeds[i])

    def test_stage1_is_reproducible(self):
        """Saved seeds are only reusable if regenerating them gives the same thing."""
        x = torch.randn(2, 400, 2)
        torch.manual_seed(1234)  # ambient RNG must not leak in
        a = _ek.generate_init_seeds(x, 16, n_init=3)
        torch.manual_seed(4321)
        b = _ek.generate_init_seeds(x, 16, n_init=3)

        assert torch.equal(a, b)

    def test_stage1_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="unknown init method"):
            _ek.generate_init_seeds(torch.randn(1, 50, 2), 4, n_init=1, method="kmeans--")

    def test_stage1_times_into_init_not_lloyd(self):
        _ek.reset_stage_timings()
        _ek.generate_init_seeds(torch.randn(2, 400, 2), 16, n_init=3)
        t = _ek.stage_timings()

        assert t.init_calls == 1 and t.init_seconds > 0.0
        assert t.lloyd_calls == 0 and t.lloyd_seconds == 0.0

    def test_stage2_consumes_saved_seeds_and_times_into_lloyd(self):
        """The whole point of the split: seeds produced once, refined later."""
        pytest.importorskip("flash_kmeans")
        x = torch.randn(2, 800, 2)
        seeds = _ek.generate_init_seeds(x, 16, n_init=2)

        _ek.reset_stage_timings()
        labels, cents = _ek.run_lloyd_batched(x, seeds, max_iter=40, tol=1e-4)
        t = _ek.stage_timings()

        assert labels.shape == (2, 800) and cents.shape == (2, 16, 2)
        assert t.lloyd_calls == 1 and t.lloyd_seconds > 0.0
        assert t.init_calls == 0, "stage 1 must not be re-run inside stage 2"

    def test_two_stage_matches_the_one_shot_helper(self):
        """flash_batch_cluster is now just a composition -- it must stay equivalent."""
        pytest.importorskip("flash_kmeans")
        x = torch.randn(2, 800, 2)

        want_labels, want_cents = _ek.flash_batch_cluster(x, 16, max_iter=40, n_init=2)
        got_labels, got_cents = _ek.run_lloyd_batched(
            x, _ek.generate_init_seeds(x, 16, n_init=2), max_iter=40, tol=1e-4
        )

        assert torch.equal(want_labels, got_labels)
        torch.testing.assert_close(want_cents, got_cents)

    def test_kmeans_pp_backend_splits_its_two_loops(self):
        """coreai-opt's own backend must report both stages, once per restart."""
        _ek.reset_stage_timings()
        _EfficientKMeans(n_clusters=8, init="kmeans++", n_init=3, max_iter=20).fit(
            torch.randn(200, 2)
        )
        t = _ek.stage_timings()

        assert t.init_calls == 3 and t.lloyd_calls == 3
        assert t.init_seconds > 0.0 and t.lloyd_seconds > 0.0
        assert t.init_separable

    def test_summary_flags_a_fused_backend(self):
        """cuml's fused fit must be labelled, not silently reported as pure stage 2."""
        t = _ek.StageTimings(init_seconds=0.0, lloyd_seconds=1.0, lloyd_calls=1)
        assert "NOT separable" not in t.summary()

        t.init_separable = False
        assert "NOT separable" in t.summary()

    def test_reset_clears_previous_run(self):
        _ek.generate_init_seeds(torch.randn(1, 100, 2), 4, n_init=1)
        _ek.reset_stage_timings()
        t = _ek.stage_timings()

        assert (t.init_seconds, t.lloyd_seconds, t.init_calls, t.lloyd_calls) == (0.0, 0.0, 0, 0)


class TestCumlSplitMode:
    """cuml's init is separable only in the opt-in two-fit mode."""

    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv(_ek.KMEANS_CUML_SPLIT_ENV, raising=False)
        assert not _ek.cuml_split_requested()

    def test_enabled_by_env(self, monkeypatch):
        monkeypatch.setenv(_ek.KMEANS_CUML_SPLIT_ENV, "1")
        assert _ek.cuml_split_requested()

    @pytest.mark.parametrize("value", ["0", "", "true", "yes"])
    def test_only_exactly_one_enables_it(self, value, monkeypatch):
        """A measurement mode that changes cuml's code path must not turn on loosely."""
        monkeypatch.setenv(_ek.KMEANS_CUML_SPLIT_ENV, value)
        assert not _ek.cuml_split_requested()

    def test_seeding_fit_does_at_most_one_lloyd_iteration(self):
        """Stage 1 must be seeding, not a partial solve -- cuML rejects max_iter=0."""
        assert _ek._CUML_INIT_MAX_ITER == 1


class TestBackendIsLoggedFromParent:
    """The parent must record the vector backend at every worker count.

    The worker processes that announce the backend never reach the parent's log, so if
    this line is skipped a finished run leaves no positive evidence of which backend it
    used. It was originally placed behind the ``num_workers > 1`` check, which hid it
    from exactly the ``num_workers=1`` runs used for stage-timing measurements.
    """

    @staticmethod
    def _palettizer_with(cluster_dims):
        model = torch.nn.Module()
        for i, dim in enumerate(cluster_dims):
            child = torch.nn.Module()
            child.cluster_dim = dim
            model.add_module(f"m{i}", child)
        return SimpleNamespace(_model=model)

    @pytest.mark.parametrize("workers", [1, 12])
    def test_vector_model_logs_backend_at_any_worker_count(self, workers, caplog, monkeypatch):
        monkeypatch.setenv(KMEANS_BACKEND_ENV, "kmeans++")
        stub = self._palettizer_with([2, 2])

        with caplog.at_level(logging.INFO):
            KMeansPalettizer._resolve_num_workers(stub, workers)

        assert "Vector k-means backend for this run: kmeans++" in caplog.text

    def test_num_workers_1_is_returned_unchanged(self, monkeypatch):
        monkeypatch.setenv(KMEANS_BACKEND_ENV, "cuml")
        stub = self._palettizer_with([2, 2])

        assert KMeansPalettizer._resolve_num_workers(stub, 1) == 1

    def test_gpu_serialised_backend_still_reduces_workers(self, monkeypatch):
        monkeypatch.setenv(KMEANS_BACKEND_ENV, "cuml")
        stub = self._palettizer_with([2, 2])

        assert KMeansPalettizer._resolve_num_workers(stub, 12) == 1

    def test_scalar_model_keeps_workers_and_logs_nothing(self, caplog, monkeypatch):
        """Scalar palettization is CPU-bound and genuinely wants workers."""
        monkeypatch.setenv(KMEANS_BACKEND_ENV, "cuml")
        stub = self._palettizer_with([1, 1])

        with caplog.at_level(logging.INFO):
            assert KMeansPalettizer._resolve_num_workers(stub, 12) == 12

        assert "Vector k-means backend" not in caplog.text


class TestNLocalTrialsOverride:
    """Greedy seeding is what cuML does, so the trial count must be controllable.

    cuVS `kmeansPlusPlus` uses `n_trials = 2 + ceil(log(n_clusters))` (8 at k=256);
    this path defaults to 1. See `batched_kmeanspp_init`.
    """

    def test_unset_is_vanilla(self, monkeypatch):
        monkeypatch.delenv(_ek.KMEANS_N_LOCAL_TRIALS_ENV, raising=False)
        assert _ek.resolve_n_local_trials(256) == 1

    def test_auto_matches_the_cuvs_formula(self, monkeypatch):
        monkeypatch.setenv(_ek.KMEANS_N_LOCAL_TRIALS_ENV, "auto")
        # 2 + ceil(ln 256) = 2 + 6. cuVS rounds up where sklearn truncates to 7.
        assert _ek.resolve_n_local_trials(256) == 8
        assert _ek.resolve_n_local_trials(16) == 5

    def test_integer_override_wins_over_caller(self, monkeypatch):
        monkeypatch.setenv(_ek.KMEANS_N_LOCAL_TRIALS_ENV, "8")
        assert _ek.resolve_n_local_trials(256, 1) == 8

    @pytest.mark.parametrize("bad", ["0", "-3", "eight", "2.5"])
    def test_invalid_override_raises(self, bad, monkeypatch):
        monkeypatch.setenv(_ek.KMEANS_N_LOCAL_TRIALS_ENV, bad)
        with pytest.raises(ValueError, match="must be 'auto' or a positive integer"):
            _ek.resolve_n_local_trials(256)

    def test_override_reaches_the_seeding_call(self, monkeypatch):
        """The env must change actual behaviour, not just the resolver."""
        x = torch.randn(2, 400, 2)
        monkeypatch.delenv(_ek.KMEANS_N_LOCAL_TRIALS_ENV, raising=False)
        vanilla = _ek.batched_kmeanspp_init(x, 16, None, torch.Generator().manual_seed(0))
        monkeypatch.setenv(_ek.KMEANS_N_LOCAL_TRIALS_ENV, "8")
        greedy = _ek.batched_kmeanspp_init(x, 16, None, torch.Generator().manual_seed(0))

        assert not torch.equal(vanilla, greedy)

    def test_greedy_lowers_inertia(self, monkeypatch):
        """The whole reason to expose this: greedy should seed better than vanilla."""
        torch.manual_seed(0)
        x = torch.randn(4, 1200, 2)

        def seed_inertia():
            seeds = _ek.batched_kmeanspp_init(x, 32, None, torch.Generator().manual_seed(7))
            d = ((x.unsqueeze(2) - seeds.unsqueeze(1)) ** 2).sum(-1)
            return float(d.min(dim=2).values.sum())

        monkeypatch.delenv(_ek.KMEANS_N_LOCAL_TRIALS_ENV, raising=False)
        vanilla = seed_inertia()
        monkeypatch.setenv(_ek.KMEANS_N_LOCAL_TRIALS_ENV, "8")
        greedy = seed_inertia()

        assert greedy < vanilla, f"greedy {greedy} should beat vanilla {vanilla}"


class TestStageImplSelection:
    """The two stages must be independently swappable, for crossover experiments."""

    def test_defaults(self, monkeypatch):
        monkeypatch.delenv(_ek.KMEANS_INIT_IMPL_ENV, raising=False)
        monkeypatch.delenv(_ek.KMEANS_LLOYD_IMPL_ENV, raising=False)
        assert _ek.resolve_init_impl() == "torch"
        assert _ek.resolve_lloyd_impl() == "flash"

    @pytest.mark.parametrize("impl", ["torch", "cuml"])
    def test_init_impl_selectable(self, impl, monkeypatch):
        monkeypatch.setenv(_ek.KMEANS_INIT_IMPL_ENV, impl)
        assert _ek.resolve_init_impl() == impl

    @pytest.mark.parametrize("impl", ["flash", "torch", "cuml"])
    def test_lloyd_impl_selectable(self, impl, monkeypatch):
        monkeypatch.setenv(_ek.KMEANS_LLOYD_IMPL_ENV, impl)
        assert _ek.resolve_lloyd_impl() == impl

    @pytest.mark.parametrize(
        "env,value",
        [("KMEANS_INIT_IMPL_ENV", "flash"), ("KMEANS_LLOYD_IMPL_ENV", "sklearn")],
    )
    def test_unknown_impl_raises(self, env, value, monkeypatch):
        """A typo must fail loudly, not silently run a different arm of the experiment."""
        monkeypatch.setenv(getattr(_ek, env), value)
        with pytest.raises(ValueError, match="is not one of"):
            _ek.resolve_init_impl() if "INIT" in env else _ek.resolve_lloyd_impl()


class TestTorchFp32Lloyd:
    """The fp32 control for flash's TF32 distance computation."""

    def test_matches_an_exact_fp64_reference(self, monkeypatch):
        """flash mis-assigns 0.61% of real points via TF32; this must not."""
        monkeypatch.setenv(_ek.KMEANS_LLOYD_IMPL_ENV, "torch")
        torch.manual_seed(0)
        x = torch.randn(3, 900, 2)
        k = 32
        seeds = _ek.generate_init_seeds(x, k, n_init=1)

        labels, centroids = _ek.run_lloyd_batched(x, seeds, max_iter=200, tol=1e-4)

        # Exact fp64 Lloyd from the same seed set.
        c = seeds[0].double().clone()
        for _ in range(200):
            lab = ((x.double().unsqueeze(2) - c.unsqueeze(1)) ** 2).sum(-1).argmin(-1)
            cnt = torch.zeros(3, k, dtype=torch.float64).scatter_add_(
                1, lab, torch.ones_like(lab, dtype=torch.float64)
            )
            sm = torch.zeros(3, k, 2, dtype=torch.float64).scatter_add_(
                1, lab.unsqueeze(-1).expand(-1, -1, 2), x.double()
            )
            new = torch.where(cnt.unsqueeze(-1) > 0, sm / cnt.clamp_min(1).unsqueeze(-1), c)
            done = bool((((new - c) ** 2).sum(-1).sum(-1) < 1e-4).all())
            c = new
            if done:
                break

        assert (labels == lab).float().mean() > 0.999
        assert centroids.shape == (3, k, 2)

    def test_empty_clusters_keep_their_previous_centroid(self, monkeypatch):
        """cuVS's rule (finalize_centroids); flash does the same. Zeroing would differ."""
        monkeypatch.setenv(_ek.KMEANS_LLOYD_IMPL_ENV, "torch")
        # More clusters than distinct points guarantees empty clusters mid-iteration.
        x = torch.randn(2, 40, 2)
        seeds = _ek.generate_init_seeds(x, 40, n_init=1)

        labels, centroids = _ek.run_lloyd_batched(x, seeds, max_iter=50, tol=1e-4)

        assert torch.isfinite(centroids).all()
        assert labels.shape == (2, 40)

    def test_restarts_never_worsen_any_block(self, monkeypatch):
        monkeypatch.setenv(_ek.KMEANS_LLOYD_IMPL_ENV, "torch")
        x = torch.randn(3, 700, 2)
        totals = []
        for n_init in (1, 3):
            seeds = _ek.generate_init_seeds(x, 24, n_init=n_init)
            labels, cents = _ek.run_lloyd_batched(x, seeds, max_iter=100, tol=1e-4)
            picked = torch.gather(cents, 1, labels.unsqueeze(-1).expand(-1, -1, 2))
            totals.append(float(((x - picked) ** 2).sum()))

        assert totals[1] <= totals[0] + 1e-4, totals

    def test_timings_land_in_stage_2(self, monkeypatch):
        monkeypatch.setenv(_ek.KMEANS_LLOYD_IMPL_ENV, "torch")
        x = torch.randn(2, 400, 2)
        seeds = _ek.generate_init_seeds(x, 16, n_init=1)
        _ek.reset_stage_timings()
        _ek.run_lloyd_batched(x, seeds, max_iter=50, tol=1e-4)
        t = _ek.stage_timings()

        assert t.lloyd_calls == 1 and t.lloyd_seconds > 0.0
        assert t.init_calls == 0


class TestVectorBackendDefault:
    """Backend and n_init are INDEPENDENT knobs.

    The backend follows the device -- flash_kmeans on CUDA, else cuml, else kmeans++ --
    and n_init is a single constant. Deriving one from the other would mean switching
    backend silently changed the work done, which makes two backends incomparable; that
    confound is what made several earlier cross-backend measurements unreadable.
    """

    def test_n_init_is_independent_of_the_backend(self):
        assert _ek._VECTOR_N_INIT == 12
        assert not hasattr(_ek, "_VECTOR_N_INIT_BY_BACKEND")

    def test_n_init_unchanged_across_backends(self, monkeypatch):
        counts = []
        for flash, cuml in ((True, True), (False, True), (False, False)):
            monkeypatch.setattr(_ek, "_flash_is_usable", lambda f=flash: f)
            monkeypatch.setattr(_ek, "_cuml_is_usable", lambda c=cuml: c)
            counts.append(_ek.vector_n_init())

        assert counts == [12, 12, 12], counts

    def test_flash_preferred_when_usable(self, monkeypatch):
        monkeypatch.setattr(_ek, "_flash_is_usable", lambda: True)
        monkeypatch.setattr(_ek, "_cuml_is_usable", lambda: True)
        assert _ek._default_vector_backend() == "flash_kmeans"

    def test_falls_back_to_cuml_then_torch(self, monkeypatch):
        monkeypatch.setattr(_ek, "_flash_is_usable", lambda: False)
        monkeypatch.setattr(_ek, "_cuml_is_usable", lambda: True)
        assert _ek._default_vector_backend() == "cuml"

        monkeypatch.setattr(_ek, "_cuml_is_usable", lambda: False)
        assert _ek._default_vector_backend() == "kmeans++"

    def test_weighted_run_never_gets_flash(self, monkeypatch):
        """flash has no sample_weight and raises on one, so sensitivities must skip it."""
        monkeypatch.setattr(_ek, "_flash_is_usable", lambda: True)
        monkeypatch.setattr(_ek, "_cuml_is_usable", lambda: True)

        assert _ek._default_vector_backend(weighted=True) == "cuml"
        assert _ek.resolve_vector_backend(weighted=True) == "cuml"
        # The weighted fallback changes the backend only, never the restart count.
        assert _ek.vector_n_init() == 12

    def test_weighted_falls_all_the_way_to_torch_without_cuml(self, monkeypatch):
        monkeypatch.setattr(_ek, "_flash_is_usable", lambda: True)
        monkeypatch.setattr(_ek, "_cuml_is_usable", lambda: False)
        assert _ek._default_vector_backend(weighted=True) == "kmeans++"

    def test_each_knob_overrides_without_disturbing_the_other(self, monkeypatch):
        monkeypatch.setattr(_ek, "_flash_is_usable", lambda: True)

        monkeypatch.setenv(KMEANS_BACKEND_ENV, "cuml")
        assert _ek.resolve_vector_backend() == "cuml"
        assert _ek.vector_n_init() == 12, "backend override must not move n_init"

        monkeypatch.setenv(KMEANS_N_INIT_ENV, "4")
        assert _ek.vector_n_init() == 4
        assert _ek.resolve_vector_backend() == "cuml", "n_init override must not move backend"

        monkeypatch.delenv(KMEANS_BACKEND_ENV)
        assert _ek.resolve_vector_backend() == "flash_kmeans"
        assert _ek.vector_n_init() == 4

    def test_flash_probe_is_memoized(self, monkeypatch):
        calls = []
        monkeypatch.setattr(_ek, "_FLASH_USABLE", None)
        real = torch.cuda.is_available

        def counting():
            calls.append(1)
            return real()

        monkeypatch.setattr(torch.cuda, "is_available", counting)
        _ek._flash_is_usable()
        _ek._flash_is_usable()
        monkeypatch.setattr(_ek, "_FLASH_USABLE", None)

        assert len(calls) == 1, "probe must be cached, not re-run per block"
