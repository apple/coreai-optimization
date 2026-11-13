# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for vector palettization k-means (``cluster_dim > 1``).

Covers the batched k-means++ seeding, exact cluster means, the relative-inertia
convergence check, backend selection, the scikit-learn CPU path, the
``VectorKMeansConfig`` settings and their plumbing into the palettizer, and (on CUDA)
the flash-kmeans and weighted GPU paths.
"""

from __future__ import annotations

import logging
import warnings
from unittest import mock

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError

from coreai_opt.palettization import (
    KMeansPalettizer,
    KMeansPalettizerConfig,
    ModuleKMeansPalettizerConfig,
)
from coreai_opt.palettization.config import OpKMeansPalettizerConfig, VectorKMeansConfig
from coreai_opt.palettization.kmeans import _vector_kmeans as ek, palettizer as palettizer_module
from coreai_opt.palettization.kmeans.kmeans_fake_palettize import _KMeansFakePalettize
from coreai_opt.palettization.spec import (
    PalettizationSpec,
    PerGroupedChannelGranularity,
    PerTensorGranularity,
)


def _clustered_blocks(num_blocks: int, points_per_center: int, n_clusters: int, seed: int = 0):
    """``(B, N, 2)`` blocks of well-separated 2D clusters, plus their true centers."""
    gen = torch.Generator().manual_seed(seed)
    centers = torch.randn(num_blocks, n_clusters, 2, generator=gen) * 10
    noise = torch.randn(num_blocks, n_clusters, points_per_center, 2, generator=gen) * 0.05
    points = (centers.unsqueeze(2) + noise).reshape(num_blocks, -1, 2)
    perm = torch.randperm(points.shape[1], generator=gen)
    return points[:, perm].contiguous(), centers


def _potential(x: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
    """``(B,)`` k-means potential: summed squared distance to the nearest seed."""
    return torch.cdist(x, seeds).min(-1).values.square().sum(-1)


class _Linear(nn.Module):
    def __init__(self, out_features: int = 64, in_features: int = 16):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.linear(x)


def _vector_spec(n_bits: int = 2, group_size: int | None = 16) -> PalettizationSpec:
    granularity = (
        PerTensorGranularity()
        if group_size is None
        else PerGroupedChannelGranularity(axis=0, group_size=group_size)
    )
    return PalettizationSpec(n_bits=n_bits, cluster_dim=2, granularity=granularity)


class TestBatchedKMeansPlusPlusInit:
    def test_seeds_are_distinct_points_of_their_own_block(self):
        x, _ = _clustered_blocks(num_blocks=3, points_per_center=20, n_clusters=8)
        for trials in (1, 4):
            seeds = ek.batched_kmeanspp_init(x, 8, trials, torch.Generator().manual_seed(0))
            assert seeds.shape == (3, 8, 2)
            for b in range(3):
                is_point = (seeds[b].unsqueeze(1) == x[b].unsqueeze(0)).all(-1).any(-1)
                assert is_point.all()
                assert torch.unique(seeds[b], dim=0).shape[0] == 8

    def test_reproducible_with_the_same_generator_seed(self):
        x, _ = _clustered_blocks(num_blocks=2, points_per_center=20, n_clusters=8)
        a = ek.batched_kmeanspp_init(x, 8, 3, torch.Generator().manual_seed(7))
        b = ek.batched_kmeanspp_init(x, 8, 3, torch.Generator().manual_seed(7))
        torch.testing.assert_close(a, b)

    def test_greedy_lowers_the_potential(self):
        gen = torch.Generator().manual_seed(0)
        x = torch.randn(32, 512, 2, generator=gen)
        standard = _potential(x, ek.batched_kmeanspp_init(x, 16, 1, gen))
        greedy = _potential(x, ek.batched_kmeanspp_init(x, 16, 8, gen))
        assert greedy.mean() < standard.mean()

    def test_zero_sensitivity_points_are_never_seeds(self):
        x = torch.randn(2, 200, 2, generator=torch.Generator().manual_seed(0))
        sensitivities = torch.zeros(2, 200)
        sensitivities[:, :50] = 1.0
        seeds = ek.batched_kmeanspp_init(x, 16, 3, torch.Generator().manual_seed(0), sensitivities)
        for b in range(2):
            idx = torch.cdist(seeds[b], x[b]).argmin(-1)
            assert (idx < 50).all()

    def test_handles_blocks_with_fewer_distinct_points_than_clusters(self):
        x = torch.ones(2, 32, 2)
        seeds = ek.batched_kmeanspp_init(x, 4, 2, torch.Generator().manual_seed(0))
        torch.testing.assert_close(seeds, torch.ones(2, 4, 2))

    def test_rejects_more_clusters_than_points(self):
        with pytest.raises(ValueError, match="exceeds"):
            ek.batched_kmeanspp_init(torch.randn(1, 4, 2), 8, 1, torch.Generator())


class TestBatchedClusterAvg:
    def test_matches_per_cluster_means(self):
        x = torch.randn(2, 30, 2)
        labels = torch.randint(0, 4, (2, 30))
        means = ek.batched_cluster_avg(x, labels, 4)
        for b in range(2):
            for k in range(4):
                if (labels[b] == k).any():
                    torch.testing.assert_close(means[b, k], x[b][labels[b] == k].mean(0))

    def test_weighted_means(self):
        x = torch.tensor([[[0.0, 0.0], [4.0, 8.0]]])
        labels = torch.zeros(1, 2, dtype=torch.long)
        means = ek.batched_cluster_avg(x, labels, 1, torch.tensor([[3.0, 1.0]]))
        torch.testing.assert_close(means, torch.tensor([[[1.0, 2.0]]]))

    def test_empty_clusters_are_zero(self):
        means = ek.batched_cluster_avg(torch.randn(1, 5, 2), torch.zeros(1, 5), 3)
        assert torch.equal(means[0, 1:], torch.zeros(2, 2))


class TestLloydUntilConverged:
    """Uses the weighted torch Lloyd loop, which also runs on CPU."""

    def _run(self, x, max_iter, tol):
        sensitivities = torch.ones(x.shape[:2])
        seeds = ek.batched_kmeanspp_init(x, 8, 1, torch.Generator().manual_seed(0), sensitivities)
        with mock.patch.object(ek, "_torch_lloyd_steps", wraps=ek._torch_lloyd_steps) as steps:
            labels = ek._lloyd_until_converged(
                x, seeds, sensitivities, max_iter, tol, use_flash=False
            )
        iterations = sum(call.args[2] for call in steps.call_args_list)
        return labels, sensitivities, iterations

    def test_stops_early_on_easy_data(self):
        x, _ = _clustered_blocks(num_blocks=4, points_per_center=50, n_clusters=8)
        _, _, iterations = self._run(x, max_iter=300, tol=1e-4)
        assert iterations < 300

    def test_tol_zero_still_stops_at_a_fixed_point(self):
        x, _ = _clustered_blocks(num_blocks=2, points_per_center=20, n_clusters=8)
        _, _, iterations = self._run(x, max_iter=300, tol=0.0)
        assert iterations < 300

    def test_unweighted_torch_loop(self):
        x, centers = _clustered_blocks(num_blocks=2, points_per_center=30, n_clusters=8)
        seeds = ek.batched_kmeanspp_init(x, 8, 2, torch.Generator().manual_seed(0))
        labels = ek._lloyd_until_converged(x, seeds, None, 300, 1e-4, use_flash=False)
        means = ek.batched_cluster_avg(x, labels, 8)
        for b in range(2):
            assert torch.cdist(centers[b], means[b]).min(-1).values.max() < 0.1

    def test_never_exceeds_max_iter(self):
        x = torch.randn(4, 4000, 2, generator=torch.Generator().manual_seed(0))
        _, _, iterations = self._run(x, max_iter=7, tol=0.0)
        assert iterations == 7

    def test_more_iterations_never_raise_inertia(self):
        x = torch.randn(4, 400, 2, generator=torch.Generator().manual_seed(0))
        short, sensitivities, _ = self._run(x, max_iter=2, tol=0.0)
        long, _, _ = self._run(x, max_iter=40, tol=0.0)
        short_inertia = ek._block_inertia(x, short, 8, sensitivities)
        long_inertia = ek._block_inertia(x, long, 8, sensitivities)
        assert (long_inertia <= short_inertia + 1e-5).all()


class TestBackendSelection:
    @pytest.fixture
    def machine(self, monkeypatch):
        def configure(cuda: bool, flash: bool, sklearn: bool) -> None:
            monkeypatch.setattr(ek.torch.cuda, "is_available", lambda: cuda)
            monkeypatch.setattr(ek, "_flash_kmeans_available", lambda: flash)
            monkeypatch.setattr(ek, "_sklearn_available", lambda: sklearn)

        return configure

    @pytest.mark.parametrize(
        ("cuda", "flash", "sklearn", "has_sensitivities", "expected"),
        [
            (True, True, True, False, "flash_kmeans"),
            (True, True, True, True, "torch"),
            (True, False, True, False, "torch"),
            (True, False, False, True, "torch"),
            (False, True, True, False, "sklearn"),
            (False, False, True, True, "sklearn"),
            (False, False, False, False, "torch"),
            (False, True, False, True, "torch"),
        ],
    )
    def test_selects_backend_from_the_machine(
        self, machine, cuda, flash, sklearn, has_sensitivities, expected
    ):
        machine(cuda=cuda, flash=flash, sklearn=sklearn)
        ek._warn_torch_cpu_fallback.cache_clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert ek.vector_kmeans_backend(has_sensitivities) == expected

    def test_cpu_without_sklearn_falls_back_to_torch_and_warns_once(self, machine):
        machine(cuda=False, flash=False, sklearn=False)
        ek._warn_torch_cpu_fallback.cache_clear()
        with pytest.warns(RuntimeWarning, match="scikit-learn is not installed") as record:
            assert ek.vector_kmeans_backend(has_sensitivities=False) == "torch"
            assert ek.vector_kmeans_backend(has_sensitivities=True) == "torch"
        assert len(record) == 1


class TestTorchCpuFallback:
    @pytest.fixture(autouse=True)
    def cpu_without_sklearn(self, monkeypatch):
        monkeypatch.setattr(ek.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(ek, "_sklearn_available", lambda: False)
        monkeypatch.setattr(ek, "_warn_torch_cpu_fallback", lambda: None)

    def test_recovers_well_separated_clusters_on_cpu(self):
        x, centers = _clustered_blocks(num_blocks=3, points_per_center=30, n_clusters=8)
        results = ek.cluster_vectors(
            list(x), None, 8, n_local_trials=None, n_init=2, max_iter=300, tol=1e-4
        )
        for (centroids, labels), block, true_centers in zip(results, x, centers, strict=True):
            assert centroids.device.type == "cpu" and labels.shape == (block.shape[0],)
            assert torch.cdist(true_centers, centroids).min(-1).values.max() < 0.1

    def test_weighted_centroids_are_weighted_means_on_cpu(self):
        x = torch.randn(1, 300, 2, generator=torch.Generator().manual_seed(0))
        sensitivities = torch.rand(1, 300, generator=torch.Generator().manual_seed(1)) + 0.1
        ((centroids, labels),) = ek.cluster_vectors(
            list(x), list(sensitivities), 8, n_local_trials=None, n_init=2, max_iter=300, tol=1e-4
        )
        expected = ek.batched_cluster_avg(x, labels.unsqueeze(0), 8, sensitivities)[0]
        torch.testing.assert_close(centroids, expected)


class TestSklearnPath:
    @pytest.fixture(autouse=True)
    def cpu_only(self, monkeypatch):
        pytest.importorskip("sklearn")
        monkeypatch.setattr(ek.torch.cuda, "is_available", lambda: False)

    def _cluster(self, blocks, sensitivities=None, **settings):
        kwargs = {"n_local_trials": None, "n_init": 2, "max_iter": 300, "tol": 1e-4}
        kwargs.update(settings)
        return ek.cluster_vectors(blocks, sensitivities, 8, **kwargs)

    def test_recovers_well_separated_clusters(self):
        x, centers = _clustered_blocks(num_blocks=3, points_per_center=30, n_clusters=8)
        results = self._cluster(list(x))
        for (centroids, labels), block, true_centers in zip(results, x, centers, strict=True):
            assert centroids.shape == (8, 2) and labels.shape == (block.shape[0],)
            assert torch.cdist(true_centers, centroids).min(-1).values.max() < 0.1

    def test_reproducible(self):
        x = torch.randn(2, 300, 2, generator=torch.Generator().manual_seed(0))
        first, second = self._cluster(list(x)), self._cluster(list(x))
        for (c1, l1), (c2, l2) in zip(first, second, strict=True):
            torch.testing.assert_close(c1, c2)
            assert torch.equal(l1, l2)

    def test_weighted_centroids_are_weighted_means(self):
        x = torch.randn(1, 300, 2, generator=torch.Generator().manual_seed(0))
        sensitivities = torch.rand(1, 300, generator=torch.Generator().manual_seed(1)) + 0.1
        ((centroids, labels),) = self._cluster(list(x), list(sensitivities))
        expected = ek.batched_cluster_avg(x, labels.unsqueeze(0), 8, sensitivities)[0]
        torch.testing.assert_close(centroids, expected)

    def test_single_restart_single_iteration(self):
        x = torch.randn(2, 300, 2, generator=torch.Generator().manual_seed(0))
        results = self._cluster(list(x), n_local_trials=1, n_init=1, max_iter=1)
        assert all(labels.max() < 8 for _, labels in results)


class TestVectorKMeansConfig:
    def test_defaults(self):
        config = VectorKMeansConfig()
        assert (config.n_local_trials, config.n_init, config.max_iter, config.tol) == (
            None,
            5,
            300,
            1e-4,
        )

    @pytest.mark.parametrize(
        "bad",
        [
            {"n_init": 0},
            {"max_iter": 0},
            {"tol": -1.0},
            {"n_local_trials": 0},
            {"init": "kmeans++"},
        ],
    )
    def test_rejects_invalid_values(self, bad):
        with pytest.raises(ValidationError):
            VectorKMeansConfig(**bad)

    def test_requires_a_vector_spec_in_the_module_config(self):
        with pytest.raises(ValidationError, match="cluster_dim > 1"):
            ModuleKMeansPalettizerConfig(
                op_state_spec={"weight": PalettizationSpec(n_bits=4)},
                vector_kmeans_config=VectorKMeansConfig(),
            )

    def test_accepts_a_vector_spec_anywhere_in_the_module_config(self):
        ModuleKMeansPalettizerConfig(
            op_state_spec={"weight": PalettizationSpec(n_bits=4)},
            op_name_config={
                "linear": OpKMeansPalettizerConfig(op_state_spec={"weight": _vector_spec()})
            },
            enable_fast_kmeans_mode=False,
            vector_kmeans_config=VectorKMeansConfig(n_init=2),
        )

    def test_parses_from_a_dict(self):
        config = KMeansPalettizerConfig.from_dict(
            {
                "kmeans_palettization_config": {
                    "global_config": {
                        "enable_fast_kmeans_mode": False,
                        "op_state_spec": {"weight": {"n_bits": 2, "cluster_dim": 2}},
                        "vector_kmeans_config": {"n_init": 1, "max_iter": 1, "n_local_trials": 1},
                    }
                }
            }
        )
        assert config.global_config.vector_kmeans_config == VectorKMeansConfig(
            n_init=1, max_iter=1, n_local_trials=1
        )

    def test_unset_gives_the_fake_palettize_module_the_defaults(self):
        fp = _KMeansFakePalettize(
            n_bits=2,
            lut_qspec=None,
            granularity=PerTensorGranularity(),
            cluster_dim=2,
            enable_per_channel_scale=False,
            enable_fast_kmeans_mode=False,
        )
        assert fp.vector_kmeans_config == VectorKMeansConfig()


class TestPalettizerPlumbing:
    @pytest.fixture(autouse=True)
    def cpu_only(self, monkeypatch):
        pytest.importorskip("sklearn")
        monkeypatch.setattr(ek.torch.cuda, "is_available", lambda: False)

    def _config(self, vector_kmeans_config=None, spec=None) -> KMeansPalettizerConfig:
        return KMeansPalettizerConfig(
            global_config=ModuleKMeansPalettizerConfig(
                op_state_spec={"weight": spec or _vector_spec()},
                enable_fast_kmeans_mode=False,
                vector_kmeans_config=vector_kmeans_config,
            )
        )

    def test_settings_reach_the_k_means_backend(self):
        settings = VectorKMeansConfig(n_local_trials=1, n_init=2, max_iter=7, tol=0.0)
        model = _Linear()
        with mock.patch(
            "coreai_opt.palettization.kmeans.kmeans_fake_palettize._cluster_vectors",
            wraps=ek.cluster_vectors,
        ) as cluster:
            KMeansPalettizer(model, self._config(settings)).prepare((torch.randn(1, 16),))
        assert cluster.call_args.kwargs == {
            "n_local_trials": 1,
            "n_init": 2,
            "max_iter": 7,
            "tol": 0.0,
        }
        # 64 rows / group_size 16 = 4 blocks, clustered in one call.
        assert len(cluster.call_args.args[0]) == 4

    def test_palettized_blocks_use_at_most_n_clusters_vectors(self):
        model = _Linear()
        prepared = KMeansPalettizer(model, self._config()).prepare((torch.randn(1, 16),))
        weight = prepared.linear.weight.detach()
        for block in weight.split(16, dim=0):
            vectors = block.transpose(0, 1).reshape(-1, 2)
            assert torch.unique(vectors, dim=0).shape[0] <= 4

    def test_prepare_is_reproducible(self):
        torch.manual_seed(0)
        model = _Linear()
        state = {k: v.clone() for k, v in model.state_dict().items()}
        first = KMeansPalettizer(model, self._config()).prepare((torch.randn(1, 16),))
        w1 = first.linear.weight.detach().clone()
        model2 = _Linear()
        model2.load_state_dict(state)
        second = KMeansPalettizer(model2, self._config()).prepare((torch.randn(1, 16),))
        torch.testing.assert_close(w1, second.linear.weight.detach())


class _InlinePool:
    """Stands in for a spawn pool: runs tasks in-process and records which tensors it got."""

    tensors: list[str] = []

    def __init__(self, processes):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap(self, fn, args):
        args = list(args)
        _InlinePool.tensors.extend(fp.tensor_fqn for fp, _ in args)
        return map(fn, args)


class _MHA(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

    def forward(self, x):
        return self.attn(x, x, x)[0]


class TestCentroidRouting:
    """Vector tensors that cluster on the GPU stay in-process; the rest use the workers."""

    @pytest.fixture(autouse=True)
    def cpu_clustering(self, monkeypatch):
        pytest.importorskip("sklearn")
        monkeypatch.setattr(ek.torch.cuda, "is_available", lambda: False)
        _InlinePool.tensors = []

    def _mixed_mha_config(self) -> KMeansPalettizerConfig:
        """One module whose two palettized states use a vector and a scalar spec."""
        return KMeansPalettizerConfig(
            global_config=None,
            module_type_configs={
                nn.MultiheadAttention: ModuleKMeansPalettizerConfig(
                    op_state_spec={
                        "in_proj_weight": _vector_spec(n_bits=2, group_size=None),
                        "weight": PalettizationSpec(n_bits=2),
                    },
                    enable_fast_kmeans_mode=False,
                ),
            },
        )

    @pytest.mark.parametrize(
        ("cluster_dim", "cuda", "expected"), [(2, True, True), (2, False, False), (1, True, False)]
    )
    def test_vector_tensors_go_to_the_gpu_when_there_is_one(self, cluster_dim, cuda, expected):
        fp = mock.Mock(cluster_dim=cluster_dim)
        with mock.patch.object(palettizer_module.torch.cuda, "is_available", return_value=cuda):
            assert palettizer_module._clusters_on_gpu(fp) == expected

    def test_gpu_states_stay_in_process_while_workers_take_the_rest(self):
        context = mock.Mock(Pool=_InlinePool)
        with (
            mock.patch.object(palettizer_module, "_clusters_on_gpu", lambda fp: fp.cluster_dim > 1),
            mock.patch.object(palettizer_module.mp, "get_context", return_value=context),
        ):
            prepared = KMeansPalettizer(_MHA(), self._mixed_mha_config()).prepare(
                (torch.randn(1, 10, 64),), num_workers=2
            )
        # Only the scalar state went to the workers; the vector state clustered in-process.
        assert len(_InlinePool.tensors) == 1
        assert _InlinePool.tensors[0].endswith("out_proj.weight")
        in_proj = prepared.attn.in_proj_weight.detach()
        assert torch.unique(in_proj.transpose(0, 1).reshape(-1, 2), dim=0).shape[0] <= 4
        assert torch.unique(prepared.attn.out_proj.weight.detach()).numel() <= 4

    def test_single_worker_creates_no_pool(self):
        with mock.patch.object(palettizer_module.mp, "get_context") as get_context:
            KMeansPalettizer(_MHA(), self._mixed_mha_config()).prepare(
                (torch.randn(1, 10, 64),), num_workers=1
            )
        get_context.assert_not_called()


@pytest.mark.cuda
class TestGPUPaths:
    def test_flash_kmeans_matches_sklearn_on_clustered_data(self):
        pytest.importorskip("flash_kmeans")
        pytest.importorskip("sklearn")
        x, centers = _clustered_blocks(num_blocks=4, points_per_center=40, n_clusters=16)
        settings = {"n_local_trials": None, "n_init": 2, "max_iter": 300, "tol": 1e-4}
        assert ek.vector_kmeans_backend(has_sensitivities=False) == "flash_kmeans"
        gpu = ek.cluster_vectors(list(x), None, 16, **settings)
        with mock.patch.object(ek.torch.cuda, "is_available", return_value=False):
            cpu = ek.cluster_vectors(list(x), None, 16, **settings)
        for (gc, _), (cc, _), true_centers in zip(gpu, cpu, centers, strict=True):
            assert torch.cdist(true_centers, gc).min(-1).values.max() < 0.1
            assert torch.cdist(true_centers, cc).min(-1).values.max() < 0.1

    def test_unweighted_without_flash_kmeans_stays_on_the_gpu(self, monkeypatch):
        monkeypatch.setattr(ek, "_flash_kmeans_available", lambda: False)
        assert ek.vector_kmeans_backend(has_sensitivities=False) == "torch"
        x, centers = _clustered_blocks(num_blocks=4, points_per_center=40, n_clusters=16)
        results = ek.cluster_vectors(
            list(x), None, 16, n_local_trials=None, n_init=2, max_iter=300, tol=1e-4
        )
        for (centroids, _), true_centers in zip(results, centers, strict=True):
            assert torch.cdist(true_centers, centroids).min(-1).values.max() < 0.1

    def test_weighted_gpu_path_recovers_clusters(self):
        x, centers = _clustered_blocks(num_blocks=4, points_per_center=40, n_clusters=16)
        sensitivities = torch.rand(x.shape[:2]) + 0.5
        results = ek.cluster_vectors(
            list(x), list(sensitivities), 16, n_local_trials=None, n_init=2, max_iter=300, tol=1e-4
        )
        for (centroids, labels), block, s, true_centers in zip(
            results, x, sensitivities, centers, strict=True
        ):
            assert torch.cdist(true_centers, centroids).min(-1).values.max() < 0.1
            expected = ek.batched_cluster_avg(block[None], labels[None], 16, s[None])[0]
            torch.testing.assert_close(centroids, expected)


class TestEmeansBackend:
    """The experimental faiss e-means backend, picked with COREAI_OPT_VECTOR_KMEANS_BACKEND."""

    def test_env_var_picks_emeans_only_for_unweighted_gpu_clustering(self, monkeypatch):
        monkeypatch.setattr(ek.torch.cuda, "is_available", lambda: True)
        monkeypatch.setenv("COREAI_OPT_VECTOR_KMEANS_BACKEND", "emeans_kmeanspp")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert ek.vector_kmeans_backend(has_sensitivities=False) == "emeans_kmeanspp"
        assert ek.vector_kmeans_backend(has_sensitivities=True) == "torch"

    def test_selecting_emeans_warns_once_per_process(self, monkeypatch):
        monkeypatch.setattr(ek.torch.cuda, "is_available", lambda: True)
        monkeypatch.setenv("COREAI_OPT_VECTOR_KMEANS_BACKEND", "emeans")
        ek._warn_emeans.cache_clear()
        with pytest.warns(RuntimeWarning, match="runs faiss e-means") as record:
            ek.vector_kmeans_backend(has_sensitivities=False)
            ek.vector_kmeans_backend(has_sensitivities=False)
        assert len(record) == 1

    def test_misspelled_backend_is_rejected(self, monkeypatch):
        monkeypatch.setenv("COREAI_OPT_VECTOR_KMEANS_BACKEND", "emean")
        with pytest.raises(ValueError, match="'emean'"):
            ek.vector_kmeans_backend(has_sensitivities=False)

    @pytest.mark.parametrize(("backend", "seeded"), [("emeans", False), ("emeans_kmeanspp", True)])
    def test_cluster_vectors_seeds_emeans_only_for_kmeanspp(self, monkeypatch, backend, seeded):
        e_means = pytest.importorskip("faiss.contrib.e_means")
        real_train = e_means.QuantizedEmeans.train
        inits = []

        def recording_train(self, x, init_centroids=None):
            inits.append(init_centroids)
            return real_train(self, x, init_centroids)

        monkeypatch.setattr(e_means.QuantizedEmeans, "train", recording_train)
        monkeypatch.setattr(ek.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(ek, "vector_kmeans_backend", lambda has_sensitivities: backend)
        x, _ = _clustered_blocks(num_blocks=3, points_per_center=40, n_clusters=16)
        ek.cluster_vectors(list(x), None, 16, n_local_trials=None, n_init=5, max_iter=300, tol=1e-4)
        assert [init is not None for init in inits] == [seeded] * 3

    def test_kmeanspp_seeded_emeans_recovers_well_separated_clusters(self):
        pytest.importorskip("faiss.contrib.e_means")
        x, centers = _clustered_blocks(num_blocks=3, points_per_center=40, n_clusters=16)
        centroids, _ = ek._cluster_emeans(x, 16, None, kmeanspp=True)
        for block_centroids, true_centers in zip(centroids, centers, strict=True):
            assert torch.cdist(true_centers, block_centroids).min(-1).values.max() < 0.1

    def test_labels_come_from_assign_not_from_train(self, monkeypatch):
        e_means = pytest.importorskip("faiss.contrib.e_means")
        real_train = e_means.QuantizedEmeans.train

        def train_returning_unusable_labels(self, x, init_centroids=None):
            centroids, _ = real_train(self, x, init_centroids)
            return centroids, torch.full((len(x),), -1)

        monkeypatch.setattr(e_means.QuantizedEmeans, "train", train_returning_unusable_labels)
        x, _ = _clustered_blocks(num_blocks=2, points_per_center=40, n_clusters=16)
        _, labels = ek._cluster_emeans(x, 16, None, kmeanspp=True)
        assert labels.shape == x.shape[:2]
        assert labels.min() >= 0
        assert labels.max() < 16


class TestEmeansBatched:
    """``emeans_batched`` runs e-means' recipe for every block of a tensor in one step."""

    @pytest.mark.parametrize(("n", "k"), [(640, 16), (1000, 7), (4096, 64), (300, 4)])
    def test_centroids_equal_faiss_per_block_on_cpu(self, n, k):
        e_means = pytest.importorskip("faiss.contrib.e_means")
        gen = torch.Generator().manual_seed(n + k)
        x = (torch.randn(3, n, 2, generator=gen) * torch.tensor([3.0, 1.0])).contiguous()
        seeds = ek.batched_kmeanspp_init(x, k, ek._resolve_n_local_trials(None, k), gen)
        expected = []
        for block, block_seeds in zip(x, seeds, strict=True):
            em = e_means.QuantizedEmeans(2, k, quantizer=None)
            em.train(block, block_seeds)
            expected.append(em.centroids)
        torch.testing.assert_close(
            ek._emeans_batched_centroids(x, seeds), torch.stack(expected), rtol=0, atol=0
        )

    def test_cluster_vectors_matches_the_per_block_path_and_recovers_clusters(self, monkeypatch):
        pytest.importorskip("faiss.contrib.e_means")
        monkeypatch.setattr(ek.torch.cuda, "is_available", lambda: False)
        x, centers = _clustered_blocks(num_blocks=3, points_per_center=40, n_clusters=16)
        settings = {"n_local_trials": None, "n_init": 5, "max_iter": 300, "tol": 1e-4}
        results = {}
        for backend in ("emeans_batched", "emeans_kmeanspp"):
            monkeypatch.setattr(ek, "vector_kmeans_backend", lambda has_sensitivities, b=backend: b)
            results[backend] = ek.cluster_vectors(list(x), None, 16, **settings)
        for (batched, labels), (per_block, per_block_labels), true_centers in zip(
            results["emeans_batched"], results["emeans_kmeanspp"], centers, strict=True
        ):
            torch.testing.assert_close(batched, per_block)
            assert torch.equal(labels, per_block_labels)
            assert torch.cdist(true_centers, batched).min(-1).values.max() < 0.1

    def test_assign_nearest_is_the_exact_nearest_centroid(self):
        x, _ = _clustered_blocks(num_blocks=2, points_per_center=40, n_clusters=16)
        centroids = torch.randn(2, 16, 2)
        expected = torch.cdist(x, centroids).argmin(-1)
        assert torch.equal(ek._assign_nearest(x, centroids), expected)

    def test_more_than_two_dimensions_use_the_per_block_path(self, monkeypatch, caplog):
        pytest.importorskip("faiss.contrib.e_means")
        monkeypatch.setattr(ek.torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(ek, "vector_kmeans_backend", lambda has_sensitivities: "emeans_batched")
        x = torch.randn(2, 600, 4)
        with caplog.at_level(logging.INFO, logger=ek.__name__):
            results = ek.cluster_vectors(
                list(x), None, 8, n_local_trials=None, n_init=5, max_iter=300, tol=1e-4
            )
        assert "e-means: 2 blocks" in caplog.text
        assert results[0][0].shape == (8, 4)


@pytest.mark.cuda
class TestEmeansEpochGraph:
    """The CUDA-graph replay of e-means' epochs computes what the plain loop computes."""

    @staticmethod
    def _plan(n, k):
        from faiss.contrib.e_means.e_means_torch import (  # noqa: PLC0415 - optional dependency
            _epoch_rows,
            _resolve_recipe,
            _scheduled_alpha,
        )

        batch, alpha0 = _resolve_recipe(n, k, 5, "quality", None, None)
        rows = _epoch_rows(n, 5)
        total = sum((r + batch - 1) // batch for r in rows)
        return batch, [_scheduled_alpha(alpha0, step, total) for step in range(total)], rows

    @pytest.mark.parametrize(("blocks", "n", "k"), [(6, 4096, 16), (3, 5000, 64), (7, 2500, 33)])
    def test_replay_matches_the_eager_loop_including_a_remainder_step(self, blocks, n, k):
        pytest.importorskip("faiss.contrib.e_means")
        gen = torch.Generator(device="cuda").manual_seed(blocks)
        batch, alphas, rows = self._plan(n, k)
        # Atomic scatter_add makes two runs of the *same* path differ by float ulps, so equal
        # means within that noise, not bit-identical.
        for _ in range(2):  # the second call replays the cached graph on different data
            x = torch.randn(blocks, n, 2, device="cuda", generator=gen).contiguous()
            seeds = ek.batched_kmeanspp_init(x, k, ek._resolve_n_local_trials(None, k), gen)
            eager = ek._centroids_eager(x, seeds, batch, alphas, rows)
            replayed = ek._centroids_graphed(x, seeds, batch, alphas, 5)
            torch.testing.assert_close(replayed, eager, rtol=1e-4, atol=1e-5)

    def test_a_cached_graph_carries_nothing_between_calls(self):
        pytest.importorskip("faiss.contrib.e_means")
        gen = torch.Generator(device="cuda").manual_seed(0)
        batch, alphas, _ = self._plan(2048, 8)
        first = torch.randn(4, 2048, 2, device="cuda", generator=gen).contiguous()
        second = torch.randn(4, 2048, 2, device="cuda", generator=gen).contiguous()
        seeds = [ek.batched_kmeanspp_init(v, 8, 3, gen) for v in (first, second)]
        fresh = ek._centroids_graphed(second, seeds[1], batch, alphas, 5)
        ek._EPOCH_GRAPHS.clear()
        ek._centroids_graphed(
            first, seeds[0], batch, alphas, 5
        )  # captures, then dirties the buffers
        reused = ek._centroids_graphed(second, seeds[1], batch, alphas, 5)
        torch.testing.assert_close(reused, fresh, rtol=1e-4, atol=1e-5)


@pytest.mark.cuda
class TestFusedKMeansPlusPlus:
    """The fused seeding steps draw what the unfused loop draws, up to float rounding."""

    def test_the_first_seed_matches_and_every_seed_is_exactly_a_point_of_its_block(self):
        gen = torch.Generator(device="cuda").manual_seed(0)
        scale = torch.tensor([4.0, 1.0], device="cuda")
        x = (torch.randn(6, 3000, 2, device="cuda", generator=gen) * scale).contiguous()
        for k in (2, 5, 16):
            trials = ek._resolve_n_local_trials(None, k)
            fused = ek._kmeanspp_seeds_fused(
                x, k, trials, torch.Generator(device="cuda").manual_seed(3)
            )
            unfused = ek.batched_kmeanspp_init(
                x, k, trials, torch.Generator(device="cuda").manual_seed(3)
            )
            # The first seed is a plain uniform draw from the same stream, so it is the same one.
            assert torch.equal(fused[:, 0], unfused[:, 0])
            assert (fused.unsqueeze(2) == x.unsqueeze(1)).all(-1).any(-1).all()

    def test_seeding_quality_matches_the_unfused_loop(self):
        x, _ = _clustered_blocks(num_blocks=8, points_per_center=60, n_clusters=24)
        x = x.cuda()
        trials = ek._resolve_n_local_trials(None, 24)
        fused = ek._kmeanspp_seeds_fused(
            x, 24, trials, torch.Generator(device="cuda").manual_seed(1)
        )
        unfused = ek.batched_kmeanspp_init(
            x, 24, trials, torch.Generator(device="cuda").manual_seed(1)
        )
        # Seeds can differ where a draw is within rounding of a tie, but the potential they leave
        # is the same to a few percent; a broken step would leave a missed cluster, ~100x more.
        torch.testing.assert_close(
            _potential(x, fused).median(), _potential(x, unfused).median(), rtol=0.05, atol=0
        )
        assert _potential(x, fused).max() < 3 * _potential(x, unfused).max()

    def test_blocks_that_run_out_of_distinct_points_still_get_seeds(self):
        x = torch.zeros(2, 64, 2, device="cuda")
        x[1, :, 0] = torch.arange(64, device="cuda") % 3  # three distinct points, ten seeds asked
        seeds = ek._kmeanspp_seeds_fused(x, 10, 3, torch.Generator(device="cuda").manual_seed(0))
        assert seeds.shape == (2, 10, 2) and torch.isfinite(seeds).all()


@pytest.mark.cuda
class TestEmeansUpdate:
    """On CUDA the EMA update is faiss' Triton kernel; it must equal the torch arithmetic."""

    @staticmethod
    def _state(blocks, k, seed):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        counts = (torch.rand(blocks, k, device="cuda", generator=gen) * 12).floor()  # some empty
        sums = torch.randn(blocks, k, 2, device="cuda", generator=gen) * counts.unsqueeze(-1)
        centroids = torch.randn(blocks, k, 2, device="cuda", generator=gen)
        usage = torch.rand(blocks, k, device="cuda", generator=gen) + 0.1
        return sums, counts, centroids, usage

    @pytest.mark.parametrize(("blocks", "k"), [(6, 16), (5, 7), (304, 256)])
    @pytest.mark.parametrize(
        ("alpha", "replace_usage"), [(0.37, False), (0.05, False), (1.0, True)]
    )
    def test_triton_update_equals_torch_update_with_host_and_device_rates(
        self, blocks, k, alpha, replace_usage
    ):
        pytest.importorskip("faiss.contrib.e_means")
        sums, counts, centroids, usage = self._state(blocks, k, seed=blocks + k)
        expected_c, expected_u = centroids.clone(), usage.clone()
        # The torch branch, forced by hiding CUDA from `_ema_update`'s check.
        cpu_c, cpu_u = expected_c.cpu(), expected_u.cpu()
        ek._ema_update(sums.cpu(), counts.cpu(), cpu_c, cpu_u, alpha, 1.0 - alpha, replace_usage)
        for rate in (
            (alpha, 1.0 - alpha),
            tuple(torch.tensor([alpha, 1.0 - alpha], dtype=torch.float32, device="cuda")),
        ):
            got_c, got_u = centroids.clone(), usage.clone()
            ek._ema_update(sums.clone(), counts.clone(), got_c, got_u, *rate, replace_usage)
            torch.testing.assert_close(got_c.cpu(), cpu_c, rtol=0, atol=0)
            torch.testing.assert_close(got_u.cpu(), cpu_u, rtol=0, atol=0)


@pytest.mark.cuda
class TestFusedAssignAccumulate:
    """The fused assign-and-accumulate kernel adds each point to its nearest centroid's stats."""

    @pytest.mark.parametrize(
        ("blocks", "n", "k", "m"), [(6, 800, 4, 64), (5, 1500, 100, 33), (3, 900, 256, 128)]
    )
    def test_matches_the_torch_step(self, blocks, n, k, m):
        from coreai_opt.palettization.kmeans import _emeans_triton  # noqa: PLC0415

        gen = torch.Generator(device="cuda").manual_seed(blocks + k)
        x = torch.randn(blocks, n, 2, device="cuda", generator=gen).contiguous()
        centroids = (
            x[:, :k] + 0.05 * torch.randn(blocks, k, 2, device="cuda", generator=gen)
        ).contiguous()
        picked = torch.randperm(n, device="cuda", generator=gen)[:m].contiguous()
        expected = torch.zeros(blocks, k, 3, device="cuda")
        ek._accumulate_batch_torch(x, picked, centroids, expected)
        got = torch.zeros(blocks, k, 3, device="cuda")
        _emeans_triton.assign_accumulate(x, picked, centroids, got)
        # Counts are exact (a point either lands in a cluster or does not); sums differ only by
        # the order of the atomic float adds.
        assert torch.equal(got[..., 2], expected[..., 2])
        torch.testing.assert_close(got[..., :2], expected[..., :2], rtol=1e-5, atol=1e-5)

    def test_ties_go_to_the_lowest_index_like_argmin(self):
        from coreai_opt.palettization.kmeans import _emeans_triton  # noqa: PLC0415

        x = torch.zeros(2, 8, 2, device="cuda")
        centroids = torch.ones(2, 70, 2, device="cuda")  # all equidistant, more than one tile
        got = torch.zeros(2, 70, 3, device="cuda")
        _emeans_triton.assign_accumulate(x, torch.arange(8, device="cuda"), centroids, got)
        assert (got[..., 2][:, 0] == 8).all() and got[..., 2][:, 1:].sum() == 0

    def test_an_epoch_is_the_same_with_the_fused_kernel_and_without_it(self, monkeypatch):
        from coreai_opt.palettization.kmeans import _emeans_triton  # noqa: PLC0415

        gen = torch.Generator(device="cuda").manual_seed(2)
        x = torch.randn(5, 3000, 2, device="cuda", generator=gen).contiguous()
        seeds = ek.batched_kmeanspp_init(x, 24, 5, gen)
        fused = ek._emeans_batched_centroids(x, seeds)
        monkeypatch.setattr(_emeans_triton, "available", lambda _: False)
        ek._EPOCH_GRAPHS.clear()  # a graph captured with the kernel must not be reused
        plain = ek._emeans_batched_centroids(x, seeds)
        torch.testing.assert_close(fused, plain, rtol=1e-3, atol=1e-3)
