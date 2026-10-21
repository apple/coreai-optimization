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
