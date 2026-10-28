# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Vector k-means for palettization (``cluster_dim > 1``).

Each palettization block is an independent k-means problem over ``cluster_dim``-long
vectors. :func:`cluster_vectors` solves all blocks of one weight tensor and picks the
backend from the device:

- **GPU**: batched k-means++ seeding (sensitivity-proportional for sensitivity-weighted
  clustering), then Lloyd iterations with all blocks of the tensor in one batch: with
  flash-kmeans (Triton; "Flash-KMeans: Fast and Memory-Efficient Exact K-Means",
  https://arxiv.org/abs/2603.09229) for unweighted clustering when it is installed,
  otherwise with a batched torch loop, which also supports per-point sensitivities.
- **CPU**: scikit-learn ``kmeans_plusplus`` and ``KMeans``, one block at a time. Without
  scikit-learn, the batched torch loop runs on CPU instead, with a warning: it gives the
  same results but is several times slower.

On every backend a block stops iterating once its inertia improves by less than ``tol``
(relative) per iteration, checked every few iterations, or once no assignment changes.
``n_init`` restarts run sequentially and the best one is kept per block. Final centroids
are always the exact (sensitivity-weighted) means of the final clusters.
"""

import functools
import math
import multiprocessing
import warnings

import numpy as np
import torch

#: Lloyd iterations between convergence checks.
_CHECK_EVERY = 5

#: Keeps the torch Lloyd loop's ``(B, chunk, K)`` distance tile near 0.5 GB.
_TORCH_DIST_ELEMENTS = 2**27


@functools.cache
def _flash_kmeans_available() -> bool:
    """Whether flash-kmeans imports with its Triton kernels, not its torch fallback."""
    try:
        from flash_kmeans import batch_kmeans_Euclid  # noqa: PLC0415 - optional dependency
    except Exception:  # noqa: BLE001 - any import failure means "not available"
        return False
    return bool(batch_kmeans_Euclid.__module__ != "flash_kmeans.torch_fallback")


@functools.cache
def _sklearn_available() -> bool:
    """Whether scikit-learn is importable."""
    try:
        import sklearn.cluster  # noqa: F401, PLC0415 - optional dependency
    except ImportError:
        return False
    return True


def vector_kmeans_backend(has_sensitivities: bool) -> str:
    """Return the backend :func:`cluster_vectors` uses on this machine.

    Args:
        has_sensitivities (bool): Whether per-point sensitivities are given.

    Returns:
        str: On a CUDA GPU, ``"flash_kmeans"`` (unweighted, flash-kmeans installed) or
        ``"torch"``. Without a GPU, ``"sklearn"``, or ``"torch"`` (on CPU, with a
        one-time warning) if scikit-learn is not installed.
    """
    if torch.cuda.is_available():
        return "flash_kmeans" if not has_sensitivities and _flash_kmeans_available() else "torch"
    if _sklearn_available():
        return "sklearn"
    _warn_torch_cpu_fallback()
    return "torch"


@functools.cache
def _warn_torch_cpu_fallback() -> None:
    """Warn, once per process, that CPU clustering falls back to the torch backend."""
    if multiprocessing.parent_process() is not None:
        # A clustering worker: the palettizer already warned in the parent process.
        return
    warnings.warn(
        "scikit-learn is not installed, so vector palettization (cluster_dim > 1) on CPU "
        "uses the torch k-means backend. It gives the same results but is several times "
        "slower; install 'scikit-learn' for faster CPU clustering.",
        RuntimeWarning,
        stacklevel=3,
    )


def _resolve_n_local_trials(n_local_trials: int | None, n_clusters: int) -> int:
    """Candidates per k-means++ step; ``None`` means scikit-learn's default.

    scikit-learn's ``kmeans_plusplus`` defaults to ``2 + int(log(n_clusters))``
    (greedy k-means++); 1 is standard k-means++.
    """
    if n_local_trials is None:
        return 2 + int(math.log(n_clusters))
    return n_local_trials


def _tensor_seed(x: torch.Tensor) -> int:
    """Deterministic random seed derived from a tensor's contents.

    Makes clustering reproducible across runs while giving each tensor its own random
    stream, so same-shaped tensors do not share seeds.
    """
    flat = x.flatten()
    probe = flat[:: max(1, flat.numel() // 64)][:64].double().tolist()
    return hash((tuple(x.shape), tuple(round(v, 6) for v in probe))) % (2**31 - 1)


def batched_kmeanspp_init(
    x: torch.Tensor,
    n_clusters: int,
    n_local_trials: int,
    generator: torch.Generator,
    sensitivities: torch.Tensor | None = None,
) -> torch.Tensor:
    """k-means++ seeding for a batch of independent blocks at once.

    Each new seed is sampled with probability proportional to ``s * d^2``: a point's
    sensitivity times its squared distance to the nearest seed chosen so far. With
    ``n_local_trials > 1`` (greedy k-means++), that many candidates are drawn per step
    and the one leaving the lowest remaining sensitivity-weighted potential is kept. The
    ``n_clusters`` steps are sequential, but each step is vectorized across blocks.

    Args:
        x (torch.Tensor): ``(B, N, D)`` vectorized blocks.
        n_clusters (int): Seeds per block. Must be ``<= N``.
        n_local_trials (int): Candidates per step; 1 is standard k-means++.
        generator (torch.Generator): Random generator on ``x``'s device.
        sensitivities (torch.Tensor | None): Optional ``(B, N)`` non-negative point sensitivities.

    Returns:
        torch.Tensor: ``(B, n_clusters, D)`` seeds, each a point of its own block.

    Raises:
        ValueError: If ``n_clusters`` exceeds the number of points per block.
    """
    b, n, d = x.shape
    if n_clusters > n:
        raise ValueError(f"n_clusters {n_clusters} exceeds {n} points per block")
    rows = torch.arange(b, device=x.device)
    x_sq = (x**2).sum(-1)
    s = torch.ones(b, n, device=x.device) if sensitivities is None else sensitivities.float()

    seeds = torch.empty(b, n_clusters, d, device=x.device, dtype=x.dtype)
    first = torch.multinomial(s.clamp_min(0) + 1e-30, 1, generator=generator).squeeze(1)
    seeds[:, 0] = x[rows, first]
    closest = ((x - seeds[:, 0:1]) ** 2).sum(-1)

    for i in range(1, n_clusters):
        potential = (s * closest).clamp_min(0)
        # Once every point of a block coincides with a seed its potential is zero and it
        # cannot be sampled from; draw uniformly for that block instead. torch.where keeps
        # this on the device: checking degenerate.any() on the host would sync every step.
        degenerate = potential.sum(dim=1, keepdim=True) <= 0
        potential = torch.where(degenerate, 1.0, potential)
        # With replacement, as scikit-learn does: valid for any n_local_trials, and never
        # picks a zero-potential point (one already at a seed) as a candidate.
        cand = torch.multinomial(potential, n_local_trials, replacement=True, generator=generator)
        cand_pts = torch.gather(x, 1, cand.unsqueeze(-1).expand(-1, -1, d))
        # (B, N, trials) squared distances via the norm trick.
        dist = (
            x_sq.unsqueeze(-1)
            - 2.0 * torch.bmm(x, cand_pts.transpose(1, 2))
            + (cand_pts**2).sum(-1).unsqueeze(1)
        ).clamp_min_(0)
        new_closest = torch.minimum(closest.unsqueeze(-1), dist)
        best = (s.unsqueeze(-1) * new_closest).sum(dim=1).argmin(dim=1)
        seeds[:, i] = cand_pts[rows, best]
        closest = new_closest[rows, :, best]
    return seeds


def batched_cluster_avg(
    x: torch.Tensor,
    labels: torch.Tensor,
    n_clusters: int,
    sensitivities: torch.Tensor | None = None,
) -> torch.Tensor:
    """Exact (sensitivity-weighted) per-cluster means for a batch of blocks.

    Args:
        x (torch.Tensor): ``(B, N, D)`` points.
        labels (torch.Tensor): ``(B, N)`` cluster index per point.
        n_clusters (int): Number of clusters.
        sensitivities (torch.Tensor | None): Optional ``(B, N)`` point sensitivities.

    Returns:
        torch.Tensor: ``(B, n_clusters, D)`` float32 means; empty clusters are zero.
    """
    b, _, d = x.shape
    idx = labels.long()
    s = torch.ones(idx.shape, device=x.device) if sensitivities is None else sensitivities.float()
    sums = torch.zeros(b, n_clusters, d, device=x.device)
    sums.scatter_add_(1, idx.unsqueeze(-1).expand(-1, -1, d), x.float() * s.unsqueeze(-1))
    mass = torch.zeros(b, n_clusters, device=x.device).scatter_add_(1, idx, s)
    return sums / mass.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)


def _block_inertia(
    x: torch.Tensor, labels: torch.Tensor, n_clusters: int, sensitivities: torch.Tensor | None
) -> torch.Tensor:
    """``(B,)`` (sensitivity-weighted) inertia of each block under the exact means of ``labels``."""
    centroids = batched_cluster_avg(x, labels, n_clusters, sensitivities)
    picked = torch.gather(centroids, 1, labels.long().unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    sq = ((x.float() - picked) ** 2).sum(-1)
    return (sq if sensitivities is None else sq * sensitivities).sum(-1)


def _flash_lloyd_steps(
    x: torch.Tensor, centroids: torch.Tensor, steps: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``steps`` flash-kmeans Lloyd iterations and return ``(labels, centroids)``."""
    from flash_kmeans import batch_kmeans_Euclid  # noqa: PLC0415 - optional dependency

    labels, centroids, _ = batch_kmeans_Euclid(
        x, n_clusters=centroids.shape[1], max_iters=steps, tol=0.0, init_centroids=centroids
    )
    return labels.long(), centroids.float()


def _torch_lloyd_steps(
    x: torch.Tensor, centroids: torch.Tensor, steps: int, sensitivities: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``steps`` sensitivity-weighted Lloyd iterations and return ``(labels, centroids)``.

    Distances use the norm trick, chunked over ``N`` so the ``(B, N, K)`` distance matrix
    is never materialized in full. An empty cluster keeps its previous centroid.
    """
    b, n, d = x.shape
    k = centroids.shape[1]
    chunk = max(1, min(n, _TORCH_DIST_ELEMENTS // max(1, b * k)))
    s = sensitivities.float()
    labels = torch.empty(b, n, dtype=torch.long, device=x.device)
    x_sq = (x**2).sum(-1)
    for _ in range(steps):
        c_sq = (centroids**2).sum(-1)
        for start in range(0, n, chunk):
            dist = (
                x_sq[:, start : start + chunk].unsqueeze(-1)
                + c_sq.unsqueeze(1)
                - 2.0 * torch.bmm(x[:, start : start + chunk], centroids.transpose(1, 2))
            )
            labels[:, start : start + chunk] = dist.argmin(-1)
        mass = torch.zeros(b, k, device=x.device).scatter_add_(1, labels, s)
        sums = torch.zeros(b, k, d, device=x.device).scatter_add_(
            1, labels.unsqueeze(-1).expand(-1, -1, d), x * s.unsqueeze(-1)
        )
        centroids = torch.where(
            mass.unsqueeze(-1) > 0, sums / mass.clamp_min(1e-30).unsqueeze(-1), centroids
        )
    return labels, centroids


def _lloyd_until_converged(
    x: torch.Tensor,
    seeds: torch.Tensor,
    sensitivities: torch.Tensor | None,
    max_iter: int,
    tol: float,
    use_flash: bool,
) -> torch.Tensor:
    """One restart of batched Lloyd iterations with a per-block relative-inertia stop.

    Runs ``_CHECK_EVERY`` iterations at a time. A block has converged once its inertia
    improved by less than ``tol`` per iteration over the last chunk, i.e.
    ``(J_prev - J) / J < tol * steps``, or once no assignment changed over the last
    chunk. The batch keeps a fixed shape, so flash-kmeans
    reuses its compiled kernels; converged blocks keep iterating, which never raises
    their inertia, and the restart ends once every block has converged or ``max_iter``
    iterations have run.

    Args:
        x (torch.Tensor): ``(B, N, D)`` vectorized blocks.
        seeds (torch.Tensor): ``(B, K, D)`` initial centroids.
        sensitivities (torch.Tensor | None): Optional ``(B, N)`` point sensitivities.
        max_iter (int): Iteration cap.
        tol (float): Relative inertia improvement per iteration below which a block stops.
        use_flash (bool): Run the iterations with flash-kmeans (unweighted only) instead
            of the torch loop.

    Returns:
        torch.Tensor: ``(B, N)`` labels from the last assignment step.
    """
    n_clusters = seeds.shape[1]
    centroids = seeds.float()
    torch_sensitivities = (
        torch.ones(x.shape[:2], device=x.device) if sensitivities is None else sensitivities
    )
    prev = torch.full((x.shape[0],), float("inf"), device=x.device)
    converged = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
    prev_labels = None
    done = 0
    while done < max_iter:
        steps = min(_CHECK_EVERY, max_iter - done)
        if use_flash:
            labels, centroids = _flash_lloyd_steps(x, centroids, steps)
        else:
            labels, centroids = _torch_lloyd_steps(x, centroids, steps, torch_sensitivities)
        done += steps
        inertia = _block_inertia(x, labels, n_clusters, sensitivities)
        # On the first chunk prev is inf, so the improvement is inf and nothing converges.
        improvement = (prev - inertia) / inertia.clamp_min(torch.finfo(torch.float32).tiny)
        converged |= improvement < tol * steps
        if prev_labels is not None:
            converged |= (labels == prev_labels).all(dim=1)
        prev, prev_labels = inertia, labels
        if bool(converged.all()):
            break
    return labels


def _cluster_torch(
    x: torch.Tensor,
    sensitivities: torch.Tensor | None,
    n_clusters: int,
    n_local_trials: int | None,
    n_init: int,
    max_iter: int,
    tol: float,
    use_flash: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched torch k-means over ``(B, N, D)`` blocks; returns ``(centroids, labels)``.

    Runs on ``x``'s device: a CUDA GPU, or the CPU when scikit-learn is unavailable.
    """
    trials = _resolve_n_local_trials(n_local_trials, n_clusters)
    base_seed = _tensor_seed(x)
    best_labels, best_inertia = None, None
    for restart in range(n_init):
        generator = torch.Generator(device=x.device)
        generator.manual_seed(base_seed + 7919 * restart)
        seeds = batched_kmeanspp_init(x, n_clusters, trials, generator, sensitivities)
        labels = _lloyd_until_converged(x, seeds, sensitivities, max_iter, tol, use_flash)
        inertia = _block_inertia(x, labels, n_clusters, sensitivities)
        if best_labels is None:
            best_labels, best_inertia = labels, inertia
        else:
            better = inertia < best_inertia
            best_labels = torch.where(better.unsqueeze(-1), labels, best_labels)
            best_inertia = torch.minimum(inertia, best_inertia)
    return batched_cluster_avg(x, best_labels, n_clusters, sensitivities), best_labels


def _cluster_sklearn(
    x: torch.Tensor,
    sensitivities: torch.Tensor | None,
    n_clusters: int,
    n_local_trials: int | None,
    n_init: int,
    max_iter: int,
    tol: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """scikit-learn k-means on one ``(N, D)`` block; returns ``(centroids, labels)``.

    Seeds come from ``kmeans_plusplus`` with the same ``n_local_trials`` as the torch
    paths (``KMeans`` itself does not expose it), so
    restarts are run here rather than through ``KMeans(n_init=...)``. Lloyd iterations
    run ``_CHECK_EVERY`` at a time with scikit-learn's own ``tol`` off, so the block stops
    on the same relative-inertia criterion as the GPU paths; scikit-learn still ends a
    chunk early when no assignment changes, which also stops the block.
    """
    from sklearn.cluster import KMeans, kmeans_plusplus  # noqa: PLC0415 - optional dependency

    def lloyd_until_converged(seeds: np.ndarray) -> KMeans:
        centers, prev, done = seeds, np.inf, 0
        while True:
            steps = min(_CHECK_EVERY, max_iter - done)
            fitted = KMeans(n_clusters, init=centers, n_init=1, max_iter=steps, tol=0.0).fit(
                points, sample_weight=s
            )
            done += fitted.n_iter_
            improvement = (prev - fitted.inertia_) / max(fitted.inertia_, np.finfo(np.float32).tiny)
            # Fewer iterations than requested means no assignment changed: a fixed point.
            if fitted.n_iter_ < steps or improvement < tol * steps or done >= max_iter:
                return fitted
            centers, prev = fitted.cluster_centers_, fitted.inertia_

    points = x.float().numpy()
    s = None if sensitivities is None else sensitivities.float().numpy()
    trials = _resolve_n_local_trials(n_local_trials, n_clusters)
    base_seed = _tensor_seed(x)
    best = None
    for restart in range(n_init):
        random_state = np.random.RandomState((base_seed + 7919 * restart) % (2**31 - 1))
        seeds, _ = kmeans_plusplus(
            points, n_clusters, sample_weight=s, random_state=random_state, n_local_trials=trials
        )
        fitted = lloyd_until_converged(seeds)
        if best is None or fitted.inertia_ < best.inertia_:
            best = fitted
    assert best is not None  # n_init >= 1
    labels = torch.from_numpy(best.labels_).long()
    batch_sensitivities = None if sensitivities is None else sensitivities.float().unsqueeze(0)
    centroids = batched_cluster_avg(
        x.float().unsqueeze(0), labels.unsqueeze(0), n_clusters, batch_sensitivities
    )
    return centroids[0], labels


def cluster_vectors(
    blocks: list[torch.Tensor],
    sensitivities: list[torch.Tensor] | None,
    n_clusters: int,
    n_local_trials: int | None,
    n_init: int,
    max_iter: int,
    tol: float,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Cluster every vectorized block of one weight tensor.

    Args:
        blocks (list[torch.Tensor]): Per-block ``(N, D)`` vectors.
        sensitivities (list[torch.Tensor] | None): Optional per-block ``(N,)`` point sensitivities.
        n_clusters (int): Clusters per block. Must be ``<= N``.
        n_local_trials (int | None): k-means++ candidates per seeding step; ``None``
            means scikit-learn's default, ``2 + int(log(n_clusters))``.
        n_init (int): Restarts; the lowest-inertia one is kept per block.
        max_iter (int): Lloyd iteration cap per restart.
        tol (float): Convergence tolerance, see the module docstring.

    Returns:
        list[tuple[torch.Tensor, torch.Tensor]]: Per block, CPU ``(centroids, labels)``
        of shapes ``(n_clusters, D)`` and ``(N,)``.
    """
    backend = vector_kmeans_backend(has_sensitivities=sensitivities is not None)
    if backend == "sklearn":
        return [
            _cluster_sklearn(
                block,
                None if sensitivities is None else sensitivities[i],
                n_clusters,
                n_local_trials,
                n_init,
                max_iter,
                tol,
            )
            for i, block in enumerate(blocks)
        ]

    # One batch per distinct block shape; all blocks of a tensor normally share one. The
    # torch backend runs on the GPU, or on the CPU when scikit-learn is unavailable.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    by_shape: dict[tuple[int, ...], list[int]] = {}
    for i, block in enumerate(blocks):
        by_shape.setdefault(tuple(block.shape), []).append(i)
    results: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for idxs in by_shape.values():
        x = torch.stack([blocks[i] for i in idxs]).to(device).float().contiguous()
        s = (
            None
            if sensitivities is None
            else torch.stack([sensitivities[i] for i in idxs]).to(device).float()
        )
        centroids, labels = _cluster_torch(
            x,
            s,
            n_clusters,
            n_local_trials,
            n_init,
            max_iter,
            tol,
            use_flash=backend == "flash_kmeans",
        )
        for j, i in enumerate(idxs):
            results[i] = (centroids[j].cpu(), labels[j].cpu())
    return [results[i] for i in range(len(blocks))]
