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

import collections
import functools
import logging
import math
import multiprocessing
import os
import time
import warnings

import numpy as np
import torch

from . import _emeans_triton, _kmeanspp_triton

#: Lloyd iterations between convergence checks.
_CHECK_EVERY = 5

#: Keeps the torch Lloyd loop's ``(B, chunk, K)`` distance tile near 0.5 GB.
_TORCH_DIST_ELEMENTS = 2**27

logger = logging.getLogger(__name__)

#: Picks faiss e-means for unweighted GPU clustering, an experimental backend: "emeans"
#: starts from e-means' random rows, "emeans_kmeanspp" from this module's greedy
#: k-means++ seeds, and "emeans_batched" is "emeans_kmeanspp" with all blocks of a tensor
#: in one step (two-number vectors). Unset or "flash" keeps the default backends.
_BACKEND_ENV = "COREAI_OPT_VECTOR_KMEANS_BACKEND"
_EMEANS_BACKENDS = ("emeans", "emeans_kmeanspp", "emeans_batched")


def _requested_backend() -> str:
    """The backend ``COREAI_OPT_VECTOR_KMEANS_BACKEND`` names; ``"flash"`` when unset.

    Raises:
        ValueError: If it names no backend, so a typo cannot silently run the default.
    """
    requested = os.environ.get(_BACKEND_ENV) or "flash"
    if requested not in ("flash", *_EMEANS_BACKENDS):
        raise ValueError(
            f"{_BACKEND_ENV}={requested!r} names no backend; "
            f"use flash, {', '.join(_EMEANS_BACKENDS)}"
        )
    return requested


@functools.cache
def _warn_emeans(backend: str) -> None:
    """Warn, on the first use in a process of each backend, that GPU vector k-means runs e-means.

    A warning rather than a log line, so it shows at any log level.
    """
    warnings.warn(
        f"{_BACKEND_ENV}={backend}: vector palettization's GPU k-means runs faiss "
        "e-means, an experimental backend",
        RuntimeWarning,
        stacklevel=3,
    )


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
        ``"torch"``; for unweighted clustering, ``COREAI_OPT_VECTOR_KMEANS_BACKEND`` can pick
        ``"emeans"``, ``"emeans_kmeanspp"`` or ``"emeans_batched"`` instead. Without a GPU,
        ``"sklearn"``, or ``"torch"`` (on CPU, with a one-time warning) if scikit-learn is not
        installed.

    Raises:
        ValueError: If ``COREAI_OPT_VECTOR_KMEANS_BACKEND`` names no backend.
    """
    requested = _requested_backend()
    if torch.cuda.is_available():
        if not has_sensitivities and requested in _EMEANS_BACKENDS:
            _warn_emeans(requested)
            return requested
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


def _clock(device: torch.device) -> float:
    """``time.perf_counter()`` once the work queued on ``device`` has finished."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _cluster_emeans(
    x: torch.Tensor, n_clusters: int, n_local_trials: int | None, kmeanspp: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """faiss e-means on each ``(N, D)`` block of ``x``; returns ``(centroids, labels)``.

    e-means solves one k-means problem per ``train`` call, so the blocks run one after
    another, each once for e-means' default 5 epochs: ``n_init``, ``max_iter`` and ``tol``
    do not apply. Assignment stays in float32 (``quantizer=None``), as the e-means README
    recommends for ``D <= 3``. With ``kmeanspp``, each block starts from this module's
    greedy k-means++ seeds instead of e-means' random rows. Labels come from e-means'
    ``assign`` against its final centroids, because the labels ``train`` returns were
    computed while the centroids were still moving. Final centroids are the exact means
    of those labels, as on the other backends.
    """
    from faiss.contrib.e_means import QuantizedEmeans  # noqa: PLC0415 - optional dependency

    seeds = None
    if kmeanspp:
        generator = torch.Generator(device=x.device)
        generator.manual_seed(_tensor_seed(x))
        trials = _resolve_n_local_trials(n_local_trials, n_clusters)
        seeds = batched_kmeanspp_init(x, n_clusters, trials, generator)
    labels, train_s, epoch_s, assign_s = [], 0.0, 0.0, 0.0
    for i, block in enumerate(x):
        em = QuantizedEmeans(x.shape[-1], n_clusters, quantizer=None)
        t0 = _clock(x.device)
        em.train(block, None if seeds is None else seeds[i])
        t1 = _clock(x.device)
        _, block_labels = em.assign(block)
        t2 = _clock(x.device)
        train_s, assign_s = train_s + t1 - t0, assign_s + t2 - t1
        epoch_s += em.iteration_stats[-1]["time"]
        labels.append(block_labels)
    logger.info(
        "e-means: %d blocks, %.3f s in train(), %.3f s of it in e-means epochs, %.3f s in assign()",
        len(x),
        train_s,
        epoch_s,
        assign_s,
    )
    stacked = torch.stack(labels)
    return batched_cluster_avg(x, stacked, n_clusters), stacked


def _assign_nearest(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """``(B, N)`` index of the nearest centroid for every point, in float32.

    On CUDA with two-number vectors this is the fused kernel the training step uses; elsewhere
    :func:`_assign_nearest_torch`, whose arithmetic the kernel reproduces.
    """
    if _emeans_triton.available(x):
        return _emeans_triton.assign_labels(x, centroids)
    return _assign_nearest_torch(x, centroids)


def _assign_nearest_torch(x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """:func:`_assign_nearest` in torch: the norm trick, chunked over ``N`` so the
    ``(B, chunk, K)`` distance tile stays near the size the torch Lloyd loop uses. The point's own
    squared norm does not change the argmin."""
    b, n, _ = x.shape
    k = centroids.shape[1]
    chunk = max(1, min(n, _TORCH_DIST_ELEMENTS // max(1, b * k)))
    c_sq = (centroids * centroids).sum(-1).unsqueeze(1)
    labels = torch.empty(b, n, dtype=torch.long, device=x.device)
    for start in range(0, n, chunk):
        part = x[:, start : start + chunk]
        labels[:, start : start + chunk] = (
            c_sq - 2.0 * torch.bmm(part, centroids.transpose(1, 2))
        ).argmin(-1)
    return labels


def _accumulate_batch_torch(
    x: torch.Tensor, picked: torch.Tensor, centroids: torch.Tensor, stats: torch.Tensor
) -> None:
    """Add the rows ``picked`` of every block to ``stats[b, j] = (sum x, sum y, count)`` of its
    nearest centroid ``j``: the reference for the fused kernel, and the path off CUDA."""
    d = x.shape[-1]
    points = x[:, picked]
    c_sq = (centroids * centroids).sum(-1)
    labels = (c_sq.unsqueeze(1) - 2.0 * torch.bmm(points, centroids.transpose(1, 2))).argmin(-1)
    stats[..., d].scatter_add_(1, labels, torch.ones_like(labels, dtype=torch.float32))
    stats[..., :d].scatter_add_(1, labels.unsqueeze(-1).expand(-1, -1, d), points)


def _emeans_step(
    x: torch.Tensor,
    picked: torch.Tensor,
    centroids: torch.Tensor,
    usage: torch.Tensor,
    alpha: float | torch.Tensor,
    decay: float | torch.Tensor,
    replace_usage: bool,
    pass_stats: torch.Tensor | None,
) -> None:
    """One e-means update of every block's centroids from the rows ``picked``, in place.

    Algorithm 1 as faiss' torch backend applies it: assign the batch to the current centroids,
    take per-cluster counts and sums, fold them into the running mass (``usage``) and position
    with step ``alpha`` (``decay`` is ``1 - alpha``), and add them to the pass totals when those
    are kept. ``alpha`` and ``decay`` are Python floats, or 0-dim tensors when captured in a
    CUDA graph. ``replace_usage`` is faiss' ``alpha == 1`` case: the old mass is discarded.
    Sums and counts live in one ``(B, K, D + 1)`` buffer, ``pass_stats`` being the same layout.
    """
    b, _, d = x.shape
    k = centroids.shape[1]
    stats = torch.zeros(b, k, d + 1, device=x.device)
    if _emeans_triton.available(x):
        _emeans_triton.assign_accumulate(x, picked, centroids, stats)
    else:
        _accumulate_batch_torch(x, picked, centroids, stats)
    if pass_stats is not None:
        pass_stats += stats
    _ema_update(stats[..., :d], stats[..., d], centroids, usage, alpha, decay, replace_usage)


def _ema_update(
    sums: torch.Tensor,
    counts: torch.Tensor,
    centroids: torch.Tensor,
    usage: torch.Tensor,
    alpha: float | torch.Tensor,
    decay: float | torch.Tensor,
    replace_usage: bool,
) -> None:
    """Fold a batch's per-cluster ``sums`` and ``counts`` into ``centroids`` and ``usage``.

    Algorithm 1, line 8 (Equation 2): ``usage`` is the running mass, updated to
    ``decay * usage + alpha * counts`` (just ``alpha * counts`` when ``replace_usage``), and an
    occupied cluster moves towards its batch mean by ``alpha * counts / mass``. On CUDA this is
    faiss' own Triton kernel, which every block's rows can share (the update is independent per
    cluster); elsewhere it is the same arithmetic in torch.
    """
    if centroids.is_cuda:
        from faiss.contrib.e_means.triton_assignment import ema_update  # noqa: PLC0415

        b, k, d = centroids.shape
        updated = ema_update(
            sums.view(b * k, d),
            counts.view(b * k),
            centroids.view(b * k, d),
            usage.view(b * k),
            alpha,
            decay,
            replace_usage,
        )
        assert updated, (
            "faiss' EMA kernel declined statistics laid out as this module lays them out"
        )
        return
    touched = counts > 0
    if replace_usage:
        retained = torch.zeros_like(usage)
    else:
        usage.mul_(decay)
        retained = usage
    mass = retained + alpha * counts
    weight = alpha * counts / torch.where(touched, mass, torch.ones_like(mass))
    mean = sums / torch.where(touched, counts, torch.ones_like(counts)).unsqueeze(-1)
    position = (mean - centroids) * weight.unsqueeze(-1) + centroids
    position = torch.where((weight == 1).unsqueeze(-1), mean, position)
    torch.where(touched.unsqueeze(-1), position, centroids, out=centroids)
    torch.where(touched, mass, usage, out=usage)


def _closing_centroids(pass_stats: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """e-means' closing centroids: the means of the last pass, or the moving centroid if empty."""
    sums, counts = pass_stats[..., :-1], pass_stats[..., -1]
    closing = sums / counts.clamp_min(1).unsqueeze(-1)
    return torch.where((counts > 0).unsqueeze(-1), closing, centroids)


def _centroids_eager(
    x: torch.Tensor,
    seeds: torch.Tensor,
    batch: int,
    alphas: list[float],
    rows_per_epoch: list[int],
) -> torch.Tensor:
    """The e-means epochs as a plain loop: one launch per op, every step."""
    from faiss.contrib.e_means.base import _INIT_USAGE  # noqa: PLC0415 - optional dependency
    from faiss.contrib.e_means.e_means_torch import (  # noqa: PLC0415 - optional dependency
        _epoch_seed,
        _torch_generator,
    )

    b, n, d = x.shape
    k = seeds.shape[1]
    centroids = seeds.float().clone()
    usage = torch.full((b, k), _INIT_USAGE, device=x.device)
    pass_stats = torch.zeros(b, k, d + 1, device=x.device)
    step = 0
    for epoch, rows in enumerate(rows_per_epoch):
        seed = _epoch_seed(1234, epoch)  # faiss' default seed, as every per-block run uses
        order = torch.randperm(n, generator=_torch_generator(x.device, seed), device=x.device)
        last = epoch == len(rows_per_epoch) - 1
        for begin in range(0, rows, batch):
            alpha = alphas[step]
            _emeans_step(
                x,
                order[begin : min(begin + batch, rows)],
                centroids,
                usage,
                alpha,
                1.0 - alpha,
                alpha == 1,
                pass_stats if last else None,
            )
            step += 1
    return _closing_centroids(pass_stats, centroids)


class _EmeansEpochGraph:
    """One epoch of batched e-means steps, captured once per shape and replayed with new inputs.

    The epoch reads and writes only the static buffers below. A caller refills them before each
    replay: the data, the epoch's permutation and step sizes, the state and the pass totals.
    The arithmetic is :func:`_emeans_step`'s, unchanged.
    """

    def __init__(self, b: int, n: int, k: int, d: int, batch: int, device: torch.device):
        self.batch = batch
        self.steps = (n + batch - 1) // batch
        self.x = torch.zeros(b, n, d, device=device)
        self.order = torch.arange(n, device=device)
        self.rates = torch.full((self.steps, 2), 0.5, device=device)
        self.centroids = torch.zeros(b, k, d, device=device)
        self.usage = torch.ones(b, k, device=device)
        self.pass_stats = torch.zeros(b, k, d + 1, device=device)
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            # Outside the capture, once per distinct step shape (the full batch and the
            # remainder): a kernel is compiled and loaded the first time it is launched.
            for step in {0, self.steps - 1}:
                self._step(step)
        torch.cuda.current_stream(device).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            for step in range(self.steps):
                self._step(step)

    def _step(self, step: int) -> None:
        begin = step * self.batch
        _emeans_step(
            self.x,
            self.order[begin : begin + self.batch],
            self.centroids,
            self.usage,
            self.rates[step, 0],
            self.rates[step, 1],
            False,
            self.pass_stats,
        )


#: Captured epochs, least recently used first. A graph is a function of shape only: every
#: buffer it reads is refilled per call, so a cache hit and a fresh capture give equal results.
_EPOCH_GRAPHS: collections.OrderedDict[tuple, _EmeansEpochGraph] = collections.OrderedDict()
_MAX_EPOCH_GRAPHS = 32


def _epoch_graph(
    b: int, n: int, k: int, d: int, batch: int, device: torch.device
) -> _EmeansEpochGraph:
    """The captured epoch for this shape, capturing it on first use."""
    key = (device, b, n, k, d, batch)
    graph = _EPOCH_GRAPHS.pop(key, None) or _EmeansEpochGraph(b, n, k, d, batch, device)
    _EPOCH_GRAPHS[key] = graph
    while len(_EPOCH_GRAPHS) > _MAX_EPOCH_GRAPHS:
        _EPOCH_GRAPHS.popitem(last=False)
    return graph


def _centroids_graphed(
    x: torch.Tensor, seeds: torch.Tensor, batch: int, alphas: list[float], nepochs: int
) -> torch.Tensor:
    """The same epochs as :func:`_centroids_eager`, replayed from a captured graph (CUDA).

    Needs every epoch to be a full pass and the step never to be 1; the caller checks.
    """
    from faiss.contrib.e_means.base import _INIT_USAGE  # noqa: PLC0415 - optional dependency
    from faiss.contrib.e_means.e_means_torch import (  # noqa: PLC0415 - optional dependency
        _epoch_seed,
        _torch_generator,
    )

    b, n, d = x.shape
    k = seeds.shape[1]
    graph = _epoch_graph(b, n, k, d, batch, x.device)
    graph.x.copy_(x)
    graph.centroids.copy_(seeds)
    graph.usage.fill_(_INIT_USAGE)
    rates = torch.tensor([(a, 1.0 - a) for a in alphas], dtype=torch.float32, device=x.device)
    for epoch in range(nepochs):
        seed = _epoch_seed(1234, epoch)
        graph.order.copy_(
            torch.randperm(n, generator=_torch_generator(x.device, seed), device=x.device)
        )
        graph.rates.copy_(rates[epoch * graph.steps : (epoch + 1) * graph.steps])
        graph.pass_stats.zero_()
        graph.graph.replay()
    return _closing_centroids(graph.pass_stats, graph.centroids)


def _emeans_batched_centroids(
    x: torch.Tensor, seeds: torch.Tensor, nepochs: int = 5
) -> torch.Tensor:
    """e-means' closing centroids ``(B, K, D)`` for ``(B, N, D)`` blocks started at ``seeds``.

    The recipe is faiss e-means': its batch law, step law, epoch schedule, cosine step,
    per-epoch permutation and mass-weighted EMA update, with the closing centroids taken as the
    means of the last pass. Only the loop is batched, so one step updates every block. All
    blocks share each epoch's permutation, as faiss' per-block runs do (one fixed seed).
    Valid for ``D <= 2``, where e-means' default turns revival of empty clusters off
    (``_NO_REVIVAL_MAX_D`` in ``faiss.contrib.e_means.base``). On CUDA the epochs are
    replayed from a captured graph when the recipe allows it, and run as a plain loop otherwise.
    """
    from faiss.contrib.e_means.e_means_torch import (  # noqa: PLC0415 - optional dependency
        _epoch_rows,
        _resolve_recipe,
        _scheduled_alpha,
    )

    n, k = x.shape[1], seeds.shape[1]
    batch, alpha0 = _resolve_recipe(n, k, nepochs, "quality", None, None)
    rows_per_epoch = _epoch_rows(n, nepochs)
    total_steps = sum((rows + batch - 1) // batch for rows in rows_per_epoch)
    alphas = [_scheduled_alpha(alpha0, step, total_steps) for step in range(total_steps)]
    if x.is_cuda and alpha0 < 1 and rows_per_epoch == [n] * nepochs:
        return _centroids_graphed(x, seeds, batch, alphas, nepochs)
    return _centroids_eager(x, seeds, batch, alphas, rows_per_epoch)


def _kmeanspp_seeds_fused(
    x: torch.Tensor, n_clusters: int, n_local_trials: int, generator: torch.Generator
) -> torch.Tensor:
    """:func:`batched_kmeanspp_init` for unweighted ``(B, N, 2)`` blocks, with fused Triton steps.

    The generator is called the same way, in the same order. Each step's candidate scores and
    nearest-seed update come from two kernels that read the points once. Distances are
    ``(x - c)^2`` summed directly, not the norm trick, so a draw or a candidate comparison that is
    within float rounding of a tie can go the other way: the seeds can differ from the unfused
    loop's, and so can every draw after the first difference.
    """
    b, n, d = x.shape
    if n_clusters > n:
        raise ValueError(f"n_clusters {n_clusters} exceeds {n} points per block")
    rows = torch.arange(b, device=x.device)
    seeds = torch.empty(b, n_clusters, d, device=x.device, dtype=x.dtype)
    first = torch.multinomial(torch.ones(b, n, device=x.device), 1, generator=generator)
    seeds[:, 0] = x[rows, first.squeeze(1)]
    closest = ((x - seeds[:, 0:1]) ** 2).sum(-1)
    total = closest.sum(dim=1)
    for i in range(1, n_clusters):
        # Once every point of a block coincides with a seed its potential is zero and it cannot
        # be sampled from; draw uniformly for that block instead, as the unfused loop does.
        potential = torch.where((total <= 0).unsqueeze(1), 1.0, closest)
        cand = torch.multinomial(potential, n_local_trials, replacement=True, generator=generator)
        cand_pts = torch.gather(x, 1, cand.unsqueeze(-1).expand(-1, -1, d))
        best = _kmeanspp_triton.candidate_potentials(x, closest, cand_pts).argmin(dim=1)
        seeds[:, i] = cand_pts[rows, best]
        total = _kmeanspp_triton.update_closest(x, closest, seeds[:, i])
    return seeds


def _cluster_emeans_batched(
    x: torch.Tensor, n_clusters: int, n_local_trials: int | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """k-means++-seeded e-means on every ``(N, D)`` block of ``x`` at once.

    One step updates all blocks, where :func:`_cluster_emeans` with ``kmeanspp=True`` updates
    one block at a time; on the CPU the two agree to floating-point order. Labels are an exact
    assignment against the closing centroids, and the returned centroids are the exact means of
    those labels.
    The labels are ``uint8`` when every index fits one, as the palettizer stores them, and
    ``int64`` otherwise.
    """
    generator = torch.Generator(device=x.device)
    generator.manual_seed(_tensor_seed(x))
    trials = _resolve_n_local_trials(n_local_trials, n_clusters)
    if _kmeanspp_triton.available(x, trials):
        seeds = _kmeanspp_seeds_fused(x, n_clusters, trials, generator)
    else:
        seeds = batched_kmeanspp_init(x, n_clusters, trials, generator)
    labels = _assign_nearest(x, _emeans_batched_centroids(x, seeds))
    centroids = batched_cluster_avg(x, labels, n_clusters)
    if n_clusters <= torch.iinfo(torch.uint8).max + 1:
        labels = labels.to(torch.uint8)
    return centroids, labels


def _stack_to_device(blocks: list[torch.Tensor], device: str) -> torch.Tensor:
    """Stack ``blocks`` into one contiguous fp32 tensor on ``device``.

    CPU blocks bound for CUDA are stacked into a pinned staging buffer, so the copy to the GPU is
    one transfer from page-locked memory. Torch caches pinned buffers by size, so after the first
    call of a shape the allocation is free.
    """
    if device == "cuda" and blocks[0].device.type == "cpu":
        staging = torch.empty(
            (len(blocks), *blocks[0].shape), dtype=blocks[0].dtype, pin_memory=True
        )
        torch.stack(blocks, out=staging)
        return staging.to(device, non_blocking=True).float().contiguous()
    return torch.stack(blocks).to(device).float().contiguous()


def _to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """``tensor`` on the CPU: one transfer into pinned memory when it comes from CUDA.

    Slicing the result per block then costs nothing, where a ``.cpu()`` per block is a transfer
    and a synchronization each.
    """
    if not tensor.is_cuda:
        return tensor.cpu()
    host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
    host.copy_(tensor, non_blocking=True)
    torch.cuda.current_stream(tensor.device).synchronize()
    return host


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
    start = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pinned = backend == "emeans_batched"  # the other backends keep their own transfers
    by_shape: dict[tuple[int, ...], list[int]] = {}
    for i, block in enumerate(blocks):
        by_shape.setdefault(tuple(block.shape), []).append(i)
    results: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for idxs in by_shape.values():
        if pinned:
            x = _stack_to_device([blocks[i] for i in idxs], device)
        else:
            x = torch.stack([blocks[i] for i in idxs]).to(device).float().contiguous()
        s = (
            None
            if sensitivities is None
            else torch.stack([sensitivities[i] for i in idxs]).to(device).float()
        )
        if backend == "emeans_batched" and x.shape[-1] <= 2:
            centroids, labels = _cluster_emeans_batched(x, n_clusters, n_local_trials)
        elif backend in _EMEANS_BACKENDS:
            centroids, labels = _cluster_emeans(
                x, n_clusters, n_local_trials, kmeanspp=backend != "emeans"
            )
        else:
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
        if pinned:
            centroids, labels = _to_cpu(centroids), _to_cpu(labels)
            for j, i in enumerate(idxs):
                results[i] = (centroids[j], labels[j])
        else:
            for j, i in enumerate(idxs):
                results[i] = (centroids[j].cpu(), labels[j].cpu())
    logger.info(
        "cluster_vectors: backend=%s blocks=%d k=%d seconds=%.3f",
        backend,
        len(blocks),
        n_clusters,
        time.perf_counter() - start,
    )
    return [results[i] for i in range(len(blocks))]
