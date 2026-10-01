# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import logging as _logging
import math as _math
import os as _os
import time as _time
from dataclasses import dataclass as _dataclass
from typing import Any as _Any

import numpy as _np
import torch as _torch

_logger = _logging.getLogger(__name__)

#: Backend selected when ``init`` does not name one explicitly. Reading this at
#: ``fit`` time lets a run switch implementations without touching call sites
#: that hardcode ``init="kmeans++"`` (e.g. ``_cluster_weights_2d``).
KMEANS_BACKEND_ENV = "COREAI_KMEANS_BACKEND"

_TORCH_BACKEND = "kmeans++"
_SKLEARN_BACKEND = "sklearn"
_SKLEARN_MINIBATCH_BACKEND = "sklearn_minibatch"
_CUML_BACKEND = "cuml"
_FLASH_BACKEND = "flash_kmeans"
_VALID_BACKENDS = (
    _TORCH_BACKEND,
    _SKLEARN_BACKEND,
    _SKLEARN_MINIBATCH_BACKEND,
    _CUML_BACKEND,
    _FLASH_BACKEND,
)

#: Fixed so repeated runs of the same configuration agree. The torch backend
#: seeds from the ambient RNG instead, so only the sklearn paths are reproducible;
#: cuML accepts the seed but reduces with GPU atomics, so it is not bit-exact.
_RANDOM_STATE = 0


@_dataclass
class StageTimings:
    """Cumulative seconds in each clustering stage, summed over every weight tensor.

    Palettization splits into two stages with very different characteristics: choosing
    initial centroids (strictly sequential in ``n_clusters``, launch-bound) and the Lloyd
    refinement loop (fully parallel per iteration, compute-bound). Their costs move
    independently with ``n_init``, ``n_clusters`` and the backend, so a single
    ``prepare()`` number hides which one dominates.

    Attributes:
        init_seconds (float): Time generating initial centroids.
        lloyd_seconds (float): Time in the assignment/update loop.
        init_calls (int): Number of init invocations.
        lloyd_calls (int): Number of Lloyd invocations.
        init_separable (bool): False once any backend reported a fused init+Lloyd, i.e.
            cuml, whose ``fit()`` is atomic and exposes no standalone seeding entry
            point. When False, ``lloyd_seconds`` includes that backend's init and
            ``init_seconds`` understates the real total.
    """

    init_seconds: float = 0.0
    lloyd_seconds: float = 0.0
    init_calls: int = 0
    lloyd_calls: int = 0
    init_separable: bool = True

    def summary(self) -> str:
        """One-line report, suitable for a log at the end of clustering."""
        total = self.init_seconds + self.lloyd_seconds
        note = "" if self.init_separable else "  [init NOT separable for cuml: folded into stage 2]"
        return (
            f"stage 1 init: {self.init_seconds:.2f}s over {self.init_calls} calls | "
            f"stage 2 lloyd: {self.lloyd_seconds:.2f}s over {self.lloyd_calls} calls | "
            f"sum {total:.2f}s{note}"
        )


#: Process-local accumulator. Only meaningful with ``num_workers=1``: worker processes
#: keep their own copy and never report it back to the parent.
_TIMINGS = StageTimings()


def reset_stage_timings() -> None:
    """Zero the stage accumulator. Call once before a palettization run."""
    global _TIMINGS
    _TIMINGS = StageTimings()


def stage_timings() -> StageTimings:
    """The accumulated stage timings for this process."""
    return _TIMINGS


#: Backend/device pairs already announced by this process. Clustering runs once
#: per block -- hundreds of times per model -- so logging every call would drown
#: the output. Centroid calculation is farmed out to `spawn`ed workers, so each
#: worker logs once, which is what makes it visible that the setting actually
#: crossed the process boundary rather than being silently dropped.
_ANNOUNCED: set[tuple[str, str]] = set()


def _resolve_backend(init: str) -> str:
    """Pick the clustering backend, letting the environment override ``init``.

    Raises:
        ValueError: If either the environment variable or ``init`` names a
            backend that does not exist. Failing loudly beats silently running
            a different algorithm than the caller asked for.
    """
    override = _os.environ.get(KMEANS_BACKEND_ENV, "").strip()
    backend = override or init
    if backend not in _VALID_BACKENDS:
        source = f"{KMEANS_BACKEND_ENV}={override!r}" if override else f"init={init!r}"
        raise ValueError(f"{source} is not one of {_VALID_BACKENDS}")
    return backend


def _announce(backend: str, device: _torch.device) -> None:
    """Log the backend and the device it will actually run on, once per pair.

    The device matters as much as the backend: the caller may hand us CUDA
    tensors, but the sklearn backends have no GPU path and fall back to CPU, so
    comparing them against the torch backend on a GPU node is comparing two
    different devices. Recording both makes that visible in the log instead of
    having to infer it.
    """
    key = (backend, device.type)
    if key in _ANNOUNCED:
        return
    _ANNOUNCED.add(key)
    override = _os.environ.get(KMEANS_BACKEND_ENV, "").strip()
    origin = f"{KMEANS_BACKEND_ENV}={override!r}" if override else f"{KMEANS_BACKEND_ENV} unset"
    if backend in (_SKLEARN_BACKEND, _SKLEARN_MINIBATCH_BACKEND):
        runs_on = "cpu (sklearn has no GPU path)"
    elif backend == _CUML_BACKEND:
        runs_on = "cuda (cuml is GPU-only)"
    elif backend == _FLASH_BACKEND:
        runs_on = f"{device} (triton kernel on cuda, torch fallback elsewhere)"
    else:
        runs_on = device
    _logger.info(
        "k-means backend: %s  [pid %d, input on %s, runs on %s, from %s]",
        backend,
        _os.getpid(),
        device,
        runs_on,
        origin,
    )


#: Whether the RAPIDS native libraries have been loaded in this process.
_RAPIDS_LIBS_LOADED = False

KMEANS_N_INIT_ENV = "COREAI_KMEANS_N_INIT"


def _resolve_n_init(n_init: int) -> int:
    """Return ``n_init``, or the environment override if one is set.

    Args:
        n_init (int): The value the caller passed.

    Returns:
        int: The number of k-means restarts to run.

    Raises:
        ValueError: If the environment variable is not a positive integer.
            Failing loudly beats silently running a different number of restarts.
    """
    override = _os.environ.get(KMEANS_N_INIT_ENV, "").strip()
    if not override:
        return n_init
    try:
        value = int(override)
    except ValueError as err:
        raise ValueError(f"{KMEANS_N_INIT_ENV}={override!r} is not an integer") from err
    if value < 1:
        raise ValueError(f"{KMEANS_N_INIT_ENV}={override!r} must be >= 1")
    return value

#: Default restarts for vector palettization, independent of which backend runs.
_VECTOR_N_INIT = 12

#: Set to ``1`` to make the cuml backend fit twice per restart -- once to obtain its
#: k-means++ seeds, once to refine them -- so stage 1 and stage 2 can be timed apart.
#: Off by default because it hoists cuML's internal ``n_init`` loop into Python, which
#: is the same algorithm but not the same code path as a single ``fit(n_init=k)``.
KMEANS_CUML_SPLIT_ENV = "COREAI_KMEANS_CUML_SPLIT"

#: ``max_iter`` for the seeding-only cuml fit. cuML rejects 0, so stage 1 unavoidably
#: includes exactly one Lloyd iteration; at ``max_iter=300`` that is <1% of stage 2.
_CUML_INIT_MAX_ITER = 1


#: Candidates per k-means++ step. ``auto`` matches cuVS's greedy
#: ``2 + ceil(log(n_clusters))`` (8 at k=256); an integer sets it directly; unset means
#: 1, i.e. vanilla -- the default 
KMEANS_N_LOCAL_TRIALS_ENV = "COREAI_KMEANS_N_LOCAL_TRIALS"


def resolve_n_local_trials(n_clusters: int, n_local_trials: int | None = None) -> int:
    """Return the k-means++ candidate count, letting the environment override the caller.

    Args:
        n_clusters (int): Needed for the ``auto`` formula.
        n_local_trials (int | None): The caller's value; ``None`` means vanilla.

    Returns:
        int: Candidates to draw per seeding step, at least 1.

    Raises:
        ValueError: If the environment variable is neither ``auto`` nor a positive int.
    """
    override = _os.environ.get(KMEANS_N_LOCAL_TRIALS_ENV, "").strip()
    if override:
        if override == "auto":
            # cuVS: cpp/src/cluster/detail/kmeans.cuh, kmeansPlusPlus().
            return 2 + _math.ceil(_math.log(n_clusters))
        if not override.isdigit() or int(override) < 1:
            raise ValueError(
                f"{KMEANS_N_LOCAL_TRIALS_ENV} must be 'auto' or a positive integer,"
                f" got {override!r}"
            )
        return int(override)
    return 1 if n_local_trials is None else max(1, n_local_trials)


#: Selects the STAGE 1 (seeding) implementation for the batched vector path:
#: ``torch`` (default, our batched k-means++) or ``cuml`` (cuML's own greedy k-means++,
#: one call per block). Exists to cross the two stages against each other and find which
#: one carries the quality gap -- see :func:`generate_init_seeds`.
KMEANS_INIT_IMPL_ENV = "COREAI_KMEANS_INIT_IMPL"

#: Selects the STAGE 2 (Lloyd) implementation: ``flash`` (default, Triton),
#: ``torch`` (ours, exact fp32) or ``cuml`` (one call per block, seeded with our
#: centroids). ``flash`` computes its distances through ``tl.dot``, which defaults to
#: **TF32**;
KMEANS_LLOYD_IMPL_ENV = "COREAI_KMEANS_LLOYD_IMPL"

_INIT_IMPLS = ("torch", "cuml")
_LLOYD_IMPLS = ("flash", "torch", "cuml")


def _resolve_impl(env_var: str, valid: tuple[str, ...], default: str) -> str:
    """Return the stage implementation named by ``env_var``, or ``default``."""
    override = _os.environ.get(env_var, "").strip()
    if not override:
        return default
    if override not in valid:
        raise ValueError(f"{env_var}={override!r} is not one of {valid}")
    return override


def resolve_init_impl() -> str:
    """Return the stage-1 implementation: ``torch`` or ``cuml``."""
    return _resolve_impl(KMEANS_INIT_IMPL_ENV, _INIT_IMPLS, "torch")


def resolve_lloyd_impl() -> str:
    """Return the stage-2 implementation: ``flash``, ``torch`` or ``cuml``."""
    return _resolve_impl(KMEANS_LLOYD_IMPL_ENV, _LLOYD_IMPLS, "flash")


def cuml_split_requested() -> bool:
    """Return True if the cuml backend should fit in two timed stages."""
    return _os.environ.get(KMEANS_CUML_SPLIT_ENV, "").strip() == "1"


#: Memoized result of the cuML usability probe.
_CUML_USABLE: bool | None = None
_FLASH_USABLE: bool | None = None


def _cuml_is_usable() -> bool:
    """Whether the cuml backend can actually run in this process.

    Deliberately stronger than an installation check. The RAPIDS wheels can all be
    present while ``libcuml.so`` still fails to load (layered environments break it
    -- see :func:`_load_rapids_native_libs`), and a default that selected a broken
    cuml would turn working runs into failures. So probe by really importing it,
    once per process, and cache the answer. The ~8s import is not wasted when the
    probe succeeds, because the backend needs it anyway.

    Returns:
        bool: True if CUDA is available and ``cuml.cluster.KMeans`` imports.
    """
    global _CUML_USABLE
    if _CUML_USABLE is not None:
        return _CUML_USABLE

    if not _torch.cuda.is_available():
        _CUML_USABLE = False
        return _CUML_USABLE

    try:
        _load_rapids_native_libs()
        from cuml.cluster import KMeans  # noqa: F401,PLC0415

        _CUML_USABLE = True
    except Exception as exc:  # noqa: BLE001 - any failure means "fall back"
        # Deliberately does NOT claim which backend will run: COREAI_KMEANS_BACKEND
        # overrides the default downstream, so asserting a fallback here produced a
        # log line that flatly contradicted the backend a run actually used.
        _logger.info("cuml not usable here (%s); it will not be chosen by default", exc)
        _CUML_USABLE = False
    return _CUML_USABLE


def _flash_is_usable() -> bool:
    """Whether the flash_kmeans backend can actually run in this process.

    Same contract as :func:`_cuml_is_usable`: probe by importing, once, and cache. The
    package silently falls back to a slow torch path if its Triton import fails, so a
    successful import is the weakest check worth making, not a guarantee of the kernel.

    Returns:
        bool: True if CUDA is available and ``flash_kmeans`` imports.
    """
    global _FLASH_USABLE
    if _FLASH_USABLE is not None:
        return _FLASH_USABLE

    if not _torch.cuda.is_available():
        _FLASH_USABLE = False
        return _FLASH_USABLE

    try:
        from flash_kmeans import batch_kmeans_Euclid  # noqa: F401,PLC0415

        _FLASH_USABLE = True
    except Exception as exc:  # noqa: BLE001 - any failure means "fall back"
        _logger.info("flash_kmeans not usable here (%s); it will not be chosen by default", exc)
        _FLASH_USABLE = False
    return _FLASH_USABLE


def _default_vector_backend(weighted: bool = False) -> str:
    """Pick the vector-palettization backend for this machine.

    Device-driven only: the fastest backend that can actually run here. On CUDA that is
    ``flash_kmeans`` (its restarts are batched across a tensor's blocks -- 252 clustering
    calls for a 4B model against 172,800), then ``cuml``, then the torch fallback. Off
    CUDA both GPU backends are unusable, so it is always ``kmeans++``.

    Measured full-model cost at ``n_init=12`` unless noted (Qwen3-4B, n8/gs32/cd2, 252
    tensors, ``num_workers=1``): ``flash_kmeans`` 4737s/17.1599, ``cuml`` at its own knee
    of ``n_init=4`` 9606s/17.2234, ``kmeans++`` at ``n_init=5`` 39228s/17.5235.

    The restart count is NOT part of this decision -- see :data:`_VECTOR_N_INIT`.

    Args:
        weighted (bool): True when sensitivities will be passed as ``sample_weight``.
            flash has no ``sample_weight`` equivalent and raises on one, so a weighted
            run skips it.

    Returns:
        str: One of :data:`_VALID_BACKENDS`.
    """
    if not weighted and _flash_is_usable():
        return _FLASH_BACKEND
    if _cuml_is_usable():
        return _CUML_BACKEND
    return _TORCH_BACKEND


#: Backends whose clustering runs on the GPU, so that extra worker processes buy no
#: parallelism (one device serialises the work) while still paying per-process CUDA
#: context switching. 
_GPU_SERIALISED_BACKENDS = (_CUML_BACKEND, _FLASH_BACKEND)


def resolve_vector_backend(weighted: bool = False) -> str:
    """Return the backend vector palettization will actually use.

    Combines the device default with the ``COREAI_KMEANS_BACKEND`` override, so callers --
    notably the palettizer's parent process -- can report and act on the real choice.
    Worker processes are where :meth:`_EfficientKMeans.fit` logs its backend, and that
    logging never reaches the parent, so without this a multi-worker run leaves no record
    of which backend it used.

    An explicitly requested backend is honoured even if it is not installed: the run then
    fails loudly rather than silently measuring something other than what it names.

    Args:
        weighted (bool): Passed through to :func:`_default_vector_backend`.

    Returns:
        str: One of :data:`_VALID_BACKENDS`.
    """
    return _resolve_backend(_default_vector_backend(weighted))


def vector_backend_is_gpu_serialised(backend: str) -> bool:
    """Whether extra worker processes hurt rather than help for this backend."""
    return backend in _GPU_SERIALISED_BACKENDS


def _load_rapids_native_libs() -> None:
    """Make RAPIDS' native libraries loadable before ``import cuml``.

    RAPIDS ships ``libcuml.so`` and its dependencies in separate ``lib*-cu12``
    wheels, in directories that are not on the dynamic loader path; each of those
    wheels exposes a ``load_library()`` shim that dlopens its own ``.so``.
    Importing cuml is supposed to trigger them, but it does not do so reliably
    once cuml resolves from a different site-packages than the importing code --
    which is what the eval's layered ``uv run --with`` environment produces -- and
    the import then dies with "libcuml.so: cannot open shared object file" even
    though every wheel installed correctly. Calling the shims explicitly, in
    dependency order, is what actually makes the import work.

    Best effort by design: a missing or silent shim is ignored here so that the
    real diagnosis comes from the cuml import in :meth:`_EfficientKMeans._fit_cuml`,
    which raises an actionable error.
    """
    global _RAPIDS_LIBS_LOADED
    if _RAPIDS_LIBS_LOADED:
        return
    _RAPIDS_LIBS_LOADED = True

    for name in ("librmm", "libraft", "libcuvs", "libcudf", "libcuml"):
        try:
            __import__(name).load_library()
        except Exception as exc:  # noqa: BLE001 - see docstring
            _logger.debug("RAPIDS %s.load_library() unavailable: %s", name, exc)


def _to_cuml_input(X: _torch.Tensor) -> _Any:
    """Convert a tensor to something cuML accepts, avoiding a host round-trip.

    cuML's input adapter reads ``.dtype`` and hands it to numpy, so passing a
    torch tensor raises ``TypeError: Cannot interpret 'torch.float32' as a data
    type``. A CUDA tensor converts to a cupy array through DLPack with no copy,
    which keeps the data resident on the device it is already on; a CPU tensor
    becomes numpy and cuML uploads it itself.

    Args:
        X (torch.Tensor): Samples to cluster.

    Returns:
        cupy.ndarray | numpy.ndarray: An array cuML's estimators accept.
    """
    if X.device.type != "cuda":
        return X.detach().cpu().numpy()

    import cupy  # noqa: PLC0415

    return cupy.from_dlpack(X.detach())


#: Overrides the Lloyd iteration cap for vector palettization.
KMEANS_MAX_ITER_ENV = "COREAI_KMEANS_MAX_ITER"

#: Lloyd iteration cap for vector palettization, matching what the callers hardcoded.
_VECTOR_MAX_ITER = 300


def _resolve_max_iter(max_iter: int) -> int:
    """Return ``max_iter``, or the environment override if one is set.

    Args:
        max_iter (int): The value the caller passed.

    Returns:
        int: Lloyd iteration cap.

    Raises:
        ValueError: If the environment variable is not a positive integer.
    """
    override = _os.environ.get(KMEANS_MAX_ITER_ENV, "").strip()
    if not override:
        return max_iter
    try:
        value = int(override)
    except ValueError as err:
        raise ValueError(f"{KMEANS_MAX_ITER_ENV}={override!r} is not an integer") from err
    if value < 1:
        raise ValueError(f"{KMEANS_MAX_ITER_ENV}={override!r} must be >= 1")
    return value


def vector_max_iter() -> int:
    """Resolved Lloyd iteration cap for vector palettization."""
    return _resolve_max_iter(_VECTOR_MAX_ITER)


def vector_n_init() -> int:
    """Resolved number of k-means restarts for vector palettization.

    :data:`_VECTOR_N_INIT` unless ``COREAI_KMEANS_N_INIT`` overrides it. Independent of
    the backend by design, so the two can be varied one at a time.

    Returns:
        int: Restart count, >= 1.
    """
    return _resolve_n_init(_VECTOR_N_INIT)


def batched_kmeanspp_init(
    x: _torch.Tensor,
    n_clusters: int,
    n_local_trials: int | None = None,
    generator: "_torch.Generator | None" = None,
) -> _torch.Tensor:
    """Greedy k-means++ seeding for a whole batch of independent problems at once.

    Each new seed is sampled with probability proportional to its squared distance to
    the nearest already-chosen seed. That is ``n_clusters`` strictly sequential steps --
    step i depends on step i-1 -- but every step is vectorised across the batch, so a
    tensor's worth of blocks costs ``n_clusters`` kernel launches instead of
    ``n_clusters * n_blocks``. 

    **Vanilla (one candidate per step) by default, and a full-model run says that is
    correct.** Greedy seeding draws
    ``n_local_trials`` candidates per step and keeps whichever most reduces the
    remaining potential. Read from cuVS source (``cpp/src/cluster/detail/kmeans.cuh``):
    ``init="k-means++"`` forces ``oversampling_factor = 0``, which dispatches to
    ``kmeansPlusPlus``, and that sets ``n_trials = 2 + ceil(log(n_clusters))`` -- **8
    candidates at k=256** -- then picks the ArgMin of each candidate's full cluster
    cost. So cuML evaluates 8 candidates at every one of the k-1 steps where this
    evaluates 1.

    Preferred over the maximin (farthest-point) scheme that :meth:`_kmeans_pp` uses. 

    Args:
        x (torch.Tensor): ``(B, N, D)`` -- B independent problems of N points.
        n_clusters (int): Seeds per problem. Must be ``<= N``.
        n_local_trials (int | None): Candidates per step. ``None`` (default) resolves
            to ``1`` (vanilla), unless ``COREAI_KMEANS_N_LOCAL_TRIALS`` is set, which
            overrides the call site like the other env knobs. Set it to ``auto`` for
            cuVS's ``2 + ceil(log(n_clusters))``.
        generator (torch.Generator | None): RNG for the sampling. **Pass one.** 

    Returns:
        torch.Tensor: ``(B, n_clusters, D)`` seeds, each a point drawn from its own
        problem.

    Raises:
        ValueError: If ``n_clusters`` exceeds the number of points per problem.
    """
    b, n, d = x.shape
    if n_clusters > n:
        raise ValueError(f"n_clusters {n_clusters} exceeds {n} points per problem")
    n_local_trials = resolve_n_local_trials(n_clusters, n_local_trials)

    rows = _torch.arange(b, device=x.device)
    x_sq = (x**2).sum(-1)
    seeds = _torch.empty(b, n_clusters, d, device=x.device, dtype=x.dtype)
    seeds[:, 0] = x[rows, _torch.randint(0, n, (b,), device=x.device, generator=generator)]
    # Running squared distance from every point to its nearest chosen seed.
    closest = ((x - seeds[:, 0:1]) ** 2).sum(-1)

    for i in range(1, n_clusters):
        weights = closest.clamp_min(0)
        # A row sums to zero only when every point coincides with a chosen seed (e.g.
        # fewer distinct values than clusters, which happens on low-entropy weights).
        # multinomial rejects that, so fall back to uniform for those rows only.
        degenerate = weights.sum(dim=1) <= 0
        if bool(degenerate.any()):
            weights = weights.clone()
            weights[degenerate] = 1.0

        cand = _torch.multinomial(weights, n_local_trials, generator=generator)
        cand_pts = _torch.gather(x, 1, cand.unsqueeze(-1).expand(-1, -1, d))
        # (B, N, trials) via the norm trick, so the (B, N, trials, D) difference is
        # never materialised.
        dist = (
            x_sq.unsqueeze(-1)
            - 2.0 * _torch.bmm(x, cand_pts.transpose(1, 2))
            + (cand_pts**2).sum(-1).unsqueeze(1)
        ).clamp_min_(0)
        # Potential that would remain after taking each candidate; keep the lowest.
        best = _torch.minimum(closest.unsqueeze(-1), dist).sum(dim=1).argmin(dim=1)
        seeds[:, i] = cand_pts[rows, best]
        closest = _torch.minimum(closest, dist[rows, :, best])

    return seeds


def batched_cluster_avg(x: _torch.Tensor, labels: _torch.Tensor, n_clusters: int) -> _torch.Tensor:
    """Exact per-cluster means for a batch, matching :meth:`_EfficientKMeans._get_cluster_avg`.

    Every backend's centroids are replaced by exact means before use, so the batched
    path must do the same or its LUTs would differ from the per-block path. Empty
    clusters yield a zero centroid, which is what the per-block version produces
    (zero sum divided by a count clamped to 1).

    Args:
        x (torch.Tensor): ``(B, N, D)``.
        labels (torch.Tensor): ``(B, N)`` cluster index per point.
        n_clusters (int): Number of clusters.

    Returns:
        torch.Tensor: ``(B, n_clusters, D)`` in ``x``'s dtype.
    """
    b, _, d = x.shape
    idx = labels.long()
    sums = _torch.zeros(b, n_clusters, d, device=x.device, dtype=_torch.float32)
    sums.scatter_add_(1, idx.unsqueeze(-1).expand(-1, -1, d), x.float())
    counts = _torch.zeros(b, n_clusters, device=x.device, dtype=_torch.float32)
    counts.scatter_add_(1, idx, _torch.ones_like(idx, dtype=_torch.float32))
    return (sums / counts.clamp_min(1.0).unsqueeze(-1)).to(x.dtype)


def generate_init_seeds(
    x: _torch.Tensor,
    n_clusters: int,
    n_init: int,
    method: str = "kmeans++",
    n_local_trials: int | None = None,
) -> _torch.Tensor:
    """STAGE 1 -- produce ``n_init`` independent sets of initial centroids.

    Separated from the Lloyd loop so the two can be timed and swapped independently.
    The two stages behave very differently: this one is ``n_clusters`` strictly
    sequential steps (launch-bound, ~50x faster batched across blocks than per block),
    while :func:`run_lloyd_batched` is a handful of fully parallel iterations.

    Each seed set gets its own deterministic generator, derived from a fingerprint of
    ``x`` so that the result is reproducible *and* decorrelated across tensors -- see
    :func:`batched_kmeanspp_init` for why both properties are needed.

    Args:
        x (torch.Tensor): ``(B, N, D)`` vectorised blocks.
        n_clusters (int): Centroids per block.
        n_init (int): Number of independent seed sets to produce.
        method (str): ``"kmeans++"`` (D^2-weighted sampling), ``"maximin"``
            (deterministic farthest-point, what ``_kmeans_pp`` uses) or ``"random"``.
        n_local_trials (int | None): Candidates per step for ``"kmeans++"``; see
            :func:`batched_kmeanspp_init`.

    Returns:
        torch.Tensor: ``(n_init, B, n_clusters, D)``.

    Raises:
        ValueError: If ``method`` is not recognised.
    """
    if resolve_init_impl() == "cuml":
        return _cuml_init_seeds(x, n_clusters, max(1, n_init))

    t0 = _time.perf_counter()
    probe = x.flatten()[:: max(1, x.numel() // 64)][:64].double().tolist()
    fingerprint = hash((tuple(x.shape), tuple(round(v, 6) for v in probe)))

    seed_sets = []
    for restart in range(max(1, n_init)):
        gen = _torch.Generator(device=x.device)
        gen.manual_seed((_RANDOM_STATE + restart * 7919 + fingerprint) % (2**63 - 1))
        if method == "kmeans++":
            seed_sets.append(batched_kmeanspp_init(x, n_clusters, n_local_trials, gen))
        elif method == "maximin":
            seed_sets.append(_batched_maximin_init(x, n_clusters, gen))
        elif method == "random":
            b, n, d = x.shape
            idx = _torch.randint(0, n, (b, n_clusters), device=x.device, generator=gen)
            seed_sets.append(_torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, d)))
        else:
            raise ValueError(f"unknown init method {method!r}")

    if x.is_cuda:
        _torch.cuda.synchronize()
    _TIMINGS.init_seconds += _time.perf_counter() - t0
    _TIMINGS.init_calls += 1
    return _torch.stack(seed_sets)


def _batched_maximin_init(
    x: _torch.Tensor, n_clusters: int, generator: "_torch.Generator | None" = None
) -> _torch.Tensor:
    """Deterministic farthest-point seeding, batched -- the scheme ``_kmeans_pp`` uses.

    Included so the init axis can be varied independently of the backend. 
    """
    b, n, d = x.shape
    rows = _torch.arange(b, device=x.device)
    seeds = _torch.empty(b, n_clusters, d, device=x.device, dtype=x.dtype)
    seeds[:, 0] = x[rows, _torch.randint(0, n, (b,), device=x.device, generator=generator)]
    closest = ((x - seeds[:, 0:1]) ** 2).sum(-1)
    for i in range(1, n_clusters):
        seeds[:, i] = x[rows, closest.argmax(dim=1)]
        closest = _torch.minimum(closest, ((x - seeds[:, i : i + 1]) ** 2).sum(-1))
    return seeds


def run_lloyd_batched(
    x: _torch.Tensor,
    seed_sets: _torch.Tensor,
    max_iter: int = 300,
    tol: float = 1e-4,
) -> tuple[_torch.Tensor, _torch.Tensor]:
    """STAGE 2 -- refine each seed set with Lloyd's algorithm, keep the best per block.

    Runs the seed sets **sequentially**. Stacking them into the batch dimension is
    tempting but costs more: flash-kmeans stops on the *max* centroid shift across the
    whole batch, so stacked restarts all iterate until the slowest converges. 

    The winner is chosen **per block**, not by one global score -- a restart that helps
    one block may hurt another.

    Args:
        x (torch.Tensor): ``(B, N, D)`` vectorised blocks.
        seed_sets (torch.Tensor): ``(n_init, B, n_clusters, D)`` from
            :func:`generate_init_seeds`.
        max_iter (int): Lloyd iteration cap.
        tol (float): Centroid-shift tolerance, compared against the batch max.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``labels (B, N)`` int64 and
        ``centroids (B, n_clusters, D)`` as exact cluster means.

    Raises:
        ModuleNotFoundError: If flash-kmeans is not installed.
    """
    impl = resolve_lloyd_impl()
    if impl == "torch":
        return _run_lloyd_torch(x, seed_sets, max_iter, tol)
    if impl == "cuml":
        return _run_lloyd_cuml(x, seed_sets, max_iter, tol)

    try:
        from flash_kmeans import batch_kmeans_Euclid  # noqa: PLC0415
    except Exception as err:
        raise ModuleNotFoundError(
            "flash-kmeans is required for batched vector clustering."
            ' To install, run: "pip install flash-kmeans".'
            f" Underlying error: {err}"
        ) from err

    n_clusters = seed_sets.shape[2]
    t0 = _time.perf_counter()
    best_labels = best_inertia = None
    for seeds in seed_sets:
        labels, _, _ = batch_kmeans_Euclid(
            x, n_clusters=n_clusters, max_iters=max_iter, tol=tol, init_centroids=seeds
        )
        labels = labels.long()
        centroids = batched_cluster_avg(x, labels, n_clusters)
        picked = _torch.gather(centroids, 1, labels.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        inertia = ((x - picked) ** 2).sum(dim=(1, 2))
        if best_labels is None:
            best_labels, best_inertia = labels, inertia
        else:
            improved = inertia < best_inertia
            best_labels = _torch.where(improved.unsqueeze(-1), labels, best_labels)
            best_inertia = _torch.minimum(best_inertia, inertia)

    if x.is_cuda:
        _torch.cuda.synchronize()
    _TIMINGS.lloyd_seconds += _time.perf_counter() - t0
    _TIMINGS.lloyd_calls += 1
    return best_labels, batched_cluster_avg(x, best_labels, n_clusters)


def _pick_best_per_block(
    x: _torch.Tensor,
    labels: _torch.Tensor,
    n_clusters: int,
    best_labels: "_torch.Tensor | None",
    best_inertia: "_torch.Tensor | None",
) -> tuple[_torch.Tensor, _torch.Tensor]:
    """Keep whichever restart wins for each block, scored on exact-mean inertia.

    Shared by all three stage-2 implementations so the restart-selection rule cannot
    drift between them -- a crossover experiment is only readable if the only thing that
    differs is the thing under test.
    """
    centroids = batched_cluster_avg(x, labels, n_clusters)
    picked = _torch.gather(centroids, 1, labels.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    inertia = ((x - picked) ** 2).sum(dim=(1, 2))
    if best_labels is None:
        return labels, inertia
    improved = inertia < best_inertia
    return (
        _torch.where(improved.unsqueeze(-1), labels, best_labels),
        _torch.minimum(inertia, best_inertia),
    )


def _run_lloyd_torch(
    x: _torch.Tensor, seed_sets: _torch.Tensor, max_iter: int, tol: float
) -> tuple[_torch.Tensor, _torch.Tensor]:
    """STAGE 2 in exact fp32 -- the control for flash's TF32 distance computation.

    flash's assignment goes through ``tl.dot``, which defaults to TF32 (10 mantissa
    bits) and feeds a norm-trick distance. On real Qwen3-4B blocks that mis-assigns
    **0.61%** of points against float64 ground truth and does not improve with
    iterations, while the identical formula in true fp32 mis-assigns **0.000%**. This
    runs the same algorithm with TF32 explicitly off.

    Also matches cuVS on two rules flash does not: empty clusters keep their previous
    centroid (``finalize_centroids`` in ``kmeans_common.cuh``), and convergence is
    tested **per block** rather than once for the whole batch.

    Distances use the norm trick, chunked over N so the ``(B, N, K)`` matrix is never
    materialised in full -- at B=128, N=40,960, K=256 that would be 5.4 GB.
    """
    b, n, d = x.shape
    n_clusters = seed_sets.shape[2]
    # Keep the working set near 0.5 GB regardless of block count.
    chunk = max(1, min(n, int(2**27 // max(1, b * n_clusters))))
    prev_tf32 = _torch.backends.cuda.matmul.allow_tf32
    _torch.backends.cuda.matmul.allow_tf32 = False
    t0 = _time.perf_counter()
    try:
        best_labels = best_inertia = None
        for seeds in seed_sets:
            centroids = seeds.clone().float()
            xf = x.float()
            labels = _torch.empty(b, n, dtype=_torch.long, device=x.device)
            for _ in range(max_iter):
                c_sq = (centroids**2).sum(-1)
                for i in range(0, n, chunk):
                    xc = xf[:, i : i + chunk]
                    dist = (
                        (xc**2).sum(-1).unsqueeze(-1)
                        + c_sq.unsqueeze(1)
                        - 2.0 * _torch.bmm(xc, centroids.transpose(1, 2))
                    )
                    labels[:, i : i + chunk] = dist.argmin(-1)
                counts = _torch.zeros(b, n_clusters, device=x.device)
                counts.scatter_add_(1, labels, _torch.ones_like(labels, dtype=counts.dtype))
                sums = _torch.zeros(b, n_clusters, d, device=x.device)
                sums.scatter_add_(1, labels.unsqueeze(-1).expand(-1, -1, d), xf)
                # cuVS rule: an empty cluster keeps its previous centroid, never zero.
                new_centroids = _torch.where(
                    counts.unsqueeze(-1) > 0,
                    sums / counts.clamp_min(1).unsqueeze(-1),
                    centroids,
                )
                shift = ((new_centroids - centroids) ** 2).sum(-1).sum(-1)
                centroids = new_centroids
                if bool((shift < tol).all()):
                    break
            best_labels, best_inertia = _pick_best_per_block(
                x, labels, n_clusters, best_labels, best_inertia
            )
    finally:
        _torch.backends.cuda.matmul.allow_tf32 = prev_tf32
    if x.is_cuda:
        _torch.cuda.synchronize()
    _TIMINGS.lloyd_seconds += _time.perf_counter() - t0
    _TIMINGS.lloyd_calls += 1
    return best_labels, batched_cluster_avg(x, best_labels, n_clusters)


def _run_lloyd_cuml(
    x: _torch.Tensor, seed_sets: _torch.Tensor, max_iter: int, tol: float
) -> tuple[_torch.Tensor, _torch.Tensor]:
    """STAGE 2 via cuML, seeded with centroids produced by stage 1.

    The crossover arm: our seeds, cuML's refinement. cuML has no batched API, so this
    loops blocks; ``init=<ndarray>`` makes it skip its own seeding entirely, and
    ``n_init=1`` is required because an explicit init leaves nothing to restart.
    """
    try:
        _load_rapids_native_libs()
        from cuml.cluster import KMeans  # noqa: PLC0415
    except Exception as err:
        raise ModuleNotFoundError(
            "cuml is required for the cuml Lloyd stage."
            ' To install, run: "pip install cuml-cu12".'
            f" Underlying error: {err}"
        ) from err

    b = x.shape[0]
    n_clusters = seed_sets.shape[2]
    t0 = _time.perf_counter()
    best_labels = best_inertia = None
    for seeds in seed_sets:
        labels = _torch.empty(b, x.shape[1], dtype=_torch.long, device=x.device)
        for blk in range(b):
            fitted = KMeans(
                n_clusters=n_clusters,
                init=_to_cuml_input(seeds[blk]),
                n_init=1,
                max_iter=max_iter,
                tol=tol,
                random_state=_RANDOM_STATE,
                output_type="numpy",
            ).fit(_to_cuml_input(x[blk]))
            labels[blk] = _torch.from_numpy(fitted.labels_).long().to(x.device)
        best_labels, best_inertia = _pick_best_per_block(
            x, labels, n_clusters, best_labels, best_inertia
        )
    if x.is_cuda:
        _torch.cuda.synchronize()
    _TIMINGS.lloyd_seconds += _time.perf_counter() - t0
    _TIMINGS.lloyd_calls += 1
    return best_labels, batched_cluster_avg(x, best_labels, n_clusters)


def _cuml_init_seeds(x: _torch.Tensor, n_clusters: int, n_init: int) -> _torch.Tensor:
    """STAGE 1 via cuML's own greedy k-means++, for the crossover experiment.

    cuML exposes no standalone seeding entry point, so each seed set comes from a fit
    capped at ``_CUML_INIT_MAX_ITER`` (it rejects ``max_iter=0``), which costs one Lloyd
    iteration on top of the seeding. ``oversampling_factor=0`` is what ``init="k-means++"``
    sets, and that is the flag that dispatches cuVS to the greedy ``kmeansPlusPlus`` with
    ``n_trials = 2 + ceil(log(k))`` rather than to ``initScalableKMeansPlusPlus``.

    Returns:
        torch.Tensor: ``(n_init, B, n_clusters, D)``, matching
        :func:`generate_init_seeds`.
    """
    try:
        _load_rapids_native_libs()
        from cuml.cluster import KMeans  # noqa: PLC0415
    except Exception as err:
        raise ModuleNotFoundError(
            "cuml is required for the cuml init stage."
            ' To install, run: "pip install cuml-cu12".'
            f" Underlying error: {err}"
        ) from err

    b, _, d = x.shape
    t0 = _time.perf_counter()
    seed_sets = _torch.empty(n_init, b, n_clusters, d, device=x.device, dtype=x.dtype)
    for restart in range(n_init):
        for blk in range(b):
            fitted = KMeans(
                n_clusters=n_clusters,
                init="k-means++",
                n_init=1,
                max_iter=_CUML_INIT_MAX_ITER,
                tol=1e-4,
                random_state=_RANDOM_STATE + restart,
                output_type="numpy",
            ).fit(_to_cuml_input(x[blk]))
            seed_sets[restart, blk] = (
                _torch.from_numpy(fitted.cluster_centers_).to(x.device).to(x.dtype)
            )
    if x.is_cuda:
        _torch.cuda.synchronize()
    _TIMINGS.init_seconds += _time.perf_counter() - t0
    _TIMINGS.init_calls += 1
    return seed_sets


def flash_batch_cluster(
    x: _torch.Tensor,
    n_clusters: int,
    max_iter: int = 300,
    tol: float = 1e-4,
    n_init: int = 1,
) -> tuple[_torch.Tensor, _torch.Tensor]:
    """Cluster a whole batch of blocks in one flash-kmeans call, k-means++ seeded.

    This is the batched counterpart of looping :meth:`_EfficientKMeans.fit` over
    blocks, and the reason the flash backend is worth having

    Restarts run **sequentially**, not stacked into the batch dimension, and the best
    is kept **per block**. Stacking looks tempting but flash-kmeans stops on the *max*
    centroid shift across the whole batch, so stacked restarts all keep iterating until
    the slowest converges. Sequential
    restarts cost exactly n_init passes and let each converge on its own schedule.

    Args:
        x (torch.Tensor): ``(B, N, D)`` vectorised blocks, same shape per block.
        n_clusters (int): Clusters per block.
        max_iter (int): Lloyd iteration cap.
        tol (float): Centroid-shift tolerance, compared against the batch max.
        n_init (int): Independent k-means++ seedings to try, keeping the best per
            block. Cost is linear in this.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``labels (B, N)`` as int64 and
        ``centroids (B, n_clusters, D)`` as exact cluster means.

    Raises:
        ModuleNotFoundError: If flash-kmeans is not installed.
    """
    seed_sets = generate_init_seeds(x, n_clusters, max(1, n_init))
    return run_lloyd_batched(x, seed_sets, max_iter=max_iter, tol=tol)


class _EfficientKMeans:
    """
    k-means clustering with a choice of backend.

    ``init`` (or the ``COREAI_KMEANS_BACKEND`` environment variable) selects:

    - ``"kmeans++"``          this module's own torch implementation, which runs
                              on whichever device the input is on.
    - ``"sklearn"``           ``sklearn.cluster.KMeans`` -- full-batch Lloyd's.
    - ``"sklearn_minibatch"`` ``sklearn.cluster.MiniBatchKMeans`` -- updates
                              centroids from random subsets, so it is far
                              cheaper per pass at some cost in solution quality.
    - ``"cuml"``              ``cuml.cluster.KMeans`` -- full-batch Lloyd's on the
                              GPU. Matches sklearn's solution quality at a
                              fraction of the cost; requires ``cuml-cu12``.
    - ``"flash_kmeans"``      ``flash_kmeans.batch_kmeans_Euclid`` -- batched Triton
                              k-means, with a torch fallback off CUDA. Its own random
                              init is NOT used: :func:`flash_batch_cluster` supplies
                              vanilla k-means++ seeds via ``init_centroids`` and runs
                              restarts sequentially. No ``sample_weight`` support.

    The sklearn backends are CPU-only; input is moved to CPU and the resulting
    labels moved back. The cuml backend is GPU-only and keeps CUDA input resident.
    Whichever backend runs, the final centroids are recomputed by
    :meth:`_get_cluster_avg` in the input dtype, so they are comparable across
    backends (and, for MiniBatchKMeans, exact cluster means rather than the
    running averages it maintains internally).
    """

    def __init__(
        self,
        n_clusters: int,
        init: str,
        n_init: int = 0,
        max_iter: int = 100,
        tol: float = 0.0001,
    ):
        self.n_clusters = n_clusters
        self.n_init = _resolve_n_init(n_init)
        self.max_iter = max_iter
        self.tol = tol
        self.labels_ = None
        self.inertia_ = None
        self.cluster_centers_ = init

        assert self.max_iter > 0
        assert self.n_clusters > 0

    @staticmethod
    def _get_cluster_avg(
        n_clusters: int,
        indices: _torch.Tensor,
        vals: _torch.Tensor,
        sample_weight: _torch.Tensor | None = None,
    ) -> _torch.Tensor:
        agg_vals = (
            vals.float() * sample_weight.float() if sample_weight is not None else vals.float()
        )
        v_sum = (
            _torch.zeros([n_clusters] + list(vals[0].size()))
            .to(vals.device)
            .index_add_(0, indices, agg_vals)
        )
        weight = (
            _torch.ones(len(vals), dtype=_torch.int).to(vals.device)
            if sample_weight is None
            else sample_weight.squeeze(1).to(vals.device)
        )
        v_numel = (
            _torch.zeros(n_clusters, dtype=weight.dtype)
            .to(vals.device)
            .index_add_(0, indices, weight)
        )
        v_numel[v_numel == 0] = 1

        v_avg = v_sum / v_numel.reshape(-1, 1)

        return v_avg.to(vals.dtype)

    def _kmeans_pp(
        self, parameters: _torch.Tensor, sample_weight: _torch.Tensor | None = None
    ) -> None:
        assert len(parameters) >= self.n_clusters

        num_update_list = []
        INIT_EXIT = 10
        self.inertia_ = int(1e9)

        # n_init trials for estimating cluster centers
        for n in range(self.n_init):
            _t_init = _time.perf_counter()
            if n % 2 and sample_weight is not None:
                centroids = parameters[
                    _np.random.choice(
                        len(parameters),
                        self.n_clusters,
                        False,
                        (sample_weight.squeeze() / sample_weight.sum()).cpu().numpy(),
                    )
                ]
            else:
                centroids = _torch.zeros(
                    (self.n_clusters, parameters.size(-1)),
                    device=parameters.device,
                    dtype=parameters.dtype,
                )
                for i in range(self.n_clusters):
                    if i == 0:
                        centroids[i] = parameters[_torch.randint(0, len(parameters), [1])]
                        d_ij_curr = _torch.cdist(centroids[:i], parameters)
                    else:
                        d_ij_prev = _torch.cdist(centroids[i - 1 : i], parameters)
                        d_ij_prev[d_ij_prev == 0] = -int(1e9)

                        d_ij_curr = _torch.cat((d_ij_curr, d_ij_prev), dim=0)

                        c_to_x = _torch.min(d_ij_curr, dim=0)
                        centroids[i] = parameters[c_to_x[0].argmax()]

            if parameters.is_cuda:
                _torch.cuda.synchronize()
            _TIMINGS.init_seconds += _time.perf_counter() - _t_init
            _TIMINGS.init_calls += 1

            _t_lloyd = _time.perf_counter()
            last_inertia = int(1e9)
            num_update = 0
            for _ in range(self.max_iter):
                min_error, labels = _torch.cdist(parameters, centroids).min(dim=-1)

                min_error = (
                    min_error * (sample_weight.T).sqrt() if sample_weight is not None else min_error
                )

                centroids.zero_()
                agg_params = parameters * sample_weight if sample_weight is not None else parameters
                weights = sample_weight.view(labels.size()) if sample_weight is not None else None
                centroids.scatter_add_(
                    0,
                    labels.view(-1, 1).expand([-1, parameters.size(-1)]),
                    agg_params,
                )
                n_centroids = _torch.bincount(
                    labels, weights=weights, minlength=self.n_clusters
                ).view(-1, 1)

                centroids /= n_centroids
                cur_inertia = min_error.square().sum()

                # update labels and cluster_centers if inertia improves
                if cur_inertia < self.inertia_:
                    num_update += 1
                    self.inertia_ = cur_inertia
                    self.labels_ = labels
                    self.cluster_centers_ = centroids

                # exit if there is no improvement in inertia within a tolerance
                elif last_inertia <= cur_inertia * (1 + self.tol):
                    break

                last_inertia = cur_inertia

            if parameters.is_cuda:
                _torch.cuda.synchronize()
            _TIMINGS.lloyd_seconds += _time.perf_counter() - _t_lloyd
            _TIMINGS.lloyd_calls += 1

            num_update_list.append(num_update)

            # In every trial, we track number of cluster centre updates.
            # If number of trials are greater than a specified value INIT_EXIT and
            # there is no update for the past INIT_EXIT number of trials,
            # it indicates that the centroids have converged
            if len(num_update_list) >= INIT_EXIT and sum(num_update_list[-INIT_EXIT:]) == 0:
                break

    def _fit_sklearn(
        self,
        X: _torch.Tensor,
        sample_weight: _torch.Tensor | None,
        minibatch: bool,
    ) -> None:
        """Cluster with scikit-learn, setting ``labels_`` and ``inertia_``.

        Args:
            X: ``(n_samples, n_features)``. Moved to CPU; sklearn has no GPU path.
            sample_weight: ``(n_samples, 1)`` as the torch backend expects it;
                sklearn wants it flat, so it is squeezed here.
            minibatch: Use ``MiniBatchKMeans`` instead of ``KMeans``.
        """
        try:
            from sklearn.cluster import KMeans, MiniBatchKMeans  # noqa: PLC0415
        except Exception as err:
            raise ModuleNotFoundError(
                "scikit-learn is required for the sklearn k-means backends."
                ' To install, run: "pip install scikit-learn".'
            ) from err

        # sklearn rejects n_init=0, which is this class's default.
        n_init = max(1, self.n_init)
        cls = MiniBatchKMeans if minibatch else KMeans
        estimator = cls(
            n_clusters=self.n_clusters,
            init="k-means++",
            n_init=n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=_RANDOM_STATE,
        )

        weights = None
        if sample_weight is not None:
            weights = sample_weight.reshape(-1).cpu().numpy()
        fitted = estimator.fit(X.detach().cpu().numpy(), sample_weight=weights)

        self.labels_ = _torch.from_numpy(fitted.labels_).long().to(X.device)
        self.inertia_ = _torch.tensor(fitted.inertia_, device=X.device)

    def _fit_cuml(self, X: _torch.Tensor, sample_weight: _torch.Tensor | None) -> None:
        """Cluster with cuML's GPU ``KMeans``, setting ``labels_`` and ``inertia_``.

        This is full-batch Lloyd's on the GPU: measured on an A100 it matches
        ``sklearn.cluster.KMeans`` inertia to within ~0.5% while running ~20x
        faster, and beats this module's own torch ``kmeans++`` by ~4x per tensor.
        cuML has no ``MiniBatchKMeans``, so there is no GPU counterpart to the
        ``sklearn_minibatch`` backend.

        Note:
            The first call in a process pays ~4.7s of one-time CUDA context, RMM
            pool and RAFT kernel-loading cost; subsequent calls do not. With
            ``num_workers > 1`` that is paid once per worker process.

        Args:
            X (torch.Tensor): ``(n_samples, n_features)``. Stays on the GPU if it
                is already there.
            sample_weight (torch.Tensor | None): ``(n_samples, 1)`` as the torch
                backend expects it; cuML wants it flat, so it is reshaped here.

        Raises:
            ModuleNotFoundError: If cuml is not installed.
        """
        try:
            _load_rapids_native_libs()
            from cuml.cluster import KMeans  # noqa: PLC0415
        except Exception as err:
            raise ModuleNotFoundError(
                "cuml is required for the cuml k-means backend."
                ' To install, run: "pip install cuml-cu12".'
                f" Underlying error: {err}"
            ) from err

        # cuML rejects n_init=0, which is this class's default.
        estimator = KMeans(
            n_clusters=self.n_clusters,
            init="k-means++",
            n_init=max(1, self.n_init),
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=_RANDOM_STATE,
            output_type="numpy",
        )

        weights = None
        if sample_weight is not None:
            weights = _to_cuml_input(sample_weight.reshape(-1))

        if cuml_split_requested():
            self._fit_cuml_two_stage(KMeans, X, weights)
            return

        # cuml's fit() performs init AND Lloyd in one call, so by default the whole
        # thing lands in stage 2 and the separability flag is cleared -- better an
        # honestly-labelled fused number than a fabricated split. Set
        # ``COREAI_KMEANS_CUML_SPLIT=1`` to measure the two stages apart.
        _t_fit = _time.perf_counter()
        fitted = estimator.fit(_to_cuml_input(X), sample_weight=weights)
        if X.is_cuda:
            _torch.cuda.synchronize()
        _TIMINGS.lloyd_seconds += _time.perf_counter() - _t_fit
        _TIMINGS.lloyd_calls += 1
        _TIMINGS.init_separable = False

        self.labels_ = _torch.from_numpy(fitted.labels_).long().to(X.device)
        self.inertia_ = _torch.tensor(fitted.inertia_, device=X.device)

    def _fit_cuml_two_stage(self, kmeans_cls: type, X: _torch.Tensor, weights: object) -> None:
        """Fit cuml as two separately-timed calls: seeding, then refinement.

        cuML has no standalone seeding entry point, but it does not need one: ``init``
        accepts an explicit centroid array (in which case cuML skips seeding entirely)
        and ``max_iter`` bounds the refinement. So one fit with ``max_iter=1`` yields
        cuML's own k-means++ seeds, and a second fit seeded with those centroids does
        only Lloyd's.

        cuML's internal ``n_init`` loop is the one part that cannot be split, so the
        restarts are hoisted into Python here: ``n_init`` independent seedings, each
        refined to convergence, lowest inertia wins. Same algorithm, different code
        path -- so the total will not exactly match a single ``fit(n_init=k)``.

        Args:
            kmeans_cls (type): ``cuml.cluster.KMeans``, passed in so the import stays
                in the caller.
            X (torch.Tensor): ``(n_samples, n_features)``.
            weights (object): cuML-ready sample weights, or None.
        """
        data = _to_cuml_input(X)
        best = None

        for restart in range(max(1, self.n_init)):
            # Vary the seed per restart, or every restart would repeat one seeding.
            seed = _RANDOM_STATE + restart

            t_init = _time.perf_counter()
            seeded = kmeans_cls(
                n_clusters=self.n_clusters,
                init="k-means++",
                n_init=1,
                max_iter=_CUML_INIT_MAX_ITER,
                tol=self.tol,
                random_state=seed,
                output_type="numpy",
            ).fit(data, sample_weight=weights)
            if X.is_cuda:
                _torch.cuda.synchronize()
            _TIMINGS.init_seconds += _time.perf_counter() - t_init
            _TIMINGS.init_calls += 1

            t_lloyd = _time.perf_counter()
            refined = kmeans_cls(
                n_clusters=self.n_clusters,
                init=seeded.cluster_centers_,
                n_init=1,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=seed,
                output_type="numpy",
            ).fit(data, sample_weight=weights)
            if X.is_cuda:
                _torch.cuda.synchronize()
            _TIMINGS.lloyd_seconds += _time.perf_counter() - t_lloyd
            _TIMINGS.lloyd_calls += 1

            if best is None or refined.inertia_ < best.inertia_:
                best = refined

        # n_init >= 1 so the loop above always ran at least once.
        assert best is not None
        self.labels_ = _torch.from_numpy(best.labels_).long().to(X.device)
        self.inertia_ = _torch.tensor(best.inertia_, device=X.device)

    def _fit_flash(self, X: _torch.Tensor, sample_weight: _torch.Tensor | None) -> None:
        """Cluster with flash-kmeans, setting ``labels_`` and ``inertia_``.

        flash-kmeans is a batched Triton k-means: its native signature takes
        ``(B, N, D)`` and solves B independent problems in one launch. This method is
        the single-block adapter, so it does **not** exploit that -- see
        :func:`flash_batch_kmeans` for the batched entry point.

        It is a thin wrapper over :func:`flash_batch_cluster` with ``B = 1``, which owns
        the k-means++ seeding, the deterministic per-tensor RNG seed and the sequential
        restarts. Reached only when the batched path declines -- per-tensor granularity
        (a single block), ragged block shapes, or a weighted run -- since otherwise
        ``_cluster_to_centroids`` clusters every block of a tensor in one call.

        Args:
            X (torch.Tensor): ``(n_samples, n_features)``. Stays on its current
                device; the Triton kernel needs CUDA, the torch fallback does not.
            sample_weight (torch.Tensor | None): Must be None.

        Raises:
            ValueError: If ``sample_weight`` is given. flash-kmeans has no weighted
                path, and silently ignoring the weights would quietly produce
                unweighted centroids for a sensitivity-weighted run.
            ModuleNotFoundError: If flash-kmeans is not installed.
        """
        if sample_weight is not None:
            raise ValueError(
                "the flash_kmeans backend does not support sample_weight; use the"
                f" {_TORCH_BACKEND!r} or {_CUML_BACKEND!r} backend for"
                " sensitivity-weighted clustering."
            )

        # Delegate to the batched implementation with a batch of one, so restarts and
        # seeding exist in exactly one place. The earlier standalone version here
        # stacked its restarts into the batch dimension, which is measurably the wrong
        # strategy: flash-kmeans stops on the *max* centroid shift across the batch, so
        # stacked restarts all iterate until the slowest converges. It also drew from the
        # ambient RNG, which made whole runs irreproducible.
        labels, centroids = flash_batch_cluster(
            X.unsqueeze(0),
            self.n_clusters,
            max_iter=self.max_iter,
            tol=self.tol,
            n_init=max(1, self.n_init),
        )

        self.labels_ = labels[0].to(X.device)
        picked = _torch.gather(centroids, 1, labels.unsqueeze(-1).expand(-1, -1, X.shape[1]))
        self.inertia_ = ((X.unsqueeze(0) - picked) ** 2).sum()

    def fit(
        self, X: _torch.Tensor, sample_weight: _torch.Tensor | None = None
    ) -> "_EfficientKMeans":
        """
        Compute k-means clustering.
        """
        N = len(X)

        assert N >= self.n_clusters, f"too many clusters {self.n_clusters} for {N} samples"

        backend = _resolve_backend(self.cluster_centers_)
        _announce(backend, X.device)

        if backend == _TORCH_BACKEND:
            self._kmeans_pp(X.float(), sample_weight=sample_weight)
        elif backend == _CUML_BACKEND:
            self._fit_cuml(X.float(), sample_weight=sample_weight)
        elif backend == _FLASH_BACKEND:
            self._fit_flash(X.float(), sample_weight=sample_weight)
        else:
            self._fit_sklearn(
                X.float(),
                sample_weight=sample_weight,
                minibatch=backend == _SKLEARN_MINIBATCH_BACKEND,
            )

        self.cluster_centers_ = _EfficientKMeans._get_cluster_avg(
            self.n_clusters, self.labels_, X, sample_weight=sample_weight
        )

        return self
