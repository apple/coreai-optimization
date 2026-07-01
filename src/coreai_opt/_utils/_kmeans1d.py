# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Pure-Python + Numba implementation of globally-optimal 1D k-means.

This is a faithful port of the C++ ``kmeans1d`` extension bundled with
coremltools (``coremltools._deps._kmeans1d``), itself an implementation of the
dynamic-programming + SMAWK algorithm of Grønlund et al. (2017),
https://cs.au.dk/~larsen/papers/1dkmeans.pdf, running in ``O(n log n + kn)``
time and ``O(kn)`` space.

It computes the same clustering as coremltools (centroids match to a tight
numerical tolerance) without depending on a compiled extension, keeping the
package platform-agnostic; Numba JIT-compiles the kernels at runtime. The
optional ``weights`` argument provides weighted (sensitive) k-means: it
minimizes ``sum_i w_i * (x_i - mu_c(i))**2``.

The unweighted and weighted paths share one kernel: unweighted clustering is
the weighted kernel with every weight set to ``1.0`` (the prefix sums then hold
counts, weighted means reduce to plain means, and the arithmetic is identical).

Equivalence with coremltools (to a tight tolerance) is pinned by
``tests/test_utils/test_kmeans1d.py``; the comments below that reference the C++
arithmetic describe *why* the port reproduces it, and those tests are the
executable proof.

Example:
    >>> result = cluster([0.0, 0.1, 9.0, 9.1], 2)
    >>> result.centroids
    array([0.05, 9.05])
"""

from __future__ import annotations

from collections import namedtuple
from collections.abc import Sequence

import numpy as np
from numba import njit

Clustered = namedtuple("Clustered", ["clusters", "centroids"])


@njit(cache=True)
def _cost(cumw, cumsum, cumsum2, lo, hi):
    """Weighted within-cluster sum of squared deviations over sorted ``[lo, hi]``."""
    if hi < lo:
        return 0.0
    total_weight = cumw[hi + 1] - cumw[lo]
    weighted_sum = cumsum[hi + 1] - cumsum[lo]
    mu = weighted_sum / total_weight
    # Expanded as sum(w*x^2) + W*mu^2 - 2*mu*sum(w*x); this algebraic form and
    # statement order reproduce coremltools' float64 result to a tight tolerance.
    result = cumsum2[hi + 1] - cumsum2[lo]
    result += total_weight * (mu * mu)
    result -= (2.0 * mu) * weighted_sum
    return result


@njit(cache=True)
def _matrix_value(row, col_candidate, prev_costs, cumw, cumsum, cumsum2):
    """Entry ``(row, col_candidate)`` of the totally-monotone DP cost matrix."""
    # ``min(row, col_candidate - 1)`` with the C++ unsigned wrap special-cased:
    # when ``col_candidate == 0`` the C++ ``j - 1`` wraps to a huge value, so the
    # minimum is ``row``. Padding (col_candidate > row) yields ``prev_costs[row]``,
    # which is what makes the matrix totally monotone for SMAWK.
    if col_candidate == 0:
        prefix_col = row
    else:
        prefix_col = col_candidate - 1
        if row < prefix_col:
            prefix_col = row
    return prev_costs[prefix_col] + _cost(cumw, cumsum, cumsum2, col_candidate, row)


@njit(cache=True)
def _smawk(rows, cols, prev_costs, cumw, cumsum, cumsum2, result):
    """Write ``result[row] = argmin_col matrix_value(row, col)`` for each row.

    Implements the SMAWK algorithm on the implicitly-defined, totally-monotone
    matrix, in ``O(num_rows + num_cols)`` per level. Tie-breaking follows
    coremltools (leftmost minimizing column wins) so the recovered cluster
    boundaries match.
    """
    num_rows = rows.shape[0]
    if num_rows == 0:
        return
    num_cols = cols.shape[0]

    # REDUCE: discard columns that can never hold a row minimum. ``>=`` keeps the
    # incumbent (smaller) column on ties, giving the leftmost argmin.
    stack = np.empty(num_rows, np.int64)
    top = 0
    for ci in range(num_cols):
        col = cols[ci]
        while top > 0:
            row = rows[top - 1]
            incumbent = stack[top - 1]
            if _matrix_value(row, col, prev_costs, cumw, cumsum, cumsum2) >= _matrix_value(
                row, incumbent, prev_costs, cumw, cumsum, cumsum2
            ):
                break
            top -= 1
        if top < num_rows:
            stack[top] = col
            top += 1
    surviving = stack[:top]

    # Recurse on the odd-indexed rows.
    num_odd = num_rows // 2
    odd_rows = np.empty(num_odd, np.int64)
    for i in range(num_odd):
        odd_rows[i] = rows[2 * i + 1]
    _smawk(odd_rows, surviving, prev_costs, cumw, cumsum, cumsum2, result)

    # INTERPOLATE: fill even-indexed rows; each search is bounded on the right by
    # the already-computed odd neighbor's argmin. Strict ``<`` keeps the leftmost.
    start = 0
    row_idx = 0
    while row_idx < num_rows:
        row = rows[row_idx]
        stop = top - 1
        if row_idx < num_rows - 1:
            stop = np.searchsorted(surviving, result[rows[row_idx + 1]])
        argmin = surviving[start]
        best = _matrix_value(row, argmin, prev_costs, cumw, cumsum, cumsum2)
        col_idx = start + 1
        while col_idx <= stop:
            value = _matrix_value(row, surviving[col_idx], prev_costs, cumw, cumsum, cumsum2)
            if value < best:
                argmin = surviving[col_idx]
                best = value
            col_idx += 1
        result[row] = argmin
        start = stop
        row_idx += 2


@njit(cache=True)
def _build_prefix_sums(sorted_array, sorted_weights):
    """Prefix sums (leading zero) of weight, weight*value, and weight*value^2."""
    n = sorted_array.shape[0]
    cumw = np.empty(n + 1)
    cumsum = np.empty(n + 1)
    cumsum2 = np.empty(n + 1)
    cumw[0] = 0.0
    cumsum[0] = 0.0
    cumsum2[0] = 0.0
    for i in range(n):
        w = sorted_weights[i]
        x = sorted_array[i]
        cumw[i + 1] = w + cumw[i]
        cumsum[i + 1] = w * x + cumsum[i]
        cumsum2[i + 1] = w * x * x + cumsum2[i]
    return cumw, cumsum, cumsum2


@njit(cache=True)
def _backtrack(boundary, cumw, sorted_array, k):
    """Recover sorted labels and centroids from the filled DP boundary table.

    Walks the cluster boundaries right-to-left. The centroid is an online
    (Welford) weighted mean rather than the closed form, which reproduces
    coremltools' float64 rounding to a tight tolerance.
    """
    n = sorted_array.shape[0]
    sorted_clusters = np.empty(n, np.int64)
    # Zero-fill matters: when the optimum needs fewer than k clusters, the loop
    # terminates with leading levels unwritten. coremltools' ctypes centroid
    # buffer is zero-filled, so those slots come back as 0.0 -- match that.
    centroids = np.zeros(k)
    right = n
    level = k - 1
    end = n - 1
    while True:
        prev_right = right
        right = boundary[level, end]
        centroid = 0.0
        for i in range(right, prev_right):
            sorted_clusters[i] = level
            point_weight = cumw[i + 1] - cumw[i]
            cumulative_weight = cumw[i + 1] - cumw[right]
            centroid += (sorted_array[i] - centroid) * point_weight / cumulative_weight
        centroids[level] = centroid
        if right <= 0:
            break
        level -= 1
        end = right - 1

    return sorted_clusters, centroids


@njit(cache=True)
def _cluster_sorted(sorted_array, sorted_weights, k):
    """Run DP + SMAWK on already-sorted values/weights, returning sorted labels."""
    n = sorted_array.shape[0]
    cumw, cumsum, cumsum2 = _build_prefix_sums(sorted_array, sorted_weights)

    # cost_matrix[level, i]: optimal cost of clustering prefix [0, i] into
    # level+1 clusters. boundary[level, i]: left edge of the last cluster.
    cost_matrix = np.empty((k, n))
    boundary = np.empty((k, n), np.int64)

    for i in range(n):
        cost_matrix[0, i] = _cost(cumw, cumsum, cumsum2, 0, i)
        boundary[0, i] = 0

    argmins = np.empty(n, np.int64)
    for level in range(1, k):
        prev_costs = cost_matrix[level - 1]
        rows = np.arange(n)
        cols = np.arange(n)
        _smawk(rows, cols, prev_costs, cumw, cumsum, cumsum2, argmins)
        for i in range(n):
            argmin = argmins[i]
            cost_matrix[level, i] = _matrix_value(i, argmin, prev_costs, cumw, cumsum, cumsum2)
            boundary[level, i] = argmin

    return _backtrack(boundary, cumw, sorted_array, k)


def cluster(
    array: Sequence[float] | np.ndarray,
    k: int,
    *,
    weights: Sequence[float] | np.ndarray | None = None,
) -> Clustered:
    """Cluster 1D data into ``k`` groups minimizing the within-cluster SSE.

    Globally optimal via dynamic programming + SMAWK. Drop-in replacement for
    ``coremltools._deps._kmeans1d.cluster``.

    Args:
        array (Sequence[float] | np.ndarray): The 1D values to cluster.
        k (int): Desired number of clusters; clamped to ``len(array)``.
        weights (Sequence[float] | np.ndarray | None): Optional per-point weights
            for weighted (sensitive) k-means. Must match ``len(array)`` if given.

    Returns:
        Clustered: Namedtuple ``(clusters, centroids)``. ``clusters`` is an int64
        array of per-point labels in original order, indexing into ``centroids``,
        an ascending float64 array of length ``min(k, len(array))``.

    Raises:
        ValueError: If ``k <= 0``; ``array`` is empty or contains non-finite
            values; or ``weights`` has the wrong length, is non-finite, contains
            a negative value, or sums to zero. (coremltools silently produces NaN
            for these; we fail loudly instead.)
    """
    values = np.ascontiguousarray(array, dtype=np.float64).ravel()
    n = values.shape[0]
    if k <= 0:
        raise ValueError(f"Invalid k: {k}")
    if n == 0:
        raise ValueError("Cannot cluster an empty array.")
    if not np.isfinite(values).all():
        raise ValueError("array must contain only finite values.")
    k = min(k, n)

    if weights is None:
        point_weights = np.ones(n, dtype=np.float64)
    else:
        point_weights = np.ascontiguousarray(weights, dtype=np.float64).ravel()
        if point_weights.shape[0] != n:
            raise ValueError(
                f"weights length ({point_weights.shape[0]}) must match array length ({n})."
            )
        if not np.isfinite(point_weights).all():
            raise ValueError("weights must contain only finite values.")
        if (point_weights < 0.0).any():
            raise ValueError("weights must be non-negative.")
        if point_weights.sum() <= 0.0:
            raise ValueError("weights must sum to a positive value.")

    order = np.argsort(values)
    sorted_clusters, centroids = _cluster_sorted(values[order], point_weights[order], k)

    clusters = np.empty(n, dtype=np.int64)
    clusters[order] = sorted_clusters
    return Clustered(clusters=clusters, centroids=centroids)
