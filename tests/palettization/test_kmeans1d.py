# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Equivalence, behavior, and source-integrity tests for the vendored 1D k-means.

The equivalence tests use coremltools (a test-only dependency) as the reference
oracle; it is imported directly, so this module requires coremltools and errors
if it is not installed. Our extension JIT-compiles ``_core.cpp`` on first use and
caches the artifact for the session, so the compile cost is paid once.
"""

import hashlib
import re
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from coremltools._deps import _kmeans1d as ct_kmeans1d

from coreai_opt.deps import _kmeans1d

_RTOL = 1e-9
_ATOL = 1e-9

# Discrete, strictly-positive weight values (with ties) for the weighted tests.
_WEIGHT_CHOICES = np.array([1.0, 2.0, 3.0, 4.0])


def _assert_equivalent(array, k, weights=None, exact_labels=True):
    """Assert our cluster() matches coremltools on centroids and reconstruction.

    Labels are compared exactly only when ``exact_labels`` is True (distinct
    input values); otherwise just the per-point centroid mapping is compared,
    since the C++ unstable sort makes tie labels non-reproducible while the
    palettized output stays identical.
    """
    reference = ct_kmeans1d.cluster(array, k, weights=weights)
    ours = _kmeans1d.cluster(array, k, weights=weights)

    ref_centroids = np.asarray(reference.centroids, dtype=np.float64)
    our_centroids = np.asarray(ours.centroids, dtype=np.float64)
    ref_clusters = np.asarray(reference.clusters)
    our_clusters = np.asarray(ours.clusters)

    np.testing.assert_allclose(our_centroids, ref_centroids, rtol=_RTOL, atol=_ATOL)

    # The centroid each point maps to must always match, ties or not.
    np.testing.assert_allclose(
        our_centroids[our_clusters], ref_centroids[ref_clusters], rtol=_RTOL, atol=_ATOL
    )

    if exact_labels:
        np.testing.assert_array_equal(our_clusters, ref_clusters)


def _inertia(points, weights, clusters, centroids):
    """Weighted within-cluster sum of squared deviations (port of coremltools' helper)."""
    points = np.asarray(points, dtype=np.float64)
    weights = np.ones_like(points) if weights is None else np.asarray(weights, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    clusters = np.asarray(clusters)
    return float(np.sum(weights * (points - centroids[clusters]) ** 2))


class TestEquivalence:
    """Our clustering must match the coremltools oracle."""

    @pytest.mark.parametrize("n", [2, 5, 50, 2_000])
    @pytest.mark.parametrize("k", [2, 4, 16, 64, 256])
    def test_equivalence_unweighted(self, n, k):
        rng = np.random.default_rng(n * 1000 + k)
        # Continuous random values are distinct with probability 1, so labels match.
        array = rng.standard_normal(n)
        _assert_equivalent(array, k, exact_labels=True)

    @pytest.mark.parametrize("n", [5, 500, 2_000])
    @pytest.mark.parametrize("k", [2, 4, 16, 64, 256])
    def test_equivalence_weighted(self, n, k):
        rng = np.random.default_rng(n + k)
        array = rng.standard_normal(n)
        # Discrete, strictly-positive weights (with ties) mirror the count-weighting path.
        weights = rng.choice(_WEIGHT_CHOICES, size=n)
        _assert_equivalent(array, k, weights=weights, exact_labels=True)

    @pytest.mark.parametrize("n", [10_000, 20_000, 50_000, 100_000])
    @pytest.mark.parametrize("k", [4, 64])
    def test_equivalence_large_n(self, n, k):
        # O(k*n) time and memory: n=1e5 x k=64 is ~100 MB per DP table. 1e6/1e7 are
        # infeasible for the exact DP (both our core and coremltools OOM identically).
        rng = np.random.default_rng(n + k)
        array = rng.standard_normal(n)
        _assert_equivalent(array, k, exact_labels=True)

    def test_equivalence_weighted_integer_counts(self):
        # Mirrors the fast/dedup path: cluster unique values weighted by counts.
        rng = np.random.default_rng(7)
        raw = np.round(rng.standard_normal(5000).astype(np.float16).astype(np.float32), 4)
        values, counts = np.unique(raw, return_counts=True)
        _assert_equivalent(values, 16, weights=counts.astype(np.float64), exact_labels=True)

    @pytest.mark.parametrize(
        "array, k",
        [
            ([5.0, 5.0, 5.0], 3),
            ([1.0, 1.0, 9.0], 3),
            ([-5.0, -5.0, -5.0], 3),
            ([2.0, 2.0, 2.0, 2.0], 4),
        ],
        ids=["all-equal", "two-distinct", "all-negative-equal", "four-equal"],
    )
    def test_collapse_fewer_than_k_clusters_matches_coremltools(self, array, k):
        # When k exceeds the number of distinct values, the optimum uses fewer than
        # k clusters and coremltools zero-fills the unused leading centroids. We must
        # match, including the non-ascending [0.0, 0.0, -5.0] padding for negatives.
        _assert_equivalent(np.array(array), k, exact_labels=False)

    def test_k_greater_than_n_is_clamped(self):
        array = np.array([1.0, 5.0, 9.0, 13.0, 17.0])
        result = _kmeans1d.cluster(array, 16)
        assert len(result.centroids) == len(array)
        _assert_equivalent(array, 16, exact_labels=True)

    def test_fp16_input_with_duplicates(self):
        # fp16 has few representable values, so duplicates (ties) are expected.
        rng = np.random.default_rng(3)
        array = rng.standard_normal(4000).astype(np.float16)
        _assert_equivalent(array, 16, exact_labels=False)

    def test_explicit_duplicates_reconstruction_matches(self):
        array = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 9.0, 9.0, 9.0, 9.0])
        _assert_equivalent(array, 3, exact_labels=False)


class TestKmeans1DBehavior:
    """Properties of the vendored cluster() surface (no oracle)."""

    def test_weights_equal_repetition(self):
        # Weighting a value by w is equivalent to repeating it w times.
        repeated = np.array([1.0, 1.0, 5.0, 5.0, 5.0, 9.0])
        unique_values = np.array([1.0, 5.0, 9.0])
        counts = np.array([2.0, 3.0, 1.0])

        repeated_result = _kmeans1d.cluster(repeated, 2)
        weighted_result = _kmeans1d.cluster(unique_values, 2, weights=counts)

        np.testing.assert_allclose(
            np.sort(np.unique(repeated_result.centroids)),
            np.sort(weighted_result.centroids),
            rtol=_RTOL,
            atol=_ATOL,
        )

    def test_centroids_are_ascending_and_labels_in_range(self):
        rng = np.random.default_rng(11)
        array = rng.standard_normal(1000)
        result = _kmeans1d.cluster(array, 16)

        centroids = np.asarray(result.centroids)
        clusters = np.asarray(result.clusters)
        assert np.all(np.diff(centroids) >= 0), "centroids must be ascending"
        assert clusters.min() >= 0 and clusters.max() < len(centroids)
        assert clusters.shape == array.shape
        assert centroids.dtype == np.float64

    def test_accepts_list_and_preserves_original_order(self):
        # A plain Python list (the doc-script call style) must work, and labels are
        # returned in the original (unsorted) order.
        result = _kmeans1d.cluster([9.0, 0.1, 9.1, 0.0], 2)
        clusters = np.asarray(result.clusters)
        centroids = np.asarray(result.centroids)
        # Points 0,2 are the large cluster; 1,3 the small one.
        assert clusters[0] == clusters[2]
        assert clusters[1] == clusters[3]
        assert clusters[0] != clusters[1]
        np.testing.assert_allclose(centroids, [0.05, 9.05], rtol=_RTOL, atol=_ATOL)


class TestsKmeans1dFromCoremltools:
    """Fixed-value oracle cases ported from coremltools' own kmeans1d test suite
    (``coremltools/deps/kmeans1d/tests/test_kmeans1d.py``).

    These assert the output against hardcoded known-good values; all inputs are
    distinct-valued, so labels are unambiguous.
    """

    def test_cluster(self):
        array = [4.0, 4.1, 4.2, -50, 200.2, 200.4, 200.9, 80, 100, 102]
        result = _kmeans1d.cluster(array, 4)
        np.testing.assert_array_equal(result.clusters, [1, 1, 1, 0, 3, 3, 3, 2, 2, 2])
        np.testing.assert_allclose(
            result.centroids, [-50.0, 4.1, 94.0, 200.5], rtol=_RTOL, atol=_ATOL
        )

    def test_cluster_with_weights(self):
        array = [4.0, 4.1, 4.2, -50, 200.2, 200.4, 200.9, 80, 100, 102]
        weights = [1, 1, 1, 0.125, 4, 1, 1, 3, 2, 2]
        result = _kmeans1d.cluster(array, 4, weights=weights)
        np.testing.assert_array_equal(result.clusters, [0, 0, 0, 0, 3, 3, 3, 1, 2, 2])
        np.testing.assert_allclose(
            result.centroids, [1.936, 80.0, 101.0, 200.35], rtol=_RTOL, atol=_ATOL
        )

    def test_weights_vs_repetition(self):
        values = [10, 24, 16, 12, 20]
        weights = [3, 1, 4, 2, 3]
        weighted = _kmeans1d.cluster(values, 2, weights=weights)
        np.testing.assert_array_equal(weighted.clusters, [0, 1, 1, 0, 1])
        np.testing.assert_allclose(weighted.centroids, [10.8, 18.5], rtol=_RTOL, atol=_ATOL)
        np.testing.assert_allclose(
            _inertia(values, weights, weighted.clusters, weighted.centroids),
            66.8,
            rtol=_RTOL,
            atol=_ATOL,
        )

        # Weighting by integer counts equals repeating each value that many times.
        repeated = np.repeat(values, weights)
        repeated_result = _kmeans1d.cluster(repeated, 2)
        np.testing.assert_allclose(
            np.sort(repeated_result.centroids),
            np.sort(weighted.centroids),
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            _inertia(repeated, None, repeated_result.clusters, repeated_result.centroids),
            66.8,
            rtol=_RTOL,
            atol=_ATOL,
        )


# Raw source of the C++ core on apple/coremltools main -- the file we vendor.
_UPSTREAM_CORE_CPP_URL = (
    "https://raw.githubusercontent.com/apple/coremltools/main/deps/kmeans1d/kmeans1d/_core.cpp"
)
_VENDORED_CORE_CPP = Path(_kmeans1d.__file__).parent / "_core.cpp"

# coremltools vendored this file from apple/coremltools main under a trailing
# "// Copyright © 2023 Apple Inc." line; ours reads "// Copyright © 2026 Apple
# Inc." (the year coreai-opt vendored it from coremltools). Normalize this one
# known, intentional difference away so the canary only fires on real drift in
# the kmeans1d logic itself.
_APPLE_COPYRIGHT_LINE_RE = re.compile(rb"// Copyright \xc2\xa9 \d{4} Apple Inc\.")


def _normalize_apple_copyright_year(source_bytes: bytes) -> bytes:
    return _APPLE_COPYRIGHT_LINE_RE.sub(b"// Copyright (c) <year> Apple Inc.", source_bytes)


class TestVendoredSourceMatchesUpstream:
    """Canary: our vendored ``_core.cpp`` must stay byte-identical to upstream
    (modulo the trailing Apple copyright year, which coreai-opt intentionally
    stamps with its own vendoring year rather than coremltools').

    Deliberately fetches over the network and does NOT guard the fetch: if the
    hashes differ, our copy drifted or Apple changed the file (time to re-vendor);
    if the network is down, the test errors rather than silently passing. Either
    outcome should turn CI red and prompt a look.
    """

    def test_vendored_core_cpp_matches_github_main(self):
        with urllib.request.urlopen(_UPSTREAM_CORE_CPP_URL, timeout=30) as response:
            upstream_bytes = response.read()

        upstream_hash = hashlib.sha256(_normalize_apple_copyright_year(upstream_bytes)).hexdigest()
        vendored_hash = hashlib.sha256(
            _normalize_apple_copyright_year(_VENDORED_CORE_CPP.read_bytes())
        ).hexdigest()

        assert vendored_hash == upstream_hash, (
            "Vendored _core.cpp no longer matches apple/coremltools main "
            f"({_UPSTREAM_CORE_CPP_URL}). Re-vendor the file and re-check the build "
            f"flags. vendored={vendored_hash} upstream={upstream_hash}"
        )
