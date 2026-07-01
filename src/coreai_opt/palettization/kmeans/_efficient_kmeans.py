# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as _np
import torch as _torch


class _EfficientKMeans:
    """
    An implementation of k-means which runs entirely on GPU.
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
        self.n_init = n_init
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

            num_update_list.append(num_update)

            # In every trial, we track number of cluster centre updates.
            # If number of trials are greater than a specified value INIT_EXIT and
            # there is no update for the past INIT_EXIT number of trials,
            # it indicates that the centroids have converged
            if len(num_update_list) >= INIT_EXIT and sum(num_update_list[-INIT_EXIT:]) == 0:
                break

    def fit(
        self, X: _torch.Tensor, sample_weight: _torch.Tensor | None = None
    ) -> "_EfficientKMeans":
        """
        Compute k-means clustering.
        """
        N = len(X)

        assert N >= self.n_clusters, f"too many clusters {self.n_clusters} for {N} samples"

        if self.cluster_centers_ != "kmeans++":
            raise ValueError(f"init must be 'kmeans++'; got {self.cluster_centers_!r}")

        self._kmeans_pp(X.float(), sample_weight=sample_weight)

        self.cluster_centers_ = _EfficientKMeans._get_cluster_avg(
            self.n_clusters, self.labels_, X, sample_weight=sample_weight
        )

        return self
