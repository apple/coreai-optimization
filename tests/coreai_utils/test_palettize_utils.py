# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from coreai_opt.coreai_utils._utils.palettize_utils import (
    _get_kmeans_lookup_table_and_weight,
)


def test_vector_kmeans_limits_clusters_to_number_of_vectors() -> None:
    """Vector K-means must not request more clusters than vector samples."""
    weight = np.arange(16, dtype=np.float32)

    lut, indices = _get_kmeans_lookup_table_and_weight(4, weight, cluster_dim=4, vector_axis=0)

    assert lut.shape == (16, 4)
    assert indices.shape == (4,)
    np.testing.assert_allclose(lut[indices], weight.reshape(-1, 4))
