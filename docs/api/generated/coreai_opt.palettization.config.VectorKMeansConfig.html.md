# coreai_opt.palettization.config.VectorKMeansConfig

### *class* coreai_opt.palettization.config.VectorKMeansConfig

Bases: `BaseModel`

K-means settings for vector palettization (`cluster_dim > 1`).

Vector palettization clusters each palettization block’s `cluster_dim`-long weight
vectors into `2 ** n_bits` centroids with k-means. Each run seeds the
centroids with k-means++, then runs Lloyd iterations (assign every vector to its
nearest centroid, move each centroid to the mean of its vectors) until the block
converges. These settings trade clustering quality for speed.

Where clustering runs:

- **CUDA GPU**: all blocks of a weight tensor are clustered together in one batch.
  Unweighted clustering (post-training palettization) uses flash-kmeans for the
  Lloyd iterations when it is installed, and otherwise a batched PyTorch loop.
  Sensitivity-weighted clustering (`calibration_mode()`) always uses the PyTorch
  loop, because flash-kmeans has no per-point weights.
- **CPU**: scikit-learn, one block at a time. If scikit-learn is not installed, the
  batched PyTorch loop runs on CPU instead and a warning is issued; it gives the
  same results but is several times slower.

Every backend uses the same seeding, the same stopping rule, and a seed derived
from each tensor’s contents, so results are reproducible across runs.

#### n_local_trials

Number of candidate centroids sampled at each
k-means++ seeding step. The candidate that most reduces the clustering
error is kept. `1` is standard k-means++. Larger values (“greedy”
k-means++) usually give lower clustering error, and seeding time grows
roughly in proportion. `None` uses scikit-learn’s default,
`2 + int(log(n_clusters))`. Default: `None`.

* **Type:**
  int | None

#### n_init

How many times k-means is run from scratch on each block, each
time starting from a different set of initial centroids. Because the result
depends on where k-means starts, more runs make it likelier to find a good
one; the run with the lowest clustering error is kept. Time grows linearly
with `n_init`. Default: 5.

* **Type:**
  int

#### max_iter

Maximum number of Lloyd iterations per run. Default: 300.

* **Type:**
  int

#### tol

Convergence tolerance. A block stops iterating once its clustering
error (inertia) improves by less than `tol`, or when `max_iter` is
reached. Default: `1e-4`.

* **Type:**
  float
