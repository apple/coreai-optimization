# coreai_opt.palettization.kmeans.dkm.DKMTrainingSpec

### *class* coreai_opt.palettization.kmeans.dkm.DKMTrainingSpec

Bases: [`TrainingStrategySpec`](coreai_opt.palettization.spec.TrainingStrategySpec.md#coreai_opt.palettization.spec.TrainingStrategySpec)

DKM (Differentiable K-Means) training-strategy hyperparameters.

#### tau

Softmax temperature for DKM’s soft centroid assignment. Smaller
values sharpen toward hard k-means; larger values soften it.
Typical range: 1e-5 to 1e-3.

#### kmeans_max_iter

Number of soft-assignment EM iterations per DKM
training forward pass.

#### enforce_zero

If True, pins the centroid nearest the origin to
exactly zero. Useful for joint pruning + palettization.

#### zero_threshold

Floor applied to the softmax assignment weights and the
per-centroid mass (denominator) for numerical stability at small
`tau`.
