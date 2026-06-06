# coreai_opt.pruning.PruningSpec

### *class* coreai_opt.pruning.PruningSpec

Bases: [`CompressionSpec`](coreai_opt.config.CompressionSpec.md#coreai_opt.config.CompressionSpec)

Specification for pruning tensors.

#### target_sparsity

Fraction of elements to prune, in `[0, 1]`.
Default: 0.5.

* **Type:**
  float

#### pruning_scheme

Structural pattern of sparsity.
Default: `Unstructured()`.

* **Type:**
  [PruningScheme](coreai_opt.pruning.spec.PruningScheme.md#coreai_opt.pruning.spec.PruningScheme)

#### pruning_algo

Pruning implementation class.
Default: `"default"` (magnitude-based pruning).

* **Type:**
  type[[PruneImplBase](coreai_opt.pruning.spec.PruneImplBase.md#coreai_opt.pruning.spec.PruneImplBase)]
