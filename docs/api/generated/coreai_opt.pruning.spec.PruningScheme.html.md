# coreai_opt.pruning.spec.PruningScheme

### *class* coreai_opt.pruning.spec.PruningScheme

Bases: `BaseModel`, `ConfigRegistryMixin`

Base class for pruning scheme specifications.

A pruning scheme defines the structural pattern of sparsity applied
to a tensor. Subclasses represent different ways of structuring the pruning.

#### axis

The axis along which structured pruning is applied.
`None` for unstructured (element-wise) pruning.

* **Type:**
  int | None
