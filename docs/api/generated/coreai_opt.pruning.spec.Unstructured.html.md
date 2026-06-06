# coreai_opt.pruning.spec.Unstructured

### *class* coreai_opt.pruning.spec.Unstructured

Bases: [`PruningScheme`](coreai_opt.pruning.spec.PruningScheme.md#coreai_opt.pruning.spec.PruningScheme)

Unstructured pruning scheme.

Individual elements are pruned independently — any element can be zeroed
regardless of its position in the tensor.
