# coreai_opt.pruning.spec.ChannelStructured

### *class* coreai_opt.pruning.spec.ChannelStructured

Bases: [`PruningScheme`](coreai_opt.pruning.spec.PruningScheme.md#coreai_opt.pruning.spec.PruningScheme)

Channel-structured pruning scheme.

Entire channels (slices along `axis`) are pruned or kept together.
Channel importance is determined by the pruning algorithm (e.g. L1 norm
of each channel for magnitude-based pruning).
