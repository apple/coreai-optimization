# coreai_opt.pruning.config.OpMagnitudePrunerConfig

### *class* coreai_opt.pruning.config.OpMagnitudePrunerConfig

Bases: [`WeightOnlyOpValidationMixin`](coreai_opt.config.WeightOnlyOpValidationMixin.md#coreai_opt.config.WeightOnlyOpValidationMixin), `OpCompressionConfig[PruningSpec]`

Operation-level pruning configuration.

Pruning is a weight-only compression technique. Only `op_state_spec`
is used to configure which state tensors (e.g. weights) to prune.

#### op_state_spec

Mapping of parameter
names to their pruning specs. Default includes `"weight"` and
`"in_proj_weight"` at 50 % sparsity.

* **Type:**
  dict[str, [PruningSpec](coreai_opt.pruning.PruningSpec.md#coreai_opt.pruning.PruningSpec) | None]

#### *classmethod* get_default_state_spec()

Provide default state spec for pruning.

* **Return type:**
  dict[str, [*PruningSpec*](coreai_opt.pruning.PruningSpec.md#coreai_opt.pruning.PruningSpec) | None]
