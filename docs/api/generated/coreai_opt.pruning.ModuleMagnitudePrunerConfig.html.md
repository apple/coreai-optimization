# coreai_opt.pruning.ModuleMagnitudePrunerConfig

### *class* coreai_opt.pruning.ModuleMagnitudePrunerConfig

Bases: [`WeightOnlyModuleValidationMixin`](coreai_opt.config.WeightOnlyModuleValidationMixin.md#coreai_opt.config.WeightOnlyModuleValidationMixin), `ModuleCompressionConfig[OpMagnitudePrunerConfig, PruningSpec]`

Module-level pruning configuration.

Manages pruning settings for an entire module, following the same
hierarchical precedence as other compression configs:

1. `op_name_config` (most specific)
2. `op_type_config`
3. `op_state_spec` (least specific)

#### op_state_spec

Default pruning
specs for state tensors in this module.

* **Type:**
  dict[str, [PruningSpec](coreai_opt.pruning.PruningSpec.md#coreai_opt.pruning.PruningSpec) | None] | None

#### op_type_config

Per-op-type overrides.

* **Type:**
  dict[str, [OpMagnitudePrunerConfig](coreai_opt.pruning.config.OpMagnitudePrunerConfig.md#coreai_opt.pruning.config.OpMagnitudePrunerConfig)]

#### op_name_config

Per-op-name overrides.

* **Type:**
  dict[str, [OpMagnitudePrunerConfig](coreai_opt.pruning.config.OpMagnitudePrunerConfig.md#coreai_opt.pruning.config.OpMagnitudePrunerConfig)]

#### module_state_spec

Specs applied
across all ops in the module.

* **Type:**
  dict[str, [PruningSpec](coreai_opt.pruning.PruningSpec.md#coreai_opt.pruning.PruningSpec) | None] | None

#### sparsity_schedule

Optional sparsity schedule.
When set, the `pruner.step()` API drives sparsity over training
steps; when `None` (default), the spec’s `target_sparsity` is
applied immediately and statically.

* **Type:**
  [SparsityScheduleBase](coreai_opt.pruning.config.SparsityScheduleBase.md#coreai_opt.pruning.config.SparsityScheduleBase) | None
