# coreai_opt.pruning.MagnitudePrunerConfig

### *class* coreai_opt.pruning.MagnitudePrunerConfig

Bases: `CompressionConfig[ModuleMagnitudePrunerConfig]`

Top-level configuration for magnitude pruning.

#### global_config

Default pruning
config applied to all modules.

* **Type:**
  [ModuleMagnitudePrunerConfig](coreai_opt.pruning.ModuleMagnitudePrunerConfig.md#coreai_opt.pruning.ModuleMagnitudePrunerConfig) | None

#### module_type_configs

Per-module-type overrides.

* **Type:**
  dict[str, [ModuleMagnitudePrunerConfig](coreai_opt.pruning.ModuleMagnitudePrunerConfig.md#coreai_opt.pruning.ModuleMagnitudePrunerConfig) | None]

#### module_name_configs

Per-module-name overrides (highest priority).

* **Type:**
  dict[str, [ModuleMagnitudePrunerConfig](coreai_opt.pruning.ModuleMagnitudePrunerConfig.md#coreai_opt.pruning.ModuleMagnitudePrunerConfig) | None]
