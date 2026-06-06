# coreai_opt.palettization.KMeansPalettizerConfig

### *class* coreai_opt.palettization.KMeansPalettizerConfig

Bases: `CompressionConfig[ModuleKMeansPalettizerConfig]`

Top-level configuration class for kmeans palettization.

This class manages the complete palettization configuration for a neural
network model, organizing module-level configurations in a hierarchical
structure. It inherits from CompressionConfig and specializes it for
palettization using ModuleKMeansPalettizerConfig.

The configuration lookup follows a hierarchical precedence (most to least
specific):

1. module_name_configs - Applies to module instances matching a name
   pattern (supports regex)
2. module_type_configs - Applies to all modules of a specific type (e.g.,
   torch.nn.modules.linear.Linear)
3. global_config - Default configuration applied to all modules not
   otherwise configured

#### global_config

Default module-level
palettization configuration applied to all modules that don’t have
a more specific configuration. When KMeansPalettizerConfig is initialized
with no arguments, a default global_config is automatically created with
standard 4-bit palettization.
Setting global_config to None disables palettization by default globally.
Default: Auto-created with 4-bit palettization spec when no args provided

* **Type:**
  [ModuleKMeansPalettizerConfig](coreai_opt.palettization.ModuleKMeansPalettizerConfig.md#coreai_opt.palettization.ModuleKMeansPalettizerConfig) | None

#### module_type_configs

Module type-specific configurations. Keys are fully-qualified module type
names (e.g., “torch.nn.modules.linear.Linear”,
“torch.nn.modules.conv.Conv2d”). Values are ModuleKMeansPalettizerConfig
objects or None to disable palettization for that module type.
Default: {} (empty dict, no type-specific configs)

* **Type:**
  dict[str, [ModuleKMeansPalettizerConfig](coreai_opt.palettization.ModuleKMeansPalettizerConfig.md#coreai_opt.palettization.ModuleKMeansPalettizerConfig) | None] | None

#### module_name_configs

Module name-specific configurations. Keys are module name patterns
(supports regex matching, e.g., “model.layer1.\*”,
“decoder.layers.0”). Values are ModuleKMeansPalettizerConfig objects or
None to disable palettization for matching modules.
Default: {} (empty dict, no name-specific configs)

* **Type:**
  dict[str, [ModuleKMeansPalettizerConfig](coreai_opt.palettization.ModuleKMeansPalettizerConfig.md#coreai_opt.palettization.ModuleKMeansPalettizerConfig) | None] | None
