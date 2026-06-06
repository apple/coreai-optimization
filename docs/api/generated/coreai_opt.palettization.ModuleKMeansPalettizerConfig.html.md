# coreai_opt.palettization.ModuleKMeansPalettizerConfig

### *class* coreai_opt.palettization.ModuleKMeansPalettizerConfig

Bases: [`WeightOnlyModuleValidationMixin`](coreai_opt.config.WeightOnlyModuleValidationMixin.md#coreai_opt.config.WeightOnlyModuleValidationMixin), `ModuleCompressionConfig[OpKMeansPalettizerConfig, PalettizationSpec]`

Configuration for palettizing a specific module using K-means clustering.

This class manages palettization settings for an entire module, including:

- Operation-level configurations (default, by type, by name)
- Module-level state (parameter) palettization

The operation configurations follow a hierarchical precedence:

1. op_name_config (most specific - applies to operations matching a name
   pattern)
2. op_type_config (applies to operations of a specific type)
3. op_state_spec (least specific - applies to all operations not
   otherwise configured)

Module-level state settings treat the module as an opaque entity,
setting palettization settings for specified tensors and ignoring op specific
palettization capabilities. Module-level settings also don’t check whether the
operation receiving the palettized tensor is a registered operation or not.
Module-level settings will override any op specific settings.

#### op_state_spec

Palettization
specifications for operation state tensors (parameters, buffers, constants)
applied to all registered operations/patterns within this module that don’t
have a more specific configuration.
Keys can be string names (e.g. “weight”, “bias”) or “\*” to refer to all
state inputs.
Values are PalettizationSpec objects or None defining how to palettize each
state tensor. None value represents disabling palettization.
Default: 4-bit palettization for “weight” and “in_proj_weight”
state tensors via `default_weight_palettization_spec()`.

* **Type:**
  dict[str, [PalettizationSpec](coreai_opt.palettization.PalettizationSpec.md#coreai_opt.palettization.PalettizationSpec) | None] | None

#### op_type_config

Operation
type-specific configurations. Keys are operation type names (e.g.,
“aten.linear.default”, “aten.conv2d.default”). Values are
OpKMeansPalettizerConfig objects or None, defining how to palettize
operations of that type. None value represents disabling palettization.
Default: {} (empty dict, no type-specific configs)

* **Type:**
  dict[str, [OpKMeansPalettizerConfig](coreai_opt.palettization.config.OpKMeansPalettizerConfig.md#coreai_opt.palettization.config.OpKMeansPalettizerConfig) | None] | None

#### op_name_config

Operation
name-specific configurations. Keys are operation name patterns
(supports regex matching). Values are OpKMeansPalettizerConfig objects or
None, defining how to palettize operations matching those names. None value
represents disabling palettization.
Default: {} (empty dict, no name-specific configs)

* **Type:**
  dict[str, [OpKMeansPalettizerConfig](coreai_opt.palettization.config.OpKMeansPalettizerConfig.md#coreai_opt.palettization.config.OpKMeansPalettizerConfig) | None] | None

#### module_state_spec

Palettization
specifications for module state tensors (parameters, buffers, and
constants). Module state settings will override op state settings for the
same state tensors.
Keys can be string names (e.g. “weight”, “bias”) or “\*” to refer to all
state inputs.
Values are PalettizationSpec objects or None. None value represents
disabling palettization.
Default: {} (empty dict, no specific module state settings)

* **Type:**
  dict[str, [PalettizationSpec](coreai_opt.palettization.PalettizationSpec.md#coreai_opt.palettization.PalettizationSpec) | None] | None

#### enable_fast_kmeans_mode

When True, enables optimizations for faster
K-means clustering by rounding the weights before clustering if data is in
float16 range. If weight dtype is float32, weights are cast to float16 and
then rounded. This is not supported with `cluster_dim > 1`. Default: True.

* **Type:**
  bool

#### rounding_precision

Number of decimal places to round to during fast
K-means clustering. Higher values preserve more precision but may reduce
speed benefits. Only used when enable_fast_kmeans_mode is True. Default: 4.

* **Type:**
  int
