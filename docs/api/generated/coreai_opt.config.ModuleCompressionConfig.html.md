# coreai_opt.config.ModuleCompressionConfig

### *class* coreai_opt.config.ModuleCompressionConfig

Bases: `BaseModel`, `Generic`[`_OpConfigT`, `_SpecT`]

Abstract base configuration class for module-level compression settings.

This generic class defines the structure for configuring compression at the
module level. Subclasses must implement the default spec providers to define
compression-specific default values. Parameterized by `_OpConfigT` (the op-level
config type, e.g., OpQuantizerConfig) and `_SpecT` (the compression spec type,
e.g., QuantizationSpec).

#### op_input_spec

Compression specifications
for operation inputs applied to all registered operations / patterns within
this module that don’t have a more specific configuration.

* **Type:**
  dict[str | int, \_SpecT | None] | None

#### op_output_spec

Compression
specifications for operation outputs applied to all registered operations /
patterns within this module that don’t have a more specific configuration.

* **Type:**
  dict[str | int, \_SpecT | None] | None

#### op_state_spec

Compression specifications for
operation state tensors applied to all registered operations / patterns
within this module that don’t have a more specific configuration.

* **Type:**
  dict[str, \_SpecT | None] | None

#### op_type_config

Operation type-specific
configurations. Keys are operation type names, values are op-level config
objects or None to disable compression for that type.

* **Type:**
  dict[str, \_OpConfigT | None] | None

#### op_name_config

Operation name-specific
configurations. Keys are operation name patterns (supports regex), values
are op-level config objects or None to disable compression.

* **Type:**
  dict[str, \_OpConfigT | None] | None

#### module_input_spec

Compression
specifications for module inputs. Module input settings treat the module
as an opaque entity.

* **Type:**
  dict[str | int, \_SpecT | None] | None

#### module_output_spec

Compression
specifications for module outputs. Module output settings treat the module
as an opaque entity.

* **Type:**
  dict[str | int, \_SpecT | None] | None

#### module_state_spec

Compression specifications
for module state tensors (parameters, buffers, constants). Module state
settings will override op state settings for the same state tensors.

* **Type:**
  dict[str, \_SpecT | None] | None
