# coreai_opt.config.OpCompressionConfig

### *class* coreai_opt.config.OpCompressionConfig

Bases: `BaseModel`, `ABC`, `Generic`[`_SpecT`]

Abstract base configuration class for op-level compression settings.

This generic class defines the structure for configuring compression at the
operation level. Subclasses must implement the default spec providers to define
compression-specific default values. Parameterized by `_SpecT`, the compression
spec type (e.g., QuantizationSpec, PalettizationSpec).

#### op_input_spec

Compression specifications
for operation inputs. Keys can be either all indices or all string names,
but not a mix of both. The special key “\*” can be used in both cases to
refer to all inputs. Values are compression spec objects or None to
disable compression.

* **Type:**
  dict[str | int, \_SpecT | None] | None

#### op_output_spec

Compression
specifications for operation outputs. Keys can be either all indices or all
string names, but not a mix of both. The special key “\*” can be used in both
cases to refer to all outputs. Values are compression spec objects or None
to disable compression.

* **Type:**
  dict[str | int, \_SpecT | None] | None

#### op_state_spec

Compression specifications for
operation state tensors (parameters, buffers, constants). Keys are string
names (e.g., “weight”, “bias”) or “\*” to refer to all state inputs.
Values are compression spec objects or None to disable compression.

* **Type:**
  dict[str, \_SpecT | None] | None

#### *abstract classmethod* get_default_input_spec()

Provide default input spec for this compression type.

Override in subclasses to define compression-specific defaults.
Return empty dict if this compression type doesn’t apply to inputs.

* **Returns:**
  Dictionary mapping input identifiers to compression specs
* **Return type:**
  dict[str | int,  *\_SpecT* | None]

#### *abstract classmethod* get_default_output_spec()

Provide default output spec for this compression type.

Override in subclasses to define compression-specific defaults.
Return empty dict if this compression type doesn’t apply to outputs.

* **Returns:**
  Dictionary mapping output identifiers to compression specs
* **Return type:**
  dict[str | int,  *\_SpecT* | None]

#### *abstract classmethod* get_default_state_spec()

Provide default state spec for this compression type.

Override in subclasses to define compression-specific defaults.
Return empty dict if this compression type doesn’t apply to state.

* **Returns:**
  Dictionary mapping state tensor names to compression specs
* **Return type:**
  dict[str,  *\_SpecT* | None]
