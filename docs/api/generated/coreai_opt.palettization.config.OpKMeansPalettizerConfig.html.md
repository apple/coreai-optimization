# coreai_opt.palettization.config.OpKMeansPalettizerConfig

### *class* coreai_opt.palettization.config.OpKMeansPalettizerConfig

Bases: [`WeightOnlyOpValidationMixin`](coreai_opt.config.WeightOnlyOpValidationMixin.md#coreai_opt.config.WeightOnlyOpValidationMixin), `OpCompressionConfig[PalettizationSpec]`

Configuration class for palettization at the operation level.

Palettization is a weight-only compression technique that doesn’t apply
to activations (inputs/outputs). Only op_state_spec is used to configure
which state tensors (e.g., weights, biases) should be palettized.

#### op_state_spec

Palettization
specifications for operation state tensors (parameters, buffers, constants).
Keys are string names (e.g., “weight”, “bias”) or “\*” to refer to all
state inputs. Values are PalettizationSpec objects or None to disable
palettization for that state tensor.
Default: 4-bit palettization for “weight” and “in_proj_weight”
state tensors via `default_weight_palettization_spec()`.

* **Type:**
  dict[str, [PalettizationSpec](coreai_opt.palettization.PalettizationSpec.md#coreai_opt.palettization.PalettizationSpec) | None] | None

#### *classmethod* get_default_state_spec()

Provide default state spec for palettization.

* **Return type:**
  dict[str, [*PalettizationSpec*](coreai_opt.palettization.PalettizationSpec.md#coreai_opt.palettization.PalettizationSpec) | None]
