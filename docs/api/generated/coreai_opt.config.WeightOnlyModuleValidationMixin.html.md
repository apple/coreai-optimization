# coreai_opt.config.WeightOnlyModuleValidationMixin

### *class* coreai_opt.config.WeightOnlyModuleValidationMixin

Bases: `object`

Mixin that adds weight-only validation to ModuleCompressionConfig subclasses.

This mixin is for compression types that only apply to weights/state tensors
and don’t compress activations (inputs/outputs).

This mixin adds a model_validator that rejects activation specs:
- op_input_spec, op_output_spec (op-level activations)
- module_input_spec, module_output_spec (module-level activations)

Only op_state_spec and module_state_spec are allowed.

#### NOTE
The mixin should come first in the inheritance list for proper MRO resolution.

#### \_\_init_\_()

### Methods

| [`validate_weight_only_module_constraint`](#coreai_opt.config.WeightOnlyModuleValidationMixin.validate_weight_only_module_constraint)()   | Ensure no activation specs are set (weight-only compression).   |
|-------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|

#### validate_weight_only_module_constraint()

Ensure no activation specs are set (weight-only compression).
