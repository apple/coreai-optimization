# coreai_opt.config.WeightOnlyOpValidationMixin

### *class* coreai_opt.config.WeightOnlyOpValidationMixin

Bases: `object`

Mixin that adds weight-only validation to OpCompressionConfig subclasses.

This mixin is for compression types that only apply to weights/state tensors
and don’t compress activations (inputs/outputs). Examples include palettization,
pruning, and low-rank decomposition.

This mixin:

1. Provides default empty implementations for get_default_input_spec and
   get_default_output_spec (satisfying abstract methods)
2. Adds a model_validator that rejects any op_input_spec or op_output_spec

#### NOTE
The mixin MUST come first in the inheritance list due to Python’s
MRO and abstract method resolution.

#### \_\_init_\_()

### Methods

| [`get_default_input_spec`](#coreai_opt.config.WeightOnlyOpValidationMixin.get_default_input_spec)()                         | Return empty dict as weight-only compression doesn't apply to inputs.   |
|-----------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| [`get_default_output_spec`](#coreai_opt.config.WeightOnlyOpValidationMixin.get_default_output_spec)()                       | Return empty dict as weight-only compression doesn't apply to outputs.  |
| [`validate_weight_only_op_constraint`](#coreai_opt.config.WeightOnlyOpValidationMixin.validate_weight_only_op_constraint)() | Ensure no input/output specs are set (weight-only compression).         |

#### *classmethod* get_default_input_spec()

Return empty dict as weight-only compression doesn’t apply to inputs.

* **Return type:**
  dict

#### *classmethod* get_default_output_spec()

Return empty dict as weight-only compression doesn’t apply to outputs.

* **Return type:**
  dict

#### validate_weight_only_op_constraint()

Ensure no input/output specs are set (weight-only compression).
