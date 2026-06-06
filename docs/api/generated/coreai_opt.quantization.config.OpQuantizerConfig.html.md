# coreai_opt.quantization.config.OpQuantizerConfig

### *class* coreai_opt.quantization.config.OpQuantizerConfig

Bases: `OpCompressionConfig[QuantizationSpec]`

Configuration class for quantization at the operation level.

This class specifies quantization settings for inputs, outputs, and state
tensors of individual operations (ops) in a neural network. Each tensor
can have its own quantization specification.

Quantization for operations will require the operation to be registered. Even if
globally all ops are configured to be quantized, ops which are not recognized will
not be quantized.

#### op_input_spec

Quantization
specifications for operation inputs. Keys can be either all indices or all
string names, but not a mix of both. The special key “\*” can be used in both
cases to refer to all inputs.
Example keys:

- int: Input index (e.g., 0 for first input, 1 for second
  input)
- str: Named input identifier (e.g., “x”, “input_0”)
- “\*”: Applies to all inputs for the operation. Other tensors
  can be explicitly mentioned to override this setting.

Values are QuantizationSpec objects or None defining how to quantize each
input. None value represents disabling quantization.
Default: {”\*”: default_activation_quantization_spec()} (int8
quantization for all inputs)

* **Type:**
  dict[str | int, [QuantizationSpec](coreai_opt.quantization.QuantizationSpec.md#coreai_opt.quantization.QuantizationSpec) | None] | None

#### op_output_spec

Quantization
specifications for operation outputs. Keys can be either all indices or all
string names, but not a mix of both. The special key “\*” can be used in both
cases to refer to all outputs.
Example keys:

- int: Output index (e.g., 0 for first output, 1 for second
  output)
- str: Named output identifier (e.g., “y”, “output_0”)
- “\*”: Applies to all outputs for the operation. Other tensors
  can be explicitly mentioned to override this setting.

Values are QuantizationSpec objects or None defining how to quantize each
output. None value represents disabling quantization.
Default: {”\*”: default_activation_quantization_spec()} (int8
quantization for all outputs)

* **Type:**
  dict[str | int, [QuantizationSpec](coreai_opt.quantization.QuantizationSpec.md#coreai_opt.quantization.QuantizationSpec) | None] | None

#### op_state_spec

Quantization
specifications for operation state tensors (parameters, buffers, constants).
Keys can be string names (e.g. “weight”, “bias”) or “\*” to refer to all
state inputs.
Values are QuantizationSpec objects or None defining how to quantize each
state tensor. None value represents disabling quantization.
Default: {“weight”: default_weight_quantization_spec()} (int8
quantization for weight inputs)

* **Type:**
  dict[str, [QuantizationSpec](coreai_opt.quantization.QuantizationSpec.md#coreai_opt.quantization.QuantizationSpec) | None] | None

#### *classmethod* get_default_input_spec()

Provide default input spec for quantization.

* **Return type:**
  dict[str | int, [*QuantizationSpec*](coreai_opt.quantization.QuantizationSpec.md#coreai_opt.quantization.QuantizationSpec) | None]

#### *classmethod* get_default_output_spec()

Provide default output spec for quantization.

* **Return type:**
  dict[str | int, [*QuantizationSpec*](coreai_opt.quantization.QuantizationSpec.md#coreai_opt.quantization.QuantizationSpec) | None]

#### *classmethod* get_default_state_spec()

Provide default state spec for quantization.

* **Return type:**
  dict[str, [*QuantizationSpec*](coreai_opt.quantization.QuantizationSpec.md#coreai_opt.quantization.QuantizationSpec) | None]
