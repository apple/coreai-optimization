# coreai_opt.quantization.ModuleQuantizerConfig

### *class* coreai_opt.quantization.ModuleQuantizerConfig

Bases: `ModuleCompressionConfig[OpQuantizerConfig, QuantizationSpec]`

Configuration class for quantization at the module level.

This class manages quantization settings for an entire module, including:

- Operation-level configurations (default, by type, by name)
- Module-level input/output quantization
- Module-level state (parameter) quantization

The operation configurations follow a hierarchical precedence:

1. op_name_config (most specific - applies to operations matching a name
   pattern)
2. op_type_config (applies to operations of a specific type)
3. op_input/output/state_spec (least specific - applies to all operations
   not otherwise configured)

Module-level input, output, and state settings treat the module as an opaque entity,
setting quantization settings for specified tensors and ignoring op specific
quantization capabilities. Module-level settings also don’t check whether the
operation receiving the quantized tensor is a registered operation or not.
Module-level settings will override any op specific settings.

#### op_input_spec

Quantization
specifications for operation inputs applied to all registered
operations/patterns within this module that don’t have a more specific
configuration.
Keys can be either all indices or all string names, but not a mix of both.
The special key “\*” can be used in both cases to refer to all inputs.
Example keys:

- int: Input index (e.g., 0 for first input, 1 for second input)
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
specifications for operation inputs applied to all registered
operations/patterns within this module that don’t have a more specific
configuration.
Keys can be either all indices or all string names, but not a mix of both.
The special key “\*” can be used in both cases to refer to all outputs.
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
specifications for operation state tensors (parameters, buffers, constants)
applied to all registered operations/patterns within this module that don’t
have a more specific configuration.
Keys can be string names (e.g. “weight”, “bias”) or “\*” to refer to all
state inputs.
Values are QuantizationSpec objects or None defining how to quantize each
state tensor. None value represents disabling quantization.
Default: {“weight”: default_weight_quantization_spec()} (int8
quantization for weight inputs)

* **Type:**
  dict[str, [QuantizationSpec](coreai_opt.quantization.QuantizationSpec.md#coreai_opt.quantization.QuantizationSpec) | None] | None

#### op_type_config

Operation
type-specific configurations. Keys are operation types (e.g.,
“linear”, “conv2d”). Generally speaking, operation types will match the torch
functional name ([https://docs.pytorch.org/docs/stable/nn.functional.html](https://docs.pytorch.org/docs/stable/nn.functional.html)) or
operation name within the torch namespace
([https://docs.pytorch.org/docs/stable/torch.html](https://docs.pytorch.org/docs/stable/torch.html))
when taking the portion of the name to the right of the last period.

For example, to refer to a Maxpool 2D operation, take the name used for the
torch functional, torch.nn.functional.max_pool2d, and use the portion of the
string after the last period: “max_pool2d”.

OpQuantizerConfig objects or None defining how to quantize operations of
that type. None value represents disabling quantization.
Default: {} (empty dict, no type-specific configs)

* **Type:**
  dict[str, [OpQuantizerConfig](coreai_opt.quantization.config.OpQuantizerConfig.md#coreai_opt.quantization.config.OpQuantizerConfig) | None] | None

#### op_name_config

Operation
name-specific configurations. Keys are operation name patterns
(supports regex matching). Values are OpQuantizerConfig objects or None
defining how to quantize operations matching those names. None value
represents disabling quantization.
Default: {} (empty dict, no name-specific configs)

* **Type:**
  dict[str, [OpQuantizerConfig](coreai_opt.quantization.config.OpQuantizerConfig.md#coreai_opt.quantization.config.OpQuantizerConfig) | None] | None

#### module_input_spec

Quantization specifications for module inputs. Module input settings treat
the module as an opaque entity, setting quantization settings for input
tensors to the module without checking whether the op receiving
the input is quantizable. Module input settings override op level
settings for the op receiving the module input.
Keys can be either all indices or all string names, but not a mix of both.
The special key “\*” can be used in both cases to refer to all module inputs.
Example keys:

- int: Input index (e.g., 0 for first input, 1 for second
  input)
- str: Named input identifier (e.g., “y”, “input_0”)
- “\*”: Applies to all inputs for the operation. Other tensors
  can be explicitly mentioned to override this setting.

Values are QuantizationSpec objects or None. None value represents disabling
quantization.
Default: {} (empty dict, no specific module input settings)

* **Type:**
  dict[str | int, [QuantizationSpec](coreai_opt.quantization.QuantizationSpec.md#coreai_opt.quantization.QuantizationSpec) | None] | None

#### module_output_spec

Quantization specifications for module outputs. Module output settings treat
the module as an opaque entity, setting quantization settings for
output tensors to the module without checking whether the op
receiving the output is quantizable. Module output settings
override op level settings for the op receiving the module output.
Keys can be:

- int: Output index (e.g., 0 for first output)
- str: Named output identifier
- “\*”: Applies to all outputs for the operation. Other tensors
  can be explicitly mentioned to override this setting.

Values are QuantizationSpec objects or None. None value represents disabling
quantization.
Default: {} (empty dict, no specific module output settings)

* **Type:**
  dict[str | int, [QuantizationSpec](coreai_opt.quantization.QuantizationSpec.md#coreai_opt.quantization.QuantizationSpec) | None] | None

#### module_state_spec

Quantization
specifications for module state tensors (parameters, buffers, and
constants). Module state settings will override op state settings for the
same state tensors.
Keys can be string names (e.g. “weight”, “bias”) or “\*” to refer to all
state inputs.
Values are QuantizationSpec objects or None. None value represents disabling
quantization.
Default: {} (empty dict, no specific module state settings)

* **Type:**
  dict[str, [QuantizationSpec](coreai_opt.quantization.QuantizationSpec.md#coreai_opt.quantization.QuantizationSpec) | None] | None

#### qat_schedule

Optional QAT schedule for controlling
observer and fake quantization state transitions during training.
When set, the `quantizer.step()` API must be used to advance the
schedule. See `QATSchedule` for details. When None (default),
both observer and fake quantization are enabled from the start of
training.

* **Type:**
  [QATSchedule](coreai_opt.quantization.config.QATSchedule.md#coreai_opt.quantization.config.QATSchedule) | None
