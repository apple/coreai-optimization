# coreai_opt.quantization.QuantizerConfig

### *class* coreai_opt.quantization.QuantizerConfig

Bases: `CompressionConfig[ModuleQuantizerConfig]`

Top-level configuration class for quantization.

This class manages the complete quantization configuration for a neural
network model, organizing module-level configurations in a hierarchical
structure. It inherits from CompressionConfig and specializes it for
quantization using ModuleQuantizerConfig.

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
quantization configuration applied to all modules that don’t have
a more specific configuration. When QuantizerConfig is initialized
with no arguments, a default global_config is automatically created with
standard int8 quantization.
Setting global_config to None disables quantization by default globally.
Default: Auto-created with int8 quantization specs when no args
provided

* **Type:**
  [ModuleQuantizerConfig](coreai_opt.quantization.ModuleQuantizerConfig.md#coreai_opt.quantization.ModuleQuantizerConfig) | None

#### module_type_configs

Module
type-specific configurations. Keys are fully-qualified module type
names (e.g., “torch.nn.modules.linear.Linear”,
“torch.nn.modules.conv.Conv2d”). Values are ModuleQuantizerConfig objects or
None to disable quantization for that module type.
Default: {} (empty dict, no type-specific configs)

* **Type:**
  dict[str, [ModuleQuantizerConfig](coreai_opt.quantization.ModuleQuantizerConfig.md#coreai_opt.quantization.ModuleQuantizerConfig) | None] | None

#### module_name_configs

Module
name-specific configurations. Keys are module name patterns
(supports regex matching, e.g., “model.layer1.\*”,
“decoder.layers.0”). Values are ModuleQuantizerConfig objects or
None to disable quantization for matching modules.
Default: {} (empty dict, no name-specific configs)

* **Type:**
  dict[str, [ModuleQuantizerConfig](coreai_opt.quantization.ModuleQuantizerConfig.md#coreai_opt.quantization.ModuleQuantizerConfig) | None] | None

#### preserved_attributes

Names of attributes of the model
which should be preserved on the prepared and finalized models, even if they
are not used in the model’s forward pass.

* **Type:**
  list[str] | None

#### execution_mode

Specifies which quantization execution
mode to use. Options are:

- ExecutionMode.GRAPH / “graph”:
  : Graph-based quantization using `torch.export` and FX graphs, built on
    `torchao`’s PT2E implementation. Requires the model to be exportable.
- ExecutionMode.EAGER / “eager”:
  : Works directly on `nn.Module` without converting to a graph representation.
    Supports dynamic control flow (if/else, loops) and doesn’t require `torch.export`.

Default: ExecutionMode.GRAPH

* **Type:**
  [ExecutionMode](coreai_opt.quantization.ExecutionMode.md#coreai_opt.quantization.ExecutionMode) | str

#### set_execution_mode(mode)

Set the quantization execution mode.

* **Parameters:**
  **mode** ([*ExecutionMode*](coreai_opt.quantization.ExecutionMode.md#coreai_opt.quantization.ExecutionMode) *|* *str*) – Execution mode to use.
  Accepts an `ExecutionMode` member (e.g. `ExecutionMode.EAGER`)
  or its string value (e.g. `"graph"`, `"eager"`).
* **Returns:**
  This config, for method chaining.
* **Return type:**
  Self
* **Raises:**
  **ValueError** – If `mode` is a string that is not a valid
      `ExecutionMode` value.
