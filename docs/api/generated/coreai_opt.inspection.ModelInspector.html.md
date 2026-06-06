# coreai_opt.inspection.ModelInspector

### *class* coreai_opt.inspection.ModelInspector(model, example_inputs, execution_mode, compressor=None, dynamic_shapes=None, export_with_no_grad=True)

Bases: `object`

Inspect operations in a PyTorch model for compression configuration.

Accepts an `nn.Module` with example inputs, auto-exports the model
(for graph mode), and provides query methods for discovering operation
names, types, and module hierarchy.

#### summary

The underlying operation summary.

* **Type:**
  [ModelSummary](coreai_opt.inspection.ModelSummary.md#coreai_opt.inspection.ModelSummary)

* **Parameters:**
  * **model** (*torch.fx.GraphModule* *|* *torch.nn.Module*) – The model to inspect.
  * **example_inputs** (*tuple* *[**Any* *,*  *...* *]*  *|* *None*) – Example inputs for tracing.
  * **execution_mode** ([*ExecutionMode*](coreai_opt.quantization.ExecutionMode.md#coreai_opt.quantization.ExecutionMode)) – Execution mode to use for model inspection.
  * **compressor** (*type* *[* *\_BaseModelCompressor* *]*  *|* *None*) – A compressor class (e.g., `Quantizer`) to filter
    ops to only those supported by that compression algorithm.
    When `None`, all ops in the model are included.
  * **dynamic_shapes** (*dict* *[**str* *,* *Any* *]*  *|* *tuple* *[**Any* *]*  *|* *list* *[**Any* *]*  *|* *None*) – Only relevant for graph execution mode.
    Optional dynamic shapes specification for torch.export.
  * **export_with_no_grad** (*bool*) – Only relevant for “graph” execution mode.
    Whether to call torch.export.export within a
    torch.no_grad() context. Defaults to True.
* **Raises:**
  * **TypeError** – If *model* is not an `nn.Module`.
  * **NotImplementedError** – If *execution_mode* is `"eager"`.
  * **RuntimeError** – If model export fails (graph mode).
  * **ValueError** – If example_inputs is None without the right model/execution_mode combination, or
        if execution_mode is not either “eager” or “graph”.

#### \_\_init_\_(model, example_inputs, execution_mode, compressor=None, dynamic_shapes=None, export_with_no_grad=True)

* **Parameters:**
  * **model** (*GraphModule* *|* *Module*)
  * **example_inputs** (*tuple* *[**Any* *,*  *...* *]*  *|* *None*)
  * **execution_mode** ([*ExecutionMode*](coreai_opt.quantization.ExecutionMode.md#coreai_opt.quantization.ExecutionMode))
  * **compressor** (*type* *[* *\_BaseModelCompressor* *]*  *|* *None*)
  * **dynamic_shapes** (*dict* *[**str* *,* *Any* *]*  *|* *tuple* *[**Any* *]*  *|* *list* *[**Any* *]*  *|* *None*)
  * **export_with_no_grad** (*bool*)
* **Return type:**
  None

### Methods

| [`format_summary`](#coreai_opt.inspection.ModelInspector.format_summary)([colorize])                                    | Format discovered operations as a module-hierarchy tree string.      |
|-------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| [`get_matched_ops_for_module_name`](#coreai_opt.inspection.ModelInspector.get_matched_ops_for_module_name)(module_name) | Return operations whose module stack contains the given module name. |
| [`get_matched_ops_for_module_type`](#coreai_opt.inspection.ModelInspector.get_matched_ops_for_module_type)(module_type) | Return operations whose module stack contains the given type.        |
| [`get_matched_ops_for_op_name`](#coreai_opt.inspection.ModelInspector.get_matched_ops_for_op_name)(pattern)             | Return operations whose name matches the given regex pattern.        |
| [`get_matched_ops_for_op_type`](#coreai_opt.inspection.ModelInspector.get_matched_ops_for_op_type)(op_type)             | Return operations matching the given op type.                        |

#### format_summary(colorize=None)

Format discovered operations as a module-hierarchy tree string.

* **Parameters:**
  **colorize** (*bool* *|* *None*) – Whether to include ANSI color codes in the
  output. `None` (default) auto-detects based on terminal
  capabilities. Pass `False` when writing to files or logs.
* **Returns:**
  The formatted tree.
* **Return type:**
  str

#### get_matched_ops_for_module_name(module_name)

Return operations whose module stack contains the given module name.

Uses `re.fullmatch` against each module FQN in the op’s module
stack, consistent with how `module_name_configs` patterns are
matched in Graph mode.

* **Parameters:**
  **module_name** (*str*) – A regex pattern to match against module FQNs
  (e.g., `"encoder.layer1"`, `"encoder\..*"`).
* **Returns:**
  Matching operations.
* **Return type:**
  tuple[[OpInfo](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo), …]
* **Raises:**
  **ValueError** – If *module_name* is not a valid regex.

#### get_matched_ops_for_module_type(module_type)

Return operations whose module stack contains the given type.

Matches using exact string equality against the fully-qualified
type name, consistent with how `module_type_configs` keys are
resolved in the quantizer.  Accepts either a class (converted via
`fqn()`) or a fully-qualified
type string (e.g., `"torch.nn.modules.conv.Conv2d"`).

* **Parameters:**
  **module_type** (*type* *|* *str*) – Module type to filter by.
* **Returns:**
  Matching operations.
* **Return type:**
  tuple[[OpInfo](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo), …]

#### get_matched_ops_for_op_name(pattern)

Return operations whose name matches the given regex pattern.

Uses `re.fullmatch`, consistent with how `op_name_config`
patterns are matched in Graph mode.

* **Parameters:**
  **pattern** (*str*) – A regex pattern to match against op names.
* **Returns:**
  Matching operations.
* **Return type:**
  tuple[[OpInfo](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo), …]
* **Raises:**
  **ValueError** – If *pattern* is not a valid regex.

#### get_matched_ops_for_op_type(op_type)

Return operations matching the given op type.

* **Parameters:**
  **op_type** (*str*) – The operation type to filter by (e.g.,
  `"conv2d"`, `"linear"`).
* **Returns:**
  Matching operations.
* **Return type:**
  tuple[[OpInfo](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo), …]

#### *property* summary *: [ModelSummary](coreai_opt.inspection.ModelSummary.md#coreai_opt.inspection.ModelSummary)*

The underlying operation summary.
