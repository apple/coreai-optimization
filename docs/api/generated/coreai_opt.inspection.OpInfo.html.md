# coreai_opt.inspection.OpInfo

### *class* coreai_opt.inspection.OpInfo(op_name, op_type, module_stack, source_frames, inputs, outputs, is_state)

Bases: `object`

Information about a single operation discovered in a model.

* **Parameters:**
  * **op_name** (*str*)
  * **op_type** (*str* *|* *None*)
  * **module_stack** (*tuple* *[*[*ModuleContext*](coreai_opt.inspection.ModuleContext.md#coreai_opt.inspection.ModuleContext) *,*  *...* *]*)
  * **source_frames** (*tuple* *[*[*SourceFrame*](coreai_opt.inspection.SourceFrame.md#coreai_opt.inspection.SourceFrame) *,*  *...* *]*)
  * **inputs** (*tuple* *[*[*InputEdge*](coreai_opt.inspection.InputEdge.md#coreai_opt.inspection.InputEdge) *,*  *...* *]*)
  * **outputs** (*dict* *[**int* *,* *tuple* *[*[*OpInfo*](#coreai_opt.inspection.OpInfo) *,*  *...* *]* *]*)
  * **is_state** (*bool*)

#### op_name

The operation name that `op_name_config` regex patterns
match against (e.g., `"add_1"`, `"linear"`).

* **Type:**
  str

#### op_type

The operation type that `op_type_config` keys match
against (e.g., `"add"`, `"linear"`). `None` if the
type could not be determined.

* **Type:**
  str | None

#### module_stack

The `nn.Module` nesting hierarchy
from outermost to innermost. The innermost entry’s `module_name` is the
string that `module_name_configs` would match, and its
`module_type` is the string that `module_type_configs`
would match.

* **Type:**
  tuple[[ModuleContext](coreai_opt.inspection.ModuleContext.md#coreai_opt.inspection.ModuleContext), …]

#### source_frames

Source code locations from outermost
`forward()` to innermost, showing the call chain that produced this op.
May be empty if source information is unavailable.

* **Type:**
  tuple[[SourceFrame](coreai_opt.inspection.SourceFrame.md#coreai_opt.inspection.SourceFrame), …]

#### inputs

Ordered input edges. Each [`InputEdge`](coreai_opt.inspection.InputEdge.md#coreai_opt.inspection.InputEdge)
carries the producing op and the output slot of that op the tensor came from.

* **Type:**
  tuple[[InputEdge](coreai_opt.inspection.InputEdge.md#coreai_opt.inspection.InputEdge), …]

#### outputs

Dictionary mapping op outputs to a tuple of ops
consuming the output.

* **Type:**
  dict[int, tuple[[OpInfo](#coreai_opt.inspection.OpInfo), …]]

#### is_state

`True` if this op represents a model parameter or
buffer rather than a computation. State ops have an empty
`module_stack` and do not appear in module tree or boundary lists.

* **Type:**
  bool

#### \_\_init_\_(op_name, op_type, module_stack, source_frames, inputs, outputs, is_state)

* **Parameters:**
  * **op_name** (*str*)
  * **op_type** (*str* *|* *None*)
  * **module_stack** (*tuple* *[*[*ModuleContext*](coreai_opt.inspection.ModuleContext.md#coreai_opt.inspection.ModuleContext) *,*  *...* *]*)
  * **source_frames** (*tuple* *[*[*SourceFrame*](coreai_opt.inspection.SourceFrame.md#coreai_opt.inspection.SourceFrame) *,*  *...* *]*)
  * **inputs** (*tuple* *[*[*InputEdge*](coreai_opt.inspection.InputEdge.md#coreai_opt.inspection.InputEdge) *,*  *...* *]*)
  * **outputs** (*dict* *[**int* *,* *tuple* *[*[*OpInfo*](#coreai_opt.inspection.OpInfo) *,*  *...* *]* *]*)
  * **is_state** (*bool*)
* **Return type:**
  None

#### inputs *: tuple[[InputEdge](coreai_opt.inspection.InputEdge.md#coreai_opt.inspection.InputEdge), ...]*

#### is_state *: bool*

#### module_stack *: tuple[[ModuleContext](coreai_opt.inspection.ModuleContext.md#coreai_opt.inspection.ModuleContext), ...]*

#### op_name *: str*

#### op_type *: str | None*

#### outputs *: dict[int, tuple[[OpInfo](#coreai_opt.inspection.OpInfo), ...]]*

#### source_frames *: tuple[[SourceFrame](coreai_opt.inspection.SourceFrame.md#coreai_opt.inspection.SourceFrame), ...]*
