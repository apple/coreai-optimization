# coreai_opt.inspection.OpInfo

### *class* coreai_opt.inspection.OpInfo(op_name, op_type, module_stack, source_frames, inputs, outputs)

Bases: `object`

Information about a single operation discovered in a model.

* **Parameters:**
  * **op_name** (*str*)
  * **op_type** (*str* *|* *None*)
  * **module_stack** (*tuple* *[*[*ModuleContext*](coreai_opt.inspection.ModuleContext.md#coreai_opt.inspection.ModuleContext) *,*  *...* *]*)
  * **source_frames** (*tuple* *[*[*SourceFrame*](coreai_opt.inspection.SourceFrame.md#coreai_opt.inspection.SourceFrame) *,*  *...* *]*)
  * **inputs** (*tuple* *[*[*OpInfo*](#coreai_opt.inspection.OpInfo) *,*  *...* *]*)
  * **outputs** (*tuple* *[*[*OpInfo*](#coreai_opt.inspection.OpInfo) *,*  *...* *]*)

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

Ordered input ops (ops, placeholders,
parameters) that feed into this op.

* **Type:**
  tuple[[OpInfo](#coreai_opt.inspection.OpInfo), …]

#### outputs

Consumer ops that receive the output
of this op, in graph order.

* **Type:**
  tuple[[OpInfo](#coreai_opt.inspection.OpInfo), …]

#### \_\_init_\_(op_name, op_type, module_stack, source_frames, inputs, outputs)

* **Parameters:**
  * **op_name** (*str*)
  * **op_type** (*str* *|* *None*)
  * **module_stack** (*tuple* *[*[*ModuleContext*](coreai_opt.inspection.ModuleContext.md#coreai_opt.inspection.ModuleContext) *,*  *...* *]*)
  * **source_frames** (*tuple* *[*[*SourceFrame*](coreai_opt.inspection.SourceFrame.md#coreai_opt.inspection.SourceFrame) *,*  *...* *]*)
  * **inputs** (*tuple* *[*[*OpInfo*](#coreai_opt.inspection.OpInfo) *,*  *...* *]*)
  * **outputs** (*tuple* *[*[*OpInfo*](#coreai_opt.inspection.OpInfo) *,*  *...* *]*)
* **Return type:**
  None

#### inputs *: tuple[[OpInfo](#coreai_opt.inspection.OpInfo), ...]*

#### module_stack *: tuple[[ModuleContext](coreai_opt.inspection.ModuleContext.md#coreai_opt.inspection.ModuleContext), ...]*

#### op_name *: str*

#### op_type *: str | None*

#### outputs *: tuple[[OpInfo](#coreai_opt.inspection.OpInfo), ...]*

#### source_frames *: tuple[[SourceFrame](coreai_opt.inspection.SourceFrame.md#coreai_opt.inspection.SourceFrame), ...]*
