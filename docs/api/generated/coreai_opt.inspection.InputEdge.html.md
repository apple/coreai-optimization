# coreai_opt.inspection.InputEdge

### *class* coreai_opt.inspection.InputEdge(op, output_idx)

Bases: `object`

One input edge into an op, pairing the producing op with its output slot.

Used as the element type of [`OpInfo.inputs`](coreai_opt.inspection.OpInfo.md#id0). Delegation properties
forward the most-accessed [`OpInfo`](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo) attributes so that code iterating
`op.inputs` does not need to go through `.op` for routine checks.

* **Parameters:**
  * **op** ([*OpInfo*](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo))
  * **output_idx** (*int* *|* *None*)

#### op

The op that produced this input tensor.

* **Type:**
  [OpInfo](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo)

#### output_idx

Which output slot of `op` this tensor came from,
or `None` for synthetic ops (registered states, ephemeral/untracked
tensors) that have no meaningful output slot.

* **Type:**
  int | None

#### \_\_init_\_(op, output_idx)

* **Parameters:**
  * **op** ([*OpInfo*](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo))
  * **output_idx** (*int* *|* *None*)
* **Return type:**
  None

#### *property* inputs *: tuple[[InputEdge](#coreai_opt.inspection.InputEdge), ...]*

#### *property* is_state *: bool*

#### *property* module_stack *: tuple[[ModuleContext](coreai_opt.inspection.ModuleContext.md#coreai_opt.inspection.ModuleContext), ...]*

#### op *: [OpInfo](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo)*

#### *property* op_name *: str*

#### *property* op_type *: str | None*

#### output_idx *: int | None*

#### *property* outputs *: dict[int, tuple[[OpInfo](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo), ...]]*
