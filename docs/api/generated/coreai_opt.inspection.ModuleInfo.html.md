# coreai_opt.inspection.ModuleInfo

### *class* coreai_opt.inspection.ModuleInfo(module_name, module_type, child_modules, ops, input_ops, output_ops)

Bases: `object`

A node in the `nn.Module` hierarchy with its directly-owned ops.

Mirrors the `nn.Module` nesting structure: each `ModuleInfo`
holds the ops that belong directly to that module and references its
child modules as nested `ModuleInfo` instances.

* **Parameters:**
  * **module_name** (*str*)
  * **module_type** (*str*)
  * **child_modules** (*dict* *[**str* *,* [*ModuleInfo*](#coreai_opt.inspection.ModuleInfo) *]*)
  * **ops** (*list* *[*[*OpInfo*](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo) *]*)
  * **input_ops** (*dict* *[**int* *,* *list* *[*[*BoundaryEdge*](coreai_opt.inspection.BoundaryEdge.md#coreai_opt.inspection.BoundaryEdge) *]* *]*)
  * **output_ops** (*dict* *[**int* *,* [*BoundaryEdge*](coreai_opt.inspection.BoundaryEdge.md#coreai_opt.inspection.BoundaryEdge) *]*)

#### module_name

Fully-qualified module name (e.g.,
`"encoder.conv1"`).  Empty string for the root module.

* **Type:**
  str

#### module_type

Fully-qualified class name of the module (e.g.,
`"torch.nn.modules.conv.Conv2d"`).

* **Type:**
  str

#### child_modules

Child modules keyed by
`module_name`, in insertion order.

* **Type:**
  dict[str, [ModuleInfo](#coreai_opt.inspection.ModuleInfo)]

#### ops

Ops directly owned by this module, in
graph order.

* **Type:**
  list[[OpInfo](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo)]

#### input_ops

Boundary edges where data enters this
module from outside. Keys are module input spec indices (positions in the
flattened module forward arguments). Values are lists of all
`(op, input_slot)` pairs that the tensor at that position feeds into inside
the module — a single module input tensor can fan out to multiple ops. Keys
are absent for positions occupied by state tensors, untracked tensors, or
unused arguments. The key is what the user passes to `module_input_spec`.

* **Type:**
  dict[int, list[[BoundaryEdge](coreai_opt.inspection.BoundaryEdge.md#coreai_opt.inspection.BoundaryEdge)]]

#### output_ops

Boundary edges where data leaves this
module. Keys are module output spec indices (positions in the flattened
module return value). Each key maps to the `(op, output_slot)` pair that
produces the tensor at that position. Keys are absent for positions occupied
by state tensors or untracked tensors. The key is what the user passes to
`module_output_spec`.

* **Type:**
  dict[int, [BoundaryEdge](coreai_opt.inspection.BoundaryEdge.md#coreai_opt.inspection.BoundaryEdge)]

#### \_\_init_\_(module_name, module_type, child_modules, ops, input_ops, output_ops)

* **Parameters:**
  * **module_name** (*str*)
  * **module_type** (*str*)
  * **child_modules** (*dict* *[**str* *,* [*ModuleInfo*](#coreai_opt.inspection.ModuleInfo) *]*)
  * **ops** (*list* *[*[*OpInfo*](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo) *]*)
  * **input_ops** (*dict* *[**int* *,* *list* *[*[*BoundaryEdge*](coreai_opt.inspection.BoundaryEdge.md#coreai_opt.inspection.BoundaryEdge) *]* *]*)
  * **output_ops** (*dict* *[**int* *,* [*BoundaryEdge*](coreai_opt.inspection.BoundaryEdge.md#coreai_opt.inspection.BoundaryEdge) *]*)
* **Return type:**
  None

### Methods

| [`all_ops`](#coreai_opt.inspection.ModuleInfo.all_ops)()                        | Return all ops within this module and its submodules in graph order.   |
|---------------------------------------------------------------------------------|------------------------------------------------------------------------|
| [`children`](#coreai_opt.inspection.ModuleInfo.children)()                      | Yield direct child modules in insertion order.                         |
| [`get_submodule`](#coreai_opt.inspection.ModuleInfo.get_submodule)(module_name) | Return a descendant module by its fully-qualified name.                |
| [`modules`](#coreai_opt.inspection.ModuleInfo.modules)()                        | Yield this module and all descendant modules in depth-first order.     |
| [`named_children`](#coreai_opt.inspection.ModuleInfo.named_children)()          | Yield `(module_name, ModuleInfo)` for direct child modules.            |
| [`named_modules`](#coreai_opt.inspection.ModuleInfo.named_modules)()            | Yield `(module_name, ModuleInfo)` for this module and all descendants. |

#### all_ops()

Return all ops within this module and its submodules in graph order.

* **Return type:**
  list[[*OpInfo*](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo)]

#### children()

Yield direct child modules in insertion order.

* **Return type:**
  *Iterator*[[*ModuleInfo*](#coreai_opt.inspection.ModuleInfo)]

#### get_submodule(module_name)

Return a descendant module by its fully-qualified name.

* **Parameters:**
  **module_name** (*str*) – Fully-qualified module name (e.g.,
  `"encoder.conv1"`).
* **Raises:**
  **KeyError** – If no module with the given name exists in this subtree.
* **Return type:**
  [*ModuleInfo*](#coreai_opt.inspection.ModuleInfo)

#### modules()

Yield this module and all descendant modules in depth-first order.

* **Return type:**
  *Iterator*[[*ModuleInfo*](#coreai_opt.inspection.ModuleInfo)]

#### named_children()

Yield `(module_name, ModuleInfo)` for direct child modules.

* **Return type:**
  *Iterator*[tuple[str, [*ModuleInfo*](#coreai_opt.inspection.ModuleInfo)]]

#### named_modules()

Yield `(module_name, ModuleInfo)` for this module and all descendants.

* **Return type:**
  *Iterator*[tuple[str, [*ModuleInfo*](#coreai_opt.inspection.ModuleInfo)]]

#### child_modules *: dict[str, [ModuleInfo](#coreai_opt.inspection.ModuleInfo)]*

#### input_ops *: dict[int, list[[BoundaryEdge](coreai_opt.inspection.BoundaryEdge.md#coreai_opt.inspection.BoundaryEdge)]]*

#### module_name *: str*

#### module_type *: str*

#### ops *: list[[OpInfo](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo)]*

#### output_ops *: dict[int, [BoundaryEdge](coreai_opt.inspection.BoundaryEdge.md#coreai_opt.inspection.BoundaryEdge)]*
