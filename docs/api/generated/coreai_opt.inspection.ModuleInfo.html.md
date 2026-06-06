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
  * **input_ops** (*list* *[*[*OpInfo*](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo) *]*)
  * **output_ops** (*list* *[*[*OpInfo*](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo) *]*)

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

Ops owned by this module, that receive data from
outside this module.

* **Type:**
  list[[OpInfo](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo)]

#### output_ops

Ops owned by this module, that send data outside
this module.

* **Type:**
  list[[OpInfo](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo)]

#### \_\_init_\_(module_name, module_type, child_modules, ops, input_ops, output_ops)

* **Parameters:**
  * **module_name** (*str*)
  * **module_type** (*str*)
  * **child_modules** (*dict* *[**str* *,* [*ModuleInfo*](#coreai_opt.inspection.ModuleInfo) *]*)
  * **ops** (*list* *[*[*OpInfo*](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo) *]*)
  * **input_ops** (*list* *[*[*OpInfo*](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo) *]*)
  * **output_ops** (*list* *[*[*OpInfo*](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo) *]*)
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

#### input_ops *: list[[OpInfo](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo)]*

#### module_name *: str*

#### module_type *: str*

#### ops *: list[[OpInfo](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo)]*

#### output_ops *: list[[OpInfo](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo)]*
