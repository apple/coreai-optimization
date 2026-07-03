# coreai_opt.inspection.BoundaryEdge

### *class* coreai_opt.inspection.BoundaryEdge(op, index)

Bases: `object`

A single data-flow edge crossing a module boundary.

Each entry in [`ModuleInfo.input_ops`](coreai_opt.inspection.ModuleInfo.md#id1) or [`ModuleInfo.output_ops`](coreai_opt.inspection.ModuleInfo.md#id5)
corresponds to one quantizer configuration point at the boundary.

* **Parameters:**
  * **op** ([*OpInfo*](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo))
  * **index** (*int*)

#### op

The op inside the module at the boundary.

* **Type:**
  [OpInfo](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo)

#### index

For input boundaries: the input slot of `op` that
receives external data, matching the index used in
`module_input_spec`. For output boundaries: the output slot of
`op` whose tensor leaves the module, matching the index used in
`module_output_spec`.

* **Type:**
  int

#### \_\_init_\_(op, index)

* **Parameters:**
  * **op** ([*OpInfo*](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo))
  * **index** (*int*)
* **Return type:**
  None

#### op *: [OpInfo](coreai_opt.inspection.OpInfo.md#coreai_opt.inspection.OpInfo)*
