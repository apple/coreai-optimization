# coreai_opt.inspection.ModuleContext

### *class* coreai_opt.inspection.ModuleContext(module_name, module_type)

Bases: `object`

One level of the `nn.Module` nesting hierarchy.

* **Parameters:**
  * **module_name** (*str*)
  * **module_type** (*str*)

#### module_name

Fully-qualified module name as it appears in
`model.named_modules()` (e.g., `"encoder.layer1"`).
This is the string used by `module_name_configs` in
`QuantizerConfig`.

* **Type:**
  str

#### module_type

Fully-qualified class name of the module (e.g.,
`"torch.nn.modules.linear.Linear"`). This is the string
used by `module_type_configs`.

* **Type:**
  str

#### \_\_init_\_(module_name, module_type)

* **Parameters:**
  * **module_name** (*str*)
  * **module_type** (*str*)
* **Return type:**
  None

#### module_name *: str*

#### module_type *: str*
