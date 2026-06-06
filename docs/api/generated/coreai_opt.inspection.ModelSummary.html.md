# coreai_opt.inspection.ModelSummary

### *class* coreai_opt.inspection.ModelSummary(model, mode)

Bases: `object`

Complete listing of operations discovered in a model.

* **Parameters:**
  * **model** ([*ModuleInfo*](coreai_opt.inspection.ModuleInfo.md#coreai_opt.inspection.ModuleInfo))
  * **mode** ([*ExecutionMode*](coreai_opt.quantization.ExecutionMode.md#coreai_opt.quantization.ExecutionMode))

#### model

Top level module summary of the module hierarchy tree containing
all discovered operations nested within their owning modules.

* **Type:**
  [ModuleInfo](coreai_opt.inspection.ModuleInfo.md#coreai_opt.inspection.ModuleInfo)

#### mode

Which discovery mode was used: `ExecutionMode.GRAPH` for exported
`GraphModule` models, `ExecutionMode.EAGER` for `nn.Module` models
traced via a forward pass.

* **Type:**
  [ExecutionMode](coreai_opt.quantization.ExecutionMode.md#coreai_opt.quantization.ExecutionMode)

#### \_\_init_\_(model, mode)

* **Parameters:**
  * **model** ([*ModuleInfo*](coreai_opt.inspection.ModuleInfo.md#coreai_opt.inspection.ModuleInfo))
  * **mode** ([*ExecutionMode*](coreai_opt.quantization.ExecutionMode.md#coreai_opt.quantization.ExecutionMode))
* **Return type:**
  None

#### mode *: [ExecutionMode](coreai_opt.quantization.ExecutionMode.md#coreai_opt.quantization.ExecutionMode)*

#### model *: [ModuleInfo](coreai_opt.inspection.ModuleInfo.md#coreai_opt.inspection.ModuleInfo)*
