# coreai_opt.quantization.spec.RunningRangeMixin

### *class* coreai_opt.quantization.spec.RunningRangeMixin(\*args, \*\*kwargs)

Bases: `object`

Mixin for calculators that maintain running min/max range buffers.

Provides `running_min` and `running_max` buffers, first-forward
initialization via `_initialize_state`, and a
`compute_qparams` implementation that delegates
the range-update rule to the abstract `update_running_range` hook.

Subclasses that want to re-use the logic of computing quantization
parameters but with different ways of updating the running statistics
can override the `update_running_range` method.

Must appear before `QParamsCalculatorBase` in the MRO so that its
`compute_qparams` and `_initialize_state`
take precedence over the base-class defaults.

* **Parameters:**
  * **args** (*object*)
  * **kwargs** (*object*)

#### \_\_init_\_(\*args, \*\*kwargs)

* **Parameters:**
  * **args** (*object*)
  * **kwargs** (*object*)
* **Return type:**
  None

### Methods

| [`compute_qparams`](#coreai_opt.quantization.spec.RunningRangeMixin.compute_qparams)(tensor, min_val, max_val)   | Update running range, persist to buffers, then compute qparams.   |
|------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| [`extra_repr`](#coreai_opt.quantization.spec.RunningRangeMixin.extra_repr)()                                     |                                                                   |
| [`update_running_range`](#coreai_opt.quantization.spec.RunningRangeMixin.update_running_range)(min_val, max_val) | Return `(updated_min, updated_max)` using subclass-specific rule. |

#### compute_qparams(tensor, min_val, max_val)

Update running range, persist to buffers, then compute qparams.

* **Parameters:**
  * **tensor** (*Tensor*)
  * **min_val** (*Tensor*)
  * **max_val** (*Tensor*)
* **Return type:**
  tuple[*Tensor*, *Tensor* | None, *Tensor* | None]

#### extra_repr()

* **Return type:**
  str

#### *abstract* update_running_range(min_val, max_val)

Return `(updated_min, updated_max)` using subclass-specific rule.

* **Parameters:**
  * **min_val** (*Tensor*)
  * **max_val** (*Tensor*)
* **Return type:**
  tuple[*Tensor*, *Tensor*]

#### running_max *: Tensor*

#### running_min *: Tensor*
