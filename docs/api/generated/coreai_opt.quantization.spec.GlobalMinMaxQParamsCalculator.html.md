# coreai_opt.quantization.spec.GlobalMinMaxQParamsCalculator

### *class* coreai_opt.quantization.spec.GlobalMinMaxQParamsCalculator(\*args, \*\*kwargs)

Bases: [`RunningRangeMixin`](coreai_opt.quantization.spec.RunningRangeMixin.md#coreai_opt.quantization.spec.RunningRangeMixin), [`QParamsCalculatorBase`](coreai_opt.quantization.spec.QParamsCalculatorBase.md#coreai_opt.quantization.spec.QParamsCalculatorBase)

Computes scale and zero point by tracking the running min/max.

Maintains `running_min` and `running_max` buffers that are updated each
forward pass via element-wise minimum and maximum:

> running_min = min(running_min, x_min)
> running_max = max(running_max, x_max)
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

| `compute_qparams`(tensor, min_val, max_val)                                                                                  | Update running range, persist to buffers, then compute qparams.   |
|------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| `extra_repr`()                                                                                                               | Return the extra representation of the module.                    |
| `forward`(tensor)                                                                                                            | Compute scale, zero point, and minval from the input tensor.      |
| `get_class`(key)                                                                                                             |                                                                   |
| `get_qparams`()                                                                                                              | Return the computed scale, zero point and minval.                 |
| `list_registry_keys`()                                                                                                       |                                                                   |
| `list_registry_values`()                                                                                                     |                                                                   |
| `register`(key)                                                                                                              | Register a virtual subclass of an ABC.                            |
| `set_export_mode`([enabled])                                                                                                 |                                                                   |
| [`update_running_range`](#coreai_opt.quantization.spec.GlobalMinMaxQParamsCalculator.update_running_range)(min_val, max_val) | Return `(updated_min, updated_max)` using subclass-specific rule. |

#### update_running_range(min_val, max_val)

Return `(updated_min, updated_max)` using subclass-specific rule.

* **Parameters:**
  * **min_val** (*Tensor*)
  * **max_val** (*Tensor*)
* **Return type:**
  tuple[*Tensor*, *Tensor*]
