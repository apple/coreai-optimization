# coreai_opt.quantization.spec.GlobalMinMaxQParamsCalculator

### *class* coreai_opt.quantization.spec.GlobalMinMaxQParamsCalculator(\*args, \*\*kwargs)

Bases: [`RunningRangeMixin`](coreai_opt.quantization.spec.RunningRangeMixin.md#coreai_opt.quantization.spec.RunningRangeMixin), [`StatefulQParamsCalculatorBase`](coreai_opt.quantization.spec.StatefulQParamsCalculatorBase.md#coreai_opt.quantization.spec.StatefulQParamsCalculatorBase)

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

| `compute_qparams`(tensor, min_val, max_val)                                                                                  | Update running range, persist to buffers, then compute qparams.              |
|------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| `extra_repr`()                                                                                                               | Return the extra representation of the module.                               |
| `forward`(tensor)                                                                                                            | Compute qparams from `tensor`; cache to buffers; return.                     |
| `get_class`(key)                                                                                                             |                                                                              |
| `get_qparams`()                                                                                                              | Return the computed scale, zero point and minval.                            |
| `list_registry_keys`()                                                                                                       |                                                                              |
| `list_registry_values`()                                                                                                     |                                                                              |
| `register`(key)                                                                                                              | Register a virtual subclass of an ABC.                                       |
| `reset_running_range`()                                                                                                      | Mark the running range as unset so the next observed range starts it afresh. |
| `resolve`(data)                                                                                                              | Resolve a string key or class type against this registry.                    |
| `set_export_mode`([enabled])                                                                                                 |                                                                              |
| [`update_running_range`](#coreai_opt.quantization.spec.GlobalMinMaxQParamsCalculator.update_running_range)(min_val, max_val) | Return `(updated_min, updated_max)` using subclass-specific rule.            |

#### update_running_range(min_val, max_val)

Return `(updated_min, updated_max)` using subclass-specific rule.

`self.running_min` / `self.running_max` may be `+inf` / `-inf` where the running
range is unset (see `reset_running_range`). Implementations must not raise on these
values; `compute_qparams` discards those entries in favor of `min_val` / `max_val`.

* **Parameters:**
  * **min_val** (*Tensor*)
  * **max_val** (*Tensor*)
* **Return type:**
  tuple[*Tensor*, *Tensor*]
