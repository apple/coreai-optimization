# coreai_opt.quantization.spec.StaticQParamsCalculator

### *class* coreai_opt.quantization.spec.StaticQParamsCalculator(\*\*kwargs)

Bases: [`StatefulQParamsCalculatorBase`](coreai_opt.quantization.spec.StatefulQParamsCalculatorBase.md#coreai_opt.quantization.spec.StatefulQParamsCalculatorBase)

Computes scale/zero-point/minval using min/max values from the current tensor.

This QParamsCalculator directly uses the min/max range from each forward pass to compute
quantization parameters. So in that sense, it does not maintain any “history” and
only computes the min/max based off of the current (most recent) tensor input.

This QParamsCalculator is typically used for weight quantization. In case of PTQ based
workflows the weights are fixed and during QAT, the min/max range is calculated using the
most recent weight tensor value.

Uses the base-class default `compute_qparams` which
directly delegates to `_compute_scale_zero_point_minval` without any running state.

#### \_\_init_\_(\*\*kwargs)

### Methods

| `compute_qparams`(tensor, min_val, max_val)   | Given the observed min/max range, return `(scale, zero_point, minval)`.   |
|-----------------------------------------------|---------------------------------------------------------------------------|
| `extra_repr`()                                | Return the extra representation of the module.                            |
| `forward`(tensor)                             | Compute qparams from `tensor`; cache to buffers; return.                  |
| `get_class`(key)                              |                                                                           |
| `get_qparams`()                               | Return the computed scale, zero point and minval.                         |
| `list_registry_keys`()                        |                                                                           |
| `list_registry_values`()                      |                                                                           |
| `register`(key)                               | Register a virtual subclass of an ABC.                                    |
| `resolve`(data)                               | Resolve a string key or class type against this registry.                 |
| `set_export_mode`([enabled])                  |                                                                           |
