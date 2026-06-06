# coreai_opt.quantization.spec.MovingAverageQParamsCalculator

### *class* coreai_opt.quantization.spec.MovingAverageQParamsCalculator(dtype, qscheme, granularity, target_dtype, quant_min, quant_max, range_calculator, float_range, averaging_constant=0.01, \*\*kwargs)

Bases: [`RunningRangeMixin`](coreai_opt.quantization.spec.RunningRangeMixin.md#coreai_opt.quantization.spec.RunningRangeMixin), [`QParamsCalculatorBase`](coreai_opt.quantization.spec.QParamsCalculatorBase.md#coreai_opt.quantization.spec.QParamsCalculatorBase)

Computes the scale and zero point using a moving average of the range.

Maintains `running_min` and `running_max` buffers that are updated each
forward pass using exponential moving average (EMA):

> a_{i} = c \* x_{i} + (1 - c) \* a_{i-1}

where `c` is the `averaging_constant`.

* **Parameters:**
  * **dtype** (*torch.dtype*)
  * **qscheme** ([*QuantizationScheme*](coreai_opt.quantization.spec.QuantizationScheme.md#coreai_opt.quantization.spec.QuantizationScheme))
  * **granularity** ([*QuantizationGranularity*](coreai_opt.quantization.spec.QuantizationGranularity.md#coreai_opt.quantization.spec.QuantizationGranularity))
  * **target_dtype** (*torch.dtype*)
  * **quant_min** (*int*)
  * **quant_max** (*int*)
  * **range_calculator** ([*RangeCalculatorBase*](coreai_opt.quantization.spec.RangeCalculatorBase.md#coreai_opt.quantization.spec.RangeCalculatorBase))
  * **float_range** (*list* *[**float* *|* *None* *]*)
  * **averaging_constant** (*float*)

#### \_\_init_\_(dtype, qscheme, granularity, target_dtype, quant_min, quant_max, range_calculator, float_range, averaging_constant=0.01, \*\*kwargs)

* **Parameters:**
  * **dtype** (*dtype*)
  * **qscheme** ([*QuantizationScheme*](coreai_opt.quantization.spec.QuantizationScheme.md#coreai_opt.quantization.spec.QuantizationScheme))
  * **granularity** ([*QuantizationGranularity*](coreai_opt.quantization.spec.QuantizationGranularity.md#coreai_opt.quantization.spec.QuantizationGranularity))
  * **target_dtype** (*dtype*)
  * **quant_min** (*int*)
  * **quant_max** (*int*)
  * **range_calculator** ([*RangeCalculatorBase*](coreai_opt.quantization.spec.RangeCalculatorBase.md#coreai_opt.quantization.spec.RangeCalculatorBase))
  * **float_range** (*list* *[**float* *|* *None* *]*)
  * **averaging_constant** (*float*)

### Methods

| `compute_qparams`(tensor, min_val, max_val)                                                                                   | Update running range, persist to buffers, then compute qparams.   |
|-------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| `extra_repr`()                                                                                                                | Return the extra representation of the module.                    |
| `forward`(tensor)                                                                                                             | Compute scale, zero point, and minval from the input tensor.      |
| `get_class`(key)                                                                                                              |                                                                   |
| `get_qparams`()                                                                                                               | Return the computed scale, zero point and minval.                 |
| `list_registry_keys`()                                                                                                        |                                                                   |
| `list_registry_values`()                                                                                                      |                                                                   |
| `register`(key)                                                                                                               | Register a virtual subclass of an ABC.                            |
| `set_export_mode`([enabled])                                                                                                  |                                                                   |
| [`update_running_range`](#coreai_opt.quantization.spec.MovingAverageQParamsCalculator.update_running_range)(min_val, max_val) | Return `(updated_min, updated_max)` using subclass-specific rule. |

#### update_running_range(min_val, max_val)

Return `(updated_min, updated_max)` using subclass-specific rule.

* **Parameters:**
  * **min_val** (*Tensor*)
  * **max_val** (*Tensor*)
* **Return type:**
  tuple[*Tensor*, *Tensor*]
