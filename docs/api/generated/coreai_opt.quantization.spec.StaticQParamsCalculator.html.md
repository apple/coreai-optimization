# coreai_opt.quantization.spec.StaticQParamsCalculator

### *class* coreai_opt.quantization.spec.StaticQParamsCalculator(dtype, qscheme, granularity, target_dtype, quant_min, quant_max, range_calculator, float_range, scale_dtype=None, \*\*kwargs)

Bases: [`QParamsCalculatorBase`](coreai_opt.quantization.spec.QParamsCalculatorBase.md#coreai_opt.quantization.spec.QParamsCalculatorBase)

Computes scale and zero point using min/max values from the current tensor.

This QParamsCalculator directly uses the min/max range from each forward pass to compute
quantization parameters. So in that sense, it does not maintain any “history” and
only computes the min/max based off of the current (most recent) tensor input.

This QParamsCalculator is typically used for weight quantization. In case of PTQ based
workflows the weights are fixed and during QAT, the min/max range is calculated using the
most recent weight tensor value.

Uses the base-class default `compute_qparams` which
directly delegates to `_compute_scale_zero_point_minval` without any running state.

* **Parameters:**
  * **dtype** (*torch.dtype*)
  * **qscheme** ([*QuantizationScheme*](coreai_opt.quantization.spec.QuantizationScheme.md#coreai_opt.quantization.spec.QuantizationScheme))
  * **granularity** ([*QuantizationGranularity*](coreai_opt.quantization.spec.QuantizationGranularity.md#coreai_opt.quantization.spec.QuantizationGranularity))
  * **target_dtype** (*torch.dtype*)
  * **quant_min** (*int*)
  * **quant_max** (*int*)
  * **range_calculator** ([*RangeCalculatorBase*](coreai_opt.quantization.spec.RangeCalculatorBase.md#coreai_opt.quantization.spec.RangeCalculatorBase))
  * **float_range** (*tuple* *[**float* *|* *None* *,* *float* *|* *None* *]*)
  * **scale_dtype** (*torch.dtype* *|* *None*)

#### \_\_init_\_(dtype, qscheme, granularity, target_dtype, quant_min, quant_max, range_calculator, float_range, scale_dtype=None, \*\*kwargs)

* **Parameters:**
  * **dtype** (*dtype*)
  * **qscheme** ([*QuantizationScheme*](coreai_opt.quantization.spec.QuantizationScheme.md#coreai_opt.quantization.spec.QuantizationScheme))
  * **granularity** ([*QuantizationGranularity*](coreai_opt.quantization.spec.QuantizationGranularity.md#coreai_opt.quantization.spec.QuantizationGranularity))
  * **target_dtype** (*dtype*)
  * **quant_min** (*int*)
  * **quant_max** (*int*)
  * **range_calculator** ([*RangeCalculatorBase*](coreai_opt.quantization.spec.RangeCalculatorBase.md#coreai_opt.quantization.spec.RangeCalculatorBase))
  * **float_range** (*tuple* *[**float* *|* *None* *,* *float* *|* *None* *]*)
  * **scale_dtype** (*dtype* *|* *None*)

### Methods

| `compute_qparams`(tensor, min_val, max_val)   | Given the observed min/max range, return `(scale, zero_point, minval)`.   |
|-----------------------------------------------|---------------------------------------------------------------------------|
| `extra_repr`()                                | Return the extra representation of the module.                            |
| `forward`(tensor)                             | Compute scale, zero point, and minval from the input tensor.              |
| `get_class`(key)                              |                                                                           |
| `get_qparams`()                               | Return the computed scale, zero point and minval.                         |
| `list_registry_keys`()                        |                                                                           |
| `list_registry_values`()                      |                                                                           |
| `register`(key)                               | Register a virtual subclass of an ABC.                                    |
| `set_export_mode`([enabled])                  |                                                                           |
