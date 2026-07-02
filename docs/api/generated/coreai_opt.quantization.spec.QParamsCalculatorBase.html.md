# coreai_opt.quantization.spec.QParamsCalculatorBase

### *class* coreai_opt.quantization.spec.QParamsCalculatorBase(dtype, qscheme, granularity, target_dtype, quant_min, quant_max, range_calculator, float_range, scale_dtype=None, \*\*kwargs)

Bases: `ClassRegistryMixin`, `Module`

Abstract base for qparams calculators — common configuration and helpers.

Concrete subclasses inherit from either:

- `StatefulQParamsCalculatorBase` — has scale/zp/minval buffers; used by
  Static, MovingAverage, GlobalMinMax.
- `StatelessQParamsCalculatorBase` — no buffers, qparams recomputed per
  forward; used by Dynamic. `FakeQuantizeImplBase` and `Quantizer`
  detect this subclass to keep the observer always on and to reject
  export.

Subclasses must implement `forward(tensor) -> (scale, zero_point, minval)`.

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

| [`compute_qparams`](#coreai_opt.quantization.spec.QParamsCalculatorBase.compute_qparams)(tensor, min_val, max_val)   | Given the observed min/max range, return `(scale, zero_point, minval)`.   |
|----------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| [`forward`](#coreai_opt.quantization.spec.QParamsCalculatorBase.forward)(tensor)                                     | Compute and return `(scale, zero_point, minval)` for `tensor`.            |
| `get_class`(key)                                                                                                     |                                                                           |
| `list_registry_keys`()                                                                                               |                                                                           |
| `list_registry_values`()                                                                                             |                                                                           |
| `register`(key)                                                                                                      | Register a virtual subclass of an ABC.                                    |
| `resolve`(data)                                                                                                      | Resolve a string key or class type against this registry.                 |

#### compute_qparams(tensor, min_val, max_val)

Given the observed min/max range, return `(scale, zero_point, minval)`.

Default implementation: pure function of the supplied range, no running state.
`RunningRangeMixin` overrides this to apply a running-range smoothing rule
before computing qparams.

* **Parameters:**
  * **tensor** (*Tensor*)
  * **min_val** (*Tensor*)
  * **max_val** (*Tensor*)
* **Return type:**
  tuple[*Tensor*, *Tensor* | None, *Tensor* | None]

#### *abstract* forward(tensor)

Compute and return `(scale, zero_point, minval)` for `tensor`.

* **Parameters:**
  **tensor** (*Tensor*)
* **Return type:**
  tuple[*Tensor*, *Tensor* | None, *Tensor* | None]

#### *property* granularity *: [QuantizationGranularity](coreai_opt.quantization.spec.QuantizationGranularity.md#coreai_opt.quantization.spec.QuantizationGranularity)*

Getter for granularity.
