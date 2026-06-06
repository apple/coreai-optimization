# coreai_opt.quantization.spec.QParamsCalculatorBase

### *class* coreai_opt.quantization.spec.QParamsCalculatorBase(dtype, qscheme, granularity, target_dtype, quant_min, quant_max, range_calculator, float_range, scale_dtype=None, \*\*kwargs)

Bases: `ClassRegistryMixin`, `Module`

Base class for implementing logic to calculate quantization parameters
(scale, zero_point, minval) given min/max values.

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
| [`extra_repr`](#coreai_opt.quantization.spec.QParamsCalculatorBase.extra_repr)()                                     | Return the extra representation of the module.                            |
| [`forward`](#coreai_opt.quantization.spec.QParamsCalculatorBase.forward)(tensor)                                     | Compute scale, zero point, and minval from the input tensor.              |
| `get_class`(key)                                                                                                     |                                                                           |
| [`get_qparams`](#coreai_opt.quantization.spec.QParamsCalculatorBase.get_qparams)()                                   | Return the computed scale, zero point and minval.                         |
| `list_registry_keys`()                                                                                               |                                                                           |
| `list_registry_values`()                                                                                             |                                                                           |
| `register`(key)                                                                                                      | Register a virtual subclass of an ABC.                                    |
| [`set_export_mode`](#coreai_opt.quantization.spec.QParamsCalculatorBase.set_export_mode)([enabled])                  |                                                                           |

#### compute_qparams(tensor, min_val, max_val)

Given the observed min/max range, return `(scale, zero_point, minval)`.

The default implementation directly computes qparams from the given
range via `_compute_scale_zero_point_minval`.  This is the correct behavior
for stateless calculators (e.g. `StaticQParamsCalculator`).

Stateful calculators override this via `RunningRangeMixin` to update
running-range buffers before computing qparams from the smoothed range.

* **Parameters:**
  * **tensor** (*Tensor*)
  * **min_val** (*Tensor*)
  * **max_val** (*Tensor*)
* **Return type:**
  tuple[*Tensor*, *Tensor* | None, *Tensor* | None]

#### extra_repr()

Return the extra representation of the module.

To print customized extra information, you should re-implement
this method in your own modules. Both single-line and multi-line
strings are acceptable.

* **Return type:**
  str

#### forward(tensor)

Compute scale, zero point, and minval from the input tensor.

On the first forward pass, initializes internal buffers using the
observed tensor shape and device. Delegates the actual qparams
calculation to `compute_qparams`.

* **Parameters:**
  **tensor** (*Tensor*)
* **Return type:**
  tuple[*Tensor*, *Tensor* | None, *Tensor* | None]

#### get_qparams()

Return the computed scale, zero point and minval.
For FP4/FP8/floating-point quantization, zero_point and minval are None.

* **Return type:**
  tuple[*Tensor*, *Tensor* | None, *Tensor* | None]

#### set_export_mode(enabled=True)

* **Parameters:**
  **enabled** (*bool*)
* **Return type:**
  None

#### *property* granularity *: [QuantizationGranularity](coreai_opt.quantization.spec.QuantizationGranularity.md#coreai_opt.quantization.spec.QuantizationGranularity)*

Getter for granularity.

#### minval *: torch.Tensor | None*

#### scale *: torch.Tensor*

#### zero_point *: torch.Tensor | None*
