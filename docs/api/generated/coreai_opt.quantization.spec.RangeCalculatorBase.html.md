# coreai_opt.quantization.spec.RangeCalculatorBase

### *class* coreai_opt.quantization.spec.RangeCalculatorBase(granularity, \*\*kwargs)

Bases: `ClassRegistryMixin`, `Module`

Base class and registry for classes used to compute the range
of a given tensor.

* **Parameters:**
  **granularity** ([*QuantizationGranularity*](coreai_opt.quantization.spec.QuantizationGranularity.md#coreai_opt.quantization.spec.QuantizationGranularity))

#### \_\_init_\_(granularity, \*\*kwargs)

* **Parameters:**
  **granularity** ([*QuantizationGranularity*](coreai_opt.quantization.spec.QuantizationGranularity.md#coreai_opt.quantization.spec.QuantizationGranularity))

### Methods

| [`forward`](#coreai_opt.quantization.spec.RangeCalculatorBase.forward)(x)   | Compute range statistics on an input and return the min/max bounds.   |
|-----------------------------------------------------------------------------|-----------------------------------------------------------------------|
| `get_class`(key)                                                            |                                                                       |
| `list_registry_keys`()                                                      |                                                                       |
| `list_registry_values`()                                                    |                                                                       |
| `register`(key)                                                             | Register a virtual subclass of an ABC.                                |

#### forward(x)

Compute range statistics on an input and return the min/max bounds.

Calls \_generate_min_max to compute range statistics and validates that
the returned min/max shapes match the original tensor number of dimensions.

* **Parameters:**
  **x** (`torch.Tensor`) – Tensor to compute range statistics upon.
* **Return type:**
  tuple[*Tensor*, *Tensor*]
