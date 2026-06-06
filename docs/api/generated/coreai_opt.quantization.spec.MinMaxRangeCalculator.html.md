# coreai_opt.quantization.spec.MinMaxRangeCalculator

### *class* coreai_opt.quantization.spec.MinMaxRangeCalculator(granularity, \*\*kwargs)

Bases: [`RangeCalculatorBase`](coreai_opt.quantization.spec.RangeCalculatorBase.md#coreai_opt.quantization.spec.RangeCalculatorBase)

Range calculator that computes the range of a given tensor as the min and max
values of the tensor.

* **Parameters:**
  **granularity** ([*QuantizationGranularity*](coreai_opt.quantization.spec.QuantizationGranularity.md#coreai_opt.quantization.spec.QuantizationGranularity))

#### \_\_init_\_(granularity, \*\*kwargs)

* **Parameters:**
  **granularity** ([*QuantizationGranularity*](coreai_opt.quantization.spec.QuantizationGranularity.md#coreai_opt.quantization.spec.QuantizationGranularity))

### Methods

| `forward`(x)             | Compute range statistics on an input and return the min/max bounds.   |
|--------------------------|-----------------------------------------------------------------------|
| `get_class`(key)         |                                                                       |
| `list_registry_keys`()   |                                                                       |
| `list_registry_values`() |                                                                       |
| `register`(key)          | Register a virtual subclass of an ABC.                                |
