# coreai_opt.quantization.spec.QuantizationComponentFactory

### *class* coreai_opt.quantization.spec.QuantizationComponentFactory

Bases: [`CompressionComponentFactoryBase`](coreai_opt.config.spec.CompressionComponentFactoryBase.md#coreai_opt.config.spec.CompressionComponentFactoryBase)

Factory class for creating quantization components from QuantizationSpec.

This factory eliminates circular dependencies between QuantizationSpec and
component classes (FakeQuantizeImplBase, QParamsCalculatorBase, RangeCalculatorBase)
by centralizing the creation logic.

#### \_\_init_\_()

### Methods

| [`construct`](#coreai_opt.quantization.spec.QuantizationComponentFactory.construct)(spec, target)                                                  | Create a fake quantizer instance from a QuantizationSpec.                                                      |
|----------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------|
| [`construct_partial`](#coreai_opt.quantization.spec.QuantizationComponentFactory.construct_partial)(spec, target)                                  | Create a fake quantizer partial object for deferred construction.                                              |
| [`create_fake_quantizer`](#coreai_opt.quantization.spec.QuantizationComponentFactory.create_fake_quantizer)(spec, quantization_target)             | Create a FakeQuantizeImplBase instance from a QuantizationSpec.                                                |
| [`create_fake_quantizer_partial`](#coreai_opt.quantization.spec.QuantizationComponentFactory.create_fake_quantizer_partial)(spec, ...)             | Create a fake quantizer partial object for deferred construction by the graph-mode prepare API (torchao PT2E). |
| [`create_qparams_calculator`](#coreai_opt.quantization.spec.QuantizationComponentFactory.create_qparams_calculator)(spec, ...)                     | Create a QParamsCalculatorBase instance from a QuantizationSpec.                                               |
| [`create_range_calculator`](#coreai_opt.quantization.spec.QuantizationComponentFactory.create_range_calculator)(spec[, ...])                       | Create a RangeCalculatorBase instance from a QuantizationSpec.                                                 |
| [`reconstruct_partial_qparams_calculator`](#coreai_opt.quantization.spec.QuantizationComponentFactory.reconstruct_partial_qparams_calculator)(...) | Return a new PartialConstructor whose qparams_calculator has attributes overridden.                            |

#### *classmethod* construct(spec, target)

Create a fake quantizer instance from a QuantizationSpec.

This method implements the base class interface and delegates to
create_fake_quantizer.

* **Parameters:**
  * **spec** ([*QuantizationSpec*](coreai_opt.quantization.QuantizationSpec.md#coreai_opt.quantization.QuantizationSpec) *|* *None*) – QuantizationSpec instance containing configuration
  * **target** ([*CompressionTargetTensor*](coreai_opt.config.spec.CompressionTargetTensor.md#coreai_opt.config.spec.CompressionTargetTensor)) – The target tensor for compression (weight or activation)
* **Returns:**
  FakeQuantizeImplBase instance configured from the spec, or None if
  spec is None
* **Return type:**
  [*FakeQuantizeImplBase*](coreai_opt.quantization.spec.fake_quantize.FakeQuantizeImplBase.md#coreai_opt.quantization.spec.fake_quantize.FakeQuantizeImplBase) | None

#### *classmethod* construct_partial(spec, target)

Create a fake quantizer partial object for deferred construction.

This method implements the base class interface and delegates to
create_fake_quantizer_partial.

* **Parameters:**
  * **spec** ([*QuantizationSpec*](coreai_opt.quantization.QuantizationSpec.md#coreai_opt.quantization.QuantizationSpec) *|* *None*) – QuantizationSpec instance containing configuration
  * **target** ([*CompressionTargetTensor*](coreai_opt.config.spec.CompressionTargetTensor.md#coreai_opt.config.spec.CompressionTargetTensor)) – The target tensor for compression (weight or activation)
* **Returns:**
  A partial object for deferred construction, or None
  if spec is None
* **Return type:**
  PartialConstructor

#### *classmethod* create_fake_quantizer(spec, quantization_target)

Create a FakeQuantizeImplBase instance from a QuantizationSpec.

This method automatically detects any extra arguments in the spec beyond
the base QuantizationSpec fields and passes them to the fake quantizer
constructor.

* **Parameters:**
  * **spec** ([*QuantizationSpec*](coreai_opt.quantization.QuantizationSpec.md#coreai_opt.quantization.QuantizationSpec)) – QuantizationSpec instance containing configuration
  * **quantization_target** ([*CompressionTargetTensor*](coreai_opt.config.spec.CompressionTargetTensor.md#coreai_opt.config.spec.CompressionTargetTensor)) – The target tensor for quantization
* **Returns:**
  FakeQuantizeImplBase instance configured from the spec
* **Return type:**
  [*FakeQuantizeImplBase*](coreai_opt.quantization.spec.fake_quantize.FakeQuantizeImplBase.md#coreai_opt.quantization.spec.fake_quantize.FakeQuantizeImplBase)

#### *classmethod* create_fake_quantizer_partial(spec, quantization_target)

Create a fake quantizer partial object for deferred construction
by the graph-mode prepare API (torchao PT2E).

* **Parameters:**
  * **spec** ([*QuantizationSpec*](coreai_opt.quantization.QuantizationSpec.md#coreai_opt.quantization.QuantizationSpec)) – QuantizationSpec instance containing configuration
  * **quantization_target** ([*CompressionTargetTensor*](coreai_opt.config.spec.CompressionTargetTensor.md#coreai_opt.config.spec.CompressionTargetTensor)) – The target tensor for quantization
* **Returns:**
  A partial object that can be used by the graph-mode
  : prepare API to construct fake quantizer instances. Each call
    to the partial will create a new instance with its own
    qparams_calculator.
* **Return type:**
  PartialConstructor

#### *classmethod* create_qparams_calculator(spec, quantization_target)

Create a QParamsCalculatorBase instance from a QuantizationSpec.

* **Parameters:**
  * **spec** ([*QuantizationSpec*](coreai_opt.quantization.QuantizationSpec.md#coreai_opt.quantization.QuantizationSpec)) – QuantizationSpec instance containing configuration
  * **quantization_target** ([*CompressionTargetTensor*](coreai_opt.config.spec.CompressionTargetTensor.md#coreai_opt.config.spec.CompressionTargetTensor)) – The target tensor for quantization (weight/activation)
* **Returns:**
  QParamsCalculatorBase instance configured from the spec
* **Return type:**
  [*QParamsCalculatorBase*](coreai_opt.quantization.spec.QParamsCalculatorBase.md#coreai_opt.quantization.spec.QParamsCalculatorBase)

#### *classmethod* create_range_calculator(spec, quantization_target=CompressionTargetTensor.WEIGHT)

Create a RangeCalculatorBase instance from a QuantizationSpec.

* **Parameters:**
  * **spec** ([*QuantizationSpec*](coreai_opt.quantization.QuantizationSpec.md#coreai_opt.quantization.QuantizationSpec)) – QuantizationSpec instance containing configuration
  * **quantization_target** ([*CompressionTargetTensor*](coreai_opt.config.spec.CompressionTargetTensor.md#coreai_opt.config.spec.CompressionTargetTensor)) – The target tensor for quantization (weight/activation).
* **Returns:**
  RangeCalculatorBase instance configured from the spec
* **Return type:**
  [*RangeCalculatorBase*](coreai_opt.quantization.spec.RangeCalculatorBase.md#coreai_opt.quantization.spec.RangeCalculatorBase)

#### *classmethod* reconstruct_partial_qparams_calculator(partial_ctr, \*\*kwargs)

Return a new PartialConstructor whose qparams_calculator has attributes overridden.

The replacement wraps the existing `qparams_calculator` callable arg so that
each freshly constructed calculator has the given attributes set before it is
returned.  All overridden attributes (e.g. `float_range`, `qscheme`) are
plain instance attributes on `QParamsCalculatorBase` that are read lazily
during `forward()`, so post-construction mutation is safe as long as no
forward pass has run yet.

* **Parameters:**
  * **partial_ctr** (*PartialConstructor*) – The existing fake-quantizer partial to update.
  * **\*\*kwargs** (*Any*) – Attribute name/value pairs to set on the calculator instance.
* **Returns:**
  A new PartialConstructor whose qparams_calculator factory applies the
  overrides.
* **Return type:**
  *PartialConstructor*
