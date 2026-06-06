# coreai_opt.config.spec.CompressionComponentFactoryBase

### *class* coreai_opt.config.spec.CompressionComponentFactoryBase

Bases: `ABC`

Abstract base class for compression component factories.

This factory provides a generic interface for creating compression
components from CompressionSpec instances. Different compression
techniques (quantization, palettization, etc.) should extend this
base class to provide their specific implementations.

#### \_\_init_\_()

### Methods

| [`construct`](#coreai_opt.config.spec.CompressionComponentFactoryBase.construct)(spec, target)                 | Create a compression component instance from a CompressionSpec.          |
|----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| [`construct_partial`](#coreai_opt.config.spec.CompressionComponentFactoryBase.construct_partial)(spec, target) | Create a compression component partial object for deferred construction. |

#### *abstract classmethod* construct(spec, target)

Create a compression component instance from a CompressionSpec.

* **Parameters:**
  * **spec** ([*CompressionSpec*](coreai_opt.config.CompressionSpec.md#coreai_opt.config.CompressionSpec) *|* *None*) – CompressionSpec instance containing configuration
  * **target** ([*CompressionTargetTensor*](coreai_opt.config.spec.CompressionTargetTensor.md#coreai_opt.config.spec.CompressionTargetTensor)) – The target tensor for compression (weight or activation)
* **Returns:**
  CompressionSimulatorBase instance configured from the spec
* **Return type:**
  [*CompressionSimulatorBase*](coreai_opt.config.spec.CompressionSimulatorBase.md#coreai_opt.config.spec.CompressionSimulatorBase) | None

#### *abstract classmethod* construct_partial(spec, target)

Create a compression component partial object for deferred construction.

* **Parameters:**
  * **spec** ([*CompressionSpec*](coreai_opt.config.CompressionSpec.md#coreai_opt.config.CompressionSpec) *|* *None*) – CompressionSpec instance containing configuration
  * **target** ([*CompressionTargetTensor*](coreai_opt.config.spec.CompressionTargetTensor.md#coreai_opt.config.spec.CompressionTargetTensor)) – The target tensor for compression (weight or activation)
* **Returns:**
  A partial object that can be used for deferred
  : construction of CompressionSimulatorBase instances.
* **Return type:**
  PartialConstructor
