# coreai_opt.config.spec.CompressionSimulatorBase

### *class* coreai_opt.config.spec.CompressionSimulatorBase(\*args, \*\*kwargs)

Bases: `ClassRegistryMixin`, `Module`

Abstract base class for compression simulators.

This base class provides a common interface for all compression
simulators, regardless of the specific compression technique. The
compression simulator takes a tensor and applies the compression
technique on the tensor, while allowing the model to be evaluated.

Subclasses should implement the forward() method to define how the
compression simulation is performed during training.

* **Parameters:**
  * **args** (*Any*)
  * **kwargs** (*Any*)

#### \_\_init_\_(\*args, \*\*kwargs)

Initialize internal Module state, shared by both nn.Module and ScriptModule.

* **Parameters:**
  * **args** (*Any*)
  * **kwargs** (*Any*)
* **Return type:**
  None

### Methods

| [`forward`](#coreai_opt.config.spec.CompressionSimulatorBase.forward)(tensor)   | Apply compression simulation to the input tensor.   |
|---------------------------------------------------------------------------------|-----------------------------------------------------|
| `get_class`(key)                                                                |                                                     |
| `list_registry_keys`()                                                          |                                                     |
| `list_registry_values`()                                                        |                                                     |
| `register`(key)                                                                 | Register a virtual subclass of an ABC.              |

#### *abstract* forward(tensor)

Apply compression simulation to the input tensor.

This method should implement the differentiable approximation of
the compression operation. The exact behavior depends on the
specific compression technique.

* **Parameters:**
  **tensor** (*Tensor*) – Input tensor to compress
* **Returns:**
  Compressed tensor (or approximation thereof) with gradients
  flowing through
* **Return type:**
  *Tensor*
