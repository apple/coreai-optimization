# coreai_opt.quantization.spec.QuantizationGranularity

### *class* coreai_opt.quantization.spec.QuantizationGranularity

Bases: `BaseModel`, `ConfigRegistryMixin`

Base class for quantization granularity specifications.

#### get_block_size(tensor_shape, quantization_target=CompressionTargetTensor.WEIGHT)

Get a list of block sizes based on the granularity.

* **Parameters:**
  * **tensor_shape** (*Size*) – Shape of the tensor being quantized.
  * **quantization_target** ([*CompressionTargetTensor*](coreai_opt.config.spec.CompressionTargetTensor.md#coreai_opt.config.spec.CompressionTargetTensor)) – Whether the tensor is a weight or an activation.
    Defaults to `WEIGHT`, which preserves the historical behavior.
* **Return type:**
  tuple[int, …]
