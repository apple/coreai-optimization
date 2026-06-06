# coreai_opt.quantization.spec.QuantizationGranularity

### *class* coreai_opt.quantization.spec.QuantizationGranularity

Bases: `BaseModel`, `ConfigRegistryMixin`

Base class for quantization granularity specifications.

#### get_block_size(tensor_shape)

Get a list of block sizes based on the granularity.

* **Parameters:**
  **tensor_shape** (*Size*)
* **Return type:**
  tuple[int, …]
