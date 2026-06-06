# coreai_opt.palettization.spec.PalettizationGranularity

### *class* coreai_opt.palettization.spec.PalettizationGranularity

Bases: `BaseModel`, `ConfigRegistryMixin`

Base class for palettization granularity specifications.

#### *abstract* get_blocks_to_cluster(weight)

Extract weight blocks to cluster based on the specified granularity.

* **Parameters:**
  **weight** (*Tensor*) – The weight tensor to split into blocks
* **Returns:**
  A list of weight tensor blocks. Each block is a view or slice of the
  original weight tensor based on the granularity configuration.
* **Raises:**
  **\_IncompatibleGranularityError** – If the tensor is incompatible with
      this granularity
* **Return type:**
  list[*Tensor*]

#### *abstract* num_blocks_to_cluster(weight)

Return the number of weight blocks to cluster based on the
specified granularity.

* **Parameters:**
  **weight** (*Tensor*) – The weight tensor to be palettized
* **Returns:**
  Number of LUTs for the weight tensor
* **Raises:**
  **\_IncompatibleGranularityError** – If the tensor is incompatible with
      this granularity
* **Return type:**
  int
