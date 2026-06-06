# coreai_opt.palettization.spec.PerTensorGranularity

### *class* coreai_opt.palettization.spec.PerTensorGranularity

Bases: [`PalettizationGranularity`](coreai_opt.palettization.spec.PalettizationGranularity.md#coreai_opt.palettization.spec.PalettizationGranularity)

Per-tensor palettization granularity.

This applies palettization to the tensor as a whole.

#### get_blocks_to_cluster(weight)

For per-tensor granularity, return the entire tensor as a single block.

* **Parameters:**
  **weight** (*Tensor*) – The weight tensor
* **Returns:**
  List containing the single weight tensor block
* **Return type:**
  list[*Tensor*]

#### num_blocks_to_cluster(weight)

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
