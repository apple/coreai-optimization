# coreai_opt.palettization.spec.PerGroupedChannelGranularity

### *class* coreai_opt.palettization.spec.PerGroupedChannelGranularity

Bases: [`PalettizationGranularity`](coreai_opt.palettization.spec.PalettizationGranularity.md#coreai_opt.palettization.spec.PalettizationGranularity)

Per-grouped-channel palettization granularity.

This applies palettization to a specific channel which is selected through the
`axis` argument. `axis` defaults to `None`, in which case the default
axis for the consuming op is used (e.g. 0 for `Linear`/`Conv`).

#### get_blocks_to_cluster(weight)

Split weight tensor into blocks along the specified axis with group_size.

* **Parameters:**
  **weight** (*Tensor*) – The weight tensor to split
* **Returns:**
  List of weight blocks, each of size group_size along the specified axis
* **Raises:**
  * **\_IncompatibleGranularityError** – If tensor is incompatible with
  * **this granularity** – 
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
