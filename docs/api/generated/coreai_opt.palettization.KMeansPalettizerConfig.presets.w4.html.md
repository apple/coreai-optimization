# w4

### w4(\*, axis=0, group_size=16)

4-bit palettization, per-grouped-channel, group_size defaults to 16.

* **Parameters:**
  * **axis** (*int*) – Channel axis to group along. Defaults to 0.
  * **group_size** (*int*) – Number of channels per palette group.
* **Returns:**
  4-bit palettization configuration.
* **Return type:**
  [KMeansPalettizerConfig](coreai_opt.palettization.KMeansPalettizerConfig.md#coreai_opt.palettization.KMeansPalettizerConfig)
