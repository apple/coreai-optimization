# w8

### w8(\*, axis=None)

int8 weight-only quantization, per-channel symmetric.

* **Parameters:**
  **axis** (*int* *|* *None*) – Channel axis for per-channel quantization.
  When `None` (default), the axis is auto-resolved based on the module type
  during quantization.
* **Returns:**
  int8 weight-only module configuration.
* **Return type:**
  [ModuleQuantizerConfig](coreai_opt.quantization.ModuleQuantizerConfig.md#coreai_opt.quantization.ModuleQuantizerConfig)
