# coreai_opt.CoreMLExportError

### *exception* coreai_opt.CoreMLExportError(message)

Bases: `ValueError`

Raised when a model cannot be exported to the CoreML backend.

* **Parameters:**
  **message** (*str*)
* **Return type:**
  None

#### *classmethod* from_config(config, context)

Build the error for an unsupported quantization config attribute (e.g. granularity).

* **Parameters:**
  * **config** (*object*)
  * **context** (*str*)
* **Return type:**
  [*CoreMLExportError*](#coreai_opt.CoreMLExportError)

#### *classmethod* from_dtype(dtype, context)

Build the error for an unsupported weight/activation/LUT dtype.

* **Parameters:**
  * **dtype** (*Any*)
  * **context** (*str*)
* **Return type:**
  [*CoreMLExportError*](#coreai_opt.CoreMLExportError)
