# coreai_opt.ExportBackend

### *class* coreai_opt.ExportBackend(value, \*args, \*\*kwargs)

Bases: `StrEnum`

Enum representing supported model export backends.

Each member is a string value representing the backend format.

* **Parameters:**
  * **value** (*object*)
  * **args** (*Any*)
  * **kwargs** (*Any*)
* **Return type:**
  Any

#### CoreML

Core ML format with compression metadata buffers.

#### CoreAI

Core AI format with custom ops.

#### \_\_init_\_(\*args, \*\*kwds)

#### MIL *: [ExportBackend](#coreai_opt.ExportBackend)* *= 'coreml'*

Deprecated. Use ExportBackend.CoreML instead.

#### MLIR *: [ExportBackend](#coreai_opt.ExportBackend)* *= 'coreai'*

Deprecated. Use ExportBackend.CoreAI instead.
