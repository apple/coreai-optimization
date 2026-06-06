# coreai_opt.config.CompressionType

### *class* coreai_opt.config.CompressionType(value, names=None, \*, module=None, qualname=None, type=None, start=1, boundary=None)

Bases: `StrEnum`

Enum representing compression techniques applied to the model.

Each member is a string value representing the compression type.

#### \_\_init_\_(\*args, \*\*kwds)

### Methods

| [`to_coreml_code`](#coreai_opt.config.CompressionType.to_coreml_code)()   | Convert to CoreML compression type code.   |
|---------------------------------------------------------------------------|--------------------------------------------|

#### to_coreml_code()

Convert to CoreML compression type code.

* **Returns:**
  CoreML-specific integer code for this compression type
* **Raises:**
  **ValueError** – If no CoreML code mapping exists for this compression type
* **Return type:**
  int
