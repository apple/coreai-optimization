# coreai_opt.config.CompressionSpec

### *class* coreai_opt.config.CompressionSpec

Bases: `BaseModel`

Base class for compression specifications.

This class provides common infrastructure for all compression techniques
including quantization, palettization, pruning, and distillation.

All concrete compression specs (QuantizationSpec, PalettizationSpec, etc.)
should inherit from this base class and set the \_compression_type private
attribute to identify their compression type.

#### model_config

Pydantic configuration making specs immutable (frozen=True)
and rejecting extra fields (extra=”forbid”)

#### \_compression_type

Private attribute that must be set by subclasses to
identify the compression type

#### get_compression_type()

Return the type of compression this spec represents.

This method reads from the \_compression_type private attribute that
must be set by each concrete subclass.

* **Returns:**
  CompressionType enum value
* **Return type:**
  [*CompressionType*](coreai_opt.config.CompressionType.md#coreai_opt.config.CompressionType)
