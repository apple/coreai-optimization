# coreai_opt.coreai_utils.CompressionGranularity

### *class* coreai_opt.coreai_utils.CompressionGranularity(value, names=None, \*, module=None, qualname=None, type=None, start=1, boundary=None)

Bases: `StrEnum`

Enum representing the granularity of quantization for Core AI weight compression.

Each member’s string value matches the granularity string accepted by Core AI
compression passes.

#### PER_TENSOR

Single set of quantization parameters for the entire tensor.

#### PER_CHANNEL

Separate quantization parameters per individual axis. The targeted axis
is pre-defined by the type of operations.

#### PER_BLOCK

Separate quantization parameters per block of axes. The targeted axes
are pre-defined by the type of operations.

#### PER_GROUPED_CHANNEL

Separate quantization parameters per group of channels.

#### \_\_init_\_(\*args, \*\*kwds)
