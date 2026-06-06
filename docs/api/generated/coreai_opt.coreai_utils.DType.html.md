# coreai_opt.coreai_utils.DType

### *class* coreai_opt.coreai_utils.DType(value, names=None, \*, module=None, qualname=None, type=None, start=1, boundary=None)

Bases: `StrEnum`

Enum representing data types for Core AI weight compression.

Each member’s string value matches the dtype string accepted by Core AI
compression utilities (e.g. `compression_types.string_to_builtin`).

#### INT2

Signed 2-bit integer.

#### UINT2

Unsigned 2-bit integer.

#### INT4

Signed 4-bit integer.

#### UINT4

Unsigned 4-bit integer.

#### INT8

Signed 8-bit integer.

#### UINT8

Unsigned 8-bit integer.

#### FP4_E2M1FN

4-bit floating-point (E2M1FN).

#### FP8_E4M3FN

8-bit floating-point (E4M3FN).

#### FP8_E5M2

8-bit floating-point (E5M2).

#### FP8_E8M0FNU

8-bit floating-point with 8 exponent bits, no mantissa, no sign
(E8M0FNU). Used as a scale dtype for FP4/FP8 quantization (MXFP format).

#### \_\_init_\_(\*args, \*\*kwds)

### Methods

| [`is_int`](#coreai_opt.coreai_utils.DType.is_int)()   | Return True if this dtype is an integer type.   |
|-------------------------------------------------------|-------------------------------------------------|

#### is_int()

Return True if this dtype is an integer type.

* **Return type:**
  bool
