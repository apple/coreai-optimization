# coreai_opt.quantization.QuantizationSpec

### *class* coreai_opt.quantization.QuantizationSpec

Bases: [`CompressionSpec`](coreai_opt.config.CompressionSpec.md#coreai_opt.config.CompressionSpec)

Specification for quantizing tensors in neural networks.

This class defines all the parameters needed to quantize a tensor, including the
target data type, quantization scheme, granularity, and the algorithms used for fake
quantization, quantization parameter calculation, and range calculation.

#### dtype

Target data type for quantization.
Valid inputs:

- Integer dtypes: torch.int8, torch.uint8, torch.int4, torch.uint4, etc.
- Floating-point dtypes: torch.float8_e4m3fn, torch.float8_e5m2,
  torch.float4_e2m1fn_x2.
  For FP8 dtypes, the notation specifies the format (e.g., in
  torch.float8_e4m3fn, ‘e4m3’ indicates 4 exponent bits and 3
  mantissa bits, ‘f’ stands for finite values only, and ‘n’ stands
  for non-standard NaN representation). For more details on FP8
  dtypes, see [https://arxiv.org/pdf/2209.05433](https://arxiv.org/pdf/2209.05433)
- String names: “int8”, “int4”, “float8_e4m3fn”, etc. Must correspond to
  an existing torch dtype

Default: torch.int8

* **Type:**
  str | torch.dtype

#### qscheme

Quantization scheme determining how values are mapped to the quantized
range.
Valid inputs:

> - “symmetric” (default), “symmetric_with_clipping”, “asymmetric”

On how it affects the quantization and dequantization formulae,
please refer to the qformulation description below.

* **Type:**
  str | coreai_opt.quantization.QuantizationScheme

#### qformulation

Quantization formula determining how values are mapped between the
quantized and dequantized domains.
Valid inputs:

- `"zp"` (default), `"minval"`
- `QuantizationFormulation.ZP`, `QuantizationFormulation.MINVAL`

Notation used in the formulae below:

- `x`: unquantized data.
- `q`: quantized data (dtype as specified by `QuantizationSpec.dtype`).
- `x'`: dequantized data (same dtype as `x`).
- `scale`: for INT quantization, defaults to the same dtype as
  `x`. For FP quantization, see the `scale_dtype` description
  below.

Formulae:

- `"zp"` — Zero-point formulation. `zero_point` has the same
  dtype as `q`.
  - `q  = clamp(round(x / scale) + zero_point, quant_min, quant_max)`
  - `x' = (q - zero_point) * scale`
- `"minval"` — Min-value formulation. `minval` has the same
  dtype as `x`.
  - `q  = clamp(round((x - minval) / scale) + quant_min, quant_min, quant_max)`
  - `x' = (q - quant_min) * scale + minval`

Default: `QuantizationFormulation.ZP`

The tables below illustrate how the joint settings across
`QuantizationSpec.dtype`, `QuantizationSpec.qscheme`,
`QuantizationSpec.qformulation` manifest in the formulae above.
(Note that the min and max values of “x” assumed below are the ones
which will be calculated based on observer settings, as specified in
`QuantizationSpec.qparam_calculator_cls`,
`QuantizationSpec.range_calculator_cls`,
`QuantizationSpec.float_range`.)

Derived quantities used in the tables:

- `max_abs     = max(|x|)`
- `max_val_pos = max(0, max(x))`
- `min_val_neg = min(0, min(x))`
- `range       = max_val_pos - min_val_neg`

For per-channel / per-block granularity, the reductions above are
taken over each quantization unit (channel slice or block) rather
than the full tensor.

**ZP formulation**, e.g. with 8 bit signed and unsigned fixed point types:

| dtype   | qscheme    | quant range   | scale           | zero_point                                          |
|---------|------------|---------------|-----------------|-----------------------------------------------------|
| INT8    | SYMMETRIC  | [-128, 127]   | max_abs / 127.5 | 0                                                   |
| INT8    | SYM_W_CLIP | [-127, 127]   | max_abs / 127   | 0                                                   |
| INT8    | ASYMMETRIC | [-128, 127]   | range / 255     | clip(-128-round(<br/>min_val_neg/scale), -128, 127) |
| UINT8   | SYMMETRIC  | [0, 255]      | max_abs / 127.5 | 128                                                 |
| UINT8   | SYM_W_CLIP | [0, 255]      | max_abs / 127.5 | 128                                                 |
| UINT8   | ASYMMETRIC | [0, 255]      | range / 255     | clip(-round(<br/>min_val_neg/scale), 0, 255)        |

And for FP4/FP8 dtypes, zero-point is always set to 0 (FP supports only the
symmetric qscheme). The scale formula depends on `scale_dtype`:

- `scale_dtype=None` (FP8 only): `scale = max_abs / fp_max`, where
  `fp_max` is the largest representable value for the target FP dtype
  (448.0 for FP8 E4M3, 57344.0 for FP8 E5M2).
- `scale_dtype=torch.float8_e8m0fnu` (FP4 and FP8):
  power-of-2 scale per OCP MX spec —
  `scale = 2^(floor(log2(max_abs)) - target_max_pow2)`, with
  `target_max_pow2` of 2 for FP4 E2M1, 8 for FP8 E4M3, 15 for FP8 E5M2.

**MINVAL formulation**, e.g. with 8 bit signed and unsigned fixed point types:

| dtype   | qscheme    | quant range   | scale           | minval      |   quant_offset |
|---------|------------|---------------|-----------------|-------------|----------------|
| INT8    | SYMMETRIC  | [-128, 127]   | max_abs / 127.5 | -max_abs    |           -128 |
| INT8    | SYM_W_CLIP | [-127, 127]   | max_abs / 127   | -max_abs    |           -127 |
| INT8    | ASYMMETRIC | [-128, 127]   | range / 255     | min_val_neg |           -128 |
| UINT8   | SYMMETRIC  | [0, 255]      | max_abs / 127.5 | -max_abs    |              0 |
| UINT8   | SYM_W_CLIP | [0, 255]      | max_abs / 127.5 | -max_abs    |              0 |
| UINT8   | ASYMMETRIC | [0, 255]      | range / 255     | min_val_neg |              0 |

`quant_offset` equals `q_min` (the lower bound of the “quant range” column).

This formulation is not allowed with FP4/FP8 dtypes.

#### NOTE
Export-backend constraints:

- CoreML export only supports `ZP`. Specs with `qformulation=MINVAL`
  are rejected during finalize with CoreML Export-backend.
- CoreAI export supports both `ZP` and `MINVAL`.

* **Type:**
  str | coreai_opt.quantization.QuantizationFormulation

#### granularity

Quantization granularity determining the scope of
quantization parameters.
Valid inputs:

- Dictionary format:
  - `{"type": "per_tensor"}` - Single scale/zero-point for entire
    tensor
  - `{"type": "per_channel", "axis": <int>}` - Per-channel
    quantization along axis
  - `{"type": "per_block", "axis": <int>, "block_size": <tuple>}` -
    Block-wise quantization along axis with specified block size
- coreai_opt.quantization.QuantizationGranularity instances:
  PerTensorGranularity(), PerChannelGranularity(axis=1), etc.

Default: PerTensorGranularity()

* **Type:**
  dict | coreai_opt.quantization.QuantizationGranularity

#### fake_quantize_cls

Fake quantization implementation class for simulating quantization.
This entity makes use of the scale and zero point computed from
qparam_calculator_cls in order to perform fake quantization (back to back
quantize/dequantize) to simulate quantization by adding quantization error
to tensors in the model.
Users may define their own fake_quantize_cls by inheriting from
coreai_opt.quantization.fake_quantize.FakeQuantizeImplBase and register
the class using the decorator
@FakeQuantizeImplBase.register(“<identifier>”), where <identifier> is a
string which can be used to refer to the registered class in
fake_quantize_cls.
Valid inputs:

- String key: “default” or custom registered class string name
- Class type:
  coreai_opt.quantization.fake_quantize._DefaultFakeQuantizeImpl
  or custom registered class type

Default: “default”

* **Type:**
  str | type[coreai_opt.quantization.fake_quantize.FakeQuantizeImplBase]

#### qparam_calculator_cls

(str | type[QParamsCalculatorBase]):
Algorithm for calculating quantization parameters (scale and zero
point).
Users may define their own qparam_calculator_cls by inheriting from
coreai_opt.quantization.qparams_calculator.QParamsCalculatorBase
and register the class using the decorator
@QParamsCalculatorBase.register(“<identifier>”), where
<identifier> is a string which can be used to refer to the
registered class in qparam_calculator_cls.
If float_range is provided, the “default”, “static”, and
“moving_average” qparam calculators will take it into account when
computing scale and zero point.
Valid inputs:

- “default”: Context-aware default:
  * For weights → StaticQParamsCalculator
  * For activations → MovingAverageQParamsCalculator
- “static”: Direct min/max quantization parameter calculation based on
  most recent calibration sample only
- “moving_average”: Uses exponential moving average for stability
- “global_minmax”: Tracks running min/max across all calibration samples
- Custom registered class string name
- coreai_opt.quantization.qparams_calculator.QParamsCalculatorBase
  class type: StaticQParamsCalculator,
  MovingAverageQParamsCalculator, or custom registered class type

Default: “default”

#### range_calculator_cls

(str | type[RangeCalculatorBase]):
Algorithm for calculating the min/max range of values to quantize.
Users may define their own range_calculator_cls by inheriting from
coreai_opt.quantization.range_calculator.RangeCalculatorBase and
register the class using the decorator
@RangeCalculatorBase.register(“identifier”), where <identifier>
is a string which can be used to refer to the registered class in
range_calculator_cls.
Valid inputs:

- “minmax”: Uses actual min/max values from the tensor
- Custom registered class string name
- coreai_opt.quantization.range_calculator.RangeCalculatorBase
  class type: MinMaxRangeCalculator or custom registered class type

Default: “minmax”

#### float_range

Custom floating-point
range [min, max] to set for quantization.
This can be used to set ranges for functions with known bounds (ReLU, Tanh,
Sigmoid, Softmax, etc.) as well as constraining certain tensors in the model
to be within a specified range if users want to exclude outliers.
float_range is used by qparams_calculator_cls. Predefined qparam classes
“default”, “static”, and “moving_average” handle float_range. If the
user defines a custom qparam_calculator_cls, float_range would need to be
handled properly within the implementation.
Default: [None, None] (no constraints, allow qparam_calculator_cls
to determine range)
Valid inputs:

- [None, None]: No range constraints (default)
- [None, float_max]: Fix float max while allowing float min to be
  determined
- [float_min, None]: Fix float min while allowing float max to be
  determined
- [float_min, float_max]: Fix both float min and max to a specific
  range

Constraints:

- Must be a list or tuple of length 2
- float_min must be <= 0
- float_max must be >= 0
- float_min < float_max

* **Type:**
  list[float | int | None]

#### scale_dtype

Data type for quantization scale factors.
Controls whether scales are constrained to power-of-2 values (e8m0 format)
or allowed to be arbitrary floating-point values.
Valid inputs:

- None: Use default scale computation via torchao’s
  choose_qparams_affine_with_min_max (integer and FP8 dtypes).
  For FP4, None is resolved to torch.float8_e8m0fnu automatically.
- torch.float8_e8m0fnu: Power-of-2 scales following OCP Microscaling (MX)
  spec. Required for FP4 quantization, optional for FP8.

Constraints:

- FP4 (float4_e2m1fn_x2): scale_dtype must be torch.float8_e8m0fnu or None
  (defaults to e8m0)
- FP8 (float8_e4m3fn, float8_e5m2): scale_dtype must be
  torch.float8_e8m0fnu or None (defaults to None)
- Integer dtypes: scale_dtype must be None (defaults to None)

Default: None

* **Type:**
  torch.dtype | None

#### get_extra_args()

Automatically detect and return fields beyond base QuantizationSpec.

This method introspects the current instance to find any additional fields
that have been added in subclasses, allowing the factory to automatically
pass them to component constructors.

* **Returns:**
  Dictionary of extra field names and their values
* **Return type:**
  Dict[str, Any]

#### *classmethod* get_n_bits_from_dtype(dtype)

Extract the number of bits from a torch dtype.

* **Parameters:**
  **dtype** (*dtype*) – The torch dtype to extract bits from
* **Returns:**
  Number of bits for the dtype
* **Raises:**
  **RuntimeError** – If unable to extract bits from the dtype
* **Return type:**
  int

#### *classmethod* get_quant_range(dtype, qscheme)

Calculate quantization range (quant_min, quant_max) for the given
dtype and scheme.

* **Parameters:**
  * **dtype** (*dtype*) – The quantization dtype
  * **qscheme** ([*QuantizationScheme*](coreai_opt.quantization.spec.QuantizationScheme.md#coreai_opt.quantization.spec.QuantizationScheme)) – The quantization scheme (symmetric, asymmetric, etc.)
* **Returns:**
  Tuple of (quant_min, quant_max) values. Returns floats for
  floating-point dtypes and ints for integer dtypes.
* **Return type:**
  tuple[int | float, int | float]

#### *classmethod* get_target_dtype(dtype)

Returns the target dtype for quantization, mapping custom dtypes
to concrete ones.

Custom integer dtypes (int1-int7, uint1-uint7) are mapped to their 8-bit
equivalents, since PyTorch doesn’t have native support for
sub-byte integer types.

FP4 (float4_e2m1fn_x2) is mapped to float8_e4m3fn, since
PyTorch support is minimal for float4_e2m1fn_x2. All FP4
representable values are exactly representable in FP8.

* **Parameters:**
  **dtype** (*dtype*) – The source dtype
* **Returns:**
  - int1, int2, …, int7 → int8
  - uint1, uint2, …, uint7 → uint8
  - float4_e2m1fn_x2 → float8_e4m3fn
  - int8, uint8, float16, float32, etc. → unchanged
* **Return type:**
  The target dtype for quantization
