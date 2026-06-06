# coreai_opt.coreai_utils.quantize_weights

### coreai_opt.coreai_utils.quantize_weights(coreai_program, dtype, qscheme=QScheme.SYMMETRIC, granularity=CompressionGranularity.PER_CHANNEL, block_size=32, weight_num_threshold=1024, scale_dtype=None, in_place=False)

Quantize weights in a Core AI AIProgram (MLIR<CoreAI> IR) by using Core AI ops.

Walks through the IR and quantizes each coreai.constant op that needs to be
compressed. Only constants consumed by ops in `_OPS_WEIGHT_NEED_COMPRESSION`
are candidates; ops that fail to be quantized are skipped with a warning.

The `granularity` and `block_size` parameters determine the effective
`block_sizes` per axis (`0` means the full axis is one block):

For a 2-D linear weight `[C_out, C_in]`:

```text
|-------------------------------|--------------------------|
| Granularity                   | block_sizes              |
|-------------------------------|--------------------------|
| PER_TENSOR                    | [0, 0]                   |
| PER_CHANNEL                   | [1, 0]                   |
| PER_BLOCK(bs=32)              | [1, 32]                  |
|-------------------------------|--------------------------|
```

For a 4-D Conv weight `[C_out, C_in, KH, KW]`:

```text
|-------------------------------|--------------------------|
| Granularity                   | block_sizes              |
|-------------------------------|--------------------------|
| PER_TENSOR                    | [0, 0, 0, 0]             |
| PER_CHANNEL                   | [1, 0, 0, 0]             |
| PER_BLOCK(bs=32)              | [1, 32, 0, 0]            |
|-------------------------------|--------------------------|
```

* **Parameters:**
  * **coreai_program** (*AIProgram*) – The model to be quantized.
  * **dtype** ([*DType*](coreai_opt.coreai_utils.DType.md#coreai_opt.coreai_utils.DType)) – Target quantized data type (e.g. `DType.INT8`,
    `DType.INT4`, `DType.FP8_E4M3FN`, `DType.FP4_E2M1FN`).
  * **qscheme** ([*QScheme*](coreai_opt.coreai_utils.common.QScheme.md#coreai_opt.coreai_utils.common.QScheme)) – Quantization scheme. Use `QScheme.SYMMETRIC` or
    `QScheme.ASYMMETRIC`. FP dtypes only support
    `QScheme.SYMMETRIC`. Defaults to `QScheme.SYMMETRIC`.
  * **granularity** ([*CompressionGranularity*](coreai_opt.coreai_utils.CompressionGranularity.md#coreai_opt.coreai_utils.CompressionGranularity)) – Quantization granularity. Supports
    `CompressionGranularity.PER_TENSOR`,
    `CompressionGranularity.PER_CHANNEL`, and
    `CompressionGranularity.PER_BLOCK`.
    Defaults to `CompressionGranularity.PER_CHANNEL`.
  * **block_size** (*int*) – Block size applied to the input channel axis. Only
    effective when `granularity` is
    `CompressionGranularity.PER_BLOCK`. Defaults to `32`.
  * **weight_num_threshold** (*int*) – Threshold of weight element count to determine
    whether to compress a weight. Defaults to `1024`.
  * **scale_dtype** ([*DType*](coreai_opt.coreai_utils.DType.md#coreai_opt.coreai_utils.DType) *|* *None*) – Data type for the scale constants. Must be
    `None` for integer `dtype` values. Must be `None` for
    `DType.FP4_E2M1FN` (scale is always stored in `DType.FP8_E8M0FNU`
    internally). For FP8 `dtype` values, `None` (default) uses the
    uncompressed weight dtype (e.g. `f16` or `f32`) for the scale;
    `DType.FP8_E8M0FNU` stores the scale in the 8-bit E8M0FNU format
    (MXFP). Defaults to `None`.
  * **in_place** (*bool*) – Whether to quantize the model in-place. Defaults to
    `False`.
* **Returns:**
  A quantized Core AI program.
* **Return type:**
  AIProgram
* **Raises:**
  * **ValueError** – If `dtype` is not in the set of supported weight dtypes.
  * **ValueError** – If `dtype` is an FP dtype and `qscheme` is
        `QScheme.ASYMMETRIC`. FP quantization only supports symmetric mode.
  * **ValueError** – If `scale_dtype` is not `None` for an integer `dtype`.
  * **ValueError** – If `scale_dtype` is not `None` for `DType.FP4_E2M1FN`.
  * **ValueError** – If `dtype` is `DType.FP4_E2M1FN` and `granularity` is not
        `CompressionGranularity.PER_BLOCK` or `block_size` is not `32`.
        FP4 weights must use per-block quantization with a block size of 32
        to produce a valid MXFP4 encoding.
