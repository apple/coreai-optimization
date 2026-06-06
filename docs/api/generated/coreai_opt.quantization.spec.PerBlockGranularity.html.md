# coreai_opt.quantization.spec.PerBlockGranularity

### *class* coreai_opt.quantization.spec.PerBlockGranularity

Bases: [`QuantizationGranularity`](coreai_opt.quantization.spec.QuantizationGranularity.md#coreai_opt.quantization.spec.QuantizationGranularity)

Per-block quantization granularity.

This applies quantization to blocks of values within the tensor. Supports two modes:

1. Single-axis mode: Quantize blocks along one specific axis
   (no blocking for axis>=2)
   - `axis`: The axis to create blocks (0 or 1)
   - `block_size`: Integer specifying block size for that axis
2. Multi-axis mode: Create blocks across multiple axes simultaneously
   - `axis`: Must be None
   - `block_size`: Tuple specifying block size for each axis
     (-1 means no blocking)

In single-axis mode, when `axis` is `None` and `block_size` is an integer,
`Quantizer.prepare()` automatically resolves the axis based on the module type
for weight quantization.

| Weight tensor shape (input)   | axis   | block_size     | Weight shape of each block (output)   |
|-------------------------------|--------|----------------|---------------------------------------|
| [C_out, C_in]                 | 1      | 32             | [1, 32]                               |
| [C_out, C_in]                 | None   | (4, 8)         | [4, 8]                                |
| [C_out, C_in, KH, KW]         | 0      | 16             | [16, 1, KH, KW]                       |
| [C_out, C_in, KH, KW]         | None   | (4, 16, 3, -1) | [4, 16, 3, KW]                        |
