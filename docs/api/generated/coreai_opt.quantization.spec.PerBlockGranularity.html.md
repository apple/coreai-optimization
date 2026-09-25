# coreai_opt.quantization.spec.PerBlockGranularity

### *class* coreai_opt.quantization.spec.PerBlockGranularity

Bases: [`QuantizationGranularity`](coreai_opt.quantization.spec.QuantizationGranularity.md#coreai_opt.quantization.spec.QuantizationGranularity)

Per-block quantization granularity.

This applies quantization to blocks of values within the tensor. Supports two modes:

1. Single-axis mode: Quantize blocks along one specific axis
   - `axis`: The axis to create blocks. May be negative (Python-style
     indexing). For weight quantization this is typically `0` or `1`
     (the channel axes); for activation quantization it is commonly the
     last / reduction axis (e.g. `-1`).
   - `block_size`: Integer specifying block size for that axis
2. Multi-axis mode: Create blocks across multiple axes simultaneously
   - `axis`: Must be None
   - `block_size`: Tuple specifying block size for each axis
     (-1 means no blocking)

In single-axis mode, when `axis` is `None` and `block_size` is an integer,
`Quantizer.prepare()` automatically resolves the axis based on the module type
for weight quantization.

Single-axis mode treats weights and activations differently:

- `WEIGHT`: only the two leading channel axes take part. Whichever of them
  is not the block axis collapses to `1` (one scale per slice), while
  trailing dimensions — e.g. conv kernel dims — keep their full size, so each
  block spans the whole kernel.
- `ACTIVATION`: every axis other than the block axis collapses to `1`, so
  the scale holds one entry per block *and* per position along all the other
  axes.

| Tensor shape (input)   | target     | axis   | block_size     | Shape of each block (output)   |
|------------------------|------------|--------|----------------|--------------------------------|
| [C_out, C_in]          | weight     | 1      | 32             | [1, 32]                        |
| [C_out, C_in]          | weight     | None   | (4, 8)         | [4, 8]                         |
| [C_out, C_in, KH, KW]  | weight     | 0      | 16             | [16, 1, KH, KW]                |
| [C_out, C_in, KH, KW]  | weight     | None   | (4, 16, 3, -1) | [4, 16, 3, KW]                 |
| [B, S, D]              | activation | -1     | 16             | [1, 1, 16]                     |
| [B, C, H, W]           | activation | 1      | 16             | [1, 16, 1, 1]                  |
