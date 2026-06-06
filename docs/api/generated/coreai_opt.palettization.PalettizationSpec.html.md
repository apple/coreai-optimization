# coreai_opt.palettization.PalettizationSpec

### *class* coreai_opt.palettization.PalettizationSpec

Bases: [`CompressionSpec`](coreai_opt.config.CompressionSpec.md#coreai_opt.config.CompressionSpec)

Specification for palettization compression of neural network weights.

Palettization is a compression technique that reduces memory usage by representing
weights using a lookup table (LUT) instead of storing full precision values.
Weights are clustered into a small number of representative values (the palette),
and each weight is replaced with an index into this palette.

This specification configures all aspects of the palettization process including
the number of bits for indices, the quantization of the lookup table,
and the granularity at which palettization is applied.

#### n_bits

Number of bits used for palette indices. Determines palette size
(2^n_bits entries). Must be one of {1, 2, 3, 4, 6, 8}. Default: 4.

#### lut_qspec

Quantization specification for the lookup table values.
If None, no quantization is applied to the LUT. When specified,
only `torch.int8`, `torch.uint8`, `torch.float8_e4m3fn`, and
`torch.float8_e5m2` dtypes are supported, and granularity must be
`PerTensorGranularity`. FP8 dtypes require symmetric quantization.
Default: None.

#### granularity

Defines how palettization is applied - per-tensor applies a
single palette to the entire tensor, per-channel applies separate
palettes to each channel. Default: PerTensorGranularity().

#### cluster_dim

The dimension of centroids for each lookup table.
The centroid is a scalar by default. When cluster_dim > 1, it indicates 2-D
clustering, and each cluster_dim length of weight vectors along the output
channel are palettized using the same 2-D centroid. The length of each entry
in the lookup tables is equal to cluster_dim. Default: 1.

#### enable_per_channel_scale

When set to True, weights are normalized along the
output channels using per-channel scales before being palettized.
Default: False.

#### model_dump_preserve_objects()

Custom model dump that preserves Pydantic BaseModel instances as objects
instead of serializing them.

This method creates a dictionary representation of the spec while keeping
all Pydantic BaseModel fields as the original Python objects rather than
serializing them to dictionaries. Non-Pydantic model fields are serialized
normally. This is useful when you want to work with actual object instances
programmatically.

* **Returns:**
  Dictionary with serialized non-Pydantic fields and preserved Pydantic objs.
* **Return type:**
  dict[str, *Any*]
