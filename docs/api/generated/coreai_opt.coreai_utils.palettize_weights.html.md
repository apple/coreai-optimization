# coreai_opt.coreai_utils.palettize_weights

### coreai_opt.coreai_utils.palettize_weights(coreai_program, lut_dtype, n_bits=4, granularity=CompressionGranularity.PER_TENSOR, group_size=32, cluster_dim=1, enable_per_channel_scale=False, weight_num_threshold=1024, num_kmeans_workers=4, enable_fast_kmeans_mode=True, rounding_precision=4, in_place=False)

Palettize weights in a Core AI AIProgram (MLIR<CoreAI> IR) by using Core AI ops.

Walks through the IR and palettizes each coreai.constant op that needs to be
compressed. Only constants consumed by ops in `_OPS_WEIGHT_NEED_COMPRESSION`
are candidates; ops that fail to be palettized are skipped with a warning.

* **Parameters:**
  * **coreai_program** (*AIProgram*) – The model to be palettized.
  * **lut_dtype** ([*DType*](coreai_opt.coreai_utils.DType.md#coreai_opt.coreai_utils.DType) *|* *None*) – The datatype for values in the look-up table.
    Can be `None` (no LUT quantization), `DType.INT8`,
    `DType.UINT8`, `DType.FP8_E4M3FN`, or `DType.FP8_E5M2`.
    Symmetric quantization is used by default. Defaults to `None`.
  * **n_bits** (*int*) – Number of bits for palettizing the weights. Defaults to `4`.
    A LUT will have `2**n_bits` entries; n_bits must be in `{1, 2, 3, 4, 6, 8}`.
  * **granularity** ([*CompressionGranularity*](coreai_opt.coreai_utils.CompressionGranularity.md#coreai_opt.coreai_utils.CompressionGranularity)) – Quantization granularity. Supports
    `CompressionGranularity.PER_TENSOR`,
    `CompressionGranularity.PER_CHANNEL`, and
    `CompressionGranularity.PER_GROUPED_CHANNEL`.
    Defaults to `CompressionGranularity.PER_TENSOR`.
  * **group_size** (*int*) – Number of channels in a group. Only effective when
    granularity is `CompressionGranularity.PER_GROUPED_CHANNEL`.
  * **cluster_dim** (*int*) – Dimension of centroids for each lookup table. When
    `cluster_dim > 1`, it indicates 2-D clustering. Defaults to 1
    (scalar palettization).
  * **enable_per_channel_scale** (*bool*) – When `True`, weights are normalized along
    output channels using per-channel scales before palettization. Not
    supported with `cluster_dim > 1`.
  * **weight_num_threshold** (*int*) – Threshold of weight element count to determine
    whether to compress a weight.
  * **num_kmeans_workers** (*int*) – Number of worker processes for k-means. Defaults
    to 4.
  * **enable_fast_kmeans_mode** (*bool*) – Whether to use weight rounding to speed up
    k-means. Weight rounding reduces precision but speeds up k-means clustering.
    Defaults to `True`.
  * **rounding_precision** (*int*) – Number of decimal places to round weights to during
    fast K-means clustering. Only effective when `enable_fast_kmeans_mode`
    is `True`. Defaults to `4`.
  * **in_place** (*bool*) – Whether to palettize the model in-place. Defaults to
    `False`.
* **Returns:**
  A palettized Core AI program.
* **Return type:**
  AIProgram
