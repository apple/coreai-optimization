# coreai_opt.coreai_utils.sparsify_weights

### coreai_opt.coreai_utils.sparsify_weights(coreai_program, target_sparsity=0.5, block_size=None, n_m_ratio=None, quantize_dtype=None, palettize_nbits=None, weight_num_threshold=1024, in_place=False)

Sparsify weights in a Core AI AIProgram (MLIR<CoreAI> IR) by using Core AI ops.

Walks through the IR and sparsifies each coreai.constant op that needs to be
compressed. Only constants consumed by ops in `_OPS_WEIGHT_NEED_COMPRESSION`
are candidates; ops that fail to be sparsified are skipped with a warning.

* **Parameters:**
  * **coreai_program** (*AIProgram*) – The model to be sparsified.
  * **target_sparsity** (*float* *|* *None*) – Percentage of sparsity in `[0, 1]`.
    `n` lowest absolute weight values are set to zero, where
    `n = floor(size * target_sparsity)`. Mutually exclusive with
    `n_m_ratio`. Defaults to `0.5`.
  * **block_size** (*int* *|* *None*) – Block size for block sparsity along the output
    channel dimension. Only applied to `linear` and `conv` layers.
    If set, must be greater than `1`. Defaults to `None`.
  * **n_m_ratio** (*tuple* *[**int* *,* *int* *]*  *|* *None*) – `(n, m)` ratio for n:m structured
    pruning along the input channel axis. Out of every `m` elements,
    the `n` with lowest magnitude are set to zero. Only applied to
    `linear` and `conv` layers. Mutually exclusive with
    `target_sparsity`. Defaults to `None`.
  * **quantize_dtype** ([*DType*](coreai_opt.coreai_utils.DType.md#coreai_opt.coreai_utils.DType) *|* *None*) – Data type for storing non-zero values (joint
    compression). Must be `None`, `DType.INT8`, `DType.UINT8`,
    `DType.FP8_E4M3FN`, or `DType.FP8_E5M2`. When set, non-zero values
    are quantized and a `coreai.blockwise_shift_scale` op dequantizes them
    back. Cannot be used with `palettize_nbits`. Defaults to `None`.
  * **palettize_nbits** (*int* *|* *None*) – Number of bits for palettizing non-zero values.
    When set, non-zero values are palettized using k-means with
    `2**palettize_nbits` clusters. Valid values: `{1, 2, 3, 4, 6, 8}`.
    Cannot be used with `quantize_dtype`. Defaults to `None`.
  * **weight_num_threshold** (*int*) – Minimum weight element count required to
    compress a weight. Defaults to `1024`.
  * **in_place** (*bool*) – Whether to sparsify the model in-place. Defaults to
    `False`.
* **Returns:**
  A sparsified Core AI program.
* **Return type:**
  AIProgram
* **Raises:**
  * **ValueError** – If both `target_sparsity` and `n_m_ratio` are set.
  * **ValueError** – If neither `target_sparsity` nor `n_m_ratio` is set.
  * **ValueError** – If both `quantize_dtype` and `palettize_nbits` are set.
  * **ValueError** – If `quantize_dtype` is not `None`, `DType.INT8`,
        `DType.UINT8`, `DType.FP8_E4M3FN`, or `DType.FP8_E5M2`.
  * **ValueError** – If `palettize_nbits` is not in `{1, 2, 3, 4, 6, 8}`.
  * **ValueError** – If `block_size` is set and not greater than `1`.
  * **ValueError** – If `n_m_ratio` does not have length 2, contains non-integers,
        has `m <= 0`, or has `n` outside `[0, m]`.
