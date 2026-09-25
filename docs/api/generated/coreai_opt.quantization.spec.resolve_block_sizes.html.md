# coreai_opt.quantization.spec.resolve_block_sizes

### coreai_opt.quantization.spec.resolve_block_sizes(tensor_shape, block_sizes)

Resolve a per-axis block partition against the shape it applies to.

`-1` takes the whole axis as one block, which is how per-channel and per-tensor are
spelled as a partition. Every other entry blocks its axis and must divide it.

* **Parameters:**
  * **tensor_shape** (*Sequence* *[**int* *]*) – Shape of the tensor being partitioned.
  * **block_sizes** (*Sequence* *[**int* *]*) – Block extent per axis, one entry per dimension.
* **Returns:**
  The resolved extents, with each `-1` replaced by its full dimension.
* **Return type:**
  list[int]
* **Raises:**
  * **ValueError** – If the two lengths differ, or an entry is neither positive nor `-1`.
  * **\_BlockSizeMismatchError** – If an entry does not divide its dimension.
