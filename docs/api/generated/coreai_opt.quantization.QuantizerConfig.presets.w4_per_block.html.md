# w4_per_block

### w4_per_block(\*, block_size=32, axis=None, execution_mode=<ExecutionMode.GRAPH: 'graph'>)

int4 weight-only quantization, per-block symmetric, block_size defaults to 32.

* **Parameters:**
  * **block_size** (*int*) – Block size along the input channel dimension (default 32).
  * **axis** (*int* *|* *None*) – Axis to apply blocks along.
    When `None` (default), the axis is auto-resolved based on the module type
    during quantization.
  * **execution_mode** ([*ExecutionMode*](coreai_opt.quantization.ExecutionMode.md#coreai_opt.quantization.ExecutionMode)) – Quantization execution mode.
    Defaults to `ExecutionMode.GRAPH`.
* **Returns:**
  int4 per-block weight-only configuration.
* **Return type:**
  [QuantizerConfig](coreai_opt.quantization.QuantizerConfig.md#coreai_opt.quantization.QuantizerConfig)
