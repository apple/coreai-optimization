# w8

### w8(\*, axis=None, execution_mode=<ExecutionMode.GRAPH: 'graph'>)

int8 weight-only quantization, per-channel symmetric.

* **Parameters:**
  * **axis** (*int* *|* *None*) – Channel axis for per-channel quantization.
    When `None` (default), the axis is auto-resolved based on the module type
    during quantization.
  * **execution_mode** ([*ExecutionMode*](coreai_opt.quantization.ExecutionMode.md#coreai_opt.quantization.ExecutionMode)) – Quantization execution mode.
    Defaults to `ExecutionMode.GRAPH`.
* **Returns:**
  int8 weight-only configuration.
* **Return type:**
  [QuantizerConfig](coreai_opt.quantization.QuantizerConfig.md#coreai_opt.quantization.QuantizerConfig)
