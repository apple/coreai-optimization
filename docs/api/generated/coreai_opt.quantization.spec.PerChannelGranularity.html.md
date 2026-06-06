# coreai_opt.quantization.spec.PerChannelGranularity

### *class* coreai_opt.quantization.spec.PerChannelGranularity

Bases: [`QuantizationGranularity`](coreai_opt.quantization.spec.QuantizationGranularity.md#coreai_opt.quantization.spec.QuantizationGranularity)

Per-channel quantization granularity.

This applies quantization to a specific channel which is selected through the
`axis` argument. When `axis` is `None` (the default), `Quantizer.prepare()`
automatically resolves it based on the module type for weight quantization.

Note: axis can be negatively indexed as per standard Python style indexing.
For example, with a block sizes list: [10, 20, 30], a valid set of axis include
-3 <= axis < 3
