# coreai_opt.quantization.spec.DynamicQParamsCalculator

### *class* coreai_opt.quantization.spec.DynamicQParamsCalculator(\*\*kwargs)

Bases: [`StatelessQParamsCalculatorBase`](coreai_opt.quantization.spec.StatelessQParamsCalculatorBase.md#coreai_opt.quantization.spec.StatelessQParamsCalculatorBase)

Dynamically computes scale/zero-point/minval from the current tensor every forward.

Typically used for activation quantization where activations vary per
inference and there is no calibration phase. Supports variable-shape scales
(e.g. LLM token-wise quantization with variable sequence length) since no
nn.Module buffers are allocated — see `StatelessQParamsCalculatorBase`
for the full stateless contract.

#### \_\_init_\_(\*\*kwargs)

### Methods

| `compute_qparams`(tensor, min_val, max_val)   | Given the observed min/max range, return `(scale, zero_point, minval)`.   |
|-----------------------------------------------|---------------------------------------------------------------------------|
| `forward`(tensor)                             | Compute qparams from `tensor` and return; no buffer state.                |
| `get_class`(key)                              |                                                                           |
| `list_registry_keys`()                        |                                                                           |
| `list_registry_values`()                      |                                                                           |
| `register`(key)                               | Register a virtual subclass of an ABC.                                    |
| `resolve`(data)                               | Resolve a string key or class type against this registry.                 |
| `set_export_mode`([enabled])                  |                                                                           |
