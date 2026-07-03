# coreai_opt.quantization.spec.StatelessQParamsCalculatorBase

### *class* coreai_opt.quantization.spec.StatelessQParamsCalculatorBase(\*\*kwargs)

Bases: [`QParamsCalculatorBase`](coreai_opt.quantization.spec.QParamsCalculatorBase.md#coreai_opt.quantization.spec.QParamsCalculatorBase)

Stateless base: no cached qparams; recomputed every forward.

Used for dynamic quantization where activations vary per inference and the
scale shape may change across forwards (e.g. LLM token-wise with variable
sequence length). `self.scale`/`zero_point`/`minval` are assigned in
forward as plain Python attributes (not buffers) for debugging visibility
only — they reflect the most recent forward and are not in `state_dict`.

- `FakeQuantizeImplBase` keeps `observer_enabled = 1` for this subclass
  so the recompute path stays live.
- `get_qparams` is undefined; `set_export_mode(True)` raises;
  `float_range=[None, None]` is required.

#### \_\_init_\_(\*\*kwargs)

### Methods

| `compute_qparams`(tensor, min_val, max_val)                                                                  | Given the observed min/max range, return `(scale, zero_point, minval)`.   |
|--------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| [`forward`](#coreai_opt.quantization.spec.StatelessQParamsCalculatorBase.forward)(tensor)                    | Compute qparams from `tensor` and return; no buffer state.                |
| `get_class`(key)                                                                                             |                                                                           |
| `list_registry_keys`()                                                                                       |                                                                           |
| `list_registry_values`()                                                                                     |                                                                           |
| `register`(key)                                                                                              | Register a virtual subclass of an ABC.                                    |
| `resolve`(data)                                                                                              | Resolve a string key or class type against this registry.                 |
| [`set_export_mode`](#coreai_opt.quantization.spec.StatelessQParamsCalculatorBase.set_export_mode)([enabled]) |                                                                           |

#### forward(tensor)

Compute qparams from `tensor` and return; no buffer state.

* **Parameters:**
  **tensor** (*Tensor*)
* **Return type:**
  tuple[*Tensor*, *Tensor* | None, *Tensor* | None]

#### set_export_mode(enabled=True)

* **Parameters:**
  **enabled** (*bool*)
* **Return type:**
  None
