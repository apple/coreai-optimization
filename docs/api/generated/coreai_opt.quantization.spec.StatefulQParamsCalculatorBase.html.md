# coreai_opt.quantization.spec.StatefulQParamsCalculatorBase

### *class* coreai_opt.quantization.spec.StatefulQParamsCalculatorBase(\*\*kwargs)

Bases: [`QParamsCalculatorBase`](coreai_opt.quantization.spec.QParamsCalculatorBase.md#coreai_opt.quantization.spec.QParamsCalculatorBase)

Stateful base: maintains scale/zero_point/minval as nn.Module buffers
across forwards.

Buffer shapes are allocated on first forward and must remain stable
(`copy_` requires shape compatibility) — use
`StatelessQParamsCalculatorBase` for variable-shape scales.

#### \_\_init_\_(\*\*kwargs)

### Methods

| `compute_qparams`(tensor, min_val, max_val)                                                                 | Given the observed min/max range, return `(scale, zero_point, minval)`.   |
|-------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| [`extra_repr`](#coreai_opt.quantization.spec.StatefulQParamsCalculatorBase.extra_repr)()                    | Return the extra representation of the module.                            |
| [`forward`](#coreai_opt.quantization.spec.StatefulQParamsCalculatorBase.forward)(tensor)                    | Compute qparams from `tensor`; cache to buffers; return.                  |
| `get_class`(key)                                                                                            |                                                                           |
| [`get_qparams`](#coreai_opt.quantization.spec.StatefulQParamsCalculatorBase.get_qparams)()                  | Return the computed scale, zero point and minval.                         |
| `list_registry_keys`()                                                                                      |                                                                           |
| `list_registry_values`()                                                                                    |                                                                           |
| `register`(key)                                                                                             | Register a virtual subclass of an ABC.                                    |
| `resolve`(data)                                                                                             | Resolve a string key or class type against this registry.                 |
| [`set_export_mode`](#coreai_opt.quantization.spec.StatefulQParamsCalculatorBase.set_export_mode)([enabled]) |                                                                           |

#### extra_repr()

Return the extra representation of the module.

To print customized extra information, you should re-implement
this method in your own modules. Both single-line and multi-line
strings are acceptable.

* **Return type:**
  str

#### forward(tensor)

Compute qparams from `tensor`; cache to buffers; return.

On the first forward pass, initializes internal buffers using the
observed tensor shape and device. Delegates the actual qparams
calculation to `compute_qparams`.

* **Parameters:**
  **tensor** (*Tensor*)
* **Return type:**
  tuple[*Tensor*, *Tensor* | None, *Tensor* | None]

#### get_qparams()

Return the computed scale, zero point and minval.
For FP4/FP8/floating-point quantization, zero_point and minval are None.

* **Return type:**
  tuple[*Tensor*, *Tensor* | None, *Tensor* | None]

#### set_export_mode(enabled=True)

* **Parameters:**
  **enabled** (*bool*)
* **Return type:**
  None

#### minval *: torch.Tensor | None*

#### scale *: torch.Tensor*

#### zero_point *: torch.Tensor | None*
