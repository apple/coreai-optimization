# coreai_opt.quantization.spec.fake_quantize.FakeQuantizeImplBase

### *class* coreai_opt.quantization.spec.fake_quantize.FakeQuantizeImplBase(dtype, qscheme, qformulation, granularity, target_dtype, quant_min, quant_max, qparams_calculator, quantization_target, n_bits=None, \*\*kwargs)

Bases: [`CompressionSimulatorBase`](coreai_opt.config.spec.CompressionSimulatorBase.md#coreai_opt.config.spec.CompressionSimulatorBase), `FakeQuantizeBase`

Base class for implementing fake quantization

* **Parameters:**
  * **dtype** (*torch.dtype*)
  * **qscheme** ([*QuantizationScheme*](coreai_opt.quantization.spec.QuantizationScheme.md#coreai_opt.quantization.spec.QuantizationScheme))
  * **qformulation** ([*QuantizationFormulation*](coreai_opt.quantization.spec.QuantizationFormulation.md#coreai_opt.quantization.spec.QuantizationFormulation))
  * **granularity** ([*QuantizationGranularity*](coreai_opt.quantization.spec.QuantizationGranularity.md#coreai_opt.quantization.spec.QuantizationGranularity))
  * **target_dtype** (*torch.dtype*)
  * **quant_min** (*int* *|* *float*)
  * **quant_max** (*int* *|* *float*)
  * **qparams_calculator** ([*QParamsCalculatorBase*](coreai_opt.quantization.spec.QParamsCalculatorBase.md#coreai_opt.quantization.spec.QParamsCalculatorBase))
  * **quantization_target** ([*CompressionTargetTensor*](coreai_opt.config.spec.CompressionTargetTensor.md#coreai_opt.config.spec.CompressionTargetTensor))
  * **n_bits** (*int* *|* *None*)

#### \_\_init_\_(dtype, qscheme, qformulation, granularity, target_dtype, quant_min, quant_max, qparams_calculator, quantization_target, n_bits=None, \*\*kwargs)

Initialize internal Module state, shared by both nn.Module and ScriptModule.

* **Parameters:**
  * **dtype** (*dtype*)
  * **qscheme** ([*QuantizationScheme*](coreai_opt.quantization.spec.QuantizationScheme.md#coreai_opt.quantization.spec.QuantizationScheme))
  * **qformulation** ([*QuantizationFormulation*](coreai_opt.quantization.spec.QuantizationFormulation.md#coreai_opt.quantization.spec.QuantizationFormulation))
  * **granularity** ([*QuantizationGranularity*](coreai_opt.quantization.spec.QuantizationGranularity.md#coreai_opt.quantization.spec.QuantizationGranularity))
  * **target_dtype** (*dtype*)
  * **quant_min** (*int* *|* *float*)
  * **quant_max** (*int* *|* *float*)
  * **qparams_calculator** ([*QParamsCalculatorBase*](coreai_opt.quantization.spec.QParamsCalculatorBase.md#coreai_opt.quantization.spec.QParamsCalculatorBase))
  * **quantization_target** ([*CompressionTargetTensor*](coreai_opt.config.spec.CompressionTargetTensor.md#coreai_opt.config.spec.CompressionTargetTensor))
  * **n_bits** (*int* *|* *None*)

### Methods

| [`calculate_qparams`](#coreai_opt.quantization.spec.fake_quantize.FakeQuantizeImplBase.calculate_qparams)()                    | Returns the computed (scale, zero_point, minval).                                                                                                                                                                                   |
|--------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`convert`](#coreai_opt.quantization.spec.fake_quantize.FakeQuantizeImplBase.convert)(model, observer_node)                    | No-op: keep fake quant nodes intact during convert_pt2e.                                                                                                                                                                            |
| [`dequantize`](#coreai_opt.quantization.spec.fake_quantize.FakeQuantizeImplBase.dequantize)(tensor, scale, zero_point, minval) | Given a quantized tensor, the scale and zero point used to perform quantization, perform de-quantization of the tensor based on the configuration in the `QuantizationSpec` and return it as a tensor with dtype as `output_dtype`. |
| [`extra_repr`](#coreai_opt.quantization.spec.fake_quantize.FakeQuantizeImplBase.extra_repr)()                                  | Return the extra representation of the module.                                                                                                                                                                                      |
| [`forward`](#coreai_opt.quantization.spec.fake_quantize.FakeQuantizeImplBase.forward)(tensor)                                  | Performs fake quantization of the given tensor using the qparams (scale, zero point, minval) computed by the QParamsCalculator.                                                                                                     |
| `get_class`(key)                                                                                                               |                                                                                                                                                                                                                                     |
| [`is_disabled`](#coreai_opt.quantization.spec.fake_quantize.FakeQuantizeImplBase.is_disabled)()                                | Return True if fake quantization has been disabled.                                                                                                                                                                                 |
| `list_registry_keys`()                                                                                                         |                                                                                                                                                                                                                                     |
| `list_registry_values`()                                                                                                       |                                                                                                                                                                                                                                     |
| [`quantize`](#coreai_opt.quantization.spec.fake_quantize.FakeQuantizeImplBase.quantize)(tensor, scale, zero_point, minval)     | Given a tensor, scale and zero point, perform quantization of the tensor based on the configuration in the `QuantizationSpec`.                                                                                                      |
| `register`(key)                                                                                                                | Register a virtual subclass of an ABC.                                                                                                                                                                                              |
| [`set_export_mode`](#coreai_opt.quantization.spec.fake_quantize.FakeQuantizeImplBase.set_export_mode)([enabled])               | Set or unset export mode.                                                                                                                                                                                                           |
| [`with_args`](#coreai_opt.quantization.spec.fake_quantize.FakeQuantizeImplBase.with_args)(\*\*kwargs)                          |                                                                                                                                                                                                                                     |

#### calculate_qparams()

Returns the computed (scale, zero_point, minval).
`zero_point` and `minval` are None for floating-point dtypes.

* **Return type:**
  tuple[*Tensor*, *Tensor* | None, *Tensor* | None]

#### convert(model, observer_node)

No-op: keep fake quant nodes intact during convert_pt2e.

If this method is not present, torchao’s convert method will try to replace
fake quant nodes with its standard quantize/dequantize ops and fails in the process

* **Parameters:**
  * **model** (*GraphModule*)
  * **observer_node** (*Node*)
* **Return type:**
  None

#### *abstract* dequantize(tensor, scale, zero_point, minval, output_dtype=torch.float32)

Given a quantized tensor, the scale and zero point used to perform quantization,
perform de-quantization of the tensor based on the configuration in the
`QuantizationSpec` and return it as a tensor with dtype as `output_dtype`.

* **Parameters:**
  * **tensor** (*Tensor*) – The tensor to dequantize
  * **scale** (*Tensor*) – The scale to use for dequantization
  * **zero_point** (*Tensor* *|* *None*) – The zero point computed by the qparams calculator
    (None for floating-point dtypes).
  * **minval** (*Tensor* *|* *None*) – The minimum representable float value of the observed
    range, computed by the qparams calculator
    (None for floating-point dtypes).
  * **output_dtype** (*dtype*) – The dtype to use for the dequantized tensor
* **Return type:**
  *Tensor*

#### extra_repr()

Return the extra representation of the module.

To print customized extra information, you should re-implement
this method in your own modules. Both single-line and multi-line
strings are acceptable.

* **Return type:**
  str

#### forward(tensor)

Performs fake quantization of the given tensor using the qparams
(scale, zero point, minval) computed by the QParamsCalculator.

* **Parameters:**
  **tensor** (*Tensor*)
* **Return type:**
  *Tensor*

#### is_disabled()

Return True if fake quantization has been disabled.

* **Return type:**
  bool

#### *abstract* quantize(tensor, scale, zero_point, minval, cast_to_target_dtype=True)

Given a tensor, scale and zero point, perform quantization of the tensor based
on the configuration in the `QuantizationSpec`.

* **Parameters:**
  * **tensor** (*Tensor*) – The tensor to quantize
  * **scale** (*Tensor*) – The scale to use for quantization
  * **zero_point** (*Tensor* *|* *None*) – The zero point computed by the qparams calculator
    (None for floating-point dtypes).
  * **minval** (*Tensor* *|* *None*) – The minimum representable float value of the observed
    range, computed by the qparams calculator
    (None for floating-point dtypes).
  * **cast_to_target_dtype** (*bool*) – If True, the quantized tensor is cast to the target_dtype.
    Otherwise, the values of the tensor are quantized to appropriate bins but the dtype
    used to represent the quantized tensor remains the same as the original tensor.
    This allows fake quantization to capture the quantization error while allowing
    gradients to backpropagate.
* **Return type:**
  *Tensor*

#### set_export_mode(enabled=True)

Set or unset export mode.

* **Parameters:**
  **enabled** (*bool*)
* **Return type:**
  None

#### *classmethod* with_args(\*\*kwargs)

* **Parameters:**
  **kwargs** (*dict*)
* **Return type:**
  *PartialConstructor*[[*FakeQuantizeImplBase*](#coreai_opt.quantization.spec.fake_quantize.FakeQuantizeImplBase)]

#### *property* granularity *: [QuantizationGranularity](coreai_opt.quantization.spec.QuantizationGranularity.md#coreai_opt.quantization.spec.QuantizationGranularity)*

Getter for granularity.
