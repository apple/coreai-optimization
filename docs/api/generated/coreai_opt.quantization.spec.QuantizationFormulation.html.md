# coreai_opt.quantization.spec.QuantizationFormulation

### *class* coreai_opt.quantization.spec.QuantizationFormulation(value, names=None, \*, module=None, qualname=None, type=None, start=1, boundary=None)

Bases: `StrEnum`

Formula used to map between quantized integers and dequantized values.

#### ZP

Standard zero-point formulation.

- `q  = clamp(round(x / scale) + zero_point, quant_min, quant_max)`
- `x' = (q - zero_point) * scale`

#### MINVAL

Min-value formulation.

- `q  = clamp(round((x - minval) / scale) + quant_min, quant_min, quant_max)`
- `x' = (q - quant_min) * scale + minval`

#### \_\_init_\_(\*args, \*\*kwds)
