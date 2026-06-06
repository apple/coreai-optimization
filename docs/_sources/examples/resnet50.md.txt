# ResNet50 model: Weight + Activation Quantization (PTQ)

In this article we will experiment with a few knobs for quantizing a model for both weight and activation quantization.
In particular we learn how to quantize using `INT8` and `FP8` quantization, as well as explore the effects that `qparams_calculator` has on determining quantization parameters and therefore affecting accuracy.

The reported metrics in this article come from applying the described quantization configurations to a pretrained [ResNet50](https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50) model from torchvision.
Data samples used are obtained from [imagenette](https://github.com/fastai/imagenette).
For evaluation, 128 samples are used.

As a reference, the pretrained FP32 model gives an eval accuracy of `78.12%`.

## W_INT8_A_INT8 quantization

### Config 1: W_INT8_A_INT8 (activation quantization with `moving_average` qparams calculator)

We begin by applying the below quantization configuration for the model.

| Tensor     | dtype  | Granularity | qparam calculator                                     |
| ---------- | ------ | ----------- | ----------------------------------------------------- |
| Weight     | `int8` | per-channel | `"static"` — computed directly from the weight tensor |
| Activation | `int8` | per-tensor  | `"moving_average"` — EMA of observed min/max          |

We instantiate a config using `QuantizerConfig()`. If no specific settings are used, the quantization configuration will default to the above settings.

```python
from coreai_opt.quantization import QuantizerConfig

config = QuantizerConfig()
```

#### Prepare

`quantizer.prepare()` inserts fake-quantize modules into the graph.
If weight-only quantization were applied, the model would be ready for evaluation after preparation.
However, as we are also using activation quantization, we must calibrate the prepared model first.

**Note**: When performing activation quantization, the example inputs used to prepare the model should be representative of the data the model would typically see.
This is due to the example inputs also serving as a starting point for quantization parameters in activation quantizers.

Further calibration is still required; however, using even a single representative data sample point as the example input can reduce the number of calibration samples needed later during calibration.

```python
from coreai_opt.quantization import Quantizer

# Instantiate the quantizer
quantizer = Quantizer(fp32_model, config)

# Prepare the model using a single representative data sample
prepared_model = quantizer.prepare(example_inputs)
```

#### Calibrate

Calibration is necessary when activation quantization is enabled. In order to determine proper quantization parameters for activation quantizers, representative data must be passed through the prepared model in the `calibration_mode()` context.

Inside `calibration_mode()`, fake quantization is disabled and observers track tensor ranges seen at each activation quantizer.
Each forward pass updates the activation scales without introducing quantization noise into the output.
On exit, observers are disabled and fake quantization is re-enabled.

```python
with quantizer.calibration_mode():
    with torch.no_grad():
        for batch in tqdm(calibration_dataloader):
            prepared_model(batch)
```

At this point, the model is ready for evaluation using the user's evaluation pipeline and dataset.
After calibrating with 896 samples outside of the evaluation dataset, we get an accuracy of `74.22%`.

Users can experiment with a different number of calibration samples - the optimal amount is model and dataset dependent.
Past a certain point, increasing the number of calibration samples will give diminishing returns.

In the next section, we demonstrate using a different `qparams_calculator` to try improving the accuracy.

### Config 2: W_INT8_A_INT8 (activation quantization with `global_minmax` qparams calculator)

For activation quantization, the API currently offers the choice of `moving_average` and `global_minmax` `qparams_calculator` for how quantization parameters are computed.

- `moving_average` computes the range as a running exponential moving average of the per-batch min/max, smoothing out transient spikes.
  An additional `averaging_constant` parameter allows users to set how sensitive the moving average is to each new batch.
- `global_minmax` sets the quantization range to the global min and max observed across all calibration batches.
  It captures the full activation range but can be sensitive to outliers.

Which calculator gives better accuracy is model-dependent, so it is worth trying both.

For weight quantization, since weights are static, `static` qparams calculator should always be used.

**Note**: When defining a `QuantizationSpec`, if `qparam_calculator_cls` is left unset, it will default to `moving_average` if used as an activation spec and `static` if used as a weight spec.

Users may also define their own `qparams_calculator` with different ways for computing quantization parameters.
Some common examples in literature include `torchao`'s [HistogramObserver](https://docs.pytorch.org/docs/2.12/generated/torch.ao.quantization.observer.HistogramObserver.html), percentile-based observers, etc.

To implement a custom `qparams_calculator`, users should extend the `QParamsCalculatorBase` class as necessary and register their custom class using `@QParamsCalculatorBase.register("<some_name>")` with a string name identifier of their choosing.

Below we define a quantization config using `global_minmax` for activation quantizers.
Weight quantizers continue to use the `static` qparams calculator.

| Tensor     | dtype  | Granularity | qparam calculator                                       |
| ---------- | ------ | ----------- | ------------------------------------------------------- |
| Weight     | `int8` | per-channel | `"static"` — computed directly from the weight tensor   |
| Activation | `int8` | per-tensor  | `"global_minmax"` — Absolute min/max from observed data |

```python
from coreai_opt.quantization import ModuleQuantizerConfig
from coreai_opt.quantization.spec import QuantizationSpec

activation_spec = QuantizationSpec(qparam_calculator_cls="global_minmax")

global_config = ModuleQuantizerConfig(
    op_input_spec={"*": activation_spec},
    op_output_spec={"*": activation_spec},
)

config = QuantizerConfig(global_config=global_config)
```

```python
# Instantiate the quantizer
quantizer = Quantizer(fp32_model, config)

# Prepare the model
prepared_model = quantizer.prepare(example_inputs)

# Calibrate the model
calibrate(quantizer, prepared_model)
```

Using `global_minmax`, we get an eval accuracy of `75.78%`. For this model, tuning the `qparams_calculator` hyperparameter allows us to achieve better accuracy.

## W_FP8_A_FP8 quantization

Next we try quantizing the model where both weights and activations are quantized to 8-bit floating-point (`float8_e4m3fn`) to see how the accuracy changes.
`FP8` is natively symmetric (zero-point = 0) and uses the same calibration workflow as `INT8`.
We continue using `global_minmax` for the activation quantizers.

Note that `FP8` quantization is not supported for `Core ML` backend.

| Tensor     | dtype           | Granularity | qparam calculator |
| ---------- | --------------- | ----------- | ----------------- |
| Weight     | `float8_e4m3fn` | per-channel | `"static"`        |
| Activation | `float8_e4m3fn` | per-tensor  | `"global_minmax"` |

`FP8` can represent a wider dynamic range than `INT8`, which can be advantageous for layers with heavy-tailed activation distributions.
However, for ranges with a more uniform distribution, integer quantization with an affine grid may perform better.

As always, different models will perform better with different quantization settings, so it is worth exploring multiple settings to find the best combinations.

To switch to `FP8` quantization, we change the `dtype` flag of the previous config as shown below.

```python
from coreai_opt.quantization.spec import PerChannelGranularity

weight_spec = QuantizationSpec(
    dtype=torch.float8_e4m3fn,
    granularity=PerChannelGranularity(),
)

activation_spec = QuantizationSpec(
    dtype=torch.float8_e4m3fn, qparam_calculator_cls="global_minmax"
)

global_config = ModuleQuantizerConfig(
    op_state_spec={"weight": weight_spec},
    op_input_spec={"*": activation_spec},
    op_output_spec={"*": activation_spec},
)

config = QuantizerConfig(global_config=global_config)
```

```python
# Instantiate the quantizer
quantizer = Quantizer(fp32_model, config)

# Prepare the model
prepared_model = quantizer.prepare(example_inputs)

# Calibrate the model
calibrate(quantizer, prepared_model)
```

The calibrated model gives an accuracy of `76.56%`, an improvement over `W_INT8_A_INT8` quantization.

## Finalize

Call `quantizer.finalize()` once the model is ready for conversion and no further updates are expected.
This modifies fake-quantized modules to produce a model compatible with `Core AI` or `Core ML` conversion.

```python
import coreai_opt as opt

finalized_model = quantizer.finalize(
    backend=opt.ExportBackend.CoreAI
)  # Use opt.ExportBackend.CoreML for Core ML conversion
```

At this point, the model is ready to be exported or traced for downstream conversion with `coreai-torch` or `coremltools`.
Refer to [Integration with Core AI](../introduction/integration_coreai.md) for more details.

## Summary

In this article we covered a sample investigative workflow which tries to find the best combination of quantization datatypes and qparams_calculator to maximize quantized model accuracy. Additional settings not covered but worth trying out include:

- Using a larger number of samples for calibration. This gives diminishing returns after a certain number (dependent on the model and dataset)
- Using a different qparams_calculator, e.g. `moving_average` vs `global_minmax`
- Using `asymmetric` quantization instead of `symmetric` (has an impact on latency)
- Using a different number of bits for quantization (has an impact on model size and latency)
- Using a different data type for quantization, e.g. `float` vs `int` (has an impact on latency)
- Quantizing a subset of the model or specific layers only while leaving layers more sensitive to quantization unquantized (has an impact on model size and latency)
- Using Quantization-Aware Training (QAT) to fine-tune model weights with quantization enabled

We also demonstrated how to take a `coreai-opt` quantized model and finalize it for downstream conversion via `coreai-torch` or `coremltools`.
