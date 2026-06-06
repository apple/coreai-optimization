# EDSR model: Weight palettization + activation quantization (PTQ)

In this article we explore two compression strategies for a super-resolution model: weight + activation quantization (W_INT8_A_INT8), and joint compression combining 4-bit palettization, with INT8 quantized LUT (look up table), and INT8 activation quantization (W_P4(INT8)\_A_INT8).
We demonstrate how to evaluate the quality trade-off between compression aggressiveness and model fidelity using PSNR.

The reported metrics in this article come from applying the described configurations to a pretrained [EDSR](https://arxiv.org/abs/1707.02921) model (`edsr_r16f64`, 1.5M parameters) sourced via [torchsr](https://github.com/Coloquinte/torchSR).
Data samples used are obtained from the B100 benchmark via `torchsr` (100 HR/LR image pairs at 2× scale).
20 images are used for calibration and the remaining 80 for evaluation.

EDSR (Enhanced Deep Super-Resolution) takes a low-resolution image and upscales it by a fixed integer factor (2×, 3×, or 4×) using a deep residual convolutional network.

We use PSNR as the metric in this example.
PSNR measures pixel-level fidelity between the model output and the ground-truth high-resolution image, expressed in dB where higher is better.
PSNR is the standard metric for super-resolution since the task has a well-defined ground truth, making it a reliable signal for comparing how much each compression configuration degrades output quality.

As a reference, the pretrained FP32 model gives a PSNR of `30.68 dB`.

## W_INT8_A_INT8 quantization

We begin with W_INT8_A_INT8 quantization using the default `QuantizerConfig()`, which applies INT8 per-channel weight quantization and INT8 per-tensor activation quantization with a `moving_average` qparams calculator.

| Tensor     | dtype  | Granularity | qparam calculator                                     |
| ---------- | ------ | ----------- | ----------------------------------------------------- |
| Weight     | `int8` | per-channel | `"static"` — computed directly from the weight tensor |
| Activation | `int8` | per-tensor  | `"moving_average"` — EMA of observed min/max          |

```python
from coreai_opt.quantization import QuantizerConfig, Quantizer

config = QuantizerConfig()
quantizer = Quantizer(fp32_model, config)

# Use a representative low-resolution (LR) image from the calibration set as example inputs
prepared_model = quantizer.prepare(example_inputs)
```

### Calibrate

Calibration is required to determine accurate activation scales.
Representative low-resolution (LR) images are passed through the model inside `calibration_mode()`.

```python
with quantizer.calibration_mode():
    with torch.no_grad():
        for lr_image in calibration_images:
            prepared_model(lr_image)
```

After calibrating with 20 B100 images, the W_INT8_A_INT8 model achieves `30.33 dB` — a drop of only `0.35 dB` from the FP32 baseline.
The model is compressed from `5.48 MB` (FP32) to approximately `1.4 MB` (INT8 weights), a 4× reduction.

## Joint compression: W_P4(INT8)\_A_INT8

The W_INT8_A_INT8 result is encouraging.
However, if further model size reduction is needed — for example to meet a binary size budget on-device — we can try compressing the weights more aggressively using 4-bit palettization while retaining INT8 activation quantization to preserve the latency benefits of quantized compute.

```python
from coreai_opt.palettization import (
    KMeansPalettizer,
    KMeansPalettizerConfig,
    ModuleKMeansPalettizerConfig,
)
from coreai_opt.palettization.spec import PalettizationSpec
from coreai_opt.quantization.spec import QuantizationSpec, QuantizationScheme

palett_spec = PalettizationSpec(
    n_bits=4,
    lut_qspec=QuantizationSpec(dtype=torch.int8, qscheme=QuantizationScheme.SYMMETRIC),
)
palett_config = KMeansPalettizerConfig(
    global_config=ModuleKMeansPalettizerConfig(
        op_state_spec={"weight": palett_spec},
    )
)
```

The activation quantization config omits `op_state_spec` since weights are already handled by the palettizer:

```python
from coreai_opt.quantization import ModuleQuantizerConfig, QuantizerConfig

act_spec = QuantizationSpec(dtype=torch.int8, qscheme=QuantizationScheme.SYMMETRIC)
quant_config = QuantizerConfig(
    global_config=ModuleQuantizerConfig(
        op_state_spec=None,
        op_input_spec={"*": act_spec},
        op_output_spec={"*": act_spec},
    )
)
```

Joint compression is applied in two sequential steps: palettize first, then quantize activations on the resulting model.

```python
import coreai_opt as opt

# Step 1: palettize weights
palettizer = KMeansPalettizer(fp32_model, palett_config)
prepared_pal = palettizer.prepare(example_inputs)

palettized = palettizer.finalize(backend=opt.ExportBackend.CoreAI)

# Step 2: quantize activations on the palettized model
quantizer = Quantizer(palettized, quant_config)
prepared_model = quantizer.prepare(example_inputs)

with quantizer.calibration_mode():
    with torch.no_grad():
        for lr_image in calibration_images:
            prepared_model(lr_image)
```

The W_P4(INT8)\_A_INT8 model achieves `29.86 dB` — a further drop of `0.47 dB` compared to W_INT8_A_INT8.

Refer to [Joint Compression](../utils/joint_compression.md) for more information on joint compression.

## Finalize

Call `quantizer.finalize()` once the model is ready for conversion and no further updates are expected.
This modifies fake-quantized modules to produce a model compatible with `Core AI` conversion.

```python
finalized_model = quantizer.finalize(backend=opt.ExportBackend.CoreAI)
```

At this point, the model is ready to be exported for downstream conversion with `coreai-torch`.
Refer to [Integration with Core AI](../introduction/integration_coreai.md) for more details.

## Summary

| Configuration              | PSNR     | Weight storage       |
| -------------------------- | -------- | -------------------- |
| FP32 baseline              | 30.68 dB | ~5.5 MB              |
| W_INT8_A_INT8              | 30.33 dB | ~1.4 MB (4× smaller) |
| W_P4(INT8)\_A_INT8 (joint) | 29.86 dB | ~0.7 MB (8× smaller) |

W_INT8_A_INT8 offers a strong compression-quality trade-off with minimal PSNR loss.
W_P4(INT8)\_A_INT8 achieves 8× weight compression at the cost of an additional `0.47 dB` drop — whether this is acceptable depends on the deployment constraints.

Users may also consider:

- Increasing the number of calibration images to improve activation scale estimates
- Applying palettization or quantization selectively to only the most compressible layers
- Using [Sensitive K-Means](../palettization/overview.md#sensitive-k-means-api) calibration to place centroids near weight values that matter most for output quality
