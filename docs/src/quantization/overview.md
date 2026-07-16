# API Overview

## Weight-Only Quantization (Data-Free PTQ)

Weight-only quantization compresses the model's weight tensors to a lower precision while keeping activations in their original precision.

```python
import coreai_opt as opt
from coreai_opt.quantization import Quantizer, QuantizerConfig
import torch

model = MyModel().eval()
example_inputs = (torch.randn(1, 3, 224, 224),)

# A built-in preset: INT8, symmetric, per-channel weight-only quantization.
config = QuantizerConfig.presets.w8()
quantizer = Quantizer(model, config)

# Compress the weights and insert fake-quantize ops.
prepared_model = quantizer.prepare(example_inputs)

# Evaluate — forward passes already reflect the compression effect.
val_metric = validate(prepared_model, val_dataset)
```

When the accuracy is acceptable, call `finalize()` for the target backend, then convert.

Core AI (`.aimodel`):

```python
from pathlib import Path
from coreai_opt.casting import cast_to_16_bit_precision
import coreai_torch

finalized_model = quantizer.finalize(backend=opt.ExportBackend.CoreAI)

# torch.export the finalized model with coreai-torch's decomposition table.
exported_program = torch.export.export(
    finalized_model, example_inputs
).run_decompositions(coreai_torch.get_decomp_table())

# Cast remaining FP32 weights/activations to FP16 (and INT32/INT64 to INT16)
# for faster inference.
cast_to_16_bit_precision(exported_program)

# Convert to a Core AI program and save it as an .aimodel.
converter = coreai_torch.TorchConverter()
converter.add_exported_program(exported_program)
ai_program = converter.to_coreai()
ai_program.optimize()
ai_program.save_asset(Path("model.aimodel"))
```

Core ML (`.mlpackage`):

```python
import coremltools as ct

finalized_model = quantizer.finalize(backend=opt.ExportBackend.CoreML)

# Trace and convert, preferably using the highest available deployment target.
traced_model = torch.jit.trace(finalized_model, example_inputs)
mlmodel = ct.convert(
    traced_model,
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS26,
)
mlmodel.save("model.mlpackage")
```

The workflow has three stages:

- **Configure.** `QuantizerConfig` specifies the quantization scheme; the `presets` namespace provides ready-made recipes (e.g. `presets.w8()` for INT8 symmetric per-channel weights). See [Quantization Config](config.md) to customize the spec or apply different settings to different layers.
- **Prepare.** `prepare()` compresses the weights according to the config and inserts fake-quantize ops. The prepared model runs like the final compressed model — its forward pass reflects the compression — so you can evaluate accuracy on it, and iterate by changing the config. For weight-only quantization this is the only step needed before export, unless you want to push accuracy further with training (see the [QAT section](#weight-and-or-activation-quantization-qat-quantization-aware-training) below).
- **Finalize and export.** `finalize(backend=...)` freezes the quantization parameters and replaces the fake-quantize ops with the chosen backend's compression ops or metadata. The finalized model is meant primarily for export and stays numerically close to the model just before `finalize()`. Export to Core AI with [`coreai-torch`](../introduction/integration_coreai.md) (the [`cast_to_16_bit_precision`](../utils/casting.md) step is recommended), or to Core ML with `coremltools`.

## Weight + Activation Quantization (Calibration-Based PTQ)

When quantizing both weights and activations, the quantizer needs to observe activation
ranges on representative data so that it can compute quantization parameters (scale, etc.) accurately.
This is done via `calibration_mode()`.

**Note**: Unlike weight-only quantization, the `example_inputs` used to prepare the quantized model with activation quantization enabled should be representative of the data the model would typically see. This is because `example_inputs` provides a starting point for the activation quantization parameters.

Further calibration is still required; however, using even a single representative data sample as the example input can reduce the number of calibration samples needed later.

```python
from coreai_opt.quantization import Quantizer, QuantizerConfig
import torch

model = MyModel().eval()
# Use a representative data sample when activation quantization is enabled.
example_inputs = (...,)

# Default config: symmetric INT8 weights (per-channel)
# + INT8 activations (per-tensor).
config = QuantizerConfig()
quantizer = Quantizer(model, config)
prepared_model = quantizer.prepare(example_inputs)

# Calibrate activation ranges on a small amount of representative data.
with quantizer.calibration_mode():
    for batch in calibration_dataloader:
        prepared_model(batch)

# Evaluate the calibrated model.
val_metric = validate(prepared_model, val_dataset)
```

Right after `prepare()`, the activation scales come only from `example_inputs`, so accuracy is usually poor at this point. `calibration_mode()` addresses this.
This is what the context manager handles:

- Inside the context:
  - activation fake-quantization is turned **off**: activation observers see undistorted activation values, so the observed ranges (and resulting scales) reflect the true distribution rather than already-quantized values.
  - weight fake-quantization stays **on**: activations flowing into each observer are produced with quantized weights upstream, matching what the deployed model will actually see.
  - range observers are turned **on**: each forward pass updates the observed activation ranges, and hence the activation quantization scales.
- After exiting the context manager, observers are turned back off and activation fake-quantization back on, leaving the model ready for evaluation.

A small amount of representative data is typically enough.

Finalize and export the calibrated `prepared_model` exactly as shown in the Weight-Only section above.

:::{note}
The finalized model is numerically close to the model just before `finalize()`, but not always bit-identical. For models with a **Conv + BatchNorm** pattern in the default [graph execution mode](#two-execution-modes-graph-and-eager), the two can differ slightly more: BatchNorm folding is handled with ops that are different between the prepared and finalized models (though algebraically equivalent). Weight quantization is matched closely between the prepared and finalized graphs, but activation quantization can still show a small numerical divergence.
:::

## Weight and/or Activation Quantization (QAT: Quantization-Aware Training)

If neither data-free weight-only quantization nor calibration-based weight + activation quantization reaches the accuracy you need — typically at the most aggressive settings (4-bit weights and below) or for models sensitive to activation quantization — fine-tune the model with quantization-aware training (QAT).

```python
from coreai_opt.quantization import Quantizer, QuantizerConfig
import torch

model = MyModel().eval()
example_inputs = (...,)  # representative data sample

config = QuantizerConfig()
quantizer = Quantizer(model, config)
prepared_model = quantizer.prepare(example_inputs)

# Fine-tune with quantization simulated in the forward pass.
for epoch in range(num_epochs):
    with quantizer.training_mode():
        train_one_epoch(prepared_model, train_dataloader, grad_optimizer)
    val_loss = validate(prepared_model, val_dataloader)
```

Inside `training_mode()`, the model is put in train mode, range observers are enabled, and fake-quantization is applied, so the loss captures the quantization error and the weights and quantization params adapt to it during training. On exit, the model returns to an evaluation-safe state (observers off, fake-quantization on), so the per-epoch validation reflects the compressed model without further changing the quantization scales. Finalize and export the fine-tuned `prepared_model` as in the Weight-Only section above.

This was a brief overview of the API for the most common usage workflow.

- See [Quantization Config](config.md) on how to modify the quantization spec (different quantization formats, range observers, etc.) and how to apply different quantization settings to different parts of the model (e.g. skip certain layers, etc.)
- See [Deeper Dive](advanced.md) on how to control the QAT schedule at a more granular level (e.g. how to freeze quantization scales during QAT, freeze batchnorm stats, etc.), and how to define custom patterns for quantizer placement, define custom logic for quantization parameter calculation, etc.

## Supported quantization types

For a detailed and current list of supported types, refer to the API reference of {class}`~coreai_opt.quantization.spec.QuantizationSpec`.

Broadly speaking, the following dtypes can be used for quantization:

- For weights: INT2/UINT2, INT4/UINT4, INT8/UINT8, FP8_E4M3, FP4_E2M1
- For activations: INT8/UINT8, FP8_E4M3

A few other dtypes (e.g. FP8_E5M2) may also be available to experiment with; however, they may have limited support on the Core AI runtime.

## Two Execution Modes: graph and eager

The {class}`~coreai_opt.quantization.Quantizer` has two internal implementations:

- **Graph mode (default).** Built on top of `torchao`'s PT2E (PyTorch 2.0 Export) implementation. Requires a `torch.export`-compatible model. The prepared model is an `fx.GraphModule`. This is recommended when performing weight + activation quantization.
- **Eager mode.** Built using the [`__torch_function__`](https://docs.pytorch.org/docs/stable/torch.overrides.html) protocol, with no graph capture. The prepared model is still an `nn.Module`.

### Code examples

```python
# Graph mode: this is the default
from coreai_opt.quantization import Quantizer, QuantizerConfig
from coreai_opt.quantization.config import ExecutionMode

config = QuantizerConfig(execution_mode=ExecutionMode.GRAPH)
model: torch.nn.Module = MyModel()
quantizer = Quantizer(model, config)
# prepare will internally export the model to a graph using torch.export
# prepared_model is a fx.GraphModule
prepared_model: fx.GraphModule = quantizer.prepare(example_inputs)
```

```python
# Eager mode (same imports as graph above)
config = QuantizerConfig(execution_mode=ExecutionMode.EAGER)
model: torch.nn.Module = MyModel()
quantizer = Quantizer(model, config)
# prepared_model is an nn.Module
prepared_model: nn.Module = quantizer.prepare(example_inputs)
```

### Choosing between graph and eager mode

#### Weights-only quantization

The two modes are expected to produce very similar models for weight-only quantization. The only difference is that in graph mode, Batchnorm weights are folded with convolution weights such that quantization happens on the fused weights, whereas they are kept separate in the eager mode. This typically should not affect accuracy/latency by much, hence eager mode can be preferred over graph mode.

A few scenarios where `eager` mode may need to be used instead of `graph`:

- If you run into any errors during the `prepare` call which, under the hood, invokes the `torch.export.export` and `torchao`'s `prepare_qat_pt2e`/`convert_pt2e` APIs. See [Graph Mode Troubleshooting](../debugging/graph_mode_troubleshooting.md) for common export errors and workarounds before falling back to eager mode.
- When `torch.nn.Module` needs to be provided as an input, instead of `ExportedProgram` to the conversion API of [coreai-torch](https://github.com/apple/coreai-torch). This happens when the `coreai-torch` conversion needs to "externalize" certain sub-modules to map them to _composite ops_ for better runtime performance.

#### Weights and activations quantization

The two modes show larger differences when activation quantization is enabled. In general, the support for node patterns and proper placement of activation fake-quantize nodes is currently much more mature for graph mode compared to eager mode. Eager mode may yield models with sub-optimal runtime performance. Thus, unless graph mode runs into any errors, it is the recommended option when quantizing activations. See below for details on the differences.

#### Detailed comparison

| Feature                                   | Graph mode (default)                                     | Eager mode                      |
| ----------------------------------------- | -------------------------------------------------------- | ------------------------------- |
| Input → output types                      | `nn.Module` → `fx.GraphModule`                           | `nn.Module` → `nn.Module`       |
| Dynamic control flow                      | Follows `torch.export.export` support                    | Supported                       |
| Conv + BatchNorm weight quantization      | BN is folded into the preceding Conv weight first        | Conv weight unfused             |
| Consecutive fake-quantize duplication     | Deduplicates (`out → fq → fq → inp` → `out → fq → inp`)  | duplication persists            |
| Pattern-based fusion boundaries           | Supported (e.g. Conv-BN-ReLU is treated as one block)    | Not supported                   |
| Shared quantizer for value-preserving ops | Supported (`maxpool`, `avgpool`, `flatten`, `concat`, …) | Not supported                   |
| Config op names                           | aten op names                                            | `__torch_function__` call sites |

- **Pattern-based fusion boundaries.** Graph mode inserts activation quantizers at the boundaries of op groups that get fused during inference — for a Conv-BN-ReLU pattern, a single pair of input/output quantizers is placed around the whole block, so no quantizer sits inside the fused region.
- **Shared quantizer for value-preserving ops.** For ops that do not change the value range (max/avg pool, flatten, concat, etc.), the same quantizer is shared across the input and output so a single set of qparams is used on both sides.

The registries that drive these behaviors live in `_graph/_annotation_pattern_registry.py` for graph mode and `_eager/supported_ops_registry.py` for eager mode.

### Pattern matching and op naming

The name strings of modules and ops used for granular control in {class}`~coreai_opt.quantization.config.QuantizerConfig` differ between the two execution modes. See [Quantization Config](config.md) for details and [How to get names + types for modules and ops](config.md#how-to-get-names--types-for-modules-and-ops) for how to retrieve them.
