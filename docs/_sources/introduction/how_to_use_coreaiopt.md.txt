# How to use coreai-opt

`coreai-opt`'s compressors share a consistent API. This page walks through the building blocks every compressor exposes — initializing it, the `prepare()` / `calibration_mode()` / `training_mode()` lifecycle, and `finalize()` for conversion. For details specific to each technique, see [Quantization](../quantization/index.md) and [Palettization](../palettization/index.md).

## Initialize a compressor

Every compressor is constructed from a model and a config, e.g.:

```python
from coreai_opt.quantization import Quantizer, QuantizerConfig
from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig

# Quantization
quantizer = Quantizer(model, QuantizerConfig.presets.w8())

# Palettization
palettizer = KMeansPalettizer(model, KMeansPalettizerConfig.presets.w4())
```

- `model` is a `torch.nn.Module`.
- The config specifies what to compress, by how much, and how — bit width, granularity, algorithm choices, per-module overrides, and so on. See [Configs](#configs) below.

## prepare()

`prepare()` is the first transformation step. It compresses the model's weights according to the config and inserts fake-quantize / fake-palettize / sparsity-mask ops where needed, so a forward on the prepared model will give outputs accounting for the compression effects. Depending on the compressor used, `prepare()` may modify the model in-place. Use the returned `prepared_model` to ensure you're using the compressed model. If you need to retain the uncompressed model, make a copy (for example, with `copy.deepcopy`) prior to calling `prepare`.

```python
example_inputs = (
    torch.randn(1, 3, 224, 224),
)  # when activation quantization is in use, use representative data instead
prepared_model = quantizer.prepare(example_inputs)

# evaluate accuracy with compression effects active
val_metric = validate(prepared_model, val_dataset)
```

For data-free workflows (e.g. weight-only quantization, K-means palettization), `prepare()` is the only step before evaluating accuracy. The config can be changed, applied to the original model and re-validated until accuracy is satisfactory.

You don't need to put the model in `.eval()` or `.train()` before calling `prepare()` — the API runs the trace internally in eval mode and restores the original mode when it returns. If the model has BatchNorm or other ops with mode-dependent behavior, set the mode you want active (typically `.eval()`) on the returned `prepared_model` before validating.

## calibration_mode()

Some compression workflows need to observe representative data — for example, to fit per-tensor activation ranges, or to compute weight sensitivities for sensitivity-based palettization. `calibration_mode()` is a context manager which updates these parameters during each forward pass:

```python
with quantizer.calibration_mode():
    for batch in calibration_dataloader:
        prepared_model(batch)

# back to evaluation-ready state outside the context
val_metric = validate(prepared_model, val_dataset)
```

The exact effect of calibration depends on the compressor and config. For example, `Quantizer.calibration_mode()` enables range observers that update activation scales as it runs forward passes of the model. `KMeansPalettizer.calibration_mode()` computes the gradient of each weight with respect to a provided loss function to compute sensitivities for a weighted k-means. See the [Quantization](../quantization/index.md) and [Palettization](../palettization/index.md) overviews for details.

The context manager places the prepared model in eval mode internally and restores the original mode on exit.

## training_mode()

When data-free and calibration-based workflows aren't enough, quantization-aware training (QAT) lets the model adapt to compression error during training. `Quantizer.training_mode()` is a context manager that puts the prepared model into a QAT-ready state: range observers track activation ranges, fake-quantize ops are enabled so the loss captures quantization error, and the model is set to `.train()` mode.

```python
for epoch in range(num_epochs):
    with quantizer.training_mode():
        train_one_epoch(prepared_model, train_dataloader, grad_optimizer)
    val_loss = validate(prepared_model, val_dataloader)
```

The context manager sets `.train()` mode internally and restores the previous mode on exit, leaving the model in an evaluation-ready state outside the context. `training_mode()` is currently provided by `Quantizer`.

## finalize()

Once the compressed model reaches the desired accuracy, `finalize()` produces a model ready for conversion to a deployment format.

```python
finalized_model = quantizer.finalize()
```

After `finalize()`, weights and compression statistics are frozen — the model is no longer expected to be modified. Depending on the compressor, `finalize()` may update the model in-place or operate on a copy; the returned `finalized_model` is what should be exported. The finalized model inherits the current training mode, so call `.eval()` on it before running inference or downstream conversion.

The finalized model can then be used to convert with [coreai-torch](https://github.com/apple/coreai-torch) to produce a Core AI model (`.aimodel`).
For details, see [Integration with Core AI](integration_coreai.md).

## Configs

Configs are how compression is specified in `coreai-opt`. They control compression at every level:

- A single specification applied globally — for example, "8-bit weight-only quantization on every supported module".
- Module-type overrides — for example, "skip all `Linear` ops; use 4-bit on `Conv`s".
- Per-module or per-op overrides, to target by name — for example, "leave the last linear layer unquantized; use 6-bit per-grouped-channel on the third self-attention module".

This granularity supports iteration on the accuracy / size / latency trade-off: configs can be changed and the model re-prepared and re-evaluated until you reach the trade-off you want. [Mixed Precision Compression](../utils/mixed_precision.md) and the [Examples](../examples/model_examples.md) section show this iteration in practice.

For the full config surface — granularity options, presets, per-module overrides, YAML files — see [Quantization Config](../quantization/config.md) and [Palettization Config](../palettization/config.md).
