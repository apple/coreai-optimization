# Graph Mode Troubleshooting

This guide helps debug common issues when using [Graph execution mode](../quantization/overview.md#two-execution-modes-graph-and-eager) in CoreAI-Opt.

A `quantizer.prepare()` failure in graph mode happens in one of two stages, and the fix differs sharply between them. The first thing to do is figure out **which** stage is failing.

## Step 1: Diagnose — does `torch.export.export` succeed?

`Quantizer.prepare()` first calls `torch.export.export` to trace the model into an FX graph, then applies quantization annotations on the resulting graph and runs `torchao`'s `prepare_qat_pt2e`. To localize the failure, run the export step directly with the same arguments `prepare()` would use:

```python
import torch

with torch.no_grad():  # matches export_with_no_grad=True (the prepare() default)
    exported_program = torch.export.export(model, example_inputs)
```

The result of this experiment determines which path to follow:

- **Export fails.** Go to [If `torch.export.export` fails](#if-torch-export-export-fails). The model isn't `torch.export`-compatible as written; the workarounds in Steps 2-3 may help.
- **Export succeeds but `prepare()` still fails.** Go to [If `prepare()` fails after a successful export](#if-prepare-fails-after-a-successful-export).

## If `torch.export.export` fails

### Step 2: Try `export_with_no_grad=False`

The default `export_with_no_grad=True` wraps the export call in `torch.no_grad()`. For some models, this context modifies tracing behavior and causes guard failures.

```python
prepared = quantizer.prepare(
    example_inputs=(input_tensor,),
    export_with_no_grad=False,
)
```

### Step 3: Use dynamic shapes for shape-related errors

If the error mentions shape constraints, guards, or symbolic dimensions, the model likely has inputs with variable dimensions (e.g., sequence length, batch size). Specify `dynamic_shapes` to tell the exporter which dimensions can vary:

```python
from torch.export.dynamic_shapes import Dim

# Example: dynamic batch dimension
prepared = quantizer.prepare(
    example_inputs=(input_tensor,),
    dynamic_shapes={"x": (Dim.AUTO, Dim.STATIC, Dim.STATIC, Dim.STATIC)},
)

# Example: dynamic sequence length with a max constraint
import torch.export

dynamic_shapes = {
    "input_ids": {1: torch.export.Dim("seq_len", max=2048)},
    "attention_mask": {1: torch.export.Dim("seq_len", max=2048)},
}
prepared = quantizer.prepare(
    example_inputs=(input_ids, attention_mask),
    dynamic_shapes=dynamic_shapes,
)
```

For full details on dynamic shapes, see the [PyTorch Export Tutorial -- Dynamic Shapes](https://docs.pytorch.org/tutorials/intermediate/torch_export_tutorial.html#constraints-dynamic-shapes).

If Steps 2-3 don't resolve the export failure (e.g., the model has data-dependent control flow that `torch.export` cannot capture), the model definition itself may need to change to become exportable — this is worth fixing at the source, since the same construct can also block conversion via [coreai-torch](https://github.com/apple/coreai-torch) later on. Otherwise, see [Fall back to EAGER execution mode](#fall-back-to-eager-execution-mode) below.

## If `prepare()` fails after a successful export

After `torch.export.export` returns, `Quantizer.prepare()` applies coreai-opt's annotation pass and then calls into torch's `prepare_qat_pt2e` API. If the error you're seeing comes from `prepare_qat_pt2e` itself, it is a torch-side issue — refer to the [`torchao` documentation](https://docs.pytorch.org/ao/stable/).

If the error does **not** come from `prepare_qat_pt2e` (i.e. it originates inside coreai-opt's annotation pass), it likely indicates a bug in coreai-opt. **Please file an issue on GitHub** with the error message and a minimal reproducer. In the meantime, [fall back to eager mode](#fall-back-to-eager-execution-mode) below — eager bypasses the entire graph-mode pipeline.

## Fall back to EAGER execution mode

EAGER mode bypasses `torch.export` entirely and uses runtime tracing instead. It is the common fallback for both export failures that can't be worked around with Steps 2-3 and post-export `prepare()` failures.

```python
from coreai_opt.quantization import ExecutionMode

config = QuantizerConfig.presets.w8()
config.execution_mode = ExecutionMode.EAGER
quantizer = Quantizer(model, config)
prepared = quantizer.prepare(example_inputs=(input_tensor,))
```

See [Choosing between graph and eager mode](../quantization/overview.md#choosing-between-graph-and-eager-mode) for the trade-offs between the two modes.

## External Resources

- [PyTorch Export Tutorial](https://docs.pytorch.org/tutorials/intermediate/torch_export_tutorial.html)
- [Dynamic Shapes](https://docs.pytorch.org/tutorials/intermediate/torch_export_tutorial.html#constraints-dynamic-shapes)
