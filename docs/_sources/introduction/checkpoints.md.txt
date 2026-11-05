# Checkpoints and workflows

A prepared model, the model returned by a compressor's [`prepare()`](how_to_use_coreaiopt.md#prepare), holds the compression state, such as palettization centroids, quantization parameters, or pruning masks. It can be saved as a checkpoint and reloaded later, with its compression state intact. For example, you can use a checkpoint to:

- Keep the results of expensive compression steps, such as k-means clustering, calibration, or training, and continue from them later instead of redoing them.
- Prepare and calibrate or train a model on one machine, and [finalize](how_to_use_coreaiopt.md#finalize) and export it on another.
- Finalize the same compressed model for more than one backend, for example to evaluate it with the [Torch backend](torch_backend.md) and then export it to [Core AI](integration_coreai.md).

This page covers how to checkpoint and reload a prepared model, and shows workflows that use checkpoints.

## Checkpointing and reloading prepared models

Every compressor's `prepare()` accepts a `state_dict` argument. Save the prepared model's `state_dict` (after calibration or training, if any), and later reload it into a fresh copy of the original model with the same compressor and config:

```python
import copy

import torch

from coreai_opt.quantization import Quantizer, QuantizerConfig

# INT8 weights and INT8 activations.
config = QuantizerConfig()

# 1. Prepare: insert fake quantization into a copy of the original model.
quantizer = Quantizer(copy.deepcopy(model), config)
prepared_model = quantizer.prepare(example_inputs)

# 2. Calibrate: observe activation ranges on representative data.
with quantizer.calibration_mode():
    for batch in calibration_dataloader:
        prepared_model(batch)

# 3. Save a checkpoint: the prepared model's state dict holds the compression
#    state, such as the calibrated quantization parameters.
torch.save(prepared_model.state_dict(), "prepared.pt")

# 4. Later, possibly on another machine: load the checkpoint.
state_dict = torch.load("prepared.pt")

# 5. Reload: prepare a fresh copy of the original model with the same config,
#    restoring the saved state instead of calibrating again.
quantizer = Quantizer(copy.deepcopy(model), config)
prepared_model = quantizer.prepare(example_inputs, state_dict=state_dict)
```

Reloading restores the saved compression state, such as palettization centroids, calibrated quantization parameters, and pruning masks, instead of redoing heavy computation such as k-means clustering. The reloaded model is in the same state as after a regular `prepare()` call, ready to evaluate, calibrate, train, or finalize for any backend.

When reloading, use the same config and model code as the run that produced the checkpoint. In graph mode, also use the same example inputs, dynamic shapes, and torch version. A mismatch raises an error rather than loading the wrong state.

## Common workflows

The workflows below show how to checkpoint and reload a prepared model in common scenarios.

### Evaluate with the Torch backend, then export to Core AI

Compress once, evaluate the compressed model quickly in PyTorch, then export the same compressed model to Core AI, without compressing it again:

```python
import copy

import torch

import coreai_opt as opt
from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig

config = KMeansPalettizerConfig.presets.w4()

# 1. Compress once and save a checkpoint before finalizing.
palettizer = KMeansPalettizer(copy.deepcopy(model), config)
prepared_model = palettizer.prepare(example_inputs)
torch.save(prepared_model.state_dict(), "prepared.pt")

# 2. Evaluate with the Torch backend.
eval_model = palettizer.finalize(backend=opt.ExportBackend.Torch)
eval_model.eval()
evaluate(eval_model)

# 3. Reload the checkpoint, without re-running k-means, and finalize for Core AI.
state_dict = torch.load("prepared.pt")
palettizer = KMeansPalettizer(copy.deepcopy(model), config)
prepared_model = palettizer.prepare(example_inputs, state_dict=state_dict)
coreai_model = palettizer.finalize(backend=opt.ExportBackend.CoreAI)
```

Convert `coreai_model` as described in [Integration with Core AI](integration_coreai.md). The same flow works for every compressor; for `Quantizer`, calibrate or train before saving the checkpoint.

### Resume pruning with fine-tuning

[Pruning with fine-tuning](../pruning/overview.md) ramps sparsity up over many training steps with a sparsity schedule. Save a checkpoint during training, and resume later from where it stopped: the schedule continues from the saved step, so the resumed run continues exactly like an uninterrupted one.

```python
import copy

import torch

from coreai_opt.pruning import (
    MagnitudePruner,
    MagnitudePrunerConfig,
    ModuleMagnitudePrunerConfig,
    PruningSpec,
)
from coreai_opt.pruning.config import PolynomialDecaySchedule

num_epochs = 10

# Ramp sparsity up to 70% over num_epochs, one pruner.step() per epoch.
config = MagnitudePrunerConfig(
    global_config=ModuleMagnitudePrunerConfig(
        op_state_spec={"weight": PruningSpec(target_sparsity=0.7)},
        sparsity_schedule=PolynomialDecaySchedule(
            begin_step=0, total_iters=num_epochs, power=3.0
        ),
    ),
)

# 1. First session: prepare, then fine-tune for a few epochs while ramping up sparsity.
pruner = MagnitudePruner(copy.deepcopy(model), config)
prepared_model = pruner.prepare(example_inputs)
optimizer = torch.optim.SGD(prepared_model.parameters(), lr=1e-3)
for epoch in range(4):
    train_one_epoch(prepared_model, train_dataloader, optimizer)
    pruner.step()

# 2. Save a checkpoint mid-schedule: the model's state dict holds the pruning masks
#    and the schedule's step count. The optimizer state is saved alongside it.
torch.save(
    {"model": prepared_model.state_dict(), "optimizer": optimizer.state_dict()},
    "pruning.pt",
)

# 3. Later session: reload the model and the optimizer.
checkpoint = torch.load("pruning.pt")
pruner = MagnitudePruner(copy.deepcopy(model), config)
prepared_model = pruner.prepare(example_inputs, state_dict=checkpoint["model"])
optimizer = torch.optim.SGD(prepared_model.parameters(), lr=1e-3)
optimizer.load_state_dict(checkpoint["optimizer"])

# 4. Continue fine-tuning: the schedule resumes from the saved step.
for epoch in range(4, num_epochs):
    train_one_epoch(prepared_model, train_dataloader, optimizer)
    pruner.step()
```

### Evaluate a palettized and activation-quantized (P4A8) model with the Torch backend, then export to Core AI

[Joint compression](../utils/joint_compression.md) applies two compressors in sequence: palettize the weights, finalize the palettizer, then quantize and calibrate the activations. To evaluate the P4A8 model at similar latency to the uncompressed model, finalize the palettizer for the Torch backend. Save a checkpoint of the palettizer before finalizing it, and reload it to export the same palettized model to Core AI, without re-running k-means:

```python
import copy

import torch

import coreai_opt as opt
from coreai_opt.palettization import KMeansPalettizer
from coreai_opt.quantization import Quantizer

# palett_config and quant_config as in Joint Compression.

# 1. Palettize once and save a checkpoint before finalizing.
palettizer = KMeansPalettizer(copy.deepcopy(model), palett_config)
palettized = palettizer.prepare(example_inputs)
torch.save(palettized.state_dict(), "palettized.pt")

# 2. Evaluate: finalize the palettizer for the Torch backend, then quantize and
#    calibrate the activations, and finalize the quantizer for the Torch backend.
palettized_model = palettizer.finalize(backend=opt.ExportBackend.Torch)
quantizer = Quantizer(palettized_model, quant_config)
prepared_model = quantizer.prepare(example_inputs)
with quantizer.calibration_mode():
    for batch in calibration_dataloader:
        prepared_model(batch)
eval_model = quantizer.finalize(backend=opt.ExportBackend.Torch)
eval_model.eval()
evaluate(eval_model)

# 3. Export: reload the palettizer checkpoint, without re-running k-means, and
#    finalize it for Core AI. Then quantize and calibrate the activations again.
palettizer = KMeansPalettizer(copy.deepcopy(model), palett_config)
palettizer.prepare(example_inputs, state_dict=torch.load("palettized.pt"))
palettized_model = palettizer.finalize(backend=opt.ExportBackend.CoreAI)
quantizer = Quantizer(palettized_model, quant_config)
prepared_model = quantizer.prepare(example_inputs)
with quantizer.calibration_mode():
    for batch in calibration_dataloader:
        prepared_model(batch)
coreai_model = quantizer.finalize(backend=opt.ExportBackend.CoreAI)
```

The activations are calibrated again for export, because the evaluation run's checkpoint can't be passed to `quantizer.prepare(state_dict=...)` on the export path: its activation quantization state is the same, but its palettized weights are stored as dense weights, while the Core AI-finalized palettizer stores them as LUT and index buffers. Calibrating with the same data produces the same quantization parameters, so the exported model matches the evaluated one. Convert `coreai_model` as described in [Integration with Core AI](integration_coreai.md).
