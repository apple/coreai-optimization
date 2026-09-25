# Applying magnitude pruning to an MNIST model

In this tutorial, we will be providing a basic introduction to pruning a model with CoreAI-Opt.

After the end of this tutorial, you should be familiar with the following:

1. [How to apply CoreAI-Opt’s post-training magnitude pruning](#Post-Training-Magnitude-Pruning)
2. [How to apply CoreAI-Opt’s magnitude pruning with a sparsity schedule and fine-tuning](#Magnitude-Pruning-with-Fine-Tuning)
3. [How to export CoreAI-Opt pruned models to Core AI](#Export-to-Core-AI)

**Table of Contents:**

- [Setup](#Setup)
  - [MNIST Dataset download](#MNIST-Dataset-download)
  - [Model definition](#Model-definition)
  - [Training and Evaluation](#Training-and-Evaluation)
  - [Train baseline model](#Train-baseline-model)
- [Post-Training Magnitude Pruning](#Post-Training-Magnitude-Pruning)
- [Magnitude Pruning with Fine-Tuning](#Magnitude-Pruning-with-Fine-Tuning)
- [Export to Core AI](#Export-to-Core-AI)

## Setup

We will be using a basic CNN model and train it on the MNIST dataset and observe its final accuracy.

Once we train this CNN model, we will apply magnitude pruning to it using `coreai-opt`, starting with a *post-training* pass (no fine-tuning), and then moving to a *scheduled* pass that ramps up sparsity while fine-tuning to recover accuracy.

### MNIST Dataset download

Helper to download the MNIST dataset with standard normalization applied.

### Model definition

A simple CNN with a single Conv2d → ReLU → MaxPool block, followed by Flatten and a Linear classifier.

### Training and Evaluation

Standard PyTorch training loop and evaluation function that computes accuracy.

The CNN model used for this tutorial contains a single Conv2d, ReLU, MaxPool, Flatten, and Linear layer. Here’s the structure:

### Train baseline model

Let’s train this model so we can get a baseline accuracy. We save the trained weights so we can reload them for each pruning experiment.

## Post-Training Magnitude Pruning

Magnitude pruning sparsifies a model by zeroing out the smallest-magnitude weights, up to a `target_sparsity`. Post-training pruning applies the full sparsity in a single shot during the `prepare()` call — no calibration data or fine-tuning is required.

Unless a model’s weights are already close to zero, post-training pruning will usually degrade accuracy. It’s most useful as a quick way to see the effect of sparsity before committing to a fine-tuning workflow.

For this tutorial, we’ll apply 50% unstructured magnitude pruning (individual elements, not whole channels) via `PruningSpec`. Refer to the [Pruning Config](https://apple.github.io/coreai-optimization/pruning/config.html) page for all options.

After calling `prepare()`, 50% of the values in each supported weight tensor are already zeroed out, so we can measure the accuracy impact immediately.

## Magnitude Pruning with Fine-Tuning

In most cases, fine-tuning is required to recover accuracy after pruning. Instead of applying the full sparsity in one shot, we configure a `sparsity_schedule` on the module config and call `pruner.step()` once per epoch to gradually ramp up sparsity while the model keeps training.

Here we target 50% sparsity, ramped in via a `PolynomialDecaySchedule` over the 5 epochs of fine-tuning.

We now fine-tune the model while incrementing the sparsity schedule. The `pruner.step()` call at the end of each epoch advances the schedule and recomputes the pruning masks against the current weight magnitudes for the next sparsity level.

## Export to Core AI

Once the pruned model is ready, call `finalize()` to prepare the sparsified modules for deployment. Pass `ExportBackend.CoreAI` to `finalize(backend=...)` to target the `.aimodel` format produced by `coreai-torch`.

We’ll export the fine-tuned, scheduled-pruning model from the previous section.

The export proceeds in three steps:

- Trace the model with `torch.export.export()` to obtain a graph representation.
- Apply `cast_to_16_bit_precision()` to cast remaining FP32 parameters to FP16 for optimal on-device performance.
- Convert the exported program to Core AI format using `coreai-torch.TorchConverter`.
