# Applying DKM palettization to an MNIST model

This tutorial shows how to palettize an MNIST model with **Differentiable K-Means (DKM)**, which learns the palettization codebook and cluster assignments during training rather than fixing them from a single one-shot K-means fit.

By the end of this tutorial you should be familiar with:

1. [How to configure DKM palettization](#DKM-Palettization)
2. [How to fine-tune a DKM-palettized model](#Fine-tuning-with-DKM)
3. [How to export the palettized model to Core AI](#Export-to-Core-AI)

**Table of Contents:**

- [Setup](#Setup)
  - [MNIST Dataset download](#MNIST-Dataset-download)
  - [Model definition](#Model-definition)
  - [Training and Evaluation](#Training-and-Evaluation)
  - [Train baseline model](#Train-baseline-model)
- [DKM Palettization](#DKM-Palettization)
  - [Fine-tuning with DKM](#Fine-tuning-with-DKM)
- [Export to Core AI](#Export-to-Core-AI)

## Setup

We train a small CNN on MNIST to get a baseline, then palettize it to 4-bit with DKM and fine-tune so the codebooks are learned jointly with the model.

### MNIST Dataset download

Helper to download the MNIST dataset with standard normalization applied.

### Model definition

A simple CNN with a single Conv2d → ReLU → MaxPool block, followed by Flatten and a Linear classifier.

### Training and Evaluation

Standard PyTorch training loop and evaluation function that computes accuracy.

The CNN model used for this tutorial contains a single Conv2d, ReLU, MaxPool, Flatten, and Linear layer. Here’s the structure:

### Train baseline model

Let’s train this model so we can get a baseline accuracy. We save the trained weights so we can reload them for the palettization experiment.

## DKM Palettization

Palettization compresses a model’s weights by clustering them into a small look-up table (LUT) of centroids, and replacing each weight with an index into that table.

Plain K-means palettization computes the LUT once, post-training, and freezes it. DKM instead makes the clustering step differentiable: during training it uses a soft assignment of weights to centroids, so gradients flow back into the palettized weights and their codebooks. This lets the codebook and assignments be learned jointly with the rest of the model, which can recover accuracy that a one-shot fit leaves behind.

DKM is enabled by setting the `training_strategy_spec` field of a `PalettizationSpec` to a `DKMTrainingSpec`. The same `KMeansPalettizer` drives it.

We use 4-bit **per-grouped-channel** palettization (each group of 2 output channels gets its own LUT of 2⁴ = 16 centroids), and attach a `DKMTrainingSpec` so the palettization is trained with DKM.

`DKMTrainingSpec` exposes a few hyperparameters (all optional): `tau` (softmax temperature for the soft assignment), `kmeans_max_iter` (soft-assignment iterations per forward), `enforce_zero`, and `zero_threshold`. Here we keep the defaults.

After `prepare()`, the model has fake-palettization modules whose codebooks are initialized from one-shot K-means. Measure the accuracy of the palettized model before DKM fine-tuning:

### Fine-tuning with DKM

We now fine-tune inside `dkm_palettizer.training_mode()`. Within this context each forward pass runs DKM’s differentiable soft assignment, so gradients flow into the palettized weights and their codebooks are updated by the optimizer.

`training_mode()` also restores the model’s original train/eval state on exit, so we can evaluate immediately afterwards.

## Export to Core AI

Once the palettized model is ready, call `finalize()` to convert the fake-palettization modules into the real LUT-based representation for deployment.

Pass `ExportBackend.CoreAI` to `finalize(backend=...)` to target the `.aimodel` format produced by `coreai-torch`.

The export proceeds in three steps:

- Trace the model with `torch.export.export()` to obtain a graph representation.
- Apply `cast_to_16_bit_precision()` to cast remaining FP32 parameters to FP16 for optimal on-device performance.
- Convert the exported program to Core AI format using `coreai-torch.TorchConverter`.
