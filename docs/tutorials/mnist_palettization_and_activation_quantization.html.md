# Applying joint palettization and activation quantization to an MNIST model

In this tutorial, we will demonstrate how to combine weight palettization with activation quantization (joint P4A8 compression) using CoreAI-Opt.

After the end of this tutorial, you should be familiar with the following:

1. [How to apply joint K-Means palettization + activation quantization](#K-Means-Palettization-+-Activation-Quantization)
2. [How to export the jointly compressed model to Core AI](#Export-to-Core-AI)

**Table of Contents:**

- [Setup](#Setup)
  - [MNIST Dataset download](#MNIST-Dataset-download)
  - [Model definition](#Model-definition)
  - [Training and Evaluation](#Training-and-Evaluation)
  - [Train baseline model](#Train-baseline-model)
- [K-Means Palettization + Activation Quantization](#K-Means-Palettization-+-Activation-Quantization)
- [Export to Core AI](#Export-to-Core-AI)

## Setup

We will be using a basic CNN model and train it on the MNIST dataset and observe its final accuracy.

Once we train this CNN model, we will apply joint compression: first palettizing the weights (with LUT quantization), then quantizing the activations to int8.

### MNIST Dataset download

Helper to download the MNIST dataset with standard normalization applied.

### Model definition

A simple CNN with a single Conv2d → ReLU → MaxPool block, followed by Flatten and a Linear classifier.

### Training and Evaluation

Standard PyTorch training loop and evaluation function that computes accuracy.

The CNN model used for this tutorial contains a single Conv2d, ReLU, MaxPool, Flatten, and Linear layer. Here’s the structure:

### Train baseline model

Let’s train this model so we can get a baseline accuracy. We save the trained weights so we can reload them for each compression experiment.

## K-Means Palettization + Activation Quantization

In this example we will combine weight palettization with activation quantization. Here we will palettize the weights such that the entries of the look-up-table (LUT) are quantized to INT8. And then we apply activation quantization to INT8. With INT8 LUT quantization, and INT8 activations, all operations may be able to run fully in INT8 arithmetic — providing speedup on certain Apple platforms. Reference the [Joint Compression](../utils/joint_compression.md) documentation for more
information.

The workflow applies the two compressors **sequentially**: palettize weights first, finalize to Core AI, then quantize activations on the palettized model.

We use 4-bit **per-grouped-channel** palettization with **LUT quantization**: each group of 2 output channels gets its own look-up table of 2⁴ = 16 centroids, and the LUT entries are quantized to int8 for additional compression.

`PalettizationSpec` controls the palette: `n_bits` sets the bits per index, `PerGroupedChannelGranularity(axis=0, group_size=2)` gives each group of 2 output channels its own LUT, and `lut_qspec` quantizes the LUT entries to int8.

The palettizer must be finalized before the quantizer can be applied. This converts the fake palettization modules into the backend-specific representation.

> **Note:** The backend specified here must match the backend used for the subsequent activation quantization step.

Now we apply activation-only quantization. Setting `op_state_spec=None` is critical — the weights are already palettized, so applying weight quantization on top would be redundant. We quantize inputs and outputs of every operation to int8 symmetric.

With both palettization and activation quantization applied, let’s measure the joint compression accuracy.

## Export to Core AI

Once the jointly compressed model is ready, call `finalize()` on the quantizer to produce the final model for deployment. We export the k-means joint-compressed model to Core AI.

The export proceeds in three steps: trace the model with `torch.export.export()`, cast remaining FP32 parameters to FP16 with `cast_to_16_bit_precision()`, then convert to Core AI format using `coreai-torch`.
