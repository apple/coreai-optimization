# Applying palettization to an MNIST model

In this tutorial, we will be providing a basic introduction to palettizing a model with CoreAI-Opt.

After the end of this tutorial, you should be familiar with the following:

1. [How to apply CoreAI-Opt’s k-means palettization](#K-Means-Palettization)
2. [How to apply CoreAI-Opt’s Sensitive K-Means (SKM) palettization](#Sensitive-K-Means-Palettization)
3. [How to export CoreAI-Opt palettized models to Core AI](#Export-to-Core-AI)

**Table of Contents:**

- [Setup](#Setup)
  - [MNIST Dataset download](#MNIST-Dataset-download)
  - [Model definition](#Model-definition)
  - [Training and Evaluation](#Training-and-Evaluation)
  - [Train baseline model](#Train-baseline-model)
- [K-Means Palettization](#K-Means-Palettization)
- [Sensitive K-Means Palettization](#Sensitive-K-Means-Palettization)
- [Export to Core AI](#Export-to-Core-AI)

## Setup

We will be using a basic CNN model and train it on the MNIST dataset and observe its final accuracy.

Once we train this CNN model, we will apply palettization to it using `coreai-opt` starting with *k-means palettization (data free)*, and then moving to apply calibration data based *Sensitive K-Means (SKM) palettization*.

### MNIST Dataset download

Helper to download the MNIST dataset with standard normalization applied.

### Model definition

A simple CNN with a single Conv2d → ReLU → MaxPool block, followed by Flatten and a Linear classifier.

### Training and Evaluation

Standard PyTorch training loop and evaluation function that computes accuracy.

The CNN model used for this tutorial contains a single Conv2d, ReLU, MaxPool, Flatten, and Linear layer. Here’s the structure:

### Train baseline model

Let’s train this model so we can get a baseline accuracy. We save the trained weights so we can reload them for each palettization experiment.

## K-Means Palettization

Palettization compresses a model’s weights by clustering them into a small look-up table (LUT) of centroids. Each weight is then replaced by an index into this table, so the weights can be stored using only a few bits per value (the *palette*).

K-means palettization uses standard k-means clustering to compute the LUT. It is data-free — no calibration data is required — and can be applied directly to a trained model, much like weight-only quantization.

For this tutorial, we will use 4-bit **per-grouped-channel** palettization: each group of 2 output channels gets its own look-up table of 2⁴ = 16 centroids. Refer to the [PalettizationSpec](../api/generated/coreai_opt.palettization.PalettizationSpec.html) reference for all options.

`KMeansPalettizerConfig` describes how to palettize each operation through `op_state_spec`, which maps a state tensor name (such as `"weight"`) to a `PalettizationSpec`.

The `PalettizationSpec` controls the palette: `n_bits` sets the bits per index, and `PerGroupedChannelGranularity(axis=0, group_size=2)` gives each group of 2 output channels its own LUT — a finer-grained alternative to the default single per-tensor LUT.

> The built-in `KMeansPalettizerConfig.presets.w4()` preset builds a similar 4-bit per-grouped-channel config. Its default `group_size` is 16, which suits larger models; here we use `group_size=2` because this toy model’s output-channel counts (12 and 10) aren’t divisible by 16. `KMeansPalettizerConfig.presets.w4(group_size=2)` would be an equivalent shortcut for the config above.

After calling `prepare()`, the model now has fake palettization modules inserted for its weight tensors. These simulate the effect of the LUT during the forward pass, so we can measure accuracy before exporting.

## Sensitive K-Means Palettization

Now, let’s try Sensitive K-Means (SKM) palettization through `coreai-opt`.

SKM uses **calibration data** to compute a per-weight importance score (sensitivity). Following the [SqueezeLLM](https://arxiv.org/pdf/2306.07629) method, it runs a backward pass and collects the squared gradients of each weight as its sensitivity. The k-means clustering is then *weighted* by these sensitivities, moving centroids closer to the weights that matter most for the model’s loss.

We now enter `calibration_mode()` with a loss function and feed it representative inputs.

For each batch, `skm.step(output, target)` computes the loss and runs a backward pass so the palettizer can collect squared gradients as sensitivities. When the context manager exits, the LUTs are recomputed using weighted k-means based on those sensitivities.

## Export to Core AI

Once the palettized model is ready, call `finalize()` to convert the fake palettization modules into the real LUT-based representation for deployment.

Pass `ExportBackend.CoreAI` to `finalize(backend=...)` to target the `.aimodel` format produced by `coreai-torch`.

We’ll export the Sensitive K-Means model from the previous section.

The export proceeds in three steps:

- Trace the model with `torch.export.export()` to obtain a graph representation.
- Apply `cast_to_16_bit_precision()` to cast remaining FP32 parameters to FP16 for optimal on-device performance.
- Convert the exported program to Core AI format using `coreai-torch.TorchConverter`.
