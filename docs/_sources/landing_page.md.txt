# Core AI Optimization Documentation

## What is `coreai-opt`?

`coreai-opt` is a Python library for compressing PyTorch models for deployment on Apple Silicon. It allows you to apply compression-based optimizations (such as quantization or palettization) to any PyTorch model, producing a transformed PyTorch model that can be converted to a Core AI model and run with the [Core AI](https://developer.apple.com/documentation/coreai) framework.

Model compression can help reduce the memory footprint of your model (disk size and at runtime), reduce inference latency, reduce power consumption, or optimize them all at once.

```{mermaid}
flowchart LR
    A[PyTorch model] --> B(coreai-opt)
    B --> C["Transformed<br/>PyTorch model<br/>(compressed)"]
    C --> D("coreai-torch<br/>(convert)")
    D --> E["Core AI model<br/>(.aimodel)"]
    style A color:#999,fill:none,stroke:none
    style C color:#999,fill:none,stroke:none
    style E color:#999,fill:none,stroke:none
    linkStyle default stroke:#999,stroke-width:1.5px
```

`coreai-opt` is built around the following ideas:

- **PyTorch native.** All APIs operate on PyTorch models. Compression is another transformation in your PyTorch workflow. The output of every compressor is itself a PyTorch model that can be validated, fine-tuned, and exported like any other model.

- **Integrates with existing PyTorch code.** Adding post-training compression, calibration-based, or compression-aware training to an existing PyTorch pipeline takes a few additional lines of code. All three use the same compressor object.

- **Aligned with Apple Silicon.** Default configurations and the majority of the available optimization options align with what the [Core AI](https://developer.apple.com/documentation/coreai) runtime executes efficiently, on one or many of the Apple Silicon platforms. Compressed PyTorch models can be seamlessly converted to `.aimodel` for deployment via Core AI.

## Types of compression

Available APIs cover the following categories of compression:

- **[Quantization](quantization/index.md)** approximates weights and/or activations using a quantization function. Weight precisions include INT2, INT4, INT8 and FP4, FP8; activation precisions include INT8 and FP8.
- **[Palettization](palettization/index.md)**, also known as codebook-style compression, clusters weights into a look-up table of centroids and stores indices in their place. Weights can be palettized to N ∈ {1, 2, 3, 4, 6, 8} bits.
- **[Pruning](pruning/index.md)** zeros out weights with the smallest magnitudes and stores the remaining weights using sparse representations.

These techniques can also be combined and applied in a hybrid fashion — for example, applying different palettization bit widths to different weights, or combining weight palettization with activation quantization — to build customized optimization recipes.

## Compression workflows

The process of applying compression to a model typically involves the following stages.

- **Data-free compression**: Weight-only compression that needs only the model — no calibration or training data. (Test data and an evaluation metric are still used to validate the result.) The fastest workflow — typically seconds to minutes even for large models. Often works well for reducing the model down to 8 bits, or even 6 or 4 bits, with only a slight decrease in accuracy. Typical approaches used for getting more aggressive compression, effective bits-per-weight (bpw) < 5 bits, involve using more granular compression (e.g. per-block quantization, per-grouped-channel palettization) and/or mixed-bit compression (assigning different bits to different weights, based on their effect on accuracy).

- **Calibration-based compression**: Post-training compression with calibration data. Often used when quantizing activations. A small amount of representative data (e.g. ~128 samples) lets compressors observe activation ranges and weight sensitivities.

- **Fine-tuning-based compression**: Compression-aware fine-tuning (e.g. quantization-aware training) with full training data. The compressor is integrated into your training loop so the model adapts to compression error as it trains. The most time-intensive workflow, but typically the only way to recover accuracy at the most aggressive compression ratios for weights (4 bits and below), and/or for models that are sensitive to activation quantization.

`coreai-opt`'s APIs allow you to easily move from one stage to the next while evaluating accuracy after each stage and escalating to a more expensive workflow only when needed.

## Getting started

For an overview of the generic structure of `coreai-opt` APIs, see [How to use coreai-opt](introduction/how_to_use_coreaiopt.md).

For end-to-end examples on API usage and common workflows, see [MNIST examples](examples/toy_models.md) and [model examples](examples/model_examples.md).

## Links to related Core AI components

- **[coreai-torch](https://github.com/apple/coreai-torch)** — Python library for converting PyTorch models to the Core AI (`.aimodel`) format.
- **[coreai-models](https://github.com/apple/coreai-models)** — GitHub repository with example models demonstrating how to convert, optimize, and re-author models for deployment with Core AI. Several of the LLMs in there are compressed to ~4–5 bits using `coreai-opt`. The repo also contains a number of AI skills, including some that wrap `coreai-opt` workflows.
- **[Core AI framework](https://developer.apple.com/documentation/coreai)** — Apple's on-device AI framework that runs `.aimodel` models.
