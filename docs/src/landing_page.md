# Core AI Optimization Documentation

## What is `coreai-opt`?

`coreai-opt` is a Python library for compressing PyTorch models for deployment on Apple silicon. It applies compression-based optimizations (such as quantization or palettization) to any PyTorch model, producing a transformed PyTorch model that can be converted to a Core AI model and run with the [Core AI](https://developer.apple.com/documentation/coreai) framework. For an overview of the Core AI ecosystem and how coreai-opt fits in, see [What is Core AI?](#what-is-core-ai).

Model compression can help reduce the memory footprint of a model (disk size and at runtime), reduce inference latency, reduce power consumption, or optimize them all at once.

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

- **PyTorch native.** All APIs operate on PyTorch models. Compression is another transformation in a PyTorch workflow. The output of every compressor is itself a PyTorch model that can be validated, fine-tuned, and exported like any other model.

- **Integrates with existing PyTorch code.** Adding post-training compression, calibration-based, or compression-aware training to an existing PyTorch pipeline takes a few additional lines of code. All three use the same compressor object.

- **Aligned with Apple silicon.** Default configurations and the majority of the available optimization options align with what the [Core AI](https://developer.apple.com/documentation/coreai) runtime executes efficiently, on one or many of the Apple silicon platforms. Compressed PyTorch models can be seamlessly converted to `.aimodel` for deployment via Core AI.

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

- **Fine-tuning-based compression**: Compression-aware fine-tuning (e.g. quantization-aware training) with full training data. The compressor is integrated into the training loop so the model adapts to compression error as it trains. The most time-intensive workflow, but typically the only way to recover accuracy at the most aggressive compression ratios for weights (4 bits and below), and/or for models that are sensitive to activation quantization.

`coreai-opt`'s APIs make it straightforward to move from one stage to the next while evaluating accuracy after each stage and escalating to a more expensive workflow only when needed.

## Getting started

For an overview of the generic structure of `coreai-opt` APIs, see [How to use coreai-opt](introduction/how_to_use_coreaiopt.md).

For end-to-end examples on API usage and common workflows, see [MNIST examples](examples/toy_models.md) and [model examples](examples/model_examples.md).

## What is Core AI?

Core AI is a set of technologies for deploying machine learning models on Apple hardware, covering the full model deployment lifecycle: from model optimization and conversion, to debugging, to app integration. Models run entirely on device on Apple silicon, with no server required.

```{image} _images/core-ai-ecosystem.png
:alt: Diagram of the Core AI ecosystem. At the top, Core AI Models provides ready-to-use models and examples. Core AI Optimization and Core AI PyTorch Extensions prepare models for deployment, producing a .aimodel file. Core AI Debugger and Xcode support integration and debugging. Core AI Framework runs models on device.
:align: center
```

The Core AI ecosystem consists of the following components:

- Convert PyTorch models to the Core AI model format (`.aimodel`) using [Core AI PyTorch Extensions](https://github.com/apple/coreai-torch)
- Compress models with quantization, palettization, and pruning using [Core AI Optimization](https://github.com/apple/coreai-optimization)
- Load and run models in an app with the [Core AI Framework](https://developer.apple.com/documentation/coreai)
- Inspect, debug, and profile models using [Core AI Debugger](https://developer.apple.com/documentation/coreai/inspecting-debugging-and-profiling-core-ai-models)
- Get popular open-source models with conversion, optimization, and Swift app integration code using [Core AI Models](https://github.com/apple/coreai-models)
