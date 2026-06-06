# Applying quantization to an MNIST model

In this tutorial, we will be providing a basic introduction to quantizing a model with CoreAI-Opt.

After the end of this tutorial, you should be familiar with the following:

1. [How to apply CoreAI-Opt’s Weight-Only quantization](#Weight-Only-Quantization)
2. [How to apply CoreAI-Opt’s Weight + Activation quantization](#Weight-and-Activation-Quantization)
3. [How to apply CoreAI-Opt’s Quantization-Aware Training](#Quantization-Aware-Training)
4. [How to export CoreAI-Opt quantized models to Core AI](#Export-to-Core-AI)

**Table of Contents:**

- [Setup](#Setup)
  - [MNIST Dataset download](#MNIST-Dataset-download)
  - [Model definition](#Model-definition)
  - [Training and Evaluation](#Training-and-Evaluation)
  - [Train unquantized model](#Train-unquantized-model)
- [Weight Only Quantization](#Weight-Only-Quantization)
- [Weight and Activation Quantization](#Weight-and-Activation-Quantization)
- [Quantization Aware Training](#Quantization-Aware-Training)
- [Export to Core AI](#Export-to-Core-AI)

## Setup

We will be using a basic CNN model and train it on the MNIST dataset and observe its final accuracy.

Once we train this CNN model, we will apply quantization to it using `coreai-opt` starting with *weight only quantization (data free)*, then moving to apply calibration data based *weight + activation quantization*, and then finally do *Quantization Aware Training (QAT)*.

### MNIST Dataset download

Helper to download the MNIST dataset with standard normalization applied.

### Model definition

A simple CNN with a single Conv2d → ReLU → MaxPool block, followed by Flatten and a Linear classifier.

### Training and Evaluation

Standard PyTorch training loop and evaluation function that computes accuracy.

The CNN model used for this tutorial contains a single Conv2d, ReLU, MaxPool2d, Flatten, and Linear layer. Here’s the structure:

### Train unquantized model

Let’s train this model (unquantized) so we can get a baseline accuracy. We save the trained weights so we can reload them for each quantization experiment.

## Weight Only Quantization

Weight-only quantization compresses model weights to a lower precision (e.g., INT8) while keeping activations in full precision (FP32). This is the simplest form of quantization — it requires **no calibration data** and can be applied directly to a trained model.

For this tutorial, we will use the `INT8` dtype. Refer to the `QuantizationSpec` reference for all options.

`QuantizerConfig` describes how to quantize each operation through three spec keys:

- `op_state_spec`: state tensors of the op (weights, biases).
- `op_input_spec`: input activations — tensors flowing into the op.
- `op_output_spec`: output activations — tensors flowing out of the op.

For weight-only quantization, only the `weight` state spec is populated; setting `op_input_spec` and `op_output_spec` to `None` tells the quantizer to leave activations in full precision.

> Note: `coreai-opt` also offers `QuantizerConfig.presets.w8()` as a shortcut for the above config.

After calling `prepare()`, the model now has FakeQuantize modules inserted for weight tensors. These simulate quantization during the forward pass.

## Weight and Activation Quantization

Now, let’s try using weight + activation quantization through `coreai-opt`.

Weight + Activation quantization compresses both the model’s weights **and** intermediate activation tensors to a lower precision. This provides a greater speedup than weight-only, but requires **calibration data** — representative inputs are run through the model so it can compute appropriate scale and zero-point values for the quantized activations.

For this tutorial, we will use the `INT8` dtype for activations. Refer to the `QuantizationSpec` reference for all options.

Two changes from the weight-only config:

- `op_input_spec` and `op_output_spec` are now populated, so activations are quantized too.
- The `"*"` key applies the same INT8 spec to every operation input/output.

We now call `calibration_mode()` and feed it representative inputs to populate scale and zero-point value.

The `calibration_mode()` context manager enables range observers (to collect activation statistics) while disabling fake quantization (so the forward pass is numerically identical to the unquantized model). After exiting the context, observers are frozen and fake quantization is re-enabled.

## Quantization Aware Training

Now, let’s try to fine-tune the weight + activation quantized model with QAT to recover any accuracy lost from quantization.

The QAT training loop wraps each epoch in `wa_quantizer.training_mode()`:

- During the context: fake quantization is enabled and observers are enabled (collecting activation statistics during training batches).
- On exit: observers are frozen; fake quantization stays enabled, so the model is ready for evaluation with quant-aware weights.

## Export to Core AI

Once the quantized model is ready, call `finalize()` to convert the fake quantization modules into the real quantized representation for deployment.

Pass `ExportBackend.CoreAI` to `finalize(backend=...)` to target the `.aimodel` format produced by `coreai-torch`.

We’ll export the finetuned weight + activation model from the previous section.

The export proceeds in three steps:

- Trace the model with `torch.export.export()` to obtain a graph representation.
- Apply `cast_to_16_bit_precision()` to cast remaining FP32 parameters to FP16 for optimal on-device performance.
- Convert the exported program to Core AI format using `coreai-torch.TorchConverter`.

<script type="application/vnd.jupyter.widget-state+json">
{"state": {"0cb2b49da1e54aa386c87a7574163883": {"model_module": "@jupyter-widgets/base", "model_module_version": "2.0.0", "model_name": "LayoutModel", "state": {"_model_module": "@jupyter-widgets/base", "_model_module_version": "2.0.0", "_model_name": "LayoutModel", "_view_count": null, "_view_module": "@jupyter-widgets/base", "_view_module_version": "2.0.0", "_view_name": "LayoutView", "align_content": null, "align_items": null, "align_self": null, "border_bottom": null, "border_left": null, "border_right": null, "border_top": null, "bottom": null, "display": null, "flex": null, "flex_flow": null, "grid_area": null, "grid_auto_columns": null, "grid_auto_flow": null, "grid_auto_rows": null, "grid_column": null, "grid_gap": null, "grid_row": null, "grid_template_areas": null, "grid_template_columns": null, "grid_template_rows": null, "height": null, "justify_content": null, "justify_items": null, "left": null, "margin": null, "max_height": null, "max_width": null, "min_height": null, "min_width": null, "object_fit": null, "object_position": null, "order": null, "overflow": null, "padding": null, "right": null, "top": null, "visibility": null, "width": null}}, "27a85aecd0f545dfb569e3d5a035439d": {"model_module": "@jupyter-widgets/base", "model_module_version": "2.0.0", "model_name": "LayoutModel", "state": {"_model_module": "@jupyter-widgets/base", "_model_module_version": "2.0.0", "_model_name": "LayoutModel", "_view_count": null, "_view_module": "@jupyter-widgets/base", "_view_module_version": "2.0.0", "_view_name": "LayoutView", "align_content": null, "align_items": null, "align_self": null, "border_bottom": null, "border_left": null, "border_right": null, "border_top": null, "bottom": null, "display": null, "flex": null, "flex_flow": null, "grid_area": null, "grid_auto_columns": null, "grid_auto_flow": null, "grid_auto_rows": null, "grid_column": null, "grid_gap": null, "grid_row": null, "grid_template_areas": null, "grid_template_columns": null, "grid_template_rows": null, "height": null, "justify_content": null, "justify_items": null, "left": null, "margin": null, "max_height": null, "max_width": null, "min_height": null, "min_width": null, "object_fit": null, "object_position": null, "order": null, "overflow": null, "padding": null, "right": null, "top": null, "visibility": null, "width": null}}, "82822052bb6b494d851768fb1a125d8f": {"model_module": "@jupyter-widgets/output", "model_module_version": "1.0.0", "model_name": "OutputModel", "state": {"_dom_classes": [], "_model_module": "@jupyter-widgets/output", "_model_module_version": "1.0.0", "_model_name": "OutputModel", "_view_count": null, "_view_module": "@jupyter-widgets/output", "_view_module_version": "1.0.0", "_view_name": "OutputView", "layout": "IPY_MODEL_0cb2b49da1e54aa386c87a7574163883", "msg_id": "", "outputs": [{"data": {"text/html": "<pre style=\\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\\">Calibrating<span style=\\"color: #800080; text-decoration-color: #800080\\">  62%</span> <span style=\\"color: #f92672; text-decoration-color: #f92672\\">\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2578</span><span style=\\"color: #3a3a3a; text-decoration-color: #3a3a3a\\">\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501</span> <span style=\\"color: #008000; text-decoration-color: #008000\\">10/16 </span> [ <span style=\\"color: #808000; text-decoration-color: #808000\\">0:00:00</span> &lt; <span style=\\"color: #008080; text-decoration-color: #008080\\">0:00:01</span> , <span style=\\"color: #800000; text-decoration-color: #800000\\">53 it/s</span> ]\\n</pre>\\n", "text/plain": "Calibrating\\u001b[35m  62%\\u001b[0m \\u001b[38;2;249;38;114m\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u001b[0m\\u001b[38;2;249;38;114m\\u2578\\u001b[0m\\u001b[38;5;237m\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u001b[0m \\u001b[32m10/16 \\u001b[0m [ \\u001b[33m0:00:00\\u001b[0m < \\u001b[36m0:00:01\\u001b[0m , \\u001b[31m53 it/s\\u001b[0m ]\\n"}, "metadata": {}, "output_type": "display_data"}], "tabbable": null, "tooltip": null}}, "8fa6ecf14b9d46a18e4f6aece19f6f20": {"model_module": "@jupyter-widgets/output", "model_module_version": "1.0.0", "model_name": "OutputModel", "state": {"_dom_classes": [], "_model_module": "@jupyter-widgets/output", "_model_module_version": "1.0.0", "_model_name": "OutputModel", "_view_count": null, "_view_module": "@jupyter-widgets/output", "_view_module_version": "1.0.0", "_view_name": "OutputView", "layout": "IPY_MODEL_27a85aecd0f545dfb569e3d5a035439d", "msg_id": "", "outputs": [{"data": {"text/html": "<pre style=\\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\\">QAT Training<span style=\\"color: #800080; text-decoration-color: #800080\\">  80%</span> <span style=\\"color: #f92672; text-decoration-color: #f92672\\">\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2578</span><span style=\\"color: #3a3a3a; text-decoration-color: #3a3a3a\\">\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501</span> <span style=\\"color: #008000; text-decoration-color: #008000\\">4/5 </span> [ <span style=\\"color: #808000; text-decoration-color: #808000\\">0:01:04</span> &lt; <span style=\\"color: #008080; text-decoration-color: #008080\\">0:00:13</span> , <span style=\\"color: #800000; text-decoration-color: #800000\\">0 it/s</span> ]\\n</pre>\\n", "text/plain": "QAT Training\\u001b[35m  80%\\u001b[0m \\u001b[38;2;249;38;114m\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u001b[0m\\u001b[38;2;249;38;114m\\u2578\\u001b[0m\\u001b[38;5;237m\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u2501\\u001b[0m \\u001b[32m4/5 \\u001b[0m [ \\u001b[33m0:01:04\\u001b[0m < \\u001b[36m0:00:13\\u001b[0m , \\u001b[31m0 it/s\\u001b[0m ]\\n"}, "metadata": {}, "output_type": "display_data"}], "tabbable": null, "tooltip": null}}}, "version_major": 2, "version_minor": 0}
</script>
