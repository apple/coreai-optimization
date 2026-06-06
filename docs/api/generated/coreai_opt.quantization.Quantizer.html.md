# coreai_opt.quantization.Quantizer

### *class* coreai_opt.quantization.Quantizer(model, config=None)

Bases: `_BaseQuantizer`

Unified quantizer API that provides a single entry point for various quantization
workflows, including:

- **Data Types**: Integer (e.g. int8, int4) and floating-point
  (e.g. float8_e4m3fn, float8_e5m2) quantization
- **Quantization Workflows**: Post-training quantization (PTQ) and
  quantization-aware training (QAT)
- **Execution Modes**: Graph mode (built on torchao’s PT2E) or eager mode

The quantizer automatically selects the appropriate underlying implementation based
on the execution_mode specified in the configuration. Defaults to graph mode. Some
of the key differences between the execution modes are summarized below:

| Feature               | Graph Mode (Default)                                                                                                                                         | Eager Mode                                                                                                                                                                   |
|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Input/Output Types    | nn.Module<br/>-> fx.GraphModule.                                                                                                                             | nn.Module -> nn.Module                                                                                                                                                       |
| Module Fusion         | Automatic pattern-based fusion<br/>(e.g., conv+bn+relu)                                                                                                      | Manual fusion required                                                                                                                                                       |
| Control Flow          | Static graph only;<br/>Requires torch.export<br/>compatible model                                                                                            | Supports dynamic<br/>control flow<br/>(if/else, loops)                                                                                                                       |
| Shared Observer Ops   | Handled correctly; ops like<br/>MaxPool that share the same<br/>observer across inputs and<br/>outputs are detected and<br/>deduplicated on the graph.       | Not supported; Ops like<br/>MaxPool have independent<br/>observers for input vs<br/>output, which can cause<br/>incorrect quantization.                                      |
| FQ Node Deduplication | Back-to-back fake-quantize<br/>nodes on the same tensor are<br/>collapsed into a single node,<br/>avoiding redundant quantization<br/>on intermediate edges. | No deduplication; if both<br/>the output of one op and<br/>the input of the next are<br/>quantized, two consecutive<br/>FQ nodes are inserted on<br/>that intermediate edge. |

As a result of above mentioned differences, the total number of fake-quantize nodes
inserted by graph and eager mode can differ for the same `QuantizerConfig`. This
means the two modes are **not guaranteed to produce equivalent quantized models**,
and final model performance (accuracy and latency) may differ between modes even
when using identical configurations.

* **Parameters:**
  * **model** (*nn.Module*) – The PyTorch model to quantize.
  * **config** ([*QuantizerConfig*](coreai_opt.quantization.QuantizerConfig.md#coreai_opt.quantization.QuantizerConfig) *|* *None*) – Quantization configuration. If None, a default configuration
    with int8 weight and activation quantization is created.

#### \_\_init_\_(model, config=None)

Initialize the model compressor.

* **Parameters:**
  * **model** (*Module*) – The PyTorch model to compress. The model will be modified in-place
    during the compression process.
  * **config** ([*QuantizerConfig*](coreai_opt.quantization.QuantizerConfig.md#coreai_opt.quantization.QuantizerConfig) *|* *None*) – Configuration parameters for the compression

### Methods

| [`calibration_mode`](#coreai_opt.quantization.Quantizer.calibration_mode)([model])                                 | Context manager for calibration-based post-training quantization.                                                   |
|--------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| [`disable_fake_quant`](#coreai_opt.quantization.Quantizer.disable_fake_quant)([module])                            | Disable fake quantization on the model or a specific module.                                                        |
| [`disable_observer`](#coreai_opt.quantization.Quantizer.disable_observer)([module])                                | Disable observers on the model or a specific module.                                                                |
| [`enable_fake_quant`](#coreai_opt.quantization.Quantizer.enable_fake_quant)([module])                              | Enable fake quantization on the model or a specific module.                                                         |
| [`enable_observer`](#coreai_opt.quantization.Quantizer.enable_observer)([module])                                  | Enable observers on the model or a specific module.                                                                 |
| [`finalize`](#coreai_opt.quantization.Quantizer.finalize)([model, backend, mmap_dir])                              | Convert quantized model to backend-specific representations.                                                        |
| [`get_compressible_op_names`](#coreai_opt.quantization.Quantizer.get_compressible_op_names)(model, execution_mode) | Return op names in *model* that this quantizer can target.                                                          |
| [`prepare`](#coreai_opt.quantization.Quantizer.prepare)(example_inputs[, dynamic_shapes, ...])                     | Prepare the model for quantization by inserting fake quantization modules.                                          |
| [`step`](#coreai_opt.quantization.Quantizer.step)()                                                                | Advance the QAT schedule by one step and apply observer/fake_quant transitions after the step has been incremented. |
| `supported_modules`()                                                                                              | Returns types of modules that are supported for compression with for a particular model optimization technique.     |
| [`training_mode`](#coreai_opt.quantization.Quantizer.training_mode)([model])                                       | Context manager for quantization-aware training (QAT) workflow.                                                     |

#### calibration_mode(model=None)

Context manager for calibration-based post-training quantization.

When entering this context, observers are enabled to collect statistics
from calibration data, and fake quantization is disabled to get accurate
statistics. When exiting, observers are disabled and fake quantization
is re-enabled for evaluation.

**When to use:**

- Required for activation quantization to achieve good accuracy.
  The model post prepare() may have poor accuracy for activation
  quantization until calibrated with representative data
- Not needed for weight-only PTQ (prepare() → finalize() is sufficient)

* **Parameters:**
  **model** (*Module* *|* *GraphModule* *|* *None*) – Optional model to setup for calibration. If None, uses
  the internal prepared model.

* **Raises:**
  **RuntimeError** – If the model has not been prepared.
* **Parameters:**
  **model** (*Module* *|* *GraphModule* *|* *None*)

#### disable_fake_quant(module=None)

Disable fake quantization on the model or a specific module.

* **Parameters:**
  **module** (*Module* *|* *None*)
* **Return type:**
  None

#### disable_observer(module=None)

Disable observers on the model or a specific module.

* **Parameters:**
  **module** (*Module* *|* *None*)
* **Return type:**
  None

#### enable_fake_quant(module=None)

Enable fake quantization on the model or a specific module.

* **Parameters:**
  **module** (*Module* *|* *None*)
* **Return type:**
  None

#### enable_observer(module=None)

Enable observers on the model or a specific module.

* **Parameters:**
  **module** (*Module* *|* *None*)
* **Return type:**
  None

#### finalize(model=None, backend=ExportBackend.CoreAI, \*, mmap_dir=None)

Convert quantized model to backend-specific representations.

Converts fake quantization modules into backend-specific quantization ops.
Only call `finalize` when exporting to a target backend. For torch-based evaluation, use
the model returned by `prepare()` directly rather than calling `finalize`.

Backend-specific processing:

- CoreAI: Prepares for CoreAI export by replacing fake quantization modules
  with Core AI specific PyTorch custom ops.
- CoreML: Prepares for CoreML export by registering compression metadata
  as buffers and removes fake quantization modules.

* **Parameters:**
  * **model** (*Module* *|* *GraphModule* *|* *None*) – Optional model to finalize. If None, uses the internal
    prepared model.
  * **backend** ([*ExportBackend*](coreai_opt.ExportBackend.md#coreai_opt.ExportBackend)) – Target export backend for the quantized model. Supports
    CoreAI (default), CoreML, and \_TORCH backends.
  * **mmap_dir** (*str* *|* *None*) – If provided, serialize finalized quantized
    weights to safetensors files under this directory and re-load
    them via mmap. Only supported in eager execution mode with the
    CoreAI backend; raises `ValueError` otherwise. The files in
    `mmap_dir` must remain in place for the lifetime of the
    returned model; removing them invalidates the mmap-backed
    weights.
* **Returns:**
  The finalized quantized model ready for deployment on the target backend.
* **Return type:**
  *Module* | *GraphModule*

#### NOTE
In graph mode, the returned `fx.GraphModule` supports calling
`.train()` and `.eval()`, but with limited effect: only dropout
and batchnorm ops are affected via FX graph rewriting. User code
branching on the `training` flag and other ops with
mode-dependent behavior are not affected.

#### NOTE
When `backend=ExportBackend.CoreAI` in execution_mode=ExecutionMode.EAGER,
finalize frees the original dense weights.

#### *classmethod* get_compressible_op_names(model, execution_mode)

Return op names in *model* that this quantizer can target.

Dispatches to the appropriate underlying quantizer based on
*execution_mode*.

* **Parameters:**
  * **model** (*nn.Module*) – The model to get compressible op names for.
  * **execution_mode** ([*ExecutionMode*](coreai_opt.quantization.ExecutionMode.md#coreai_opt.quantization.ExecutionMode)) – The execution mode.
* **Returns:**
  Op names that can be compressed via quantization.
* **Return type:**
  set[str]

#### prepare(example_inputs, dynamic_shapes=None, export_with_no_grad=True)

Prepare the model for quantization by inserting fake quantization modules.

**Graph Mode:**
Exports the model using torch.export, applies quantization annotations, and
sets up fake quantization modules. Returns an fx.GraphModule.

**Eager Mode:**
Uses \_\_torch_function_\_ to trace model execution and insert fake quantizers
during the forward pass. Returns an nn.Module.

**Important Notes:**

- For weight-only PTQ: The prepared model can be directly finalized
  (prepare() → finalize() workflow).
- For activation quantization: The prepared model should be calibrated using
  calibration_mode() before finalization to collect statistics and achieve
  good accuracy.

* **Parameters:**
  * **example_inputs** (*tuple* *[**Any* *,*  *...* *]*) – Tuple of example inputs for model tracing. When
    activation quantization is in use, these should be
    representative of the data the model would typically see.
  * **dynamic_shapes** (*dict* *[**str* *,* *Any* *]*  *|* *tuple* *[**Any* *]*  *|* *list* *[**Any* *]*  *|* *None*) – Dynamic shapes specification (graph mode only).
    Ignored in EAGER mode.
  * **export_with_no_grad** (*bool*) – Whether to export with no_grad (graph mode
    only). Ignored in EAGER mode.
* **Returns:**
  The prepared model with fake quantization modules inserted, ready for
  calibration or training. This is a data-free PTQ compressed model.
* **Return type:**
  *Module* | *GraphModule*

#### NOTE
In graph mode, the returned `fx.GraphModule` supports calling
`.train()` and `.eval()`, but with limited effect: only dropout
and batchnorm ops are affected via FX graph rewriting. User code
branching on the `training` flag and other ops with
mode-dependent behavior are not affected.

#### step()

Advance the QAT schedule by one step and apply observer/fake_quant
transitions after the step has been incremented.

Must be called inside a training_mode() context. Increments \_step_count
(monotonically; never reset between training loops), then applies the
absolute observer/fake_quant state corresponding to the new step count.

* **Raises:**
  **RuntimeError** – If called outside a training_mode() context.
* **Warns:**
  **UserWarning** – If no qat_schedule is configured on any module.
* **Return type:**
  None

#### training_mode(model=None)

Context manager for quantization-aware training (QAT) workflow.

When entering this context, the model is configured for training with both
observers and fake quantization enabled (default behavior), or with the state
determined by the current step count if a QATSchedule is configured.
This allows the model to:

1. Set the model in training mode (model.training is set to True)
2. Enable the observers and activate the fake quantization
3. Using the observers, simulate quantization during forward/backward passes

When exiting the context, observers are disabled and fake quantization is
enabled (regardless of schedule).

The step count is not reset when re-entering training_mode() — it resumes
from the last value, so schedule state is restored from the accumulated count.

Nested calls to training_mode() are not allowed and will raise a RuntimeError.

**When to use:**

- For quantization-aware training (QAT) to fine-tune a prepared model
- The prepared model from prepare() may have poor accuracy for
  weight-only quantization. Fine-tuning the model with the quantization
  enabled will help the weights adapt to the effects of quantization.
- Upon calibrating an activation-quantized model, there wasn’t enough
  improvement in model accuracy. Fine-tuning the weights to adapt to
  the effect of activation (and weight) quantization can help recover
  the lost accuracy.

* **Parameters:**
  **model** (*Module* *|* *GraphModule* *|* *None*) – Optional model to setup for training. If None, uses
  the internal prepared model.

* **Raises:**
  * **RuntimeError** – If the model has not been prepared.
  * **RuntimeError** – If called while already inside a training_mode() context.
  * **TypeError** – If the provided model is not a torch.fx.GraphModule (graph mode).
* **Parameters:**
  **model** (*Module* *|* *GraphModule* *|* *None*)
