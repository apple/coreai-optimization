# coreai_opt.palettization.KMeansPalettizer

### *class* coreai_opt.palettization.KMeansPalettizer(model, config=None)

Bases: `_BasePalettizer`, `EagerCompressionComponentBuilderMixin`

K-means palettizer with integrated supported operations strategy.

* **Parameters:**
  * **model** (*Module*)
  * **config** ([*KMeansPalettizerConfig*](coreai_opt.palettization.KMeansPalettizerConfig.md#coreai_opt.palettization.KMeansPalettizerConfig) *|* *None*)

#### \_\_init_\_(model, config=None)

Initialize the KMeans palettizer.

* **Parameters:**
  * **model** (*Module*) – The PyTorch model to palettize.
  * **config** ([*KMeansPalettizerConfig*](coreai_opt.palettization.KMeansPalettizerConfig.md#coreai_opt.palettization.KMeansPalettizerConfig) *|* *None*) – Optional palettization configuration. If None, default configuration
    will be used.

### Methods

| [`calibration_mode`](#coreai_opt.palettization.KMeansPalettizer.calibration_mode)([model, sensitivity_path])   | Context manager for calibration using Sensitive K-Means clustering.                                             |
|----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| [`finalize`](#coreai_opt.palettization.KMeansPalettizer.finalize)([model, backend, mmap_dir])                  | Convert palettized model to backend-specific representations.                                                   |
| [`get_op_type_resolver`](#coreai_opt.palettization.KMeansPalettizer.get_op_type_resolver)()                    | Return a function that maps a torch function to its palettizable op type.                                       |
| [`prepare`](#coreai_opt.palettization.KMeansPalettizer.prepare)(example_inputs[, sensitivity_path, ...])       | Prepare the model for palettization.                                                                            |
| [`save_sensitivities`](#coreai_opt.palettization.KMeansPalettizer.save_sensitivities)(path)                    | Save sensitivity values from the prepared model to a file.                                                      |
| `supported_modules`()                                                                                          | Returns types of modules that are supported for compression with for a particular model optimization technique. |
| `training_mode`([model])                                                                                       | Context manager for training time compression workflow.                                                         |

#### calibration_mode(model=None, \*, loss_fn, sensitivity_path=None)

Context manager for calibration using Sensitive K-Means clustering.

This method implements sensitivity-based palettization as described in
“SqueezeLLM: Dense-and-Sparse Quantization”
([https://arxiv.org/pdf/2306.07629.pdf](https://arxiv.org/pdf/2306.07629.pdf)). The loss function is used to compute
gradients via backpropagation, and the squared gradients are collected as
sensitivity values for each weight element.

These sensitivity values indicate how sensitive a given weight element is:
the more sensitive an element, the larger the impact palettizing it has on the
model’s loss function. This means that weighted k-means moves the clusters
closer to the sensitive weight values, allowing them to be represented more
exactly. This leads to a lower degradation in model performance after
palettization.

* **Parameters:**
  * **loss_fn** (*Callable*) – Loss function that takes (output, target) and returns a scalar
    loss. The loss is used for gradient computation, where the squared
    gradients serve as sensitivity weights for kmeans clustering.
  * **sensitivity_path** (*str* *|* *None*) – Optional path for saving the sensitivity
    of weights. Defaults to None.
  * **model** (*Module* *|* *None*) – Optional model to calibrate. If None, uses self._model.

#### finalize(model=None, backend=ExportBackend.CoreAI, \*, mmap_dir=None)

Convert palettized model to backend-specific representations.

Only call `finalize` when exporting to a target backend. For torch-based
evaluation, use the model returned by `prepare()` directly rather than
calling `finalize`.

* **Parameters:**
  * **model** (*nn.Module* *|* *None*) – Model to finalize. If None, uses the
    internal prepared model.
  * **backend** ([*ExportBackend*](coreai_opt.ExportBackend.md#coreai_opt.ExportBackend)) – Target export backend for the palettized
    model. Supports CoreAI (default) and CoreML backends.
  * **mmap_dir** (*str* *|* *None*) – If provided, finalized palettized weights are
    written under this directory and re-loaded as mmap-backed
    tensors so they don’t have to be held in RAM. Only supported
    with the CoreAI backend; raises `ValueError` otherwise. The
    files in `mmap_dir` must remain in place for the lifetime of
    the returned model; removing them invalidates the mmap-backed
    weights.
* **Returns:**
  The finalized palettized model ready for deployment.
* **Return type:**
  torch.nn.Module

#### NOTE
When `backend=ExportBackend.CoreAI`, finalize frees the original
dense weights in place: on each parametrized weight,
`parametrizations[...].original` is replaced with a zero-size
placeholder so its storage can be released.

#### *classmethod* get_op_type_resolver()

Return a function that maps a torch function to its palettizable op type.

* **Return type:**
  *Callable*[[*Callable*], str | None]

#### prepare(example_inputs, sensitivity_path=None, num_workers=1)

Prepare the model for palettization.

* **Parameters:**
  * **example_inputs** (*tuple* *[**Tensor* *]*) – Sample inputs to trace the model and configure
    palettizers
  * **sensitivity_path** (*str* *|* *None*) – Optional path to precomputed sensitivity values for
    weighted k-means clustering. These sensitivity values indicate the
    importance of each weight element and can be computed using
    calibration_mode(). When provided, k-means clustering will place
    centroids closer to more sensitive weight values. If None (default),
    vanilla (non-weighted) k-means clustering is used.
  * **num_workers** (*int*) – `1` runs clustering sequentially. Values greater than
    `1` use `torch.multiprocessing` to parallelize clustering
    across layers. It is recommended to use more than one worker
    process to parallelize the clustering, especially when multiple
    CPUs are available. Defaults to `1`.
* **Returns:**
  The prepared nn.Module with fake palettization
  modules inserted. This is a data-free PTP compressed model.
* **Raises:**
  * **RuntimeError** – If the model has already been prepared.
  * **ValueError** – If `num_workers` is less than 1.
* **Return type:**
  *Module*

#### save_sensitivities(path)

Save sensitivity values from the prepared model to a file.

This method extracts the sensitivity values currently set in the model’s
\_KMeansFakePalettize modules and saves them to the specified path. This is
useful when sensitivities were computed via calibration_mode() but not
saved at that time.

The saved sensitivities can later be loaded using prepare(sensitivity_path=…)
to apply the same weighted k-means clustering to a fresh model.

* **Parameters:**
  **path** (*str*) – File path where sensitivities will be saved
* **Raises:**
  * **RuntimeError** – If the model has not been prepared yet
  * **ValueError** – If no sensitivities are found in the model
* **Return type:**
  None
