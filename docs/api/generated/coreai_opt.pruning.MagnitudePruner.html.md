# coreai_opt.pruning.MagnitudePruner

### *class* coreai_opt.pruning.MagnitudePruner(model, config=None)

Bases: `_BasePruner`, `EagerCompressionComponentBuilderMixin`

Apply magnitude-based pruning to a model.

This pruner zeros out the smallest-magnitude weight elements to reach a
configurable sparsity target. The model is parsed in an eager fashion and
the pruner registers parametrizations for each candidate parameter to be
pruned. The mask is applied on every forward pass while parametrizations
are active.

When a `sparsity_schedule` is configured on a module’s config, `step()`
advances the schedule and recomputes the mask for that module’s
parametrizations. Without a schedule, the spec’s `target_sparsity` is
applied statically.

* **Parameters:**
  * **model** (*torch.nn.Module*) – Model to prune.
  * **config** ([*MagnitudePrunerConfig*](coreai_opt.pruning.MagnitudePrunerConfig.md#coreai_opt.pruning.MagnitudePrunerConfig) *|* *None*) – Pruning configuration. When
    `None`, a default config with 50 % sparsity is used.

#### \_\_init_\_(model, config=None)

Initialize the model compressor.

* **Parameters:**
  * **model** (*Module*) – The PyTorch model to compress. The model will be modified in-place
    during the compression process.
  * **config** ([*MagnitudePrunerConfig*](coreai_opt.pruning.MagnitudePrunerConfig.md#coreai_opt.pruning.MagnitudePrunerConfig) *|* *None*) – Configuration parameters for the compression

### Methods

| `calibration_mode`([model])                                                  | Context manager for calibration data-based compression workflow.                                                |
|------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| [`finalize`](#coreai_opt.pruning.MagnitudePruner.finalize)([model, backend]) | Finalize the model to be lowered to the target backend.                                                         |
| [`prepare`](#coreai_opt.pruning.MagnitudePruner.prepare)(example_inputs)     | Prepare the model for pruning.                                                                                  |
| [`step`](#coreai_opt.pruning.MagnitudePruner.step)()                         | Advance the sparsity schedule by one step.                                                                      |
| `supported_modules`()                                                        | Returns types of modules that are supported for compression with for a particular model optimization technique. |
| `training_mode`([model])                                                     | Context manager for training time compression workflow.                                                         |

#### finalize(model=None, backend=ExportBackend.CoreAI)

Finalize the model to be lowered to the target backend.

* **Parameters:**
  * **model** (*torch.nn.Module* *|* *None*) – Model to finalize. Uses the model
    passed at construction time when `None`.
  * **backend** ([*ExportBackend*](coreai_opt.ExportBackend.md#coreai_opt.ExportBackend)) – Target export backend.
* **Returns:**
  The finalized model ready for the target backend.
* **Return type:**
  torch.nn.Module
* **Raises:**
  **RuntimeError** – If the model has not been prepared.

#### prepare(example_inputs)

Prepare the model for pruning.

* **Parameters:**
  **example_inputs** (*tuple* *[**torch.Tensor* *]*) – Sample inputs to trace the
  model and configure pruning parametrizations.
* **Returns:**
  The prepared model with pruning parametrizations.
* **Return type:**
  torch.nn.Module
* **Raises:**
  **RuntimeError** – If the model has already been prepared.

#### step()

Advance the sparsity schedule by one step.

Increments the step count, then recomputes and applies the mask for
every parametrization with a configured `sparsity_schedule`. Safe to
call when no schedule is configured (no-op).

* **Return type:**
  None
