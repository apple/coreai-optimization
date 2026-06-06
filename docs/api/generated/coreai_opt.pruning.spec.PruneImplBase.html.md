# coreai_opt.pruning.spec.PruneImplBase

### *class* coreai_opt.pruning.spec.PruneImplBase(target_sparsity, pruning_scheme, \*\*kwargs)

Bases: [`CompressionSimulatorBase`](coreai_opt.config.spec.CompressionSimulatorBase.md#coreai_opt.config.spec.CompressionSimulatorBase)

Abstract base for pruning parametrizations that mask a layer’s weight.

Subclasses implement [`compute_mask()`](#coreai_opt.pruning.spec.PruneImplBase.compute_mask) — a pure static function from
`(weight, sparsity, pruning_scheme)` to a binary mask. The base class
handles the mask buffer and optional schedule-driven sparsity updates.

* **Parameters:**
  * **target_sparsity** (*float*)
  * **pruning_scheme** ([*PruningScheme*](coreai_opt.pruning.spec.PruningScheme.md#coreai_opt.pruning.spec.PruningScheme))
  * **kwargs** (*Any*)

#### \_\_init_\_(target_sparsity, pruning_scheme, \*\*kwargs)

Initialize internal Module state, shared by both nn.Module and ScriptModule.

* **Parameters:**
  * **target_sparsity** (*float*)
  * **pruning_scheme** ([*PruningScheme*](coreai_opt.pruning.spec.PruningScheme.md#coreai_opt.pruning.spec.PruningScheme))
  * **kwargs** (*Any*)

### Methods

| [`compute_mask`](#coreai_opt.pruning.spec.PruneImplBase.compute_mask)(weight, sparsity, pruning_scheme)   | Compute a binary pruning mask for the given weight tensor.                        |
|-----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| [`forward`](#coreai_opt.pruning.spec.PruneImplBase.forward)(weight)                                       | Compute / re-compute the mask if stale, and then apply it to the weight.          |
| `get_class`(key)                                                                                          |                                                                                   |
| `list_registry_keys`()                                                                                    |                                                                                   |
| `list_registry_values`()                                                                                  |                                                                                   |
| `register`(key)                                                                                           | Register a virtual subclass of an ABC.                                            |
| [`update_sparsity`](#coreai_opt.pruning.spec.PruneImplBase.update_sparsity)(step_count)                   | Update the sparsity based on the configured schedule and the provided step count. |
| [`with_args`](#coreai_opt.pruning.spec.PruneImplBase.with_args)(\*\*kwargs)                               | Create a partial constructor with pre-filled arguments.                           |

#### *abstract static* compute_mask(weight, sparsity, pruning_scheme)

Compute a binary pruning mask for the given weight tensor.

* **Parameters:**
  * **weight** (*torch.Tensor*) – The weight tensor to compute a mask for.
  * **sparsity** (*float*) – Fraction of elements to prune, in [0, 1].
  * **pruning_scheme** ([*PruningScheme*](coreai_opt.pruning.spec.PruningScheme.md#coreai_opt.pruning.spec.PruningScheme)) – Structural pattern of sparsity.
* **Returns:**
  Binary mask with the same shape as *weight* (1 = keep,
  0 = prune).
* **Return type:**
  torch.Tensor

#### forward(weight)

Compute / re-compute the mask if stale, and then apply it to the weight.

* **Parameters:**
  **weight** (*Tensor*)
* **Return type:**
  *Tensor*

#### update_sparsity(step_count)

Update the sparsity based on the configured schedule and the provided step count.

* **Raises:**
  **RuntimeError** – If no schedule is attached. This method should be
      invoked only after setting the `schedule` property.
* **Parameters:**
  **step_count** (*int*)
* **Return type:**
  None

#### *classmethod* with_args(\*\*kwargs)

Create a partial constructor with pre-filled arguments.

* **Parameters:**
  **kwargs** (*Any*)
* **Return type:**
  *PartialConstructor*[[*PruneImplBase*](#coreai_opt.pruning.spec.PruneImplBase)]

#### schedule *: [SparsityScheduleBase](coreai_opt.pruning.config.SparsityScheduleBase.md#coreai_opt.pruning.config.SparsityScheduleBase) | None* *= None*

#### *property* sparsity *: float*

Sparsity that the current mask reflects. Use `update_sparsity` to change.
