# coreai_opt.pruning.config.SparsityScheduleBase

### *class* coreai_opt.pruning.config.SparsityScheduleBase

Bases: `BaseModel`, `ConfigRegistryMixin`

Abstract base for sparsity schedules used by `MagnitudePruner`.

A sparsity schedule defines how the sparsity applied during pruning
evolves over training steps. Instead of applying the full target sparsity
immediately, a schedule lets sparsity rise gradually so the model can
adapt to it during training. Each schedule is a pure function of the
pruner’s step count and the spec’s target sparsity.

#### *abstract* compute_sparsity(step_count, target_sparsity, prev_sparsity=None)

Return the sparsity that should be applied at *step_count*.

* **Parameters:**
  * **step_count** (*int*) – The current step count of the pruner
    (monotonically increasing).
  * **target_sparsity** (*float*) – The final sparsity we want to reach at the
    end of the pruning schedule.
  * **prev_sparsity** (*float* *|* *None*) – Sparsity from the previous invocation.
    Schedules that don’t need this can ignore it; schedules that do
    (e.g. `PolynomialDecaySchedule` with an `update_frequency`
    gap) raise `ValueError` when omitted.
* **Returns:**
  The sparsity level to apply at the current step.
* **Return type:**
  float
