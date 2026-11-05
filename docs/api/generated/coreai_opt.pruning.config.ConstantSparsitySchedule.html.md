# coreai_opt.pruning.config.ConstantSparsitySchedule

### *class* coreai_opt.pruning.config.ConstantSparsitySchedule

Bases: [`SparsityScheduleBase`](coreai_opt.pruning.config.SparsityScheduleBase.md#coreai_opt.pruning.config.SparsityScheduleBase)

Step function: zero before `begin_step`, `target_sparsity` at and after.

#### begin_step

Step at which to switch from 0 to `target_sparsity`.
Default: 0.

* **Type:**
  int

#### compute_sparsity(step_count, target_sparsity)

Return the sparsity that should be applied at *step_count*.

* **Parameters:**
  * **step_count** (*int*) – The current step count of the pruner
    (monotonically increasing).
  * **target_sparsity** (*float*) – The final sparsity we want to reach at the
    end of the pruning schedule.
* **Returns:**
  The sparsity level to apply at the current step.
* **Return type:**
  float
