# coreai_opt.pruning.config.ConstantSparsitySchedule

### *class* coreai_opt.pruning.config.ConstantSparsitySchedule

Bases: [`SparsityScheduleBase`](coreai_opt.pruning.config.SparsityScheduleBase.md#coreai_opt.pruning.config.SparsityScheduleBase)

Step function: zero before `begin_step`, `target_sparsity` at and after.

#### begin_step

Step at which to switch from 0 to `target_sparsity`.
Default: 0.

* **Type:**
  int

#### compute_sparsity(step_count, target_sparsity, prev_sparsity=None)

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
