# coreai_opt.pruning.config.PolynomialDecaySchedule

### *class* coreai_opt.pruning.config.PolynomialDecaySchedule

Bases: [`SparsityScheduleBase`](coreai_opt.pruning.config.SparsityScheduleBase.md#coreai_opt.pruning.config.SparsityScheduleBase)

Polynomial schedule from `initial_sparsity` to `target_sparsity`.

Inspired by PyTorch’s `torch.optim.lr_scheduler.PolynomialLR` and the paper
[“To prune or not to prune”](https://arxiv.org/pdf/1710.01878.pdf).

Behavior by step:

- `step < begin_step` → `initial_sparsity`
- `begin_step <= step < begin_step + total_iters` → scheduled value
- `step >= begin_step + total_iters` → `target_sparsity`

Formula at update index $i \in [0, n\_updates - 1]$:

$$
t = i / \max(n\_updates - 1, 1)

sparsity = target + (initial - target) \cdot (1 - t)^{power}
$$

#### begin_step

Step at which the schedule starts. Default: 0.

* **Type:**
  int

#### total_iters

Length of the schedule in steps. Must be positive.

* **Type:**
  int

#### power

Polynomial exponent. `1.0` is linear; higher values
keep sparsity low for longer before climbing. Default: 3.0.

* **Type:**
  float

#### initial_sparsity

Sparsity before and at the start of the
schedule, in `[0, 1]`. Default: 0.0.

* **Type:**
  float

#### update_frequency

Steps between sparsity updates within the
schedule. Must be >= 1. Default: 1 (update every step).

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
