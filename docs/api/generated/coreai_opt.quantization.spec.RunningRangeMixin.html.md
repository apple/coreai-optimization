# coreai_opt.quantization.spec.RunningRangeMixin

### *class* coreai_opt.quantization.spec.RunningRangeMixin(\*args, \*\*kwargs)

Bases: `object`

Mixin for calculators that maintain running min/max range buffers.

Provides `running_min` and `running_max` buffers, first-forward
initialization via `_initialize_state`, and a
`compute_qparams` implementation that delegates
the range-update rule to the abstract `update_running_range` hook.

Subclasses that want to re-use the logic of computing quantization
parameters but with different ways of updating the running statistics
can override the `update_running_range` method.

Must appear before `StatefulQParamsCalculatorBase` in the MRO so that its
`compute_qparams` and `_initialize_state`
take precedence over the base-class defaults.

* **Parameters:**
  * **args** (*object*)
  * **kwargs** (*object*)

#### \_\_init_\_(\*args, \*\*kwargs)

* **Parameters:**
  * **args** (*object*)
  * **kwargs** (*object*)
* **Return type:**
  None

### Methods

| [`compute_qparams`](#coreai_opt.quantization.spec.RunningRangeMixin.compute_qparams)(tensor, min_val, max_val)   | Update running range, persist to buffers, then compute qparams.              |
|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| [`extra_repr`](#coreai_opt.quantization.spec.RunningRangeMixin.extra_repr)()                                     |                                                                              |
| [`reset_running_range`](#coreai_opt.quantization.spec.RunningRangeMixin.reset_running_range)()                   | Mark the running range as unset so the next observed range starts it afresh. |
| [`update_running_range`](#coreai_opt.quantization.spec.RunningRangeMixin.update_running_range)(min_val, max_val) | Return `(updated_min, updated_max)` using subclass-specific rule.            |

#### compute_qparams(tensor, min_val, max_val)

Update running range, persist to buffers, then compute qparams.

* **Parameters:**
  * **tensor** (*Tensor*)
  * **min_val** (*Tensor*)
  * **max_val** (*Tensor*)
* **Return type:**
  tuple[*Tensor*, *Tensor* | None, *Tensor* | None]

#### extra_repr()

* **Return type:**
  str

#### reset_running_range()

Mark the running range as unset so the next observed range starts it afresh.

Sets `running_min` to `+inf` and `running_max` to `-inf`: an empty range that
observed data can’t produce, since observed minimums are clamped to `<= 0` and
maximums to `>= 0`.

The buffers keep their shape, so this stays `torch.compile`
friendly and survives `state_dict` round trips. The current qparams (scale, zero
point) stay usable. Subclasses that keep extra running statistics should override this
(calling `super()`) to clear them too.

* **Return type:**
  None

#### *abstract* update_running_range(min_val, max_val)

Return `(updated_min, updated_max)` using subclass-specific rule.

`self.running_min` / `self.running_max` may be `+inf` / `-inf` where the running
range is unset (see `reset_running_range`). Implementations must not raise on these
values; `compute_qparams` discards those entries in favor of `min_val` / `max_val`.

* **Parameters:**
  * **min_val** (*Tensor*)
  * **max_val** (*Tensor*)
* **Return type:**
  tuple[*Tensor*, *Tensor*]

#### running_max *: Tensor*

#### running_min *: Tensor*
