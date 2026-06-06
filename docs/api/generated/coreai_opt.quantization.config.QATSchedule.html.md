# coreai_opt.quantization.config.QATSchedule

### *class* coreai_opt.quantization.config.QATSchedule

Bases: `BaseModel`

Schedule for controlling observer and fake quantization state in QAT.

Defines step thresholds for enabling/disabling observers and fake quantization
during quantization-aware training. Must be used in conjunction with the
`quantizer.step()` API to advance the schedule.

The step values correspond to the cadence at which `quantizer.step()` is
called. For example, if `step()` is called once per batch, the thresholds
represent batch steps; if called once per epoch, they represent epochs.

Calling `step()` increments the step counter and immediately applies
the corresponding observer/fake-quantization state. Where you place
`step()` in your training loop determines when the model sees the
new state.

#### enable_observer

Step count at which observers are enabled. Must be >= 0.

#### enable_fake_quant

Step count at which fake quantization is enabled.
Must be >= enable_observer.

#### disable_observer

Step count at which observers are disabled. Must be
> enable_observer and >= enable_fake_quant if provided. None means
observers are never disabled by the schedule.

#### NOTE
In graph execution mode, when consecutive modules both quantize the
intermediate edge (one via `op_output_spec`, the next via
`op_input_spec`), graph mode deduplicates them into a single
fake-quantize node. The schedule of the consuming module is always
applied to the deduplicated node, irrespective of the choice of
deduplication made by the graph preparation.

#### NOTE
When two modules share a weight parameter and have different
schedules, the schedule of the first module encountered in the
module tree is applied. A warning is emitted for the conflict if
there is no fake-quantize node deduplication happening (in Eager
execution mode).
