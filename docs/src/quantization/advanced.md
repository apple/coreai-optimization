# Deeper Dive

## QAT: API overview with custom schedule

The QAT workflow shown in [Quantization Overview](overview.md#weight-andor-activation-quantization-qat-quantization-aware-training) demonstrates a typical scenario.
However, one of the hyperparameters that may need to be tuned, for better accuracy, is the "QAT schedule".
Through an example, let us see how the API lets us do that.

Let's say, here is what we want to do during QAT:

- We want to do QAT for 30 epochs. Each epoch comprises 100 mini-batch grad steps. (Total: 3k steps).
- For 1.5 epochs, i.e. 150 steps, keep the observers ON (so that quant scales and zp can get updated based on data distribution) while keeping fake quant OFF. This phase would be different than PTQ calibration in that the weights will continue to get updated.
- Turn on fake quant after 150 steps, so that the model output and loss incorporate the effect of quantization, and gradient updates start to change the weights to adjust to quantization effects. The weight updates will continue till the end of training (30th epoch).
- At epoch 20 (i.e. step count 2k), we want to turn the observers OFF, so that quant scales and zp stop updating, and the weights can continue to adapt to a model with fixed qparams.

Now let's see what this would look like in pseudo-code:

```python
from coreai_opt.quantization import Quantizer, QuantizerConfig, ModuleQuantizerConfig
from coreai_opt.quantization.config import QATSchedule
import torch

model = MyModel()
example_inputs = (
    ...
)  # use a representative data sample when activation quantization is in use

# default INT8_INT8 config
config = QuantizerConfig(
    global_config=ModuleQuantizerConfig(
        qat_schedule=QATSchedule(
            enable_observer=0,  # step 0: observers ON, fake quant OFF (to start with)
            enable_fake_quant=150,  # step 150: observers already ON, fake quant turned ON
            disable_observer=2000,  # step 2000: observers turned OFF, fake quant stays ON
        )
    )
)

"""
# equivalent yaml config
quantization_config:
  global_config:
    qat_schedule:
      enable_observer: 0
      enable_fake_quant: 150
      disable_observer: 2000
"""

quantizer = Quantizer(model, config)
prepared_model = quantizer.prepare(example_inputs)

# Training loop

optimizer = torch.optim.Adam(prepared_model.parameters(), lr=0.01)
# define the learning rate scheduler (e.g., reduce LR by half every 5 epochs)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

for epoch in range(30):
    with quantizer.training_mode():
        # enter QAT context.
        #  - It will apply the QAT schedule specified in the config.
        #  - It also moves the model to the train mode and resets it back to whatever it was on exiting the context
        for batch, target in train_dataloader:
            optimizer.zero_grad()
            output = prepared_model(batch)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()  # update weights
            # advance the QAT schedule (inside the mini-batch loop as QATschedule is specified in units of 'steps')
            quantizer.step()

        # update the LR. Outside of the mini-batch loop (for per-epoch schedulers)
        scheduler.step()

    # outside "training_mode" : prepared model restored to validation state (observers off, fake quantization on)
    val_metric = validate(prepared_model, val_dataloader)
```

If the schedule we wanted to try was slightly different: turn fake quant ON after one epoch, we could use purely epoch-based counting, similar to the LR scheduler:

```python
# default INT8_INT8 config
config = QuantizerConfig(
    global_config=ModuleQuantizerConfig(
        qat_schedule=QATSchedule(
            enable_observer=0,  # turn observer ON at 0-th epoch
            enable_fake_quant=1,  # turn fake quant ON after 1st epoch
            disable_observer=20,  # turn observer OFF after 20-th epoch
        )
    )
)

for epoch in range(30):
    with quantizer.training_mode():
        # enter QAT context. It will apply the QAT schedule specified in the config.
        for batch, target in train_dataloader:
            optimizer.zero_grad()
            output = prepared_model(batch)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()  # update weights

        # update the LR. Outside of the mini-batch loop (for per-epoch schedulers)
        scheduler.step()
        # update the QAT schedule counts. Outside of the mini-batch loop
        quantizer.step()

    # outside "training_mode" : prepared model restored to validation state (observers off, fake quantization on)
    val_metric = validate(prepared_model, val_dataloader)
```

Since `QATSchedule` is a property of {class}`~coreai_opt.quantization.config.ModuleQuantizerConfig`, different ones can be defined for different modules, to customize the schedule for different parts of the model.

If not provided it defaults to `{enable_observer=0, enable_fake_quant=0, disable_observer=∞}`, which matches the behavior shown in the [QAT example](overview.md#weight-andor-activation-quantization-qat-quantization-aware-training).

If for a certain reason (e.g. if the observer/fake_quantize enablement/disablement needs to be tuned based on values of loss or validation metric instead of a predefined schedule), you do not want to use the `QATSchedule` and want to do it explicitly, it can be done by using the methods `enable_observer`, `disable_observer`, `disable_fake_quant` and `enable_fake_quant`, without invoking the `training_mode` context manager.

```python
from coreai_opt.quantization import Quantizer, QuantizerConfig

# default INT8_INT8 config
config = QuantizerConfig()
quantizer = Quantizer(model, config)
prepared_model = quantizer.prepare(
    example_inputs
)  # prepared model: observers OFF, fake quant ON

# Training loop
for epoch in range(30):
    if epoch < 1:
        quantizer.enable_observer()  # update quant scales
        quantizer.disable_fake_quant()  # loss/output does not have effect of quantization
    elif epoch < 20:
        quantizer.enable_observer()  # update quant scales
        quantizer.enable_fake_quant()  # weight update adjusting to quantization effect
    else:
        quantizer.disable_observer()  # freeze quant scales
        quantizer.enable_fake_quant()

    prepared_model.train()

    for batch, target in train_dataloader:
        optimizer.zero_grad()
        output = prepared_model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # validation code

    # get the model in a state apt for validation
    prepared_model.eval()
    quantizer.disable_observer()
    quantizer.enable_fake_quant()
    val_metric = validate(prepared_model, val_dataloader)
```

## Symmetric vs asymmetric quantization

Quantization maps floating-point values to a fixed set of discrete values (**bins**). The number of bins is determined by the dtype — for example, `int8` has 256 bins (`-128` to `127`), `uint8` has 256 bins (`0` to `255`), and `int4` has 16 bins (`-8` to `7`). Each bin represents a distinct quantized value that a floating-point value can be mapped to.

The `qscheme` controls how these bins are distributed around zero, by determining where the **zero point** is placed:

- **Symmetric**: the zero point is fixed at `0` for signed types (`int8`, `int4`) or at the midpoint for unsigned types (`128` for `uint8`, `8` for `uint4`). This places an equal number of bins on both sides of the zero point, which works well when the data distribution is roughly centered around zero.
- **Asymmetric**: the zero point is chosen based on the observed data distribution (running statistics during calibration). This typically places an unequal number of bins on each side of the zero point, which allows the quantized range to better fit skewed distributions — for example, activations that are always non-negative after `relu`.

## Quantization Defaults for Known-Range Activations

In graph mode, certain activation ops have analytically known output ranges. For these ops, the quantizer overrides the `qscheme` and `float_range` of the qparams calculator at prepare time, regardless of what the user configured. The user's `dtype` is always preserved — these adjustments do not change the number of bits or the signed/unsigned choice.

The scale and zero point values in the table below assume the default `int8` dtype. For other dtypes, the same formulas apply with the appropriate `quant_min` / `quant_max`.

| Op            | Output range        | `qscheme`  | `float_range` | Scale (int8) | Zero point (int8) |
| ------------- | ------------------- | ---------- | ------------- | ------------ | ----------------- |
| `hardsigmoid` | [0, 1]              | asymmetric | (0, 1)        | 1 / 255      | −128              |
| `hardtanh`    | Depends (see below) | Depends    | Depends       | Depends      | Depends           |
| `relu`        | \[0, ∞)             | asymmetric | (0, None)     | dynamic      | −128              |
| `relu6`       | [0, 6]              | asymmetric | (0, 6)        | 6 / 255      | −128              |
| `sigmoid`     | [0, 1]              | asymmetric | (0, 1)        | 1 / 255      | −128              |
| `tanh`        | [−1, 1]             | symmetric  | (−1, 1)       | 2 / 255      | 0                 |

**Relu**: The lower bound of `float_range` is pinned to 0 and `qscheme` is set to asymmetric. Because the observed minimum is always 0, the zero point is fixed at `quant_min` (−128 for int8) and stays there regardless of calibration data. The upper bound remains `None` (data-driven), so the scale continues to update during calibration.

**Sigmoid and hardsigmoid**: Both `qscheme` and `float_range` are fully pinned. Scale and zero point are entirely determined by the dtype and the fixed output range — calibration data has no effect on them.

**Tanh**: `qscheme` (symmetric) and `float_range` (−1, 1) are fully pinned. Scale and zero point are entirely determined by the dtype and range.

**Hardtanh**: Bounds are read from the op's node arguments at prepare time, so the effective range and qscheme depend on how the op was configured. If `min_val == −max_val` the range is symmetric around zero and `qscheme` is set to symmetric; otherwise `qscheme` is set to asymmetric. Both ends of `float_range` are pinned to the configured bounds. `relu6` is a special case of `hardtanh(0, 6)` and is handled identically.

> **Motivation for asymmetric treatment**: Symmetric quantization places the zero point at the center of the quantized range. For `relu`, `sigmoid`, and `hardsigmoid`, whose outputs are always non-negative, symmetric quantization places half the bins in negative territory that the op never produces — wasting half the available resolution. Asymmetric quantization shifts the zero point to the edge of the range so that all bins cover values the op actually generates. For `tanh` and symmetric `hardtanh`, the output is centered at zero so both halves of the range are used equally, and symmetric quantization is appropriate.

Eager mode does not perform these adjustments — all activations are quantized uniformly using the user-configured spec.

## Customization options

### Custom patterns for placement of activation quantizers

The graph-mode quantizer uses a pattern registry (`_AnnotationPatternRegistry`) to determine where activation quantizers are placed in the graph. When a pattern matches a subgraph, the quantizer inserts quantize/dequantize nodes around the matched ops' inputs and outputs.

To add activation quantization for a custom op with an activation function, subclass `NAryActPattern` and register it. For example, to quantize `div -> activation` as a fused pair:

```python
from coreai_opt.quantization._graph._annotation_pattern_registry import (
    NAryActPattern,
    _AnnotationPatternRegistry,
    _get_all_patterns_from_base_ops,
)


@_AnnotationPatternRegistry.register("div_act")
class DivActPattern(NAryActPattern):
    @classmethod
    def generate_patterns(cls):
        return _get_all_patterns_from_base_ops(
            {torch.div, operator.truediv}, use_act=True
        )
```

`use_act=True` generates patterns for every supported activation function (relu, silu, gelu, etc.) appended to the base op. The quantizer then treats the `div -> activation` pair as a single unit, placing quantizers only on the inputs and final output.

For multi-op chains longer than 2, `NAryActPattern` does not support this — the annotation function raises an error for chains longer than 2. You need to subclass `BaseAnnotationPattern` directly with a custom annotator function. Note that sequential partition matching requires each op type in the chain to be unique (e.g., `mul -> sub` works but `mul -> mul -> sub` does not):

```python
from coreai_opt.quantization._graph._annotation_pattern_registry import (
    BaseAnnotationPattern,
    _AnnotationPatternRegistry,
)
from coreai_opt.quantization._graph._annotation_utils import (
    OpsListPattern,
    Q_ANNOTATION_KEY,
    _get_call_function_node_from_partition,
    _get_input_qspec_map,
    _get_output_qspec,
    is_any_annotated,
    mark_nodes_as_annotated,
    match_pattern_with_sequential_partitions,
)
from torch.ao.quantization.quantizer import QuantizationAnnotation


def _annotate_multi_op_match(
    annotator_match, quantization_config, shared_observer_nodes
):
    """Annotate a multi-op chain: quantize inputs of first op, output of last op."""
    nodes = [_get_call_function_node_from_partition(p) for p in annotator_match]
    if is_any_annotated(nodes):
        return

    first_node, last_node = nodes[0], nodes[-1]
    input_qspec_map = _get_input_qspec_map(
        first_node.all_input_nodes, quantization_config
    )
    output_qspec = _get_output_qspec(
        last_node, quantization_config, shared_observer_nodes
    )

    # Annotate first op with input specs only
    first_node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        _annotated=True,
    )
    # Mark intermediate ops as annotated (no q/dq inserted between them)
    mark_nodes_as_annotated(nodes[1:-1])
    # Annotate last op with output spec only
    last_node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        output_qspec=output_qspec,
        _annotated=True,
    )


@_AnnotationPatternRegistry.register("add_mul_sub")
class AddMulSubPattern(BaseAnnotationPattern):
    @classmethod
    def get_annotator_func(cls):
        return _annotate_multi_op_match

    @classmethod
    def generate_patterns(cls):
        # _get_function_or_string_set converts torch functions to string names
        # for torch >= 2.8 compatibility
        add_names = _get_function_or_string_set({torch.add, operator.add})
        mul_names = _get_function_or_string_set({torch.mul, operator.mul})
        sub_names = _get_function_or_string_set({torch.sub, operator.sub})
        return [
            OpsListPattern([a, m, s])
            for a in add_names
            for m in mul_names
            for s in sub_names
        ]

    @classmethod
    def match_single_pattern(cls, model, pattern):
        return match_pattern_with_sequential_partitions(model, pattern)
```

Registering with a key that already exists overwrites the previous pattern with a warning. There is no `unregister` method — to remove a pattern, delete it directly from the registry dict:

```python
del _AnnotationPatternRegistry.REGISTRY["add_mul_sub"]
```

### Custom hooks for quantization param calculator

The {class}`~coreai_opt.quantization.spec.QuantizationSpec` fields `qparam_calculator_cls`, `range_calculator_cls`, and `fake_quantize_cls` are all pluggable via the same registry pattern. To create a custom component:

1. Subclass the corresponding base class
2. Register it with `@BaseClass.register("my_key")`
3. Reference it by string key in {class}`~coreai_opt.quantization.spec.QuantizationSpec`

The following example implements a custom qparam calculator that tracks the maximum observed range across all calibration batches (as opposed to {class}`~coreai_opt.quantization.spec.MovingAverageQParamsCalculator` which uses EMA smoothing):

```python
import torch
from coreai_opt.quantization.spec import (
    QParamsCalculatorBase,
    RunningRangeMixin,
)


@QParamsCalculatorBase.register("max_range")
class MaxRangeQParamsCalculator(RunningRangeMixin, QParamsCalculatorBase):
    """Track the widest observed min/max range across all calibration batches."""

    def update_running_range(
        self, min_val: torch.Tensor, max_val: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        running_min = torch.minimum(self.running_min, min_val)
        running_max = torch.maximum(self.running_max, max_val)
        return running_min, running_max
```

Use it by passing the registered key to {class}`~coreai_opt.quantization.spec.QuantizationSpec`:

```python
from coreai_opt.quantization import QuantizationSpec

spec = QuantizationSpec(
    dtype=torch.int8,
    qparam_calculator_cls="max_range",
)
```

The `RunningRangeMixin` provides `running_min`/`running_max` buffers and handles initialization on the first forward pass. Subclasses only need to implement `update_running_range()` to define how the running statistics are updated each batch. For stateless calculators that don't need running state, subclass `QParamsCalculatorBase` directly and override `compute_qparams()`.

The same pattern applies to the other pluggable fields:

- `range_calculator_cls`: subclass `RangeCalculatorBase`, register, and reference by key
- `fake_quantize_cls`: subclass `FakeQuantizeImplBase`, register, and reference by key
