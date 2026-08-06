# Inspecting PyTorch Model Structure

`coreai-opt` configs reference module names, module types, op names, and op types to target specific parts of a model. Before writing a config, you need to know exactly which strings your model exposes. {class}`~coreai_opt.inspection.ModelInspector` discovers these automatically and provides query methods corresponding to each config key type (`op_type_config`, `op_name_config`, `module_name_configs`, `module_type_configs`).

## Execution Modes

`ModelInspector` supports two execution modes, selected via the `execution_mode` argument:

- **Graph mode** (`execution_mode="graph"`): Exports the model with `torch.export` and walks the resulting FX graph. Op names are global identifiers assigned during export (for example, `"linear"`, `"linear_1"`). The compressor must be `Quantizer` or `None`.
- **Eager mode** (`execution_mode="eager"`): Intercepts operations during a live forward pass. Op names are module-qualified identifiers that reflect the module hierarchy (for example, `"linear1.linear"`, `"linear2.linear"`). This mode supports both `Quantizer` and `KMeansPalettizer` as the compressor.

If you plan to compress the model using one of `coreai-opt`'s compression techniques, choose the `execution_mode` you plan to use when compressing for inspection in order to identify the correct op and module names to use in the compression config.

For more information on `graph` mode vs. `eager` mode, see [here](../quantization/overview.md#two-execution-modes-graph-and-eager).

## Basic Usage

```python
import torch
import torch.nn as nn
from coreai_opt.inspection import ModelInspector
from coreai_opt.quantization import Quantizer


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


model = MyModel()

inspector = ModelInspector(
    model,
    example_inputs=(torch.randn(1, 10),),
    execution_mode="graph",
    compressor=Quantizer,
)

# Print a module-hierarchy tree showing ops, connectivity, and source locations
print(inspector.format_summary())
```

Pass `colorize=False` to suppress ANSI color codes, for example when writing to a file.

:::{note}
Note the use of `compressor=Quantizer`. This filters the captured and displayed ops to those registered as compressible by `Quantizer`. Omit this argument to capture and display all ops.
:::

The above code produces output like the following (colors omitted for brevity):

```text
Legend:
  ■ module_name (module_type)  ◆ op_name [op_type]

  op inputs:  {I: producer[N]}  —  I = op_input_spec index; N = output slot of the producing op
  op states:     param_name    —  model parameter or buffer
  op outputs: {N: [consumers]} —  N = output slot index; consumers = ops receiving that output
  untracked_N                  —  input tensor whose producer was not intercepted (e.g. raw attribute or global tensor); still quantizable via op_input_spec
  module inputs:  {I: [op[N], ...]}  —  I = module_input_spec index; op[N] = op and its input slot receiving data from outside; absent keys = non-quantizable
  module outputs: {I: op[N]}         —  I = module_output_spec index; op[N] = op and its output slot leaving the module; absent keys = non-quantizable

(__main__.MyModel)
    module inputs:  {0: [linear[0]]}
    module outputs: {0: linear_1[0]}
├── ■ linear1 (torch.nn.modules.linear.Linear)
│       module inputs:  {0: [linear[0]]}
│       module outputs: {0: linear[0]}
│   └── ◆ linear [linear]
│         op inputs:  {0: x[0]}
│         op states:  weight, bias
│         op outputs: {0: [relu]}
├── ■ relu (torch.nn.modules.activation.ReLU)
│       module inputs:  {0: [relu[0]]}
│       module outputs: {0: relu[0]}
└── ■ linear2 (torch.nn.modules.linear.Linear)
        module inputs:  {0: [linear_1[0]]}
        module outputs: {0: linear_1[0]}
    └── ◆ linear_1 [linear]
          op inputs:  {0: relu[0]}
          op states:  weight, bias
          op outputs: {0: [output]}
```

Note that `relu` does not appear as an operation (`◆`) within the `relu` module, because `ReLU` is not a compressible op in `Quantizer`. It still appears as a module node (`■`) and in connectivity lines such as `op outputs: {0: [relu]}` and `op inputs: {0: relu[0]}`, because the relu tensor passes through and connects the two linear ops.

## Reading the Tree

### Module lines

Module lines use the form `■ module_name (module_type)`. For example, `■ linear1 (torch.nn.modules.linear.Linear)`:

- `"linear1"` is the module name, usable in `module_name_configs`.
- `"torch.nn.modules.linear.Linear"` is the module type, usable in `module_type_configs`.

**Module boundaries** appear indented under the module header:

- `module inputs: {I: [op[N], ...]}` — The activations entering this module from outside. `I` is the position in the module's input spec (matching `module_input_spec` in a config), `op` is the name of the first compressible op inside the module that receives data at that position, and `N` is the input slot on that op. A single external input can fan out to multiple ops. Keys absent from this dict correspond to non-quantizable positions (for example, state tensors or unused arguments).
- `module outputs: {I: op[N]}` — The activations leaving this module. `I` is the position in the module's output spec, `op` is the compressible op producing that output, and `N` is the op's output slot. Absent keys correspond to non-quantizable positions.

### Op lines

Op lines use the form `◆ op_name [op_type]`. For example, `◆ linear_1 [linear]`:

- `"linear_1"` is the op name, usable in `op_name_config`.
- `"linear"` is the op type, usable in `op_type_config`.

**Op connectivity** appears indented under the op header:

- `op inputs: {I: producer[N]}` — Activation inputs only (parameters and buffers are on a separate line). `I` is the argument position (matching `op_input_spec` in a config), `producer` is the name of the op that produced this tensor, and `N` is the output slot of that producer. For example, `{0: relu[0]}` means argument 0 comes from output slot 0 of the `relu` op.
- `op states: param_name, ...` — Model parameters and buffers consumed by this op. This line is omitted if the op takes no states.
- `op outputs: {N: [consumer1, consumer2, ...]}` — `N` is the output slot index, and the list contains the names of all ops consuming that output.
- `untracked_N` — Appears in place of a producer name when the input tensor's origin was not intercepted (for example, a raw module attribute or global tensor). These tensors are still quantizable via `op_input_spec`.
- `filepath` and `code` — Source file and line of the call that produced the op, shown as dim text.

## Eager Mode

To inspect using eager mode, pass `execution_mode="eager"`. The same `MyModel` example above yields:

```text
(__main__.MyModel)
    module inputs:  {0: [linear1.linear[0]]}
    module outputs: {0: linear2.linear[0]}
├── ■ linear1 (torch.nn.modules.linear.Linear)
│       module inputs:  {0: [linear1.linear[0]]}
│       module outputs: {0: linear1.linear[0]}
│   └── ◆ linear1.linear [linear]
│         op inputs:  {0: input_0}
│         op states:  weight, bias
│         op outputs: {0: [relu.relu]}
│         filepath:  my_model.py:16
├── ■ relu (torch.nn.modules.activation.ReLU)
│       module inputs:  {0: [relu.relu[0]]}
│       module outputs: {0: relu.relu[0]}
└── ■ linear2 (torch.nn.modules.linear.Linear)
        module inputs:  {0: [linear2.linear[0]]}
        module outputs: {0: linear2.linear[0]}
    └── ◆ linear2.linear [linear]
          op inputs:  {0: relu.relu[0]}
          op states:  weight, bias
          op outputs: {0: [output_0]}
          filepath:  my_model.py:18
```

## Querying Operations by Config Key

Once you have reviewed the full summary to see what names and types are present, use the query methods to check which operations a specific pattern matches. This is useful for verifying that a config targets the intended ops before applying compression.

Each query method returns a tuple of {class}`~coreai_opt.inspection.OpInfo` objects matching the filter. The method names correspond directly to the config keys they help populate.

From the graph mode summary above, this model exposes:

- **Op types**: `linear`
- **Op names**: `linear`, `linear_1`
- **Module types**: `torch.nn.modules.linear.Linear`, `torch.nn.modules.activation.ReLU`
- **Module names**: `linear1`, `relu`, `linear2`

Op names and module names can be passed as a literal string or as a regex following [Python re syntax](https://docs.python.org/3/library/re.html) for wildcard matching. The pattern is matched against the full string. The matching behavior is identical to how compression config entries match modules and ops, so you can see exactly which ops a given pattern would select.

**By op type** — exact-string match against `op_type_config` keys:

```python
inspector.get_matched_ops_for_op_type("linear")  # matches both linear ops
```

**By op name** — regex against `op_name_config` keys:

```python
inspector.get_matched_ops_for_op_name("linear_1")  # matches just linear_1
inspector.get_matched_ops_for_op_name(".*linear.*")  # matches both linear and linear_1
```

**By module name** — regex against `module_name_configs` keys:

```python
inspector.get_matched_ops_for_module_name(
    "linear1"
)  # matches the op in module "linear1"
inspector.get_matched_ops_for_module_name(
    "linear[12]"
)  # matches ops in "linear1" and "linear2"
```

Each returned {class}`~coreai_opt.inspection.OpInfo` provides `op_name`, `op_type`, `module_stack`, `inputs`, `outputs`, and `is_state`. The `module_stack` is a tuple of {class}`~coreai_opt.inspection.ModuleContext` entries from outermost to innermost module:

```python
>>> for op in inspector.get_matched_ops_for_op_type("linear"):
...     print(f"  op_name={op.op_name}, op_type={op.op_type}")
...     print(f"  module: {op.module_stack[-1].module_name} ({op.module_stack[-1].module_type})")
  op_name=linear, op_type=linear
  module: linear1 (torch.nn.modules.linear.Linear)
  op_name=linear_1, op_type=linear
  module: linear2 (torch.nn.modules.linear.Linear)
```

`OpInfo.inputs` is a tuple of {class}`~coreai_opt.inspection.InputEdge` objects, one per input argument position. Each `InputEdge` carries the producing `OpInfo` and the output slot index (`output_idx`) of that producer. State inputs (parameters, buffers) are interleaved in the tuple at their actual argument positions, and their corresponding `InputEdge` objects have `is_state=True`.

Using these strings directly in a config:

```python
config = QuantizerConfig(
    # Target a specific module by name
    module_name_configs={
        "linear1": ModuleQuantizerConfig(...),
    },
    # Target all modules of a given type
    module_type_configs={
        "torch.nn.modules.linear.Linear": ModuleQuantizerConfig(...),
    },
)

# Op-level targeting within a ModuleQuantizerConfig
config = QuantizerConfig(
    global_config=ModuleQuantizerConfig(
        # Target a specific op by name
        op_name_config={
            "linear_1": OpQuantizerConfig(...),
        },
        # Target all ops of a given type
        op_type_config={
            "linear": OpQuantizerConfig(...),
        },
    ),
)
```

## Navigating the Module Hierarchy

For programmatic access to the inspector's data structures, the {class}`~coreai_opt.inspection.ModelSummary` exposes a {class}`~coreai_opt.inspection.ModuleInfo` tree that mirrors the `nn.Module` hierarchy. These types are publicly exported from `coreai_opt.inspection` for use in custom analysis or tooling.

```python
>>> root = inspector.summary.model
>>> for name, child in root.named_children():
...     print(f"{name}: {child.module_type}, {len(child.ops)} direct ops")
linear1: torch.nn.modules.linear.Linear, 1 direct ops
relu: torch.nn.modules.activation.ReLU, 0 direct ops
linear2: torch.nn.modules.linear.Linear, 1 direct ops
```

```python
# Look up a specific submodule
linear2_module = root.get_submodule("linear2")

# Get all ops under this subtree (depth-first)
linear2_ops = linear2_module.all_ops()
```

`ModuleInfo` supports the same iteration patterns as `nn.Module`: `children()`, `named_children()`, `modules()`, `named_modules()`, and `get_submodule()`.

`ModuleInfo` also exposes the module boundary connectivity described in the tree:

- `input_ops` — dict mapping module input spec index to a list of {class}`~coreai_opt.inspection.BoundaryEdge` objects, each holding the op and input slot receiving data from outside the module.
- `output_ops` — dict mapping module output spec index to a single {class}`~coreai_opt.inspection.BoundaryEdge`, holding the op and output slot whose tensor leaves the module.

```python
# Inspect boundary connectivity for a submodule
linear1_module = root.get_submodule("linear1")
for idx, edges in linear1_module.input_ops.items():
    for edge in edges:
        print(f"  module input {idx} -> {edge.op.op_name}[{edge.index}]")
```
