# Inspecting PyTorch Model Structure

`coreai-opt` configs reference module names, module types, op names, and op types to target specific parts of a model. Before writing a config, you need to know exactly which strings your model exposes. {class}`~coreai_opt.inspection.ModelInspector` discovers these automatically and provides query methods corresponding to each config key type (`op_type_config`, `op_name_config`, `module_name_configs`, `module_type_configs`).

:::{note}
`ModelInspector` currently supports **graph execution mode only**. Eager mode support is planned. For eager mode op naming, see [How to get names + types](../quantization/config.md#how-to-get-names--types-for-modules-and-ops).
:::

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

# Print a module-hierarchy tree showing ops, source locations, and connectivity
print(inspector.format_summary())
```

Note the use of `compressor=Quantizer` in the list of arguments to `ModelInspector`. This filters the list of ops captured and displayed by `ModelInspector` to only those operations which are registered for compressibility by `Quantizer`. Omitting this argument allows for all ops to be captured and displayed.

The above code produces output like the following (colors omitted for brevity):

```text
Legend:  ■ module_name (module_type)  ◆ op_name [op_type]

(MyModel)
    module inputs:  linear
    module outputs: linear_1
├── ■ linear1 (torch.nn.modules.linear.Linear)
│       module inputs:  linear
│       module outputs: linear
│   └── ◆ linear [linear]
│         op inputs:  x, linear1_weight, linear1_bias
│         op outputs: relu
│         filepath:  my_model.py:16
│         code:      x = self.linear1(x)
├── ■ relu (torch.nn.modules.activation.ReLU)
│       module inputs:  relu
│       module outputs: relu
└── ■ linear2 (torch.nn.modules.linear.Linear)
        module inputs:  linear_1
        module outputs: linear_1
    └── ◆ linear_1 [linear]
          op inputs:  relu, linear2_weight, linear2_bias
          op outputs: output
          filepath:  my_model.py:18
          code:      x = self.linear2(x)
```

The output shows the model's module hierarchy and the ops within each module. Note in particular that since `relu` is not a registered compressible op by `Quantizer`, it does not show up as an operation within the `ReLU` module.

Reading the tree:

- **Module name** and **module type** appear on module lines: `■ module_name (module_type)`. For example, `■ linear1 (torch.nn.modules.linear.Linear)` — `"linear1"` is the module name (usable in `module_name_configs`) and `"torch.nn.modules.linear.Linear"` is the module type (usable in `module_type_configs`).
- **Op name** and **op type** appear on operation lines: `◆ op_name [op_type]`. For example, `◆ linear_1 [linear]` — `"linear_1"` is the op name (usable in `op_name_config`) and `"linear"` is the op type (usable in `op_type_config`).
- **Op inputs/outputs** show connectivity between operations.
- **filepath** and **code** (shown for user-defined modules) show where in your source code the operation originates.

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

Pass `colorize=False` to suppress ANSI color codes (e.g., when writing to a file).

## Querying Operations by Config Key

Once you have reviewed the full summary to see what names and types are present, you can use query methods to check which operations would be matched by a specific name or type pattern. This is useful for verifying your config will target the intended ops before applying compression.

Each query method returns a tuple of {class}`~coreai_opt.inspection.OpInfo` objects matching the filter. The method names correspond directly to the config keys they help populate.

From the Basic Usage summary, this model exposes:

- **Op types**: `linear`
- **Op names**: `linear`, `linear_1`
- **Module types**: `torch.nn.modules.linear.Linear`, `torch.nn.modules.activation.ReLU`
- **Module names**: `linear1`, `relu`, `linear2`

Op names and module names can be passed as a literal name or as a regex following [Python re syntax](https://docs.python.org/3/library/re.html) for wildcard matching; the pattern is matched against the entire string. The matching methodology is identical to how compression config entries match modules and ops in a model, allowing the user to see exactly which modules or ops would be matched given a particular string.

Each query method returns a tuple of {class}`~coreai_opt.inspection.OpInfo` objects matching the filter:

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
)  # matches ops in modules "linear1" and "linear2"
```

Each returned {class}`~coreai_opt.inspection.OpInfo` provides `op_name`, `op_type`, and `module_stack` (the nesting of modules containing the op):

```python
>>> for op in inspector.get_matched_ops_for_op_type("linear"):
...     print(f"  op_name={op.op_name}, op_type={op.op_type}")
...     print(f"  module: {op.module_stack[-1].module_name} ({op.module_stack[-1].module_type})")
  op_name=linear, op_type=linear
  module: linear1 (torch.nn.modules.linear.Linear)
  op_name=linear_1, op_type=linear
  module: linear2 (torch.nn.modules.linear.Linear)
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
# look up a specific submodule
linear2_module = root.get_submodule("linear2")

# get all ops under this subtree (depth-first)
linear2_ops = linear2_module.all_ops()
```

`ModuleInfo` supports the same iteration patterns as `nn.Module`: `children()`, `named_children()`, `modules()`, `named_modules()`, and `get_submodule()`.
