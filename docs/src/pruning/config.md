# Config API

Pruning Configs follow the same philosophy as the [Palettization Config](../palettization/config.md).
They are simpler as pruning applies only to the weights in the model.
(Hence there are no `op_input_spec` and `op_output_spec` fields in the {class}`~coreai_opt.pruning.config.ModuleMagnitudePrunerConfig` and {class}`~coreai_opt.pruning.config.OpMagnitudePrunerConfig`.)

## PruningSpec

{class}`~coreai_opt.pruning.spec.PruningSpec` defines the following key properties (for full list see API reference):

- `target_sparsity`: Fraction of elements to zero, in `[0, 1]`. Default: 0.5.
- `pruning_scheme`: Structural pattern of sparsity. Allowed: {class}`~coreai_opt.pruning.spec.Unstructured`() or {class}`~coreai_opt.pruning.spec.ChannelStructured`(axis=...), defaults to the former.

```python
from coreai_opt.pruning import PruningSpec
from coreai_opt.pruning.spec import (
    ChannelStructured,
    default_weight_pruning_spec,
)

# 50% unstructured magnitude pruning (default)
spec = default_weight_pruning_spec()

# 75% unstructured
spec = PruningSpec(target_sparsity=0.75)

# 50% channel-structured along axis 0 — entire channels are pruned together
spec = PruningSpec(
    target_sparsity=0.5,
    pruning_scheme=ChannelStructured(axis=0),
)
```

:::{note}
**Realized sparsity for `ChannelStructured`**:

Channel-structured pruning prunes whole channels along `axis`, so the realized sparsity is rounded down to the nearest multiple of `1/num_channels`. For `num_channels=10` and `target_sparsity=0.5`, exactly 5 channels are pruned and the realized sparsity matches the target. For `num_channels=7` and the same target sparsity, only 3 channels are pruned, giving 3/7 ≈ 43% realized sparsity. `Unstructured` rounds at the element level, so this is only a concern for `ChannelStructured`.
:::

## Config classes and their defaults

The pruning config system mirrors palettization's three-class hierarchy:

- {class}`~coreai_opt.pruning.MagnitudePrunerConfig` — the top-level config for the entire model. It holds a `global_config`, plus optional `module_type_configs` and `module_name_configs` for overrides. Same precedence as palettization: `module_name_configs` > `module_type_configs` > `global_config`.

- {class}`~coreai_opt.pruning.ModuleMagnitudePrunerConfig` — controls pruning for all ops within a module's scope (or all modules if used as a `global_config`). Like {class}`~coreai_opt.palettization.config.ModuleKMeansPalettizerConfig`, it specifies a default `op_state_spec` for ops in the module and allows overrides via `op_type_config`, `op_name_config`, and `module_state_spec`. For a given op's weight, the spec is resolved in this priority order (highest first): `module_state_spec`, the matching entry in `op_name_config`, the matching entry in `op_type_config`, then the module's `op_state_spec`. It also exposes a `sparsity_schedule` field — when set, `pruner.step()` ramps sparsity over training (see [Pruning with Fine-Tuning](overview.md#pruning-with-fine-tuning)); when unset, the spec's `target_sparsity` is applied statically.

- {class}`~coreai_opt.pruning.config.OpMagnitudePrunerConfig` — controls pruning for a specific op type or op name. Only `op_state_spec` is used.

### Default behavior when no arguments are provided

Creating any of these config classes with no arguments gives you a ready-to-use **50% unstructured magnitude pruning** configuration:

```python
# All three of these produce equivalent default pruning settings:
config = MagnitudePrunerConfig()
# is equivalent to:
config = MagnitudePrunerConfig(global_config=ModuleMagnitudePrunerConfig())
# which is equivalent to:
config = MagnitudePrunerConfig(
    global_config=ModuleMagnitudePrunerConfig(
        op_state_spec={
            "weight": default_weight_pruning_spec(),
            "in_proj_weight": default_weight_pruning_spec(),
        },
    )
)

op_config = OpMagnitudePrunerConfig()
# is equivalent to:
op_config = OpMagnitudePrunerConfig(
    op_state_spec={
        "weight": default_weight_pruning_spec(),
        "in_proj_weight": default_weight_pruning_spec(),
    },
)
```

- The default applies `default_weight_pruning_spec()` — 50% target sparsity, unstructured, magnitude-based — to parameters named `"weight"` and `"in_proj_weight"`. Other state tensors (e.g., `"bias"`) are left uncompressed.

- If you need different behavior — such as pruning custom parameter names, excluding certain modules, or applying different sparsity targets to different layers — see the [Examples](#examples) section below.

## Examples

Several examples below configure specific module types or module names. To determine these for your model, see [How to get names + types](../quantization/config.md#how-to-get-names--types-for-modules-and-ops). Since pruning only supports eager execution mode, only the eager mode guidance in that section is relevant.

### Apply 50% pruning globally, 75% to linear layers

Apply 50% magnitude pruning to all supported layers, and override `linear` layers to 75%.

```python
# programmatic
import torch.nn as nn
from coreai_opt.pruning import (
    MagnitudePrunerConfig,
    ModuleMagnitudePrunerConfig,
    PruningSpec,
)

# 50% on all supported layers globally (the default)
config = MagnitudePrunerConfig()

# override Linear layers to 75%
config.set_module_type(
    nn.Linear,
    ModuleMagnitudePrunerConfig(
        op_state_spec={"weight": PruningSpec(target_sparsity=0.75)},
    ),
)
```

The snippet above applies 50% pruning globally (covering Conv2d and all other supported modules), then overrides Linear layers to 75%.

#### Config chaining

The setters also return the config itself, so multiple modifications can be chained into a single expression. The snippet above is equivalent to:

```python
config = MagnitudePrunerConfig().set_module_type(
    nn.Linear,
    ModuleMagnitudePrunerConfig(
        op_state_spec={"weight": PruningSpec(target_sparsity=0.75)},
    ),
)
```

```yaml
# yaml
magnitude_pruning_config:
  global_config:
    op_state_spec:
      weight:
        target_sparsity: 0.5
        pruning_scheme: { type: unstructured }
  module_type_configs:
    torch.nn.modules.linear.Linear:
      op_state_spec:
        weight:
          target_sparsity: 0.75
          pruning_scheme: { type: unstructured }
```

### Apply pruning to specific module types only

When you want to prune only specific module types and leave everything else uncompressed, construct the config explicitly without a `global_config`. Each module type gets its own `ModuleMagnitudePrunerConfig`, and modules not listed in `module_type_configs` are skipped.

```python
# programmatic — explicit (scoped to specific module types)
from coreai_opt.pruning import (
    MagnitudePrunerConfig,
    ModuleMagnitudePrunerConfig,
    PruningSpec,
)

config = MagnitudePrunerConfig(
    module_type_configs={
        "torch.nn.modules.linear.Linear": ModuleMagnitudePrunerConfig(
            op_state_spec={"weight": PruningSpec(target_sparsity=0.75)},
        ),
        "torch.nn.modules.conv.Conv2d": ModuleMagnitudePrunerConfig(
            op_state_spec={"weight": PruningSpec(target_sparsity=0.5)},
        ),
    },
)
```

```yaml
# yaml
magnitude_pruning_config:
  module_type_configs:
    torch.nn.modules.linear.Linear:
      op_state_spec:
        weight:
          target_sparsity: 0.75
          pruning_scheme: { type: unstructured }
    torch.nn.modules.conv.Conv2d:
      op_state_spec:
        weight:
          target_sparsity: 0.5
          pruning_scheme: { type: unstructured }
```
