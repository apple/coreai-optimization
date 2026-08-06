# Config API

Palettization Configs follow the same philosophy as the [Quantization Config](../quantization/config.md).
They are simpler as palettization applies only to the weights in the model.
(Hence there are no `op_input_spec` and `op_output_spec` fields in the {class}`~coreai_opt.palettization.config.ModuleKMeansPalettizerConfig` and {class}`~coreai_opt.palettization.config.OpKMeansPalettizerConfig`.)

## PalettizationSpec

{class}`~coreai_opt.palettization.spec.PalettizationSpec` defines the following key properties, among others (for full list see API reference):

- `n_bits`: Number of bits per LUT index. LUT size = 2^n_bits centroids (default: 4)
- `granularity`: controls whether there is one LUT for the entire weight tensor or one LUT per group of channels. Allowed: {class}`~coreai_opt.palettization.spec.PerTensorGranularity`() or {class}`~coreai_opt.palettization.spec.PerGroupedChannelGranularity`(), defaults to the former
- `cluster_dim`: Dimension of the cluster centers, i.e. the entries in the LUT. Defaults to 1; if > 1, it results in vector palettization.
- `lut_qspec`: Optional {class}`~coreai_opt.quantization.spec.QuantizationSpec` describing how to quantize the LUT centroid values themselves. Defaults to `None` (LUT centroids are stored at the same dtype as the uncompressed weights). Supported dtypes are `torch.int8`, `torch.uint8`, `torch.float8_e4m3fn`, and `torch.float8_e5m2`.

```python
from coreai_opt.palettization import PalettizationSpec
from coreai_opt.palettization.spec import (
    PerGroupedChannelGranularity,
    default_weight_palettization_spec,
)
from coreai_opt.quantization import QuantizationSpec

# 4-bit per-tensor (default — 16 centroids)
spec = default_weight_palettization_spec()

# 2-bit per group of 8 channels — finer granularity, better accuracy
spec = PalettizationSpec(
    n_bits=2,
    granularity=PerGroupedChannelGranularity(axis=0, group_size=8),
)

# 4-bit with quantized LUT entries — LUT centroids stored as INT8
spec = PalettizationSpec(
    n_bits=4,
    lut_qspec=QuantizationSpec(dtype=torch.int8, qscheme="symmetric"),
)

# Cluster pairs of weight values instead of individual scalars
spec = PalettizationSpec(n_bits=4, cluster_dim=2)
```

:::{note}
**Reproducibility with `cluster_dim > 1`**:

When `cluster_dim > 1`, palettization uses vector k-means whose centroid initialization relies on `numpy.random` and `torch.randint`, so it is non-deterministic.

```python
# model gets different weights when we run it multiple times
model_1 = KMeansPalettizer(model, config).prepare(example_inputs)
model_2 = KMeansPalettizer(model, config).prepare(example_inputs)
```

To obtain reproducible results, seed `numpy` and `torch` before each call to `prepare()`:

```python
seed = 42

# models now have identical palettized weights
np.random.seed(seed)
torch.manual_seed(seed)
model_1 = KMeansPalettizer(model, config).prepare(example_inputs)

np.random.seed(seed)
torch.manual_seed(seed)
model_2 = KMeansPalettizer(model, config).prepare(example_inputs)
```

When `prepare()` is called with `num_workers > 1`, k-means runs in spawned worker processes that do not inherit the parent's RNG state, so the seeding advice above is only effective with `num_workers=1`. Use the sequential path if you need reproducible vector palettization.

Scalar palettization (`cluster_dim == 1`, the default) is deterministic and does not require seeding.
:::

## Config classes and their defaults

The palettization config system mirrors quantization's three-class hierarchy, but scoped to weights only (no activation quantization):

- {class}`~coreai_opt.palettization.config.KMeansPalettizerConfig` — the top-level config for the entire model. It holds a `global_config`, plus optional `module_type_configs` and `module_name_configs` for overrides. Same precedence as quantization: `module_name_configs` > `module_type_configs` > `global_config`.

- {class}`~coreai_opt.palettization.config.ModuleKMeansPalettizerConfig` — controls palettization for all ops within a module's scope (or all modules if used as a `global_config`). Like {class}`~coreai_opt.quantization.config.ModuleQuantizerConfig`, it specifies a default `op_state_spec` for ops in the module and allows overrides via `op_type_config`, `op_name_config`, and `module_state_spec`. Since palettization is weight-only, the activation/IO fields (`op_input_spec`, `op_output_spec`, `module_input_spec`, `module_output_spec`) are absent. For a given op's weight, the spec is resolved in this priority order (highest first): `module_state_spec`, the matching entry in `op_name_config`, the matching entry in `op_type_config`, then the module's `op_state_spec`.

- {class}`~coreai_opt.palettization.config.OpKMeansPalettizerConfig` — controls palettization for a specific op type or op name. Only `op_state_spec` is used.

### Default behavior when no arguments are provided

Creating any of these config classes with no arguments gives you a ready-to-use **4-bit weight palettization** configuration:

```python
# All three of these produce equivalent default palettization settings:
config = KMeansPalettizerConfig()
# is equivalent to:
config = KMeansPalettizerConfig(global_config=ModuleKMeansPalettizerConfig())
# which is equivalent to:
config = KMeansPalettizerConfig(
    global_config=ModuleKMeansPalettizerConfig(
        op_state_spec={
            "weight": default_weight_palettization_spec(),
            "in_proj_weight": default_weight_palettization_spec(),
        },
    )
)

op_config = OpKMeansPalettizerConfig()
# is equivalent to:
op_config = OpKMeansPalettizerConfig(
    op_state_spec={
        "weight": default_weight_palettization_spec(),
        "in_proj_weight": default_weight_palettization_spec(),
    },
)
```

- The default applies `default_weight_palettization_spec()` — 4-bit, per-tensor granularity, scalar clustering — to parameters named `"weight"` and `"in_proj_weight"`. Other state tensors (e.g., `"bias"`) are left uncompressed.

- If you need different behavior — such as palettizing custom parameter names, excluding certain modules, or applying different bit widths to different layers, see the [Examples](#examples) section.

## Examples

Several examples below configure specific module types or module names. To determine these for your model, use {class}`~coreai_opt.inspection.ModelInspector` with `execution_mode="eager"` — see [Inspecting Model Structure](../debugging/model_inspection.md). Palettization supports eager mode only.

### Apply 4-bit palettization globally, 8-bit to linear layers

Apply 4-bit palettization to all supported layers, and override `linear` layers to 8-bit.

```python
# programmatic — using presets
import torch.nn as nn
from coreai_opt.palettization import (
    KMeansPalettizerConfig,
    ModuleKMeansPalettizerConfig,
)

# define a config that applies 4-bit per-grouped-channel palettization to all supported layers, using one of the "pre-defined" presets.
config = KMeansPalettizerConfig.presets.w4()

# then update this config, to change the palettization for just the linear layers: to 8-bit per-tensor
config.set_module_type(nn.Linear, ModuleKMeansPalettizerConfig.presets.w8())
```

The snippet above applies 4-bit palettization globally (covering Conv2d and all other supported modules), then overrides Linear layers to 8-bit.

#### Config chaining

The setters also return the config itself, so multiple modifications can be chained into a single expression. The snippet above is equivalent to:

```python
config = KMeansPalettizerConfig.presets.w4().set_module_type(
    nn.Linear, ModuleKMeansPalettizerConfig.presets.w8()
)
```

```yaml
# yaml
kmeans_palettization_config:
  global_config:
    op_state_spec:
      weight:
        n_bits: 4
        granularity: { type: per_grouped_channel, axis: 0, group_size: 16 }
  module_type_configs:
    torch.nn.modules.linear.Linear:
      op_state_spec:
        weight:
          n_bits: 8
          granularity: { type: per_tensor }
```

### Apply 4-bit palettization to conv layers, 8-bit to linear layers

When you want to palettize only specific module types and leave everything else uncompressed, construct the config explicitly without a `global_config`. Each module type gets its own `ModuleKMeansPalettizerConfig`, and modules not listed in `module_type_configs` are skipped.

```python
# programmatic — explicit (scoped to specific module types)
from coreai_opt.palettization import (
    KMeansPalettizerConfig,
    ModuleKMeansPalettizerConfig,
    PalettizationSpec,
)

config = KMeansPalettizerConfig(
    module_type_configs={
        "torch.nn.modules.linear.Linear": ModuleKMeansPalettizerConfig(
            op_state_spec={"weight": PalettizationSpec(n_bits=8)},
        ),
        "torch.nn.modules.conv.Conv2d": ModuleKMeansPalettizerConfig(
            op_state_spec={"weight": PalettizationSpec(n_bits=4)},
        ),
    },
)
```

```yaml
# yaml
kmeans_palettization_config:
  module_type_configs:
    torch.nn.modules.linear.Linear:
      op_state_spec:
        weight:
          n_bits: 8
          granularity: { type: per_tensor }
    torch.nn.modules.conv.Conv2d:
      op_state_spec:
        weight:
          n_bits: 4
          granularity: { type: per_tensor }
```
