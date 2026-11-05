# Config API

Palettization Configs follow the same philosophy as the [Quantization Config](../quantization/config.md).
They are simpler as palettization applies only to the weights in the model.
(Hence there are no `op_input_spec` and `op_output_spec` fields in the {class}`~coreai_opt.palettization.config.ModuleKMeansPalettizerConfig` and {class}`~coreai_opt.palettization.config.OpKMeansPalettizerConfig`.)

To estimate what a config costs in terms of bits per weight (BPW), use the {func}`~coreai_opt.inspection.bits_per_weight` utility to get an analytical BPW estimate for a *prepared* `coreai-opt` model. See [Utility for computing analytical Bits Per Weight (BPW)](../utils/mixed_precision.md#utility-for-computing-analytical-bits-per-weight-bpw) for details.

## PalettizationSpec

{class}`~coreai_opt.palettization.spec.PalettizationSpec` defines the following key properties, among others (for full list see API reference):

- `n_bits`: Number of bits per LUT index. LUT size = 2^n_bits centroids (default: 4)
- `granularity`: controls whether there is one LUT for the entire weight tensor or one LUT per group of channels. Allowed: {class}`~coreai_opt.palettization.spec.PerTensorGranularity`() or {class}`~coreai_opt.palettization.spec.PerGroupedChannelGranularity`(), defaults to the former
- `cluster_dim`: Dimension of the cluster centers, i.e. the entries in the LUT. Defaults to 1; if > 1, it results in vector palettization.
- `lut_qspec`: Optional {class}`~coreai_opt.quantization.spec.QuantizationSpec` describing how to quantize the LUT centroid values themselves. Defaults to `None` (LUT centroids are stored at the same dtype as the uncompressed weights). Supported dtypes are `torch.int8`, `torch.uint8`, `torch.float8_e4m3fn`, and `torch.float8_e5m2`.
- `sparsity_detection_threshold`: Optional float in [0, 1]. A weight with at least this fraction of exact zeros (and at least one) is treated as already sparse, for example from a pruned model, and is palettized jointly with its sparsity: only the nonzero values are palettized, the zeros stay exactly zero, and the exported model stores the weight in a sparse representation together with the palettized ops for the nonzero values. Defaults to `None`, which palettizes every weight densely. See [Palettizing already-sparse weights](#palettizing-already-sparse-weights).

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

# Palettize pruned weights jointly with their sparsity: weights with at least 40% exact zeros
spec = PalettizationSpec(n_bits=4, sparsity_detection_threshold=0.4)
```

:::{note}
**Reproducibility with `cluster_dim > 1`**:

Vector k-means derives its random seeds from each weight tensor's contents, so repeated calls to `prepare()` produce the same palettized weights without any manual seeding, with or without `num_workers > 1`. On CPU the results are reproducible on the same machine. On a GPU the seeding is reproducible too, but floating-point reductions in the GPU kernels can cause very small run-to-run differences.

Scalar palettization (`cluster_dim == 1`, the default) is deterministic.
:::

### Palettizing already-sparse weights

When `sparsity_detection_threshold` is set, the first forward pass of `prepare()` detects which weights are already sparse. For each of them, the nonzero mask is fixed, k-means fits the palette on the nonzero values only, and the zeros stay exactly zero. `finalize()` then exports the weight as a sparse representation (the mask and the nonzero values) with the palettization ops applied to the nonzero values only, so zeros need no indices.

A few details depend on the configuration:

- With `PerGroupedChannelGranularity`, Core AI maps contiguous blocks of the nonzero values to the groups, so every group has to store the same number of values. Groups with fewer nonzeros than the fullest group also store some of their zeros, which costs a little of the sparse savings. For scalar palettization the palette then holds an entry for exactly `0.0`.
- With vector palettization (`cluster_dim > 1`), the nonzero values of each group are clustered in vectors of `cluster_dim` consecutive values. If a group's nonzero count is not a multiple of `cluster_dim`, its smallest nonzeros are set to zero (truncated). With per-group vector palettization, the aforementioned padding zeros can share a vector with a nonzero value and result in a nonzero value when de-palettized; therefore this could reduce the percentage of sparsity by a small amount.

Supported configurations for exporting an already-sparse weight:

| Option                      | CoreML | Core AI | Notes                                                                                                              |
| --------------------------- | ------ | ------- | ------------------------------------------------------------------------------------------------------------------ |
| Per-tensor                  | Yes    | Yes     |                                                                                                                    |
| Grouped along axis 0        | Yes    | Yes     | For Core AI, groups are padded with stored zeros so each group stores the same number of values.                   |
| Grouped along another axis  | Yes    | No      | Core AI needs contiguous groups, which only holds along axis 0.                                                    |
| Per-channel scale           | Yes    | Yes     |                                                                                                                    |
| Quantized LUT (`lut_qspec`) | No     | Yes     | For Core AI, the zero point must be 0 (a symmetric `lut_qspec`). Not supported together with vector palettization. |
| Vector (`cluster_dim > 1`)  | No     | Yes     | For Core AI, per-tensor or grouped along axis 0.                                                                   |

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

### Enable palettization at different training steps per module (PATSchedule)

For palettization-aware training, you can defer when each module's palettizer turns on using a {class}`~coreai_opt.palettization.config.PATSchedule` set on that module's {class}`~coreai_opt.palettization.config.ModuleKMeansPalettizerConfig`. Its `enable_fake_palettize` field is the step-count threshold at which that module's palettizer becomes active; until then the module trains uncompressed. See [Using PATSchedule](overview.md#using-patschedule) for the training loop that drives the schedule via `palettizer.training_mode()` and `palettizer.step()`.

```python
from coreai_opt.palettization import (
    KMeansPalettizerConfig,
    ModuleKMeansPalettizerConfig,
    PalettizationSpec,
)
from coreai_opt.palettization.config import PATSchedule

# All layers palettized 4-bit, but each named module's palettizer turns on at a different training step:
# module1 after 100 steps, module2 after 500.
config = KMeansPalettizerConfig(
    global_config=ModuleKMeansPalettizerConfig(
        op_state_spec={"weight": PalettizationSpec(n_bits=4)},
    ),
    module_name_configs={
        "module1": ModuleKMeansPalettizerConfig(
            op_state_spec={"weight": PalettizationSpec(n_bits=4)},
            pat_schedule=PATSchedule(enable_fake_palettize=100),
        ),
        "module2": ModuleKMeansPalettizerConfig(
            op_state_spec={"weight": PalettizationSpec(n_bits=4)},
            pat_schedule=PATSchedule(enable_fake_palettize=500),
        ),
    },
)
```

```yaml
# yaml
kmeans_palettization_config:
  global_config:
    op_state_spec:
      weight:
        n_bits: 4
  module_name_configs:
    module1:
      op_state_spec:
        weight:
          n_bits: 4
      pat_schedule:
        enable_fake_palettize: 100
    module2:
      op_state_spec:
        weight:
          n_bits: 4
      pat_schedule:
        enable_fake_palettize: 500
```

### Tune vector palettization k-means (VectorKMeansConfig)

Vector palettization (`cluster_dim > 1`) runs k-means on each palettization block, and its speed/quality trade-off is set with a {class}`~coreai_opt.palettization.config.VectorKMeansConfig` on the module's {class}`~coreai_opt.palettization.config.ModuleKMeansPalettizerConfig`. Scalar palettization solves 1D k-means exactly and does not use these settings; setting `vector_kmeans_config` on a module config without any spec that has `cluster_dim > 1` raises an error. If `vector_kmeans_config` is not set, the defaults below are used.

| Field            | Default | Effect                                                                                                                                                                                                                                                                                                                                                                                               |
| ---------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `n_local_trials` | `None`  | Candidate centroids sampled at each k-means++ seeding step; the one that most reduces clustering error is kept. `1` is standard k-means++; larger values (greedy k-means++) usually lower the clustering error, at proportionally higher seeding cost. `None` uses scikit-learn's default, `2 + int(log(n_clusters))` (6 for `n_bits=6`, 7 for `n_bits=8`), as in `sklearn.cluster.kmeans_plusplus`. |
| `n_init`         | `5`     | Number of k-means restarts; the best result is kept per block. Time grows linearly with it.                                                                                                                                                                                                                                                                                                          |
| `max_iter`       | `300`   | Maximum Lloyd iterations per restart.                                                                                                                                                                                                                                                                                                                                                                |
| `tol`            | `1e-4`  | A block stops once its clustering error improves by less than `tol` (relative) per iteration, or once no assignment changes. Same on GPU and CPU.                                                                                                                                                                                                                                                    |

Clustering runs where the hardware allows:

```text
              vector k-means (all blocks of one weight tensor)
                                  |
           +------- CUDA GPU -----+------ CPU -------+
           |                                         |
  batched k-means++ seeding                     scikit-learn
   (shared by PTP and SKM)                (seeding + Lloyd, per block)
           |                                         |
      +----+----+                                    |
     PTP       SKM                                   |
 flash-kmeans  PyTorch                               |
 (or PyTorch)   |                                    |
      +----+----+                                    |
           +-------------------+---------------------+
                               |
                     stop each block at tol
                               |
                  best of n_init restarts -> LUT
```

- **GPU**: all vector tensors cluster on the GPU. PTP (data-free post-training palettization) runs its Lloyd iterations with [flash-kmeans](https://github.com/svg-project/flash-kmeans) (Yang et al., 2026, [arXiv:2603.09229](https://arxiv.org/abs/2603.09229)) when it is installed, which is much faster, and otherwise with a PyTorch loop. SKM (Sensitive K-Means, see `calibration_mode()`) always uses the PyTorch loop, because flash-kmeans has no per-point weights.
- **CPU**: scikit-learn. As for scalar palettization, `prepare(num_workers=N)` clusters `N` layers in parallel.

With a GPU, vector tensors run in the main process, one at a time, while the `num_workers` CPU workers handle the scalar tensors.

```python
from coreai_opt.palettization import (
    KMeansPalettizerConfig,
    ModuleKMeansPalettizerConfig,
    PalettizationSpec,
)
from coreai_opt.palettization.config import VectorKMeansConfig
from coreai_opt.palettization.spec import PerGroupedChannelGranularity

spec = PalettizationSpec(
    n_bits=8,
    cluster_dim=2,
    granularity=PerGroupedChannelGranularity(axis=0, group_size=32),
)

# Defaults: greedy k-means++, 5 restarts, convergence-based stopping.
config = KMeansPalettizerConfig(
    global_config=ModuleKMeansPalettizerConfig(
        op_state_spec={"weight": spec},
        enable_fast_kmeans_mode=False,
    )
)

# Fast, approximate clustering, for example to check that a model exports.
fast_config = KMeansPalettizerConfig(
    global_config=ModuleKMeansPalettizerConfig(
        op_state_spec={"weight": spec},
        enable_fast_kmeans_mode=False,
        vector_kmeans_config=VectorKMeansConfig(n_init=1, max_iter=1),
    )
)
```

```yaml
# yaml
kmeans_palettization_config:
  global_config:
    enable_fast_kmeans_mode: false
    op_state_spec:
      weight:
        n_bits: 8
        cluster_dim: 2
        granularity: { type: per_grouped_channel, axis: 0, group_size: 32 }
    vector_kmeans_config:
      n_init: 1
      max_iter: 1
```
