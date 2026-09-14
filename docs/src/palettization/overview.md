# API Overview

## Vanilla K-means API

```python
import coreai_opt as opt
from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig
import torch

model = MyModel().eval()
example_inputs = (torch.randn(1, 3, 224, 224),)

# define config
# here we use a config that applies 4-bit per-grouped-channel palettization to all supported layers.
# this can be done by using one of the several available "pre-defined" configs, accessible via the "presets" namespace.
config = KMeansPalettizerConfig.presets.w4()

# palettize weights in the model with the config
palettizer = KMeansPalettizer(model, config)
prepared_model = palettizer.prepare(example_inputs)

# ---------- validate --------------------
# use prepared_model to check accuracy on validation data.
# forward pass will include the effect of weight compression.
val_metric = validate(prepared_model, val_dataset)

# ----------- deployment ------------------
# same as with Quantizer:
# invoke the 'finalize' API to update the PyTorch model and make it compatible for conversion
# with either coreai or coremltools

finalized_model_for_coreai = palettizer.finalize(backend=opt.ExportBackend.CoreAI)
# OR
finalized_model_for_coreml = palettizer.finalize(backend=opt.ExportBackend.CoreML)
```

## Sensitive K-means API

Sensitivity-based palettization uses calibration data to compute per-weight importance
scores (based on the [SqueezeLLM](https://arxiv.org/pdf/2306.07629) method).

```python
from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig
import torch.nn.functional as F

model = MyModel().eval()
example_inputs = (torch.randn(1, 3, 224, 224),)

config = KMeansPalettizerConfig()  # defaults to 4 bit palettization for all weights
palettizer = KMeansPalettizer(model, config)
prepared_model = palettizer.prepare(example_inputs)

# compute the clusters/LUTs with weighted-kmeans
# weights of the prepared_model will get updated
with palettizer.calibration_mode(loss_fn=F.cross_entropy) as skm:
    for batch, target in calibration_dataloader:
        output = prepared_model(batch)
        skm.step(output, target)

# ---------- validate --------------------
# use prepared_model to check accuracy on validation data.
# forward pass will include the effect of weight compression.
val_metric = validate(prepared_model, val_dataset)

# ----------- deployment ------------------
# same as before
```

To save the importance-scores (aka sensitivities) for the weights to reuse later, specify a `sensitivity_path` argument in the `calibration_mode` context:

```python
# provide path to save weight sensitivities
with palettizer.calibration_mode(
    loss_fn=F.cross_entropy, sensitivity_path="sensitivities.pt"
) as skm:
    for batch, target in calibration_dataloader:
        output = prepared_model(batch)
        skm.step(output, target)
```

Precomputed sensitivities can be loaded during preparation in a future new run to avoid needing to recalibrate:

```python
prepared_model = palettizer.prepare(example_inputs, sensitivity_path="sensitivities.pt")

# ---------- validate --------------------
val_metric = validate(prepared_model, val_dataset)

# ----------- deployment ------------------
# same as before
```

Note that `Sensitive K-means` shares the same `KMeansPalettizer` and `KMeansPalettizerConfig` as `Vanilla K-means`. `Sensitive K-means` is achieved simply by passing representative data samples through the model while inside the `calibration_mode` context.

For more details on how to use {class}`~coreai_opt.palettization.config.KMeansPalettizerConfig`, {class}`~coreai_opt.palettization.config.ModuleKMeansPalettizerConfig` to apply different settings to different weights in the model, see [Palettization Config](config.md).

## Training a Palettized model

A `KMeansPalettizer` palettized model can still be fine-tuned in a training pipeline. As palettization is a hard assignment lookup, gradients cannot be propagated for palettized weights, meaning any parameter which is palettized will not update during `optimizer.step()` (the palettization codebook and index assignments will also be fixed).

Any parameters not being palettized can still update and learn so that one can fine-tune any non-palettized portion of the model while being palettization-aware, i.e. adapting to other parameters which are palettized.

Run the training loop inside `palettizer.training_mode()`, which places the model in train mode and restores its original train/eval state on exit:

```python
from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig, ModuleKMeansPalettizerConfig, PalettizationSpec

kmeans_palettizer_config = KMeansPalettizerConfig(...)

palettizer = KMeansPalettizer(model, kmeans_palettizer_config)
prepared_model = palettizer.prepare(example_inputs)

with palettizer.training_mode():
    for ...
        # Perform whatever training forward pass / loss calculation / optimizer gradient application is desired
        ...
```

### Using PATSchedule

Users may also choose to enable palettization of parameters at a future point during training instead of right from the start. Weights which are marked for palettization, but have not yet had their palettizers enabled, will still be able to update during training. When a weight's palettizer is switched from disabled to enabled, its codebook and index assignments are recomputed using the most current weight values.

Since palettizer enabling can be configured on a module by module basis, this gives users more flexibility on training to be palettization-aware by allowing weights to update in response to palettization noise from other palettized weights before themselves being palettized.

For example, users could choose to apply the following workflow:

1. Begin training with all parameters marked to be palettized but with all palettizers disabled
2. Fine-tune the model with no palettization noise for a few epochs
3. Enable a subset of palettizers while keeping the rest of the palettizers disabled
4. Fine-tune the model with palettization noise from the subset of enabled palettizers (any weights with enabled palettizers will not update further, but any weights with palettizers not yet enabled will continue training)
5. Enable the rest of palettizers in the model and evaluate the final accuracy

Steps 3 and 4 can be repeated as many times as desired with more and more palettizers being enabled.

To achieve this, add a `PATSchedule` to the relevant module configs and call `palettizer.step()` inside the training loop to advance the schedule:

```python
from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig, ModuleKMeansPalettizerConfig, PalettizationSpec
from coreai_opt.palettization.config import PATSchedule

# Create a KMeansPalettizer config with 2 modules being palettized using different PATSchedules
kmeans_palettizer_config = KMeansPalettizerConfig(
    global_config = ...,
    module_name_configs = {
        "module1": ModuleKMeansPalettizerConfig(
            ...,
            pat_schedule=PATSchedule(
                enable_fake_palettize=2
            )
        ),
        "module2": ModuleKMeansPalettizerConfig(
            ...,
            pat_schedule=PATSchedule(
                enable_fake_palettize=4
            )
        )
    }
)

palettizer = KMeansPalettizer(model, kmeans_palettizer_config)
prepared_model = palettizer.prepare(example_inputs)

# Track the palettizer's step count so palettizer state is managed
# in accordance with each module's PATSchedule.
with palettizer.training_mode():
    for ...
        # Perform whatever training forward pass / loss calculation / optimizer gradient application is desired
        ...

        # Call palettizer.step() to increase palettizer's step count by 1
        palettizer.step()
```

Each time `palettizer.step()` is called, the palettizer checks each module's `PATSchedule` to see whether any palettizer should be enabled. With the above config settings of `enable_fake_palettize` = 2 and 4 for `module1` and `module2` respectively, this results in both palettizers being disabled to start, `module1`'s palettizer enabled after `palettizer.step()` has been called twice, and `module2`'s palettizer enabled after `palettizer.step()` has been called 4 times in total.

Note that `palettizer.step()` can be called whenever the user desires, whether it be per-batch, per-epoch, or on some other cadence.
