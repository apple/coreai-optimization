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

To save the importance-scores (aka sensitivities) for the weights to reuse later:

```python
# provide path to save weight sensitivities
with palettizer.calibration_mode(
    loss_fn=F.cross_entropy, sensitivity_path="sensitivities.pt"
) as skm:
    for batch, target in calibration_dataloader:
        output = prepared_model(batch)
        skm.step(output, target)
```

And then in a future new run, load precomputed sensitivities during prepare:

```python
prepared_model = palettizer.prepare(example_inputs, sensitivity_path="sensitivities.pt")

# ---------- validate --------------------
val_metric = validate(prepared_model, val_dataset)

# ----------- deployment ------------------
# same as before
```

For more details on how to use {class}`~coreai_opt.palettization.config.KMeansPalettizerConfig`, {class}`~coreai_opt.palettization.config.ModuleKMeansPalettizerConfig` to apply different settings to different weights in the model, see [Palettization Config](config.md).
