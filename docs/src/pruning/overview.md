# API Overview

## Post-training Pruning

Post-training pruning sparsifies the model in a single shot. The mask is computed during the `prepare()` call and the prepared model immediately reflects the effects of sparsity.

Unless the original PyTorch model already has a large fraction of weights close to zero across all of its weight parameters, post-training pruning will almost always degrade accuracy. It is most useful as a quick way to evaluate the impact of sparsity on model size and inference latency before committing to a fine-tuning workflow.

```python
import coreai_opt as opt
from coreai_opt.pruning import MagnitudePruner, MagnitudePrunerConfig
import torch

model = MyModel().eval()
example_inputs = (torch.randn(1, 3, 224, 224),)

# Default config: 50% unstructured magnitude pruning on every supported weight.
config = MagnitudePrunerConfig()

# Apply sparsity to the model. After the prepare API is called,
# the model will have 50% sparsity on every supported weight parameter
pruner = MagnitudePruner(model, config)
prepared_model = pruner.prepare(example_inputs)

# Validate the model with the effects of sparsity
val_metric = validate(prepared_model, val_dataset)

# Deployment is similar to the Quantizer.
# We invoke the 'finalize' API to update the PyTorch model and make it compatible
# for conversion with either CoreAI or CoreML

finalized_model_for_coreai = pruner.finalize(backend=opt.ExportBackend.CoreAI)
# OR
finalized_model_for_coreml = pruner.finalize(backend=opt.ExportBackend.CoreML)
```

## Pruning with Fine-Tuning

In most cases, fine-tuning is required to recover good accuracy after pruning. We can use a sparsity schedule on the module config and call `pruner.step()` to gradually ramp up sparsity over training.

```python
from coreai_opt.pruning import (
    MagnitudePruner,
    MagnitudePrunerConfig,
    ModuleMagnitudePrunerConfig,
    PruningSpec,
)
from coreai_opt.pruning.config import PolynomialDecaySchedule
import torch

model = MyModel()
example_inputs = (...,)
num_epochs = 5

# 70% target sparsity ramped in via a polynomial schedule over num_epochs.
# Schedule starts at step 0 (sparsity=0) and advances on every pruner.step()
# call along a cubic curve until reaching target_sparsity at step total_iters.
config = MagnitudePrunerConfig(
    global_config=ModuleMagnitudePrunerConfig(
        op_state_spec={"weight": PruningSpec(target_sparsity=0.7)},
        sparsity_schedule=PolynomialDecaySchedule(
            begin_step=0, total_iters=num_epochs, power=3.0
        ),
    ),
)
pruner = MagnitudePruner(model, config)
prepared_model = pruner.prepare(example_inputs)

# ---------- training loop --------------------
# We fine-tune the model while incrementing the sparsity schedule.
# The pruner.step() API advances the sparsity schedule and recomputes
# the masks against the current weight magnitudes for the next sparsity level.
# The step() API can be called at the epoch frequency or at the batch step frequency
# based on the configuration of the schedule
optimizer = torch.optim.SGD(prepared_model.parameters(), lr=1e-3)
for epoch in range(num_epochs):
    prepared_model.train()
    for batch, target in train_dataloader:
        optimizer.zero_grad()
        loss = criterion(prepared_model(batch), target)
        loss.backward()
        optimizer.step()
    pruner.step()

    val_metric = validate(prepared_model, val_dataloader)

# ----------- deployment ------------------
# same as before
```

For more details on how to use {class}`~coreai_opt.pruning.MagnitudePrunerConfig`, {class}`~coreai_opt.pruning.ModuleMagnitudePrunerConfig` to apply different settings to different weights in the model, see [Pruning Config](config.md).
