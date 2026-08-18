# Quantization with Composite Ops

Core AI recognizes certain well-known building blocks, such as SDPA or RMSNorm, as _composite ops_ and applies optimized implementations for them at runtime. These composite ops are available via the [coreai_torch.composite_ops](https://apple.github.io/coreai-torch/main/api/composite-ops.html) API. When a PyTorch model uses one of these ops, it can be converted to Core AI using one of the following APIs:

- **Using `TorchConverter.add_pytorch_module()` along with the `externalize_modules` arg**: As described [here](https://apple.github.io/coreai-torch/main/guides/composite-ops.html), this approach takes as input an `nn.Module` torch model, along with the composite op torch module, specified via the *[externalize_modules](https://apple.github.io/coreai-torch/main/guides/externalization.html)* arg. Any `coreai-opt` optimizer that yields a torch model of type `nn.Module` can be converted using this API.
- **Using `TorchConverter.add_exported_program()`** : This [coreai-torch API](https://apple.github.io/coreai-torch/main/api/TorchConverter.html#add-exported-program) operates on an already exported torch program. When using `coreai-opt`'s quantizer with [graph execution mode](overview.md#two-execution-modes-graph-and-eager), which results in a torch `ExportedProgram`, there are a few additional steps required to ensure correct conversion and externalization of the composite op modules. This process is explained below with an example.

## Example

This example uses the same `RMSNormComposite` op from the [Externalization](https://apple.github.io/coreai-torch/main/guides/externalization.html) guide, however, the same process applies for all composite ops with their respective `ExternalizeSpec`s.

First define the model that uses a composite op:

```python
import torch
import torch.nn as nn


# The composite op
class RMSNormComposite(nn.Module):
    def __init__(self, axes=-1, eps=1e-5, version=1):
        super().__init__()
        self.axes = axes
        self.eps = eps
        self.version = version

    def forward(self, input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        x_f32 = input.to(torch.float32)
        inv_rms = torch.rsqrt((x_f32 * x_f32).mean(self.axes, keepdim=True) + self.eps)
        return (input * inv_rms).to(input.dtype) * scale


# A model that uses the composite op
class Model(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.norm = RMSNormComposite()
        self.norm_weight = nn.Parameter(torch.ones(dim))
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        return self.out(self.norm(self.proj(x), self.norm_weight))


original_model = Model().eval()
example_inputs = (torch.randn(1, 32),)
```

Next, apply `coreai-opt` quantization with graph mode, and convert using `coreai-torch`:

```python
import coreai_opt as opt
from coreai_opt.quantization import Quantizer, QuantizerConfig
from coreai_opt.quantization.config import ExecutionMode
import coreai_torch
from coreai_torch import (
    ExternalizeSpec,
    TorchConverter,
    _patch_model_for_externalization,
    _subexport_and_restore,
)

# Patch the model in-place
# to mark RMSNormComposite module for externalization
_patch_model_for_externalization(
    original_model,
    targets=[
        ExternalizeSpec(
            target_class=RMSNormComposite,
            composite_op_name="rms_norm",
            composite_attrs=["axes", "eps", "version"],
        )
    ],
)

# Usual flow using coreai-opt APIs for quantization
config = QuantizerConfig.presets.w8(execution_mode=ExecutionMode.GRAPH)
quantizer = Quantizer(original_model, config)
prepared_model = quantizer.prepare(example_inputs)
finalized_quantized_model = quantizer.finalize(backend=opt.ExportBackend.CoreAI)

# Export to Core AI
exported_program = torch.export.export(
    finalized_quantized_model, example_inputs
).run_decompositions(coreai_torch.get_decomp_table())
composite_module_exported_programs = _subexport_and_restore(
    original_model, exported_program
)
coreai_program = (
    TorchConverter()
    .add_exported_program(
        exported_program,
        _externalized_exported_programs=composite_module_exported_programs,
    )
    .to_coreai()
)
```

Compared to the [usual flow](../introduction/integration_coreai.md), there are *two* key differences:

## Patch original model with custom ops

In graph mode execution, `coreai-opt`'s `quantizer.prepare` invokes the `torch.export.export` API. By default, this causes the composite op body to be inlined in the graph and we lose the information about it's boundary for externalization. In order to preserve this information, the `_patch_model_for_externalization` API modifies the original `torch` model in-place, before preparing the model for graph mode quantization using `coreai-opt`. Specifically, it replaces the forward method of the `nn.module`s specified in `ExternalizeSpec(target_class)` with a `torch.library.custom_op` that, in turn, invokes the original forward method of the composite op. This keeps the model functionally the same, however, now the sub-module (`RMSNormComposite` in this example) becomes opaque to the `torch.export.export` API, i.e., a single node instead of the composite op body being inlined in the exported graph.

Note that in addition to replacing the `forward` method of the composite ops with a `torch.library.custom_op`, the `_patch_model_for_externalization` API also registers some additional information/metadata as attributes on the composite op sub-modules. This metadata is then used by `_subexport_and_restore` as explained below, after which the model is restored by removing this metadata along with the `torch.library.custom_op` from the model.

## Extract exported programs for submodules

Once a model is patched using the `_patch_model_for_externalization` API, it can be quantized as usual using `coreai-opt`'s graph mode (either for PTQ or QAT) and finalized to `CoreAI` backend.

Now in order to convert the finalized model into a coreai graph (`aimodel`), the composite op sub-module bodies need to be exported as torch exported programs in order to be included in the graph as externalized function calls.
The `_subexport_and_restore` takes in the **original torch model** and does exactly that.
It returns a list of torch exported programs corresponding to all the composite op sub-modules that were _patched_ for externalization by `_patch_model_for_externalization`.
These torch exported programs are then passed to the `add_exported_program` API via the `_externalized_exported_programs` argument to prepare the coreai graph (`aimodel`).
After this process, the coreai graph contains both the main graph and the sub-graphs corresponding to the composite op submodules preserved as function calls.

## Notes

- Quantization cannot be applied *inside* the composite op body because the `_patch_model_for_externalization` API replaces the composite op body with a torch custom op, thus making it opaque to `coreai-opt`'s graph mode quantizer.
- However, the boundaries (incoming and outgoing tensors) of the composite ops can be quantized as usual, using the `module_input_spec` and `module_output_spec` config kwargs as described in the documentation on `coreai-opt` configs [here](config.md).

:::{warning} The externalization APIs used below, `_patch_model_for_externalization` and `_subexport_and_restore` in coreai-torch are currently experimental. :::
