# Integration with Core AI

`coreai-opt` is built to optimize PyTorch models in a way that lets them be easily converted into the Core AI model format. Most of the compression configurations it exposes map directly to compression modes that the [Core AI](https://developer.apple.com/documentation/coreai) runtime executes efficiently on Apple Silicon, with a smaller set kept available for general experimentation.

Core AI deployment uses `ExportBackend.CoreAI` with `finalize()`, followed by conversion with `coreai-torch`:

```python
import coreai_opt as opt

# After prepare() / calibration / training, finalize for Core AI export
# "backend" defaults to opt.ExportBackend.CoreAI
finalized_model = quantizer.finalize(backend=opt.ExportBackend.CoreAI)
```

Under the hood, `finalize()` replaces coreai-opt's internal fake-quantize/fake-palettize ops with PyTorch custom ops whose definitions match the corresponding compression ops in the Core AI dialect. This allows `coreai-torch` to recognize the ops and map them correctly in the Core AI representation.

From here, conversion follows the same steps as for any uncompressed model conversion:

```python
from pathlib import Path

from coreai_opt.casting import cast_to_16_bit_precision
import coreai_torch
import torch

# torch.export the finalized model with coreai-torch's decomposition table.
exported_program = torch.export.export(
    finalized_model, example_inputs
).run_decompositions(coreai_torch.get_decomp_table())

# Cast remaining FP32 weights/activations to FP16 (and INT32/INT64 to INT16) so
# the model runs more efficiently.
cast_to_16_bit_precision(exported_program)

# Convert to a Core AI program and save it as an .aimodel.
converter = coreai_torch.TorchConverter()
converter.add_exported_program(exported_program)
ai_program = converter.to_coreai()
ai_program.optimize()
ai_program.save_asset(Path("model.aimodel"))
```

The `cast_to_16_bit_precision` step is recommended as part of the default Core AI workflow: it rewrites the exported graph so FP32 tensors are cast to FP16, for faster inference. See [16-bit PyTorch Model Casting](../utils/casting.md) for details.

For more on `coreai-torch` itself — supported ops, the decomposition table, and the `AIProgram` API — see the [`coreai-torch` documentation](https://github.com/apple/coreai-torch).

For compatibility purposes, setting `backend` in `finalize()` to `ExportBackend.CoreML` produces a torch model that can be converted with [coremltools](https://github.com/apple/coremltools) and deployed with [Core ML](https://developer.apple.com/documentation/coreml). Not all `coreai-opt` compression configurations are supported by Core ML — for example, FP4/FP8 weight/activation quantization are Core AI-only and will not convert with `coremltools`.
