# 16-bit PyTorch Model Casting

Running a model in 16-bit precision roughly halves the size of its parameters on disk and in memory, and typically runs faster on hardware that has native 16-bit support. The tradeoff is reduced numerical precision: some models lose accuracy when run in fp16/int16, and the impact is workload-dependent. It is good practice to validate accuracy on representative data after casting.

The following casting utilities imported from `coreai_opt.casting` transform a `torch.export.ExportedProgram` so that its computations and weights run in 16-bit precision. Three public APIs are available:

```{list-table}
:header-rows: 1

* - API
  - Scope
* - `cast_to_16_bit_precision`
  - FP32→FP16 and INT32/INT64→INT16 (top-level, recommended)
* - `cast_fp32_to_fp16`
  - FP32→FP16 only
* - `cast_int32_to_int16`
  - INT32/INT64→INT16 only
```

Together, these passes rewrite the exported graph in place: parameters/buffers, user inputs, in-graph tensor creation ops (e.g. `torch.zeros`, `torch.arange`), and op output dtypes may all be updated. The downstream `coreai_torch` converter therefore sees a graph whose tensors are already in 16-bit precision rather than an FP32 graph wrapped with per-op casts.

When updating the graph, certain criteria must be met in order for casting to take place for a particular state or operation. For example, parameters will not be cast if they contain values exceeding the FP16 maximum threshold value. Integer casting will not take place for operations whose outputs are used as indices to avoid potential overflow.

:::{note}
These passes mutate the `ExportedProgram` in place and may change the dtypes of user inputs and outputs. Calling code may need to be updated so that input tensors are passed as `fp16`/`int16` (as appropriate) and that the new output dtypes are handled correctly, for any inputs and outputs that were modified.
:::

## How the casting passes select what to cast

The two casting passes apply different selection criteria, resulting in different levels of aggressiveness.

**FP32 → FP16.** The FP pass is relatively aggressive: it attempts to cast all floating-point state and operations to FP16 by default. The main exception is numerical safety — tensors whose values exceed the FP16 representable range (approximately ±65504) are left in FP32 to avoid overflow.

**INT32/INT64 → INT16.** The INT pass is more conservative. Rather than casting all integer tensors, it targets only those that are likely to benefit without risking correctness. A tensor is skipped if any of the following apply:

- It is **constant-foldable**: its value is fully determined at compile time and will be folded away, so a dtype change has no effect.
- It **feeds into an indexing operation**: casting could cause overflow and produce incorrect indices.
- It is **not consumed by a computationally intensive op**: for tensors used only in lightweight operations, the overhead of a cast outweighs the benefit.

## Casting an uncompressed model

The example below shows the typical flow: define a model, export it, cast to 16-bit, run the cast program, and convert with `coreai_torch`.

```python
from pathlib import Path
import torch
from coreai_opt.casting import cast_to_16_bit_precision

model = MyModel().eval()
example_input = torch.randn(1, 3, 224, 224)

# 1. Export to a torch.export.ExportedProgram with coreai_torch decompositions
import coreai_torch

exported_program = torch.export.export(model, (example_input,)).run_decompositions(
    coreai_torch.get_decomp_table()
)

# 2. Cast FP32 -> FP16 and INT32/INT64 -> INT16 in place.
cast_to_16_bit_precision(exported_program)

# 3. Run the cast program. User inputs are now expected as fp16, and
#    outputs may also be fp16. Update calling code accordingly.
output = exported_program.module()(example_input.half())

# 4. Convert to a deployable format. The graph is already 16-bit, so the
#    converter sees fp16/int16 tensors directly with no extra casts.
converter = coreai_torch.TorchConverter()
converter.add_exported_program(exported_program)
ai_program = converter.to_coreai()
ai_program.optimize()
ai_program.save_asset(Path("model.aimodel"))
```

If only one type of casting is needed, replace `cast_to_16_bit_precision` with `cast_fp32_to_fp16` or `cast_int32_to_int16`.

## Casting a compressed model

If model compression is to be performed, it should be done prior to casting.

The below example shows application of `cast_to_16_bit_precision` after weight and activation quantization has been applied. Any quantized int8 buffers are left untouched; any remaining FP32 weights move to FP16.

```python
from pathlib import Path
import torch
from coreai_opt import ExportBackend
from coreai_opt.casting import cast_to_16_bit_precision
from coreai_opt.quantization import Quantizer, QuantizerConfig

model = MyModel().eval()
example_input = torch.randn(1, 3, 224, 224)

# 1. Quantize weights and activations to INT8 and finalize for the Core AI backend.
config = QuantizerConfig()
quantizer = Quantizer(model, config)
prepared_model = quantizer.prepare((example_input,))

with quantizer.calibration_mode():
    # Perform forward passes for calibration
    prepared_model(...)

finalized_model = quantizer.finalize(backend=ExportBackend.CoreAI)

# 2. Export with coreai_torch decompositions
import coreai_torch

exported_program = torch.export.export(
    finalized_model, (example_input,)
).run_decompositions(coreai_torch.get_decomp_table())

# 3. Cast remaining FP32/INT32 portions of the graph. Int8 quantized buffers
#    are preserved.
cast_to_16_bit_precision(exported_program)

# 4. Convert to a deployable format. The graph is already 16-bit, so the
#    converter sees fp16/int16 tensors directly with no extra casts.
converter = coreai_torch.TorchConverter()
converter.add_exported_program(exported_program)
ai_program = converter.to_coreai()
ai_program.optimize()
ai_program.save_asset(Path("model.aimodel"))
```

## Difference vs. PyTorch's `.half()` and `torch.autocast`

PyTorch ships two related mechanisms for fp16 execution; neither produces a true 16-bit model artifact:

```{list-table}
:header-rows: 1

* - Mechanism
  - What it changes
  - Produces a 16-bit model?
* - `model.half()` / `model.to(torch.float16)`
  - Only module parameters and registered buffers are cast.<br>Tensor creation ops and user inputs are untouched and can still result in large portions of the model in FP32.
  - Partial — tensors created inside `forward()` (e.g. `torch.zeros`, `torch.arange`, `torch.tensor`) remain FP32 and must be cast by hand.
* - `torch.autocast`
  - Runtime: auto-casts op inputs in a context
  - No — the underlying model is still FP32; autocast wraps each op with cast-down/cast-up at run time.
* - `cast_to_16_bit_precision`
  - Graph rewrite on `ExportedProgram`
  - Yes — parameters, user inputs, in-graph creation ops, and op output dtypes are all updated to 16-bit.
```

Because the casting utility rewrites tensor dtypes directly in the exported graph, downstream converters see a 16-bit graph and do not need to insert per-op cast wrappers.
