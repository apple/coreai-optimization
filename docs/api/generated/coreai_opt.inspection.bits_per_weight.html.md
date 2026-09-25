# coreai_opt.inspection.bits_per_weight

### coreai_opt.inspection.bits_per_weight(model)

Compute the average bits-per-weight of a prepared `coreai-opt` model.

Walks the module tree once. For each parametrized weight, the dense original
tensor is counted at its effective compressed cost (eager mode quantization or
palettization). Every other directly-owned parameter (biases, norms) and every
buffer (BatchNorm running stats, RoPE caches, etc.) are counted at their
full-precision dtype cost, regardless of `persistent=`.

* **Parameters:**
  **model** (*torch.nn.Module*) – A full-precision, eager-mode quantized, or
  palettized prepared model.
* **Returns:**
  Overall bpw, per-module breakdown, and the total
  number of bits and weights used to derive them.
* **Return type:**
  [BitsPerWeightResult](coreai_opt.inspection.BitsPerWeightResult.md#coreai_opt.inspection.BitsPerWeightResult)
* **Raises:**
  **NotImplementedError** – If `model` is a graph-mode prepared model (a
      `torch.fx.GraphModule`) or a `torch.export.ExportedProgram`, or if it
      contains a weight compression whose storage cost this utility cannot
      compute.
