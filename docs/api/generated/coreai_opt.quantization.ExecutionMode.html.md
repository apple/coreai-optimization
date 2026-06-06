# coreai_opt.quantization.ExecutionMode

### *class* coreai_opt.quantization.ExecutionMode(value, \*args, \*\*kwargs)

Bases: `StrEnum`

Enum representing quantization execution modes.

Each member is a string value representing the execution mode used
for quantization.

* **Parameters:**
  * **value** (*object*)
  * **args** (*Any*)
  * **kwargs** (*Any*)
* **Return type:**
  Any

#### GRAPH

Graph-based quantization using `torch.export` to capture the model as an FX graph,
then applying quantization on top. Built on `torchao`’s PT2E implementation. Requires
the model to be exportable via `torch.export.export`. Recommended default.

#### EAGER

Eager-mode quantization that works directly on `nn.Module` without graph capture.
Supports dynamic control flow (if/else, loops) and is the fallback when a model is not
exportable.

#### \_\_init_\_(\*args, \*\*kwds)

#### PT2E *: [ExecutionMode](#coreai_opt.quantization.ExecutionMode)* *= 'graph'*

Deprecated. Use `ExecutionMode.GRAPH` instead.
