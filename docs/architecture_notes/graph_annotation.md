# Graph Annotation

This note explains how graph-mode quantization annotation works: how a `QuantizerConfig` becomes a set of `QuantizationAnnotation` entries on an exported fx graph, and why the algorithm is shaped the way it is.

Code lives in `src/coreai_opt/quantization/_graph/`. The entry point is `_qspec_reconcile.py`'s `annotate_via_reconciliation`, called from `_AnnotationHandler.annotate`.

## Problem Formulation

A quantization config allows the user to express how each quantizable tensor in the graph (weight or activation) should be quantized. However, many tensors are implicitly referred to by multiple nodes. For example, the output of a node becomes the input to another node. If both nodes independently quantized their respective references, we would end up with a quant -> dequant -> quant, even if that output is only consumed in one place. Ideally, we would quantize the tensor only once.

Furthermore, passthrough operations, like `concat`, imply some agreement between their outputs and all of their inputs. In the case of `concat`, all inputs should have the same dtype and, if the output has per-tensor scales, we should jointly compute the scale and zero point across all inputs. A motivating example in pseudo-code:

```text
x = concat(conv(a), sigmoid(b))
```

Here each input to the concat has a different range. Independently, `conv(a)` would compute its range from training data, while `sigmoid(b)` would use a fixed (0, 1) range. However, because they're used in the same `concat`, we want to jointly quantize them. Thus, we should use the range of `conv(a)` even for `sigmoid(b)`.

We want to ensure that the actual annotations written to the graph obey the user's intent (or communicate when we cannot obey the user's intent), while complying with the constraints that the graph imposes on shared tensors.

## Constraint-relaxing algorithm

1. Pattern match the graph against the config to get a ranked list of matched patterns, higher-priority configs earlier.
2. Use the ranked patterns to construct a *provisional* qspec for each quantizable tensor, stored in the provisional qspec map. A provisional qspec answers "what would this tensor's qspec be if it didn't need to worry about any other tensor?"
3. Find the relationships between tensors and generate a *constraints* for each. Enqueue them.
4. While the queue is not empty:
   1. Pop the next constraint. Fetch the provisional qspecs it references. If they already satisfy it, do nothing.
   2. Otherwise reconcile them, field by field — see [Per-field policies](#per-field-policies).
   3. If the constraint requires the tensors to share a *quantizer* rather than merely compatible settings, point all of them at one shared provisional qspec.
   4. If anything changed, re-generate constraints for the affected tensors and re-enqueue.
5. Reassemble a spec from each settled field map and annotate the graph.

### Why it terminates

Most constraints only ever *relax* a field or *raise* its priority-ness, and priority-ness is monotonic — reconciliation lowers a field's priority number and never raises it. We will only enqueue new constraints when a field actually changes.

`InheritFields` is the exception: it copies values at unchanged priority. It terminates because the relation follows fx edges and an fx graph is a DAG, so a chain cannot cycle back on itself. Thus, there is a limit on the number of `InheritFields` constraints we can create.

Empirically, we measured that constraint reconciliation takes `O(#edges)` iterations, with a constant close to 2.

## Per-field policies

The fields are exactly the settable inputs of `coreai_opt...QuantizationSpec`, plus `QUANTIZATION_TARGET`. Reconciliation settles each independently and `_qspec_resolution._build_concrete_spec` reassembles a spec from the results.

`_qspec_constraints._FIELD_POLICY` is the authority. The reasoning:

| Field | Policy | Why |
| --- | --- | --- |
| `DTYPE` | Highest-priority proposal wins | Two tensors sharing an observer can't have different dtypes, and there's no safe join — int4 and int8 aren't compatible in either direction. Config precedence decides. |
| `QFORMULATION`, `FAKE_QUANTIZE_CLS`, `QPARAM_CALCULATOR_CLS`, `RANGE_CALCULATOR_CLS`, `SCALE_DTYPE`, `QSCHEME`, `GRANULARITY` | Highest-priority proposal wins | Ordinary config choices. |
| `FLOAT_RANGE` | Widest of the proposals, per bound | A shared observer must cover every member's data, so a pin survives only while every member agrees on it. Priority is deliberately not consulted: covering the data is a correctness constraint, not a preference. This is also where "fixed qparams" lives — a fixed observer is one whose `float_range` is pinned on both bounds. |
| `QUANTIZATION_TARGET` | Highest-priority proposal wins | This is not actually set as part of the config, but comes from whether the tensor is a weight or an activation. If we share a qspec between a weight and an activation, using the higher-priority one creates a coherent set of fields that matches at least one target's original provisional qspec. |

Some properties are deliberately *not* fields. `quant_min` / `quant_max` / `n_bits` / `target_dtype` are computed by `QuantizationSpec` from `dtype` and `qscheme`, so they fall out of reassembly; reconciling them alongside their own inputs would let them contradict those inputs. `ch_axis` lives inside `GRANULARITY`.

## Constraint Types

Tensors can be related in a few different ways, and they need different treatment.

The first is **two tensors forced onto one observer** — `cat`'s inputs, or the several consumers of one shared weight. Here the single observer has to cover every member's data, so the members' settings must be reconciled: for `cat([relu(a), b])` relu's `[0, ∞)` pin is *relaxed* to cover `b`'s negatives, because keeping it would clip `b`. `ShareObserverInstance` expresses this.

The second is **two tensors which must agree on some fields**. In this case, they may have different QParams, but must agree in other ways. We cover this with `ShareFields`

The third is **one tensor seen at two points**, which is what a chain of shape-only ops produces. `relu -> view -> linear` has one tensor throughout: `view` changes no values, so the tensor at `linear`'s input is relu's output and is still in `[0, ∞)`. Ideally both points would share one observer, since they are measuring identical data.

They cannot. A `QParamsCalculator` is bound to a single tensor rank: it caches its resolved axis on first use and sizes its running buffers to the first tensor it sees, an invariant `QParamsCalculatorBase._resolve_axis` states directly. `view`, `reshape`, `squeeze`, `unsqueeze`, `select` and `expand` all change rank, so one observer spanning them fails at runtime — and not only under per-channel granularity, since a per-tensor observer that first saw a 4-D tensor holds `[1,1,1,1]` buffers and cannot accept 3-D.

So the two points keep separate, correctly-shaped observers, and what transfers between them is the *knowledge* rather than the observer. `ShareFields` is the wrong tool for that: it reconciles, and reconciliation is bidirectional, so a downstream slot's default range would widen the upstream pin instead of adopting it. What is needed is a one-way copy, which is `InheritFields`:

| Constraint | Relation | Effect |
| --- | --- | --- |
| `ShareObserverInstance` | different tensors, one observer | reconcile — the observer must cover every member |
| `ShareFields` | different tensors, separate observers | agree on the named fields, nothing else |
| `InheritFields` | one tensor, separate observers | copy, upstream to downstream |

Without it, `linear`'s input relearns a range it was already given and quantizes provably non-negative data as symmetric, spending half the integer range on values that cannot occur.

`InheritFields` carries only `QSCHEME` and `FLOAT_RANGE` — properties that are facts about the data rather than choices about it. `DTYPE` is excluded: it is a config choice, and copying it downstream would override a higher-priority config.

Propagation resolves chain-wise rather than hop-wise. The intermediate shape-only ops aren't quantizable patterns, so they're absent from `winning_configs` and hold no fields, leaving nothing to carry a fact from one hop to the next.

## Declined slots

A slot is *declined* when a key in the config named it and held `None`. The referencing node does not want to quantize the tensor, but the tensor may still be quantized by other references.

What a decline does then depends on which relation it lands in.

**Adjacent edges drop the constraint.** An edge is one tensor with two slots. If the producer is declined while the consumer wants its input quantized, there is nothing to deduplicate: the consumer observes, the producer doesn't, and the edge carries one observer either way.

**Groups tied by op semantics let quantization win.** For `flatten`, `cat`, `maxpool` and the rank-preserving shape-only ops, the tied slots are one tensor of one op. "Quantize my input but not my output" is not a coherent instruction about such a tensor, so any member asking to be observed outranks a decline, in either direction, whatever the priorities. The group goes unobserved only when nothing in it asks for quantization — which is what makes a weight-only config leave every `flatten` and `cat` in float, since there both sides are disabled.

**Shared state resolves by priority.** One weight reached by several consumers is also one tensor, but each consumer's config independently states whether *it* quantizes, so ordinary config precedence applies and the highest-priority proposal decides — decline or not. This is what lets `op_name_config` exclude a single invocation of a reused module: that entry outranks the global config, so its `None` wins and the shared weight goes unobserved. Ties go to the decline.
