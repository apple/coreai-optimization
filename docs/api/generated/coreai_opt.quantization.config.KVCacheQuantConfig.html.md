# coreai_opt.quantization.config.KVCacheQuantConfig

### *class* coreai_opt.quantization.config.KVCacheQuantConfig

Bases: `BaseModel`

Enables KV-cache buffer storage in a quantized dtype for one cache-update op type.

Carries the cache-update op’s quantization spec inline (via `op_quantizer_config`)
and is the finalize-side switch that promotes the resulting fake-quant to stored
quantization: the dequantize is relocated from the op’s input to its output and
the cache buffer is retyped to the quantized dtype.

Used as a value in `QuantizerConfig.kv_cache_quant_configs`, which maps the
short op-type name to a config.

Precondition on the cache op: it must commute with quantize/dequantize —
i.e. a pure data-movement op (slicing, narrowing, copy). Arithmetic on
cached values would silently produce a numerically wrong model.

During `prepare`, the cache spec is applied as a global-only knob with
highest priority: any prior annotation on the cache op — from any scope, by
any mechanism (including module-scope `op_input_spec={"*": ...}`
wildcards) — is **hard-overridden**, not merged. A warning is emitted at
config-construction time when explicit `op_type_config[op]` collisions
are detected. See `QuantizerConfig._validate_kv_cache_quant_configs()`.

#### op_quantizer_config

`OpQuantizerConfig` whose `op_input_spec` selects the
input edge to quantize. The int key indexes the op’s tensor-valued,
deduplicated inputs (`node.all_input_nodes`). `op_input_spec` must
contain exactly one key (no `"*"`, no multi-key dicts) because the
finalize-side relocation needs a single, unambiguous input edge to act on.
`op_output_spec` and `op_state_spec` must be explicitly set to None
(or empty); `OpQuantizerConfig` otherwise applies non-None defaults
which the validator rejects.

#### *property* quant_input_idx *: int*

The single int key in `op_quantizer_config.op_input_spec`.

Indexes the cache op’s `all_input_nodes` (tensor-valued, deduplicated
inputs) — see `op_quantizer_config` for details.
