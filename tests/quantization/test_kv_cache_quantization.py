# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
KV-cache quantization tests

Covers ``KVCacheQuantConfig`` validation, the prepare-time annotation override,
the finalize-side dequantize relocation + cache-buffer retyping, and
interactions with module-scope configs.
"""

import logging

import pytest
import torch
import torch.nn as nn
from pydantic import ValidationError

from coreai_opt import ExportBackend
from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.quantization import (
    ModuleQuantizerConfig,
    Quantizer,
    QuantizerConfig,
)
from coreai_opt.quantization._graph._annotation_pattern_registry import (
    _AnnotationPatternRegistry,
)
from coreai_opt.quantization.config.quantization_config import (
    KVCacheQuantConfig,
    OpQuantizerConfig,
)
from coreai_opt.quantization.spec import (
    PerTensorGranularity,
    QuantizationSpec,
    default_activation_quantization_spec,
    default_weight_quantization_spec,
)
from coreai_opt.quantization.spec.fake_quantize import FakeQuantizeImplBase

# Module-scoped test ops. ``torch.library`` registrations are process-global
# and have no deregistration API, so these ops persist in the torch op registry
# for the entire test session. The ``coreai_opt_test_kv_cache`` namespace is
# reserved for this file — do not register other ops under the same names
# elsewhere or pytest collection will fail with a duplicate-registration error.


@torch.library.custom_op("coreai_opt_test_kv_cache::key_cache_update", mutates_args=["x"])
def _test_key_cache_update(
    x: torch.Tensor,
    update: torch.Tensor,
    begin: torch.Tensor,
    end: torch.Tensor,
    layer_idx: int,
    seq_len: int,
) -> torch.Tensor:
    return x.narrow(0, layer_idx, 1).squeeze(0).narrow(2, 0, seq_len).clone()


@_test_key_cache_update.register_fake
def _(x, update, begin, end, layer_idx, seq_len):
    return torch.empty((1, x.shape[2], seq_len, x.shape[4]), dtype=x.dtype)


@torch.library.custom_op("coreai_opt_test_kv_cache::value_cache_update", mutates_args=["x"])
def _test_value_cache_update(
    x: torch.Tensor,
    update: torch.Tensor,
    begin: torch.Tensor,
    end: torch.Tensor,
    layer_idx: int,
    seq_len: int,
) -> torch.Tensor:
    return x.narrow(0, layer_idx, 1).squeeze(0).narrow(2, 0, seq_len).clone()


@_test_value_cache_update.register_fake
def _(x, update, begin, end, layer_idx, seq_len):
    return torch.empty((1, x.shape[2], seq_len, x.shape[4]), dtype=x.dtype)


class _KVCacheAttentionModel(nn.Module):
    """Two cache-update ops feeding SDPA."""

    def forward(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        q: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> torch.Tensor:
        begin = torch.zeros(5, dtype=torch.int32)
        end = torch.zeros(5, dtype=torch.int32)
        k = torch.ops.coreai_opt_test_kv_cache.key_cache_update(k_cache, k_new, begin, end, 0, 4)
        v = torch.ops.coreai_opt_test_kv_cache.value_cache_update(v_cache, v_new, begin, end, 0, 4)
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


class _AttentionModel(nn.Module):
    def __init__(self, in_dim: int = 8, hidden: int = 32):
        super().__init__()
        self.k_proj = nn.Linear(in_dim, hidden)
        self.v_proj = nn.Linear(in_dim, hidden)
        self.attn = _KVCacheAttentionModel()

    def forward(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        q: torch.Tensor,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> torch.Tensor:
        return self.attn(
            k_cache,
            v_cache,
            q,
            self.k_proj(k_new).unsqueeze(0),
            self.v_proj(v_new).unsqueeze(0),
        )


def _kv_cache_attention_example_inputs() -> tuple[torch.Tensor, ...]:
    """Example inputs for ``_AttentionModel``.

    Shapes match the cache layout (2 layers, 1 batch, 8 heads, 16 seq, 32 dim).
    ``_AttentionModel`` projects k_new/v_new (1, 8, 4, in_dim=8) via its Linears
    to (1, 8, 4, 32), unsqueezes to (1, 1, 8, 4, 32), and feeds those plus
    k_cache / v_cache / q to the inner cache ops + SDPA. q is built at the
    post-cache shape (1, 8, 4, 32) so SDPA accepts q / k / v together.
    """
    return (
        torch.zeros(2, 1, 8, 16, 32, dtype=torch.float32),  # k_cache
        torch.zeros(2, 1, 8, 16, 32, dtype=torch.float32),  # v_cache
        torch.randn(1, 8, 4, 32, dtype=torch.float32),  # q
        torch.randn(1, 8, 4, 8, dtype=torch.float32),  # k_new
        torch.randn(1, 8, 4, 8, dtype=torch.float32),  # v_new
    )


# Cache op short names that the multi-op tests configure together.
_CACHE_OP_NAMES = ("key_cache_update", "value_cache_update")


def _make_act_spec(dtype, qscheme: str = "symmetric") -> QuantizationSpec:
    """Activation QuantizationSpec with the given dtype/qscheme; other fields
    match ``default_activation_quantization_spec()``."""
    return QuantizationSpec(
        dtype=dtype,
        qscheme=qscheme,
        granularity=PerTensorGranularity(),
        fake_quantize_cls="default",
        qparam_calculator_cls="moving_average",
        range_calculator_cls="minmax",
    )


def _default_cache_config(spec) -> KVCacheQuantConfig:
    """Cache config with ``op_input_spec={1: spec}`` and both other specs None."""
    return KVCacheQuantConfig(
        op_quantizer_config=OpQuantizerConfig(
            op_input_spec={1: spec},
            op_output_spec=None,
            op_state_spec=None,
        ),
    )


def _build_quantizer_config(
    kv_cache_quant_configs=None, execution_mode: str = "graph"
) -> QuantizerConfig:
    return QuantizerConfig(
        execution_mode=execution_mode,
        global_config=ModuleQuantizerConfig(
            op_state_spec={"weight": default_weight_quantization_spec()},
            op_input_spec=None,
            op_output_spec=None,
        ),
        kv_cache_quant_configs=kv_cache_quant_configs,
    )


def _get_single_cache_op_node(graph_module, op_short_name: str) -> torch.fx.Node:
    """Find and return the sole ``call_function`` node for the named cache op.

    Asserts exactly one match — the test models register one call site per op.
    """
    matches = [
        n
        for n in graph_module.graph.nodes
        if n.op == "call_function" and op_short_name in str(n.target)
    ]
    assert len(matches) == 1, f"expected exactly one {op_short_name} node, got {len(matches)}"
    return matches[0]


class TestKVCacheQuantization:
    @staticmethod
    def _prepare_model_with_kv_cache(config):
        model = _AttentionModel().eval()
        quantizer = Quantizer(model, config)
        return quantizer.prepare(_kv_cache_attention_example_inputs())

    @pytest.mark.parametrize(
        "op_state_spec,op_input_spec,op_output_spec,regex_match",
        [
            (
                None,
                {
                    0: default_activation_quantization_spec(),
                    1: default_activation_quantization_spec(),
                },
                None,
                "must contain exactly one key",
            ),
            (
                None,
                {"*": default_activation_quantization_spec()},
                None,
                "must contain exactly one key",
            ),
            (
                None,
                {-1: default_activation_quantization_spec()},
                None,
                "non-negative int",
            ),
            (
                None,
                {1: None},
                None,
                "non-None QuantizationSpec",
            ),
            (
                {"weight": default_weight_quantization_spec()},
                {0: default_activation_quantization_spec()},
                None,
                "op_state_spec must be empty or None",
            ),
            (
                None,
                {0: default_activation_quantization_spec()},
                {0: default_activation_quantization_spec()},
                "op_output_spec must be empty or None",
            ),
        ],
        ids=[
            "two_int_keys",
            "string_key",
            "negative_int_key",
            "none_input_spec_value",
            "non_none_state_spec",
            "non_none_output_spec",
        ],
    )
    def test_kv_cache_quant_invalid_configs(
        self, op_state_spec, op_input_spec, op_output_spec, regex_match
    ):
        with pytest.raises(ValidationError, match=regex_match):
            _ = KVCacheQuantConfig(
                op_quantizer_config=OpQuantizerConfig(
                    op_input_spec=op_input_spec,
                    op_output_spec=op_output_spec,
                    op_state_spec=op_state_spec,
                ),
            )

    def test_kv_cache_quant_rejected_in_eager_mode(self):
        """``kv_cache_quant_configs`` is graph-mode only; constructing a
        ``QuantizerConfig`` with eager mode and any cache config must raise.
        """
        with pytest.raises(ValidationError, match="only supported with ExecutionMode.GRAPH"):
            _build_quantizer_config(
                execution_mode="eager",
                kv_cache_quant_configs={
                    "key_cache_update": _default_cache_config(
                        default_activation_quantization_spec()
                    ),
                },
            )

    def test_prepare_does_not_mutate_global_pattern_registry(self):
        """``prepare()`` must not add a cache-update entry to the process-global
        ``_AnnotationPatternRegistry``: the per-run pattern is scoped to the
        ``_AnnotationHandler`` instance via ``extra_patterns``.
        """
        baseline_keys = set(_AnnotationPatternRegistry.REGISTRY.keys())
        config = _build_quantizer_config(
            kv_cache_quant_configs={
                "key_cache_update": _default_cache_config(default_activation_quantization_spec()),
            },
        )
        self._prepare_model_with_kv_cache(config)

        after_keys = set(_AnnotationPatternRegistry.REGISTRY.keys())
        assert after_keys == baseline_keys, (
            f"global registry was mutated by prepare(); leaked keys: {after_keys - baseline_keys}"
        )

    def test_user_config_survives_prepare(self):
        """``prepare()`` must not mutate the user's ``QuantizerConfig``."""
        config = _build_quantizer_config(
            kv_cache_quant_configs={
                "key_cache_update": _default_cache_config(default_activation_quantization_spec()),
            },
        )
        before = config.model_dump()

        self._prepare_model_with_kv_cache(config)
        after = config.model_dump()
        assert before == after, "prepare() mutated the user's QuantizerConfig in place."

    @pytest.mark.parametrize(
        "in_global,in_module_type,in_module_name,expected_substrings",
        [
            (True, False, False, ["global_config.op_type_config"]),
            (False, True, False, ["module_type_configs"]),
            (False, False, True, ["module_name_configs"]),
            (
                True,
                True,
                True,
                [
                    "global_config.op_type_config",
                    "module_type_configs",
                    "module_name_configs",
                ],
            ),
        ],
        ids=[
            "global_only",
            "module_type_only",
            "module_name_only",
            "all_scopes",
        ],
    )
    def test_kv_cache_quant_warns_on_duplicate_op_type_config(
        self, caplog, in_global, in_module_type, in_module_name, expected_substrings
    ):
        """Validator must warn (not reject) when a ``kv_cache_quant_configs`` key
        also appears in any ``op_type_config`` scope. Pins down the
        "kv_cache_quant_configs wins; duplicates ignored" UX contract.
        """
        kv_cache_op = "key_cache_update"
        op_qcfg = OpQuantizerConfig(
            op_input_spec={1: default_activation_quantization_spec()},
            op_output_spec=None,
            op_state_spec=None,
        )

        global_cfg = ModuleQuantizerConfig(
            op_state_spec={"weight": default_weight_quantization_spec()},
            op_input_spec=None,
            op_output_spec=None,
            op_type_config={kv_cache_op: op_qcfg} if in_global else {},
        )
        module_type_cfgs = (
            {
                "torch.nn.modules.linear.Linear": ModuleQuantizerConfig(
                    op_type_config={kv_cache_op: op_qcfg},
                )
            }
            if in_module_type
            else {}
        )
        module_name_cfgs = (
            {"linear1": ModuleQuantizerConfig(op_type_config={kv_cache_op: op_qcfg})}
            if in_module_name
            else {}
        )

        with caplog.at_level(
            logging.WARNING,
            logger="coreai_opt.quantization.config.quantization_config",
        ):
            QuantizerConfig(
                execution_mode="graph",
                global_config=global_cfg,
                module_type_configs=module_type_cfgs,
                module_name_configs=module_name_cfgs,
                kv_cache_quant_configs={
                    kv_cache_op: KVCacheQuantConfig(op_quantizer_config=op_qcfg),
                },
            )

        matching = [
            r.getMessage() for r in caplog.records if "kv_cache_quant_configs" in r.getMessage()
        ]
        assert len(matching) == 1, (
            f"expected exactly one duplicate-key warning, got {len(matching)}: {matching!r}"
        )
        for substring in expected_substrings:
            assert substring in matching[0], (
                f"expected {substring!r} in warning, got: {matching[0]}"
            )

    def test_none_kv_cache_quant_config(self):
        """Without a cache config, ``prepare()`` leaves the cache op's update
        input unobserved.
        """
        config = _build_quantizer_config(kv_cache_quant_configs=None)

        prepared = self._prepare_model_with_kv_cache(config)
        cache_op = _get_single_cache_op_node(prepared, "key_cache_update")
        update_input = cache_op.all_input_nodes[1]

        assert "activation_post_process_" not in update_input.name, (
            f"cache-update op's update input node should not be a "
            f"fake-quantize observer; got {update_input.op}/{update_input.target}"
        )

    def test_prepare_rejects_out_of_range_quant_input_idx(self):
        """``prepare()`` must fail when ``op_input_spec``'s int key is
        out of range for the matched op's ``all_input_nodes``.
        """
        bad_idx = 99
        config = _build_quantizer_config(
            kv_cache_quant_configs={
                "key_cache_update": KVCacheQuantConfig(
                    op_quantizer_config=OpQuantizerConfig(
                        op_input_spec={bad_idx: default_activation_quantization_spec()},
                        op_output_spec=None,
                        op_state_spec=None,
                    ),
                ),
            },
        )
        with pytest.raises(
            ValueError, match=rf"(?s)kv_cache_quant_configs.*op_input_spec.*{bad_idx}"
        ):
            self._prepare_model_with_kv_cache(config)

    def test_kv_cache_lifecycle(self):
        """End-to-end prepare → calibrate → finalize with two cache ops, each
        configured with a different dtype (key=int8, value=uint8) to prove
        ``kv_cache_quant_configs`` dispatches per op type instead of applying
        a single spec to every matched cache op. Asserts per-op observer
        dtypes after prepare and per-op cache-buffer dtypes after finalize.
        """
        op_specs = {
            "key_cache_update": _make_act_spec(torch.int8),
            "value_cache_update": _make_act_spec(torch.uint8, qscheme="asymmetric"),
        }
        config = _build_quantizer_config(
            kv_cache_quant_configs={
                op: _default_cache_config(spec) for op, spec in op_specs.items()
            },
        )
        example_inputs = _kv_cache_attention_example_inputs()
        quantizer = Quantizer(_AttentionModel().eval(), config)

        prepared = quantizer.prepare(example_inputs)

        for op_short_name, spec in op_specs.items():
            cache_op = _get_single_cache_op_node(prepared, op_short_name)

            # The new K/V at all_input_nodes[1] must be wrapped by a
            # FakeQuantize module whose dtype matches the configured op_input_spec.
            update_input = cache_op.all_input_nodes[1]
            assert "activation_post_process_" in update_input.name, (
                f"{op_short_name}'s update input should be fed by a "
                f"activation_post_process node; got {update_input.op}/{update_input.target}"
            )
            observer = prepared.get_submodule(str(update_input.target))
            assert observer.dtype == spec.dtype, (
                f"{op_short_name}'s observer dtype should be {spec.dtype} "
                f"(per-op dispatch); got {observer.dtype}"
            )

            # Cache state at all_input_nodes[0] must NOT be observed.
            cache_input = cache_op.all_input_nodes[0]
            assert cache_input.op == "placeholder", (
                f"{op_short_name}'s cache input should be a placeholder; got "
                f"{cache_input.op}/{cache_input.target}"
            )

            # Begin/end index tensors at args[2:4] must NOT be observed —
            # non-float inputs are skipped by _get_input_qspec_map.
            for idx in (2, 3):
                arg = cache_op.args[idx]
                assert arg.op != "call_module", (
                    f"{op_short_name}'s args[{idx}] is an int index tensor; should "
                    f"not be observed, but got call_module {arg.target}"
                )

        # Global op_state_spec must still apply to non-cache-op modules: each
        # Linear's weight should get a weight-target observer.
        weight_observers = [
            m
            for m in prepared.modules()
            if isinstance(m, FakeQuantizeImplBase)
            and m.quantization_target == CompressionTargetTensor.WEIGHT
        ]
        assert len(weight_observers) == 2, (
            f"expected one weight observer per Linear (2 total), got {len(weight_observers)}"
        )

        finalized = quantizer.finalize(backend=ExportBackend.CoreAI)
        for op_short_name, spec in op_specs.items():
            cache_op = _get_single_cache_op_node(finalized, op_short_name)

            # The new K/V at all_input_nodes[1] should now be a coreai.quantize
            # node (the input-side dq was deleted).
            update_input = cache_op.all_input_nodes[1]
            assert update_input.target is torch.ops.coreai.quantize, (
                f"{op_short_name}'s update input should be coreai.quantize; "
                f"got {update_input.target}"
            )

            # The cache op's sole user should be the relocated coreai.dequantize.
            users = list(cache_op.users)
            assert len(users) == 1, f"expected one user of {op_short_name}, got {users}"
            new_dq = users[0]
            assert new_dq.target is torch.ops.coreai.dequantize, (
                f"{op_short_name}'s user should be coreai.dequantize; got {new_dq.target}"
            )

            # Cache placeholder at all_input_nodes[0] must be retyped to the
            # op's configured dtype.
            cache_placeholder = cache_op.all_input_nodes[0]
            assert cache_placeholder.op == "placeholder"
            assert cache_placeholder.meta["val"].dtype == spec.dtype, (
                f"{op_short_name}'s cache placeholder dtype should be {spec.dtype}; "
                f"got {cache_placeholder.meta['val'].dtype}"
            )

    def test_kv_cache_spec_wins_over_module_scope_shadowing(self):
        """``kv_cache_quant_configs`` is a global-only knob with the highest
        priority. When configured cache ops live inside a module covered by
        ``module_name_configs``, the cache spec still wins via a
        post-annotation override — the module-scope wildcard does not silently
        shadow the cache op's input edges.

        Setup: K/V projections live on the outer model; only the cache ops
        (and SDPA) live under ``attn``. The ``attn`` wildcard asks for fp8 on
        every input edge of every op it covers; the per-op cache configs
        request int8 on the update edge. After the override:

        - both cache ops' update inputs have int8 observers (override beat
          the wildcard's fp8)
        - both cache ops' cache-state inputs are unobserved (override stripped
          the wildcard's wholesale "*" claim from those edges too)
        """
        int8_spec = _make_act_spec(torch.int8)
        fp8_spec = _make_act_spec(torch.float8_e4m3fn)
        config = QuantizerConfig(
            execution_mode="graph",
            global_config=ModuleQuantizerConfig(
                op_state_spec={"weight": default_weight_quantization_spec()},
                op_input_spec=None,
                op_output_spec=None,
            ),
            module_name_configs={
                "attn": ModuleQuantizerConfig(
                    op_input_spec={"*": fp8_spec},
                ),
            },
            kv_cache_quant_configs={
                op: KVCacheQuantConfig(
                    op_quantizer_config=OpQuantizerConfig(
                        op_input_spec={1: int8_spec},
                        op_output_spec=None,
                        op_state_spec=None,
                    ),
                )
                for op in _CACHE_OP_NAMES
            },
        )
        example_inputs = _kv_cache_attention_example_inputs()
        prepared = Quantizer(_AttentionModel().eval(), config).prepare(example_inputs)

        for op_short_name in _CACHE_OP_NAMES:
            cache_op = _get_single_cache_op_node(prepared, op_short_name)

            # Update input: override wrote int8 over the wildcard's fp8 demand.
            update_input = cache_op.all_input_nodes[1]
            assert update_input.op == "call_module", (
                f"{op_short_name}'s update input should be wrapped by an observer; "
                f"got {update_input.op}/{update_input.target}"
            )
            observer = prepared.get_submodule(str(update_input.target))
            assert observer.dtype == torch.int8, (
                f"{op_short_name}'s observer dtype should be int8 (cache spec "
                f"overrode the attn wildcard's fp8); got {observer.dtype}"
            )

            # Cache-state input: wildcard would have claimed it, override stripped it.
            cache_input = cache_op.all_input_nodes[0]
            assert cache_input.op != "call_module", (
                f"{op_short_name}'s cache-state input should not be observed; "
                f"got {cache_input.op}/{cache_input.target}"
            )
