# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Quantization tests parametrized across execution modes (graph and eager).

Each test class covers a distinct scenario or feature area of the quantizer.
Tests are parametrized via the ``execution_mode`` fixture so every scenario
runs for both graph and eager mode in a single test definition.
"""

import pytest
import torch
import torch.nn as nn

from coreai_opt import ExportBackend
from coreai_opt.quantization import (
    ModuleQuantizerConfig,
    QuantizationSpec,
    Quantizer,
    QuantizerConfig,
)
from coreai_opt.quantization._graph.quantizer import GraphQuantizer
from coreai_opt.quantization.config.quantization_config import QATSchedule
from coreai_opt.quantization.spec import (
    PerTensorGranularity,
    QuantizationScheme,
    default_activation_quantization_spec,
    default_weight_quantization_spec,
)
from coreai_opt.quantization.spec.fake_quantize import FakeQuantizeImplBase
from coreai_opt.quantization.spec.qparams_calculator import (
    DynamicQParamsCalculator,
    MovingAverageQParamsCalculator,
)
from tests.models.simple import SimpleLinearModel

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

_EXECUTION_MODES = ["graph", "eager"]


def _make_w8a8_config(execution_mode: str) -> QuantizerConfig:
    return QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={"weight": default_weight_quantization_spec()},
            op_input_spec={"*": default_activation_quantization_spec()},
            op_output_spec=None,
        )
    ).set_execution_mode(execution_mode)


@pytest.fixture(params=_EXECUTION_MODES, ids=_EXECUTION_MODES)
def execution_mode(request) -> str:
    return request.param


@pytest.fixture
def example_input() -> torch.Tensor:
    return torch.randn(2, 8)


def _count_fake_quant_modules(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, FakeQuantizeImplBase))


def _make_module_name_config(
    execution_mode: str,
    module_name_configs: dict,
) -> QuantizerConfig:
    """W8A8 config with the given module_name_configs applied."""
    return QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={"weight": default_weight_quantization_spec()},
            op_input_spec={"*": default_activation_quantization_spec()},
            op_output_spec=None,
        ),
        module_name_configs=module_name_configs,
    ).set_execution_mode(execution_mode)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAliasedSubmoduleQuantization:
    """
    The quantizer must correctly handle models where a submodule is reachable
    under more than one attribute name — the pattern used by HuggingFace wrappers
    that hoist backbone children to the top level (e.g. ClipModule).

    Eager mode: expected to pass — hook-based matching is by object identity.
    Graph mode: currently expected to fail — alias name "encoder" ends up in
                module_configs (via named_children() recursion in
                _set_config_for_module) but is absent from
                module_name_to_state_names_map (built with named_modules()
                which deduplicates), causing an AssertionError inside
                _match_and_annotate_state_node.

    For the model used here:
      canonical name: "_model.encoder"
      alias name:     "encoder"
    """

    def _make_model(self) -> nn.Module:
        class _EncoderModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(8, 8)
                self.fc2 = nn.Linear(8, 8)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc2(torch.relu(self.fc1(x)))

        class _BackboneModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = _EncoderModel()
                self.proj = nn.Linear(8, 8)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.proj(self.encoder(x))

        class AliasedSubmoduleModel(nn.Module):
            """
            Wrapper that stores a backbone as self._model and also hoists one of its
            children (encoder) as a top-level alias self.encoder — the same pattern
            used by HuggingFace model wrappers such as ClipModule.

            self._model   — the full backbone (registered first)
            self.encoder  — alias: same object as self._model.encoder
            """

            def __init__(self):
                super().__init__()
                self._model = _BackboneModel()
                self.encoder = self._model.encoder  # alias hoisted to top level
                self.head = nn.Linear(8, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.encoder(x)  # uses the alias path
                x = self._model.proj(x)
                return self.head(x)

        return AliasedSubmoduleModel().eval()

    def _config_excluding_submodule(self, execution_mode: str, module_name: str) -> QuantizerConfig:
        """Global W8A8 config with the named submodule excluded (config=None)."""
        return QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec={"weight": default_weight_quantization_spec()},
                op_input_spec={"*": default_activation_quantization_spec()},
            ),
            module_name_configs={module_name: None},
        ).set_execution_mode(execution_mode)

    def test_fake_quant_inserted_in_aliased_submodule(self, execution_mode, example_input):
        """Fake-quant modules must be inserted for ops inside the aliased submodule."""
        prepared = Quantizer(self._make_model(), _make_w8a8_config(execution_mode)).prepare(
            (example_input,)
        )
        assert _count_fake_quant_modules(prepared) > 0

    def test_canonical_and_alias_exclusion_same_result(self, execution_mode, example_input):
        """
        Excluding by canonical name and by alias name must produce the same
        fake-quant count — they refer to the same submodule.
        """
        full = Quantizer(self._make_model(), _make_w8a8_config(execution_mode)).prepare(
            (example_input,)
        )
        excl_canonical = Quantizer(
            self._make_model(), self._config_excluding_submodule(execution_mode, "_model.encoder")
        ).prepare((example_input,))
        excl_alias = Quantizer(
            self._make_model(), self._config_excluding_submodule(execution_mode, "encoder")
        ).prepare((example_input,))
        assert _count_fake_quant_modules(excl_canonical) < _count_fake_quant_modules(full)
        assert _count_fake_quant_modules(excl_canonical) == _count_fake_quant_modules(excl_alias)

    def test_later_module_name_config_wins_over_earlier(self, execution_mode, example_input):
        """
        When module_name_configs contains multiple entries that resolve to the same
        module (via canonical name or alias), the later entry in the dict wins and
        is applied to all invocations of that module.

        We verify this by comparing against single-entry configs that represent
        what the winning entry alone would produce:
          - {"_model.encoder": None, "encoder": W8A8} → later W8A8 wins
            → same count as {"encoder": W8A8} alone
          - {"encoder": W8A8, "_model.encoder": None} → later None wins
            → same count as {"_model.encoder": None} alone
        """
        global_mod_config = ModuleQuantizerConfig(
            op_state_spec={"weight": default_weight_quantization_spec()},
            op_input_spec={"*": default_activation_quantization_spec()},
            op_output_spec=None,
        )

        # Later entry is W8A8 → equivalent to only having "encoder": W8A8
        later_incl = Quantizer(
            self._make_model(),
            _make_module_name_config(
                execution_mode, {"_model.encoder": None, "encoder": global_mod_config}
            ),
        ).prepare((example_input,))
        only_incl = Quantizer(
            self._make_model(),
            _make_module_name_config(execution_mode, {"encoder": global_mod_config}),
        ).prepare((example_input,))

        # Later entry is None → equivalent to only having "_model.encoder": None
        later_excl = Quantizer(
            self._make_model(),
            _make_module_name_config(
                execution_mode, {"encoder": global_mod_config, "_model.encoder": None}
            ),
        ).prepare((example_input,))
        only_excl = Quantizer(
            self._make_model(), _make_module_name_config(execution_mode, {"_model.encoder": None})
        ).prepare((example_input,))

        assert _count_fake_quant_modules(later_incl) == _count_fake_quant_modules(only_incl)
        assert _count_fake_quant_modules(later_excl) == _count_fake_quant_modules(only_excl)
        assert _count_fake_quant_modules(later_excl) < _count_fake_quant_modules(later_incl)


class TestReusedModuleQuantization:
    """
    The quantizer must correctly handle models where the same module object is
    used (called) multiple times — the pattern seen in EDSR where a single
    nn.ReLU is shared across 16 residual blocks.

    Here we use a shared nn.Linear so that the reused module is a quantizable op.

    ReusedLinearModel has:
      self.fc       — canonical name, same object as self.fc_alias
      self.fc_alias — alias, same object as self.fc

    forward() calls both self.fc(x) and self.fc_alias(x), producing two separate
    sets of nodes in the exported graph (one per call site).

    Key behaviors to verify:
    - Both call sites get fake-quant nodes (one per invocation, not shared).
    - module_name_config applies to ALL invocations regardless of which name
      (canonical or alias) is used — configuring by module object identity.
    - op_name_config (graph mode) can target individual call-site nodes
      independently, enabling per-invocation configuration.
    """

    def _make_model(self) -> nn.Module:
        class ReusedLinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(8, 8)
                self.fc_alias = self.fc  # same object, different name
                self.head = nn.Linear(8, 4)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.fc(x)  # first invocation
                x = self.fc_alias(x)  # second invocation (same module)
                return self.head(x)

        return ReusedLinearModel().eval()

    def _config_excluding_module(self, execution_mode: str, module_name: str) -> QuantizerConfig:
        return QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec={"weight": default_weight_quantization_spec()},
                op_input_spec={"*": default_activation_quantization_spec()},
            ),
            module_name_configs={module_name: None},
        ).set_execution_mode(execution_mode)

    def test_fake_quant_inserted_for_each_invocation(self, execution_mode, example_input):
        """
        Each call site of the shared module must get its own fake-quant nodes.
        Excluding the shared module by name must reduce the count compared to
        the fully-quantized baseline.
        """
        full = Quantizer(self._make_model(), _make_w8a8_config(execution_mode)).prepare(
            (example_input,)
        )
        excl = Quantizer(
            self._make_model(), self._config_excluding_module(execution_mode, "fc")
        ).prepare((example_input,))
        assert _count_fake_quant_modules(excl) < _count_fake_quant_modules(full)

    def test_module_name_config_any_alias_excludes_all_invocations(
        self, execution_mode, example_input
    ):
        """
        Excluding by canonical name ('fc') or alias name ('fc_alias') must
        both exclude ALL invocations and produce the same fake-quant count —
        module_name_config applies by module object identity.
        """
        full = Quantizer(self._make_model(), _make_w8a8_config(execution_mode)).prepare(
            (example_input,)
        )
        excl_canonical = Quantizer(
            self._make_model(), self._config_excluding_module(execution_mode, "fc")
        ).prepare((example_input,))
        excl_alias = Quantizer(
            self._make_model(), self._config_excluding_module(execution_mode, "fc_alias")
        ).prepare((example_input,))
        assert _count_fake_quant_modules(full) > _count_fake_quant_modules(excl_canonical)
        assert _count_fake_quant_modules(excl_canonical) == _count_fake_quant_modules(excl_alias)

    def test_op_name_config_allows_per_invocation_config(self, example_input):
        """
        op_name_config targets individual graph nodes by name, enabling
        per-invocation configuration of a shared module (graph mode only).

        We prepare once to discover node names, then build a config that
        excludes just one of the two fc invocations and verify the fake-quant
        count falls strictly between the full and fully-excluded counts.
        """

        # Discover compressible op names from the prepared graph
        discovery_prepared = Quantizer(self._make_model(), _make_w8a8_config("graph")).prepare(
            (example_input,)
        )
        compressible = GraphQuantizer.get_compressible_op_names(discovery_prepared)
        assert len(compressible) >= 2, f"Expected >=2 compressible ops, got: {compressible}"

        # Exclude just the first op via op_name_config
        first_op = sorted(compressible)[0]
        op_excl_config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec={"weight": default_weight_quantization_spec()},
                op_input_spec={"*": default_activation_quantization_spec()},
                op_name_config={first_op: None},
            )
        ).set_execution_mode("graph")

        full = Quantizer(self._make_model(), _make_w8a8_config("graph")).prepare((example_input,))
        excl_one = Quantizer(self._make_model(), op_excl_config).prepare((example_input,))
        excl_all = Quantizer(
            self._make_model(), self._config_excluding_module("graph", "fc")
        ).prepare((example_input,))

        # Excluding one invocation: count strictly between full and fully-excluded
        assert _count_fake_quant_modules(excl_all) < _count_fake_quant_modules(excl_one)
        assert _count_fake_quant_modules(excl_one) < _count_fake_quant_modules(full)

    def test_later_module_name_config_wins_all_invocations(self, execution_mode, example_input):
        """
        When module_name_configs contains multiple entries resolving to the same
        shared module, the later entry wins and applies to ALL call sites of that
        module (both invocations of self.fc and self.fc_alias).

        Verified by comparing against single-entry equivalent configs:
          - {"fc": None, "fc_alias": W8A8} → later W8A8 wins → same as {"fc_alias": W8A8}
          - {"fc_alias": W8A8, "fc": None} → later None wins → same as {"fc": None}
        """
        global_mod_config = ModuleQuantizerConfig(
            op_state_spec={"weight": default_weight_quantization_spec()},
            op_input_spec={"*": default_activation_quantization_spec()},
            op_output_spec=None,
        )

        later_incl = Quantizer(
            self._make_model(),
            _make_module_name_config(execution_mode, {"fc": None, "fc_alias": global_mod_config}),
        ).prepare((example_input,))
        only_incl = Quantizer(
            self._make_model(),
            _make_module_name_config(execution_mode, {"fc_alias": global_mod_config}),
        ).prepare((example_input,))

        later_excl = Quantizer(
            self._make_model(),
            _make_module_name_config(execution_mode, {"fc_alias": global_mod_config, "fc": None}),
        ).prepare((example_input,))
        only_excl = Quantizer(
            self._make_model(), _make_module_name_config(execution_mode, {"fc": None})
        ).prepare((example_input,))

        assert _count_fake_quant_modules(later_incl) == _count_fake_quant_modules(only_incl)
        assert _count_fake_quant_modules(later_excl) == _count_fake_quant_modules(only_excl)
        assert _count_fake_quant_modules(later_excl) < _count_fake_quant_modules(later_incl)


class TestDynamicActivationQuantization:
    """Lifecycle and finalize-rejection tests for dynamic activation quantization.

    Uses a 2-Linear model (l1=dynamic, l2=moving-average) and walks setup →
    calibration → fake-quant inference. Verifies dynamic qparams change per
    inference while moving-average qparams stay frozen post-calibration.
    """

    @staticmethod
    def _get_activation_fq(prepared_model, execution_mode, layer_prefix, calculator_cls):
        """Find the unique input-activation FakeQuantize for a layer.

        Eager identifies by submodule name (``<layer>_quantize_input``). Graph
        identifies by ``qparams_calculator`` class — each layer in this test
        has a distinct activation spec, so the calculator class is unique.
        """
        if execution_mode == "eager":
            for name, mod in prepared_model.named_modules():
                if name.startswith(layer_prefix) and name.endswith("quantize_input"):
                    return mod
            raise AssertionError(f"No input activation FQ for prefix {layer_prefix!r}")

        matches = [
            m
            for m in prepared_model.modules()
            if isinstance(m, FakeQuantizeImplBase)
            and isinstance(m.qparams_calculator, calculator_cls)
        ]
        assert len(matches) == 1, (
            f"Expected exactly 1 FakeQuantize with {calculator_cls.__name__}, got {len(matches)}"
        )
        return matches[0]

    def _make_mixed_dynamic_static_config(
        self, execution_mode: str, qat_schedule: QATSchedule | None = None
    ) -> QuantizerConfig:
        """l1: dynamic activation; l2: moving-average activation; both: static weight."""
        weight_spec = default_weight_quantization_spec()
        dynamic_act_spec = QuantizationSpec(
            dtype=torch.int8,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerTensorGranularity(),
            qparam_calculator_cls="dynamic",
        )
        moving_avg_act_spec = default_activation_quantization_spec()
        return QuantizerConfig(
            global_config=None,
            module_name_configs={
                "l1": ModuleQuantizerConfig(
                    op_state_spec={"weight": weight_spec},
                    op_input_spec={"*": dynamic_act_spec},
                    op_output_spec=None,
                    qat_schedule=qat_schedule,
                ),
                "l2": ModuleQuantizerConfig(
                    op_state_spec={"weight": weight_spec},
                    op_input_spec={"*": moving_avg_act_spec},
                    op_output_spec=None,
                    qat_schedule=qat_schedule,
                ),
            },
        ).set_execution_mode(execution_mode)

    def test_dynamic_qparams_lifecycle(self, execution_mode):
        # 1. Initialize model + mixed-spec config and prepare.
        config = self._make_mixed_dynamic_static_config(execution_mode)
        quantizer = Quantizer(SimpleLinearModel(), config)
        prepared_model = quantizer.prepare((torch.randn(4, 64),))

        # 2. Verify calculator types are wired correctly per layer.
        dynamic_fq = self._get_activation_fq(
            prepared_model, execution_mode, "l1", DynamicQParamsCalculator
        )
        moving_avg_fq = self._get_activation_fq(
            prepared_model, execution_mode, "l2", MovingAverageQParamsCalculator
        )
        assert isinstance(dynamic_fq.qparams_calculator, DynamicQParamsCalculator)
        assert isinstance(moving_avg_fq.qparams_calculator, MovingAverageQParamsCalculator)

        # 3. Calibration: feed several batches inside calibration_mode().
        torch.manual_seed(0)
        with quantizer.calibration_mode():
            for _ in range(5):
                prepared_model(torch.randn(4, 64))

        dyn_scale_before_forward = dynamic_fq.qparams_calculator.scale.clone()
        moving_avg_scale_before_forward = moving_avg_fq.qparams_calculator.scale.clone()

        # 4. Fake-quant mode (default after calibration_mode exits): run an input
        #    with deterministically larger magnitude than calibration so dynamic's
        #    recomputed scale provably differs from its calibrated value.
        prepared_model(torch.randn(4, 64) * 10.0)
        dyn_scale_after_forward = dynamic_fq.qparams_calculator.scale.clone()
        moving_avg_scale_after_forward = moving_avg_fq.qparams_calculator.scale.clone()

        # Dynamic recomputes scale per inference; moving-average is frozen post-calibration.
        assert not torch.equal(dyn_scale_before_forward, dyn_scale_after_forward)
        assert torch.equal(moving_avg_scale_before_forward, moving_avg_scale_after_forward)

    @pytest.mark.parametrize(
        "backend,is_supported",
        [
            (ExportBackend.CoreAI, False),
            (ExportBackend.CoreML, False),
            (ExportBackend._TORCH, True),
        ],
    )
    def test_finalize_rejects_dynamic_for_non_torch_backends(
        self, execution_mode, backend, is_supported
    ):
        """``finalize`` must reject CoreAI/CoreML for dynamic FakeQuantize
        modules. ``_TORCH`` is allowed since it returns the prepared model as-is."""
        config = self._make_mixed_dynamic_static_config(execution_mode)
        quantizer = Quantizer(SimpleLinearModel(), config)
        prepared_model = quantizer.prepare((torch.randn(4, 64),))

        if is_supported:
            finalized = quantizer.finalize(prepared_model, backend=backend)
            assert finalized is not None
        else:
            with pytest.raises(NotImplementedError, match="dynamic quantization"):
                quantizer.finalize(prepared_model, backend=backend)

    def test_qat_schedule_does_not_disable_dynamic_observer(self, execution_mode):
        """QAT schedule's ``disable_observer`` transition must skip dynamic FQs."""
        config = self._make_mixed_dynamic_static_config(
            execution_mode,
            qat_schedule=QATSchedule(enable_observer=0, enable_fake_quant=1, disable_observer=2),
        )
        quantizer = Quantizer(SimpleLinearModel(), config)
        prepared_model = quantizer.prepare((torch.randn(4, 64),))

        dynamic_fq = self._get_activation_fq(
            prepared_model, execution_mode, "l1", DynamicQParamsCalculator
        )
        moving_avg_fq = self._get_activation_fq(
            prepared_model, execution_mode, "l2", MovingAverageQParamsCalculator
        )

        # Step past disable_observer=2 inside training_mode (which is what
        # actually invokes the schedule via _maybe_apply_qat_schedule).
        with quantizer.training_mode():
            for _ in range(5):
                quantizer.step()

        assert moving_avg_fq.observer_enabled.item() == 0
        assert dynamic_fq.observer_enabled.item() == 1


class TestSharedWeightQuantization:
    class _LeafA(nn.Module):
        def __init__(self):
            super().__init__()
            self.my_weight = nn.Parameter(torch.randn(2, 2))

        def forward(self, x):
            return torch.nn.functional.linear(x, self.my_weight)

    class _LeafB(nn.Module):
        def __init__(self):
            super().__init__()
            self.other_weight = nn.Parameter(torch.randn(2, 2))

        def forward(self, x):
            return torch.nn.functional.linear(x, self.other_weight)

    class _SharedWeightModel(nn.Module):
        """Two leaves whose state tensors alias the same parameter.

        ``linear2.other_weight is linear1.my_weight`` after construction.
        """

        def __init__(self):
            super().__init__()
            self.linear1 = TestSharedWeightQuantization._LeafA()
            self.linear2 = TestSharedWeightQuantization._LeafB()
            self.linear2.other_weight = self.linear1.my_weight

        def forward(self, x):
            return self.linear2(self.linear1(x))

    # In the below model, the add op consumes a state tensor which is referenced by multiple local
    # names: "my_weight" through leaf_a and "other_weight" through leaf_b.
    class _AddModelSharedStateInput(torch.nn.Module):
        """Model with add op consuming a state tensor which is referenced by multiple local
        names: "my_weight" through leaf_a and "other_weight" through leaf_b.
        """

        def __init__(self):
            super().__init__()
            self.leaf_a = TestSharedWeightQuantization._LeafA()
            self.leaf_b = TestSharedWeightQuantization._LeafB()
            self.leaf_b.other_weight = self.leaf_a.my_weight

        def forward(self, inp):
            x = self.leaf_a.my_weight + inp
            return x

    @staticmethod
    def _w4_spec() -> QuantizationSpec:
        return QuantizationSpec(dtype=torch.int4)

    @pytest.mark.parametrize(
        "config, expected_dtype",
        [
            (
                QuantizerConfig(
                    global_config=None,
                    module_type_configs={
                        _LeafA: ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec={"my_weight": default_weight_quantization_spec()},
                        ),
                        _LeafB: ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec={"other_weight": _w4_spec()},
                        ),
                    },
                ),
                torch.int4,
            ),
            (
                QuantizerConfig(
                    global_config=None,
                    module_type_configs={
                        _LeafB: ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec={"other_weight": _w4_spec()},
                        ),
                        _LeafA: ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec={"my_weight": default_weight_quantization_spec()},
                        ),
                    },
                ),
                default_weight_quantization_spec().dtype,
            ),
            (
                QuantizerConfig(
                    global_config=None,
                    module_name_configs={
                        "linear1": ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec={"my_weight": default_weight_quantization_spec()},
                        ),
                    },
                    module_type_configs={
                        _LeafB: ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec={"other_weight": _w4_spec()},
                        ),
                    },
                ),
                default_weight_quantization_spec().dtype,
            ),
            (
                QuantizerConfig(
                    global_config=None,
                    module_name_configs={
                        "linear2": ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec={"other_weight": _w4_spec()},
                        ),
                    },
                    module_type_configs={
                        _LeafA: ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec={"my_weight": default_weight_quantization_spec()},
                        ),
                    },
                ),
                torch.int4,
            ),
            (
                QuantizerConfig(
                    global_config=None,
                    module_name_configs={
                        "linear1": ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec={"my_weight": default_weight_quantization_spec()},
                        ),
                    },
                    module_type_configs={
                        _LeafB: ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec=None,
                            module_state_spec={"other_weight": _w4_spec()},
                        ),
                    },
                ),
                torch.int4,
            ),
            (
                QuantizerConfig(
                    global_config=None,
                    module_type_configs={
                        _LeafA: ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec=None,
                            module_state_spec={"my_weight": default_weight_quantization_spec()},
                        ),
                        _LeafB: ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec=None,
                            module_state_spec={"other_weight": _w4_spec()},
                        ),
                    },
                ),
                torch.int4,
            ),
            (
                QuantizerConfig(
                    global_config=None,
                    module_type_configs={
                        _LeafB: ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec=None,
                            module_state_spec={"other_weight": _w4_spec()},
                        ),
                        _LeafA: ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec=None,
                            module_state_spec={"my_weight": default_weight_quantization_spec()},
                        ),
                    },
                ),
                default_weight_quantization_spec().dtype,
            ),
            (
                QuantizerConfig(
                    global_config=None,
                    module_name_configs={
                        "linear1": ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec=None,
                            module_state_spec={"my_weight": default_weight_quantization_spec()},
                        ),
                    },
                    module_type_configs={
                        _LeafB: ModuleQuantizerConfig(
                            op_input_spec=None,
                            op_output_spec=None,
                            op_state_spec=None,
                            module_state_spec={"other_weight": _w4_spec()},
                        ),
                    },
                ),
                default_weight_quantization_spec().dtype,
            ),
        ],
    )
    def test_shared_weight_quantization(self, config, expected_dtype, execution_mode):
        """
        Test shared weight quantization for different variations and orderings of configurations
        applied to the same shared weight.
        """
        model = self._SharedWeightModel()
        inp = (torch.randn(1, 2),)

        config = config.set_execution_mode(execution_mode)
        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(inp)

        if execution_mode == "graph":
            node_dict = {node.name: node for node in prepared_model.graph.nodes}
            leaf_a_linear_weight = getattr(
                prepared_model, node_dict["linear"].all_input_nodes[1].name
            )
            assert leaf_a_linear_weight.qparams_calculator.dtype == expected_dtype
            leaf_b_linear_weight = getattr(
                prepared_model, node_dict["linear_1"].all_input_nodes[1].name
            )
            assert leaf_a_linear_weight is leaf_b_linear_weight
        else:
            assert (
                prepared_model.linear1.parametrizations["my_weight"][0].qparams_calculator.dtype
                == expected_dtype
            )
            assert (
                prepared_model.linear1.parametrizations["my_weight"][0]
                is prepared_model.linear2.parametrizations["other_weight"][0]
            )

    @pytest.mark.parametrize(
        "config",
        [
            QuantizerConfig(
                global_config=None,
                module_name_configs={
                    "": ModuleQuantizerConfig(
                        op_input_spec=None,
                        op_output_spec=None,
                        op_state_spec={"my_weight": default_weight_quantization_spec()},
                    ),
                },
            ),
            QuantizerConfig(
                global_config=None,
                module_name_configs={
                    "": ModuleQuantizerConfig(
                        op_input_spec=None,
                        op_output_spec=None,
                        op_state_spec={"other_weight": default_weight_quantization_spec()},
                    ),
                },
            ),
        ],
    )
    @pytest.mark.parametrize("execution_mode", ["graph", "eager"])
    def test_independent_state_tensor_usage(self, config, execution_mode):
        """
        Test that modules using a state tensor can configure the tensor using a local name which is
        used by the module owning the tensor.
        """

        model = self._AddModelSharedStateInput()
        inp = (torch.randn(1, 2),)

        config = config.set_execution_mode(execution_mode)
        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(inp)

        quantizers = [
            module
            for module in prepared_model.modules()
            if isinstance(module, FakeQuantizeImplBase)
        ]
        assert len(quantizers) == 1

        if execution_mode == "graph":
            node_dict = {node.name: node for node in prepared_model.graph.nodes}
            assert "activation_post_process" in node_dict["add"].all_input_nodes[0].name
        else:
            assert (
                prepared_model.leaf_a.parametrizations["my_weight"][0]
                is prepared_model.leaf_b.parametrizations["other_weight"][0]
            )

    def test_op_state_spec_last_key_wins_for_aliased_state(self, execution_mode):
        """Last key in op_state_spec wins when a state tensor matches multiple alias keys."""
        model = self._SharedWeightModel()
        inp = (torch.randn(1, 2),)

        config = QuantizerConfig(
            global_config=None,
            module_name_configs={
                "": ModuleQuantizerConfig(
                    op_input_spec=None,
                    op_output_spec=None,
                    op_state_spec={
                        "my_weight": default_weight_quantization_spec(),
                        "other_weight": self._w4_spec(),
                    },
                ),
            },
        ).set_execution_mode(execution_mode)

        quantizer = Quantizer(model, config)
        prepared_model = quantizer.prepare(inp)

        if execution_mode == "graph":
            node_dict = {node.name: node for node in prepared_model.graph.nodes}
            weight = getattr(prepared_model, node_dict["linear"].all_input_nodes[1].name)
            assert weight.qparams_calculator.dtype == torch.int4
        else:
            assert (
                prepared_model.linear1.parametrizations["my_weight"][0].qparams_calculator.dtype
                == torch.int4
            )
