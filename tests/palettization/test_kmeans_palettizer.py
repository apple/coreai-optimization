# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import copy
import logging

import pytest
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
from torch.nn.utils.parametrize import is_parametrized

from coreai_opt import ExportBackend
from coreai_opt.palettization import (
    KMeansPalettizer,
    KMeansPalettizerConfig,
    ModuleKMeansPalettizerConfig,
)
from coreai_opt.palettization.config import OpKMeansPalettizerConfig
from coreai_opt.palettization.kmeans.kmeans_fake_palettize import _KMeansFakePalettize
from coreai_opt.palettization.spec import (
    PalettizationSpec,
    PerGroupedChannelGranularity,
    PerTensorGranularity,
    default_weight_palettization_spec,
)
from coreai_opt.palettization.spec.errors import _IncompatibleGranularityError
from coreai_opt.palettization.spec.fake_palettize import (
    _disable_observer,
    _enable_observer,
    _FakePalettizeImplBase,
)
from coreai_opt.quantization.spec import QuantizationScheme, QuantizationSpec


@pytest.fixture
def basic_config():
    return KMeansPalettizerConfig(
        global_config=ModuleKMeansPalettizerConfig(
            op_state_spec={
                "weight": default_weight_palettization_spec(),
            }
        )
    )


class TestKMeansPalettizer:
    """Test cases for KMeansPalettizer class."""

    def test_init_with_config(self, simple_conv_linear_model, basic_config):
        """
        Test the KMeansPalettizer with a simple model and the basic config
        and check that the model and config are getting propagated. Also check
        if model is prepared.
        """
        palettizer = KMeansPalettizer(simple_conv_linear_model, basic_config)
        assert palettizer._model is simple_conv_linear_model
        assert palettizer._config is basic_config
        assert not palettizer._is_model_prepared(palettizer._model)

    def test_init_without_config(self, simple_conv_linear_model):
        """
        Test KMeansPalettizer with no config passed in (uses default config)
        """
        palettizer = KMeansPalettizer(simple_conv_linear_model)
        assert palettizer._model is simple_conv_linear_model
        assert isinstance(palettizer._config, KMeansPalettizerConfig)
        assert palettizer._config == KMeansPalettizerConfig()
        assert not palettizer._is_model_prepared(palettizer._model)

    def test_prepare_basic(self, simple_conv_linear_model, basic_config, simple_model_input):
        """
        Check that the model is getting prepared properly and we can pass an example
        input through the model and get the correct shape output.

        Also check that the conv and linear layers have weight palettizers
        (through parametrization).
        """
        palettizer = KMeansPalettizer(simple_conv_linear_model, basic_config)

        prepared_model = palettizer.prepare((simple_model_input,))

        assert palettizer._is_model_prepared(prepared_model)

        assert prepared_model is simple_conv_linear_model
        assert is_parametrized(prepared_model.conv, "weight")
        assert is_parametrized(prepared_model.linear, "weight")

        # Check weight palettizers are inserted
        assert isinstance(
            prepared_model.conv.parametrizations["weight"][0],
            _FakePalettizeImplBase,
        )

        assert isinstance(
            prepared_model.linear.parametrizations["weight"][0],
            _FakePalettizeImplBase,
        )

        output = prepared_model(simple_model_input)
        assert output.shape == (1, 10)

    def test_multiple_prepare_calls(
        self, simple_conv_linear_model, basic_config, simple_model_input
    ):
        """
        If we re-prepare the model, it should throw a warning
        """
        palettizer = KMeansPalettizer(simple_conv_linear_model, basic_config)

        palettizer.prepare((simple_model_input,))

        with pytest.raises(RuntimeError):
            palettizer.prepare((simple_model_input,))

    def test_finalize_before_prepare(self, simple_conv_linear_model, basic_config):
        """
        If we call finalize before prepare, we should raise an error
        """
        palettizer = KMeansPalettizer(simple_conv_linear_model, basic_config)

        with pytest.raises(RuntimeError):
            palettizer.finalize()

    def test_module_name_configs(self, simple_conv_linear_model, simple_model_input):
        """
        Test that we can configure the KMeans palettizer with module names
        """
        config = KMeansPalettizerConfig(
            global_config=None,
            module_name_configs={
                "conv": ModuleKMeansPalettizerConfig(
                    op_state_spec={
                        "weight": default_weight_palettization_spec(),
                    }
                )
            },
        )

        palettizer = KMeansPalettizer(simple_conv_linear_model, config)
        prepared_model = palettizer.prepare((simple_model_input,))
        assert not is_parametrized(prepared_model.linear, "weight")
        assert is_parametrized(prepared_model.conv, "weight")

        fp_mods = [
            i for i in prepared_model.named_modules() if isinstance(i[1], _FakePalettizeImplBase)
        ]
        assert len(fp_mods) == 1

    def test_module_type_configs(self, simple_conv_linear_model, simple_model_input):
        """
        Test that we can configure the KMeans palettizer with module types
        """
        config = KMeansPalettizerConfig(
            global_config=None,
            module_type_configs={
                torch.nn.Linear: ModuleKMeansPalettizerConfig(
                    op_state_spec={
                        "weight": default_weight_palettization_spec(),
                    }
                )
            },
        )

        palettizer = KMeansPalettizer(simple_conv_linear_model, config)
        prepared_model = palettizer.prepare((simple_model_input,))
        assert is_parametrized(prepared_model.linear, "weight")
        assert not is_parametrized(prepared_model.conv, "weight")

        fp_mods = [
            i for i in prepared_model.named_modules() if isinstance(i[1], _FakePalettizeImplBase)
        ]
        assert len(fp_mods) == 1

    def test_wildcard_with_bias_excluded(self, simple_conv_linear_model, simple_model_input):
        """Test that wildcard '*' palettizes all states but 'bias': None excludes bias."""
        config = KMeansPalettizerConfig(
            global_config=ModuleKMeansPalettizerConfig(
                op_state_spec={
                    "*": default_weight_palettization_spec(),
                    "bias": None,
                }
            )
        )

        palettizer = KMeansPalettizer(simple_conv_linear_model, config)
        prepared_model = palettizer.prepare((simple_model_input,))

        assert is_parametrized(prepared_model.conv, "weight")
        assert is_parametrized(prepared_model.linear, "weight")
        assert not is_parametrized(prepared_model.conv, "bias")
        assert not is_parametrized(prepared_model.linear, "bias")

        output = prepared_model(simple_model_input)
        assert output.shape == (1, 10)

    def test_skip_layer_config(self, simple_conv_linear_model, simple_model_input):
        """Test skipping specific layers from palettization."""
        config = KMeansPalettizerConfig(
            # default global config
            module_name_configs={
                "linear": None,  # Skip linear layer
            },
        )

        original_linear_weight = simple_conv_linear_model.linear.weight.clone()

        palettizer = KMeansPalettizer(simple_conv_linear_model, config)
        prepared_model = palettizer.prepare((simple_model_input,))

        # Conv should be parametrized, linear should not
        assert is_parametrized(prepared_model.conv, "weight")
        assert not is_parametrized(prepared_model.linear, "weight")

        # Linear weight should be unchanged
        assert torch.equal(original_linear_weight, prepared_model.linear.weight)

    def test_weight_palettization(self, simple_conv_linear_model, basic_config, simple_model_input):
        """
        Test that weight palettization is taking place
        """
        simple_conv_linear_model.eval()
        conv_weight = simple_conv_linear_model.conv.weight.clone()
        linear_weight = simple_conv_linear_model.linear.weight.clone()

        palettizer = KMeansPalettizer(simple_conv_linear_model, basic_config)
        prepared_model = palettizer.prepare((simple_model_input,))

        # Check that we have some palettization (limited unique values)
        conv_unique_before = len(torch.unique(conv_weight))
        conv_unique_after = len(torch.unique(prepared_model.conv.weight))
        linear_unique_before = len(torch.unique(linear_weight))
        linear_unique_after = len(torch.unique(prepared_model.linear.weight))

        # After palettization, should have fewer unique values
        assert conv_unique_after <= 16  # n_bits=4, so max 16 values
        assert linear_unique_after <= 16
        assert conv_unique_after < conv_unique_before
        assert linear_unique_after < linear_unique_before

        # Check that we have some reconstruction error due to palettization
        assert (conv_weight - prepared_model.conv.weight).abs().sum() > 1e-4
        assert (linear_weight - prepared_model.linear.weight).abs().sum() > 1e-4

    def test_per_grouped_channel_axis_default_resolved_per_op(
        self, simple_conv_linear_model, simple_model_input
    ):
        """PerGroupedChannelGranularity with axis omitted resolves to each op's mixin default."""
        spec = PalettizationSpec(
            n_bits=2,
            granularity=PerGroupedChannelGranularity(group_size=2),
            cluster_dim=1,
        )
        config = KMeansPalettizerConfig(
            global_config=ModuleKMeansPalettizerConfig(op_state_spec={"weight": spec}),
        )

        palettizer = KMeansPalettizer(simple_conv_linear_model, config)
        prepared_model = palettizer.prepare((simple_model_input,))

        for module in (prepared_model.conv, prepared_model.linear):
            fake_palettize = module.parametrizations["weight"][0]
            assert isinstance(fake_palettize.granularity, PerGroupedChannelGranularity)
            assert fake_palettize.granularity.axis == fake_palettize.reshape_strategy.default_axis

    def test_disabled_fake_palett_removed_after_prepare(self, caplog):
        """Disabled FakePalettize modules are removed during prepare, compatible ones kept."""
        # out_features=1000: 1000 % 32 != 0 → incompatible, disabled and removed
        # out_features=1024: 1024 % 32 == 0 → compatible, stays
        model = nn.Sequential(
            nn.Linear(768, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1024),
        )
        example_inputs = (torch.randn(1, 768),)

        spec = PalettizationSpec(
            n_bits=4,
            granularity=PerGroupedChannelGranularity(axis=0, group_size=32),
        )
        config = KMeansPalettizerConfig(
            global_config=ModuleKMeansPalettizerConfig(
                op_state_spec={"weight": spec},
            )
        )

        palettizer = KMeansPalettizer(model, config)
        with caplog.at_level(logging.WARNING):
            prepared_model = palettizer.prepare(example_inputs)

        assert any("Skipping palettization" in msg for msg in caplog.messages)

        # Incompatible layer should have its parametrization removed
        assert not is_parametrized(prepared_model[0], "weight")

        # Compatible layer should retain its FakePalettize parametrization
        assert is_parametrized(prepared_model[2], "weight")
        assert isinstance(prepared_model[2].parametrizations.weight[0], _FakePalettizeImplBase)
        assert not prepared_model[2].parametrizations.weight[0].is_disabled()

    def test_prepared_model_supports_torch_inference(
        self, simple_conv_linear_model, simple_model_input
    ):
        """Verify torch-based evaluation uses the prepared model directly."""
        palettizer = KMeansPalettizer(simple_conv_linear_model)
        prepared_model = palettizer.prepare(example_inputs=(simple_model_input,))

        # Prepared model retains palettization parametrizations for torch-based evaluation
        assert is_parametrized(prepared_model.conv, "weight")
        assert is_parametrized(prepared_model.linear, "weight")

        # Prepared model supports torch-based inference directly
        output = prepared_model(simple_model_input)
        assert output.shape == (1, 10)

    def test_finalize_torch_backend(self, simple_conv_linear_model, simple_model_input):
        palettizer = KMeansPalettizer(simple_conv_linear_model)

        prepared_model = palettizer.prepare(example_inputs=(simple_model_input,))

        prepared_output = prepared_model(simple_model_input)

        finalized_model = palettizer.finalize(backend=ExportBackend._TORCH)

        assert finalized_model is not None

        finalized_output = finalized_model(simple_model_input)

        assert torch.equal(prepared_output, finalized_output)

    def test_finalize_mil_backend(self, simple_conv_linear_model, simple_model_input):
        palettizer = KMeansPalettizer(simple_conv_linear_model)

        palettizer.prepare(example_inputs=(simple_model_input,))

        finalized_model = palettizer.finalize(backend=ExportBackend.CoreML)

        assert finalized_model is not None

    def test_finalize_mlir_backend(self, simple_conv_linear_model, simple_model_input):
        palettizer = KMeansPalettizer(simple_conv_linear_model)

        palettizer.prepare(example_inputs=(simple_model_input,))

        finalized_model = palettizer.finalize(backend=ExportBackend.CoreAI)

        assert finalized_model is not None

    @staticmethod
    def _palettize_model(model, palettization_config, example_input, mmap_path, backend):
        palettizer = KMeansPalettizer(model, palettization_config)
        palettizer.prepare(example_inputs=(example_input,))

        finalized_model = palettizer.finalize(
            backend=backend,
            mmap_dir=mmap_path,
        )
        return finalized_model

    @pytest.mark.parametrize(
        "backend",
        [ExportBackend._TORCH, ExportBackend.CoreML],
    )
    def test_finalize_mmap_value_error_unsupported_backend(
        self, simple_conv_linear_model, simple_model_input, basic_config, tmp_path, backend
    ):
        """``mmap_dir`` is only supported with the CoreAI backend; finalize
        raises ``ValueError`` for other backends."""
        with pytest.raises(
            ValueError,
            match="mmap_dir is only supported with backend=ExportBackend.CoreAI",
        ):
            self._palettize_model(
                simple_conv_linear_model,
                basic_config,
                simple_model_input,
                str(tmp_path),
                backend,
            )

    @pytest.mark.parametrize(
        "backend",
        [ExportBackend._TORCH, ExportBackend.CoreAI, ExportBackend.CoreML],
    )
    def test_finalize_weight_parametrization_state_per_backend(
        self, backend, simple_conv_linear_model, simple_model_input, basic_config
    ):
        """Post-finalize parametrization state is backend-specific:

        - _TORCH: parametrization preserved with dense ``.original`` intact
        - CoreAI: parametrization replaced with palettize module; ``.original``
          cleared to a zero-size placeholder
        - CoreML: parametrization removed entirely
        """
        finalized = self._palettize_model(
            simple_conv_linear_model, basic_config, simple_model_input, None, backend
        )

        weight_modules = [m for m in finalized.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
        assert weight_modules, "fixture regressed: no Conv2d/Linear to check"

        if backend is ExportBackend._TORCH:
            for m in weight_modules:
                assert P.is_parametrized(m, "weight")
                assert m.parametrizations["weight"].original.numel() > 0
        elif backend is ExportBackend.CoreAI:
            for m in weight_modules:
                assert P.is_parametrized(m, "weight")
                assert m.parametrizations["weight"].original.numel() == 0
        elif backend is ExportBackend.CoreML:
            for m in weight_modules:
                assert not P.is_parametrized(m, "weight")
                assert m.weight.numel() > 0

    @pytest.mark.parametrize("use_mmap", [False, True])
    def test_finalize_mmap_files_exist(
        self, use_mmap, simple_conv_linear_model, simple_model_input, basic_config, tmp_path
    ):
        """Finalize with CoreAI backend writes one safetensors file per
        palettized module when ``mmap_dir`` is set, and nothing otherwise
        When written, each file must be a valid non-empty safetensors blob."""
        from safetensors.torch import load_file  # noqa: PLC0415

        mmap_dir = str(tmp_path) if use_mmap else None
        self._palettize_model(
            simple_conv_linear_model,
            basic_config,
            simple_model_input,
            mmap_dir,
            ExportBackend.CoreAI,
        )

        files = sorted(p.name for p in tmp_path.iterdir())
        if use_mmap:
            assert files == ["conv.weight.safetensors", "linear.weight.safetensors"]
            for fname in files:
                loaded = load_file(str(tmp_path / fname), device="cpu")
                assert loaded, f"{fname} is empty"
                assert all(t.numel() > 0 for t in loaded.values()), (
                    f"{fname} contains zero-size tensors"
                )
                assert "lut" in loaded, f"{fname} missing 'lut' key"
                assert "indices" in loaded, f"{fname} missing 'indices' key"
        else:
            assert files == []

    def test_finalize_mmap_matches_non_mmap_output(
        self, simple_conv_linear_model, simple_model_input, basic_config, tmp_path
    ):
        """Finalize with and without ``mmap_dir`` produces numerically identical
        outputs."""
        model_no_mmap = copy.deepcopy(simple_conv_linear_model)
        model_with_mmap = copy.deepcopy(simple_conv_linear_model)

        finalized_no_mmap = self._palettize_model(
            model_no_mmap, basic_config, simple_model_input, None, ExportBackend.CoreAI
        )
        finalized_with_mmap = self._palettize_model(
            model_with_mmap,
            basic_config,
            simple_model_input,
            str(tmp_path),
            ExportBackend.CoreAI,
        )

        with torch.no_grad():
            out_no_mmap = finalized_no_mmap(simple_model_input)
            out_with_mmap = finalized_with_mmap(simple_model_input)

        assert torch.equal(out_no_mmap, out_with_mmap)

    def test_finalize_state_dict_safetensors_roundtrip(
        self, simple_conv_linear_model, simple_model_input, basic_config, tmp_path
    ):
        """An mmap-finalized model survives a state_dict save → load_file →
        load_state_dict round-trip with identical forward outputs. The reloaded
        tensors come back mmap-backed by the bundled safetensors file (per
        safetensors' default behavior)."""
        from safetensors.torch import load_file, save_file  # noqa: PLC0415

        finalized = self._palettize_model(
            simple_conv_linear_model,
            basic_config,
            simple_model_input,
            str(tmp_path / "per_layer_mmap"),
            ExportBackend.CoreAI,
        )

        with torch.no_grad():
            out_before_roundtrip = finalized(simple_model_input)

        # Bundle the full state_dict into a single safetensors file.
        bundled = tmp_path / "full_state.safetensors"
        save_file(
            {
                k: v.contiguous()
                for k, v in finalized.state_dict().items()
                if isinstance(v, torch.Tensor)
            },
            str(bundled),
        )

        # Reload via mmap and reassign onto the existing model.
        reloaded_sd = load_file(str(bundled), device="cpu")
        finalized.load_state_dict(reloaded_sd, assign=True)

        with torch.no_grad():
            out_after_roundtrip = finalized(simple_model_input)

        assert torch.equal(out_before_roundtrip, out_after_roundtrip)


class TestKMeansPalettizerOpAndModuleConfigs:
    """Tests for op_type_config, op_name_config, and module_state_spec support."""

    @staticmethod
    def _get_fake_palett(module, attr="weight"):
        return module.parametrizations[attr][0]

    def test_op_type_config_targets_only_matching_op(
        self, simple_conv_linear_model, simple_model_input
    ):
        """op_type_config keyed by registry func_type palettizes only matching ops."""
        config = KMeansPalettizerConfig(
            global_config=ModuleKMeansPalettizerConfig(
                op_state_spec=None,
                op_type_config={
                    "linear": OpKMeansPalettizerConfig(
                        op_state_spec={"weight": PalettizationSpec(n_bits=2)},
                    ),
                },
            ),
        )

        palettizer = KMeansPalettizer(simple_conv_linear_model, config)
        prepared_model = palettizer.prepare((simple_model_input,))

        assert is_parametrized(prepared_model.linear, "weight")
        assert not is_parametrized(prepared_model.conv, "weight")
        assert self._get_fake_palett(prepared_model.linear).n_bits == 2

    def test_op_name_config_overrides_op_type_config(
        self, simple_conv_linear_model, simple_model_input
    ):
        """op_name_config takes precedence over op_type_config for matching ops."""
        config = KMeansPalettizerConfig(
            global_config=ModuleKMeansPalettizerConfig(
                op_state_spec=None,
                op_type_config={
                    "linear": OpKMeansPalettizerConfig(
                        op_state_spec={"weight": PalettizationSpec(n_bits=4)},
                    ),
                },
                op_name_config={
                    ".*linear": OpKMeansPalettizerConfig(
                        op_state_spec={"weight": PalettizationSpec(n_bits=1)},
                    ),
                },
            ),
        )

        palettizer = KMeansPalettizer(simple_conv_linear_model, config)
        prepared_model = palettizer.prepare((simple_model_input,))

        # op_name (n_bits=1) should win over op_type (n_bits=4).
        assert self._get_fake_palett(prepared_model.linear).n_bits == 1

    def test_module_state_spec_overrides_op_state_spec(
        self, simple_conv_linear_model, simple_model_input
    ):
        """module_state_spec on a module config overrides op_state_spec for that module."""
        config = KMeansPalettizerConfig(
            global_config=ModuleKMeansPalettizerConfig(
                op_state_spec={"weight": PalettizationSpec(n_bits=4)},
            ),
            module_type_configs={
                nn.Linear: ModuleKMeansPalettizerConfig(
                    op_state_spec={"weight": PalettizationSpec(n_bits=4)},
                    module_state_spec={"weight": PalettizationSpec(n_bits=2)},
                ),
            },
        )

        palettizer = KMeansPalettizer(simple_conv_linear_model, config)
        prepared_model = palettizer.prepare((simple_model_input,))

        # conv keeps op_state_spec (n_bits=4); linear's weight is overridden by
        # module_state_spec (n_bits=2).
        assert self._get_fake_palett(prepared_model.conv).n_bits == 4
        assert self._get_fake_palett(prepared_model.linear).n_bits == 2

    def test_module_state_spec_disables_palettization(
        self, simple_conv_linear_model, simple_model_input
    ):
        """module_state_spec={"weight": None} disables palettization for that state."""
        config = KMeansPalettizerConfig(
            global_config=ModuleKMeansPalettizerConfig(
                op_state_spec={"weight": default_weight_palettization_spec()},
            ),
            module_type_configs={
                nn.Linear: ModuleKMeansPalettizerConfig(
                    op_state_spec={"weight": default_weight_palettization_spec()},
                    module_state_spec={"weight": None},
                ),
            },
        )

        palettizer = KMeansPalettizer(simple_conv_linear_model, config)
        prepared_model = palettizer.prepare((simple_model_input,))

        assert is_parametrized(prepared_model.conv, "weight")
        assert not is_parametrized(prepared_model.linear, "weight")

    @pytest.mark.parametrize(
        "spec_field",
        [
            "module_state_spec",
            "op_state_spec",
        ],
    )
    def test_shared_weight_uses_priority(
        self, spec_field, shared_params_model, shared_params_model_input
    ):
        """Shared weight resolution must honor MODULE_NAME > MODULE_TYPE precedence.

        ``shared_params_model`` defines ``shared_linear``, ``layer1``, and ``layer2``
        as nn.Linear siblings whose weights all alias the same tensor. With both a
        MODULE_TYPE config (matches every nn.Linear, n_bits=4) and a MODULE_NAME
        config (matches only ``layer2``, n_bits=2), MODULE_NAME should win.

        Parametrized over which spec field expresses the override:

        - ``module_state_spec``: the priority dict picks layer2's spec over the
          others (layer2 has the higher-precedence MODULE_NAME config). Without a
          properly built priority dict, insertion order
          [shared_linear, layer1, layer2] would win and pick n_bits=4. This case
          guards that regression.
        - ``op_state_spec``: layer2's spec with n_bits=2 gets picked as the
          higher priority spec in accordance with config priority rules.
        """
        type_kwargs = {"op_state_spec": None, "module_state_spec": None}
        type_kwargs[spec_field] = {"weight": PalettizationSpec(n_bits=4)}

        name_kwargs = {"op_state_spec": None, "module_state_spec": None}
        name_kwargs[spec_field] = {"weight": PalettizationSpec(n_bits=2)}

        config = KMeansPalettizerConfig(
            global_config=None,
            module_type_configs={
                nn.Linear: ModuleKMeansPalettizerConfig(**type_kwargs),
            },
            module_name_configs={
                "layer2": ModuleKMeansPalettizerConfig(**name_kwargs),
            },
        )

        palettizer = KMeansPalettizer(shared_params_model, config)
        prepared_model = palettizer.prepare((shared_params_model_input,))

        # All three modules share the same parametrization (same underlying weight).
        shared_palett = self._get_fake_palett(prepared_model.shared_linear)
        assert shared_palett is self._get_fake_palett(prepared_model.layer1)
        assert shared_palett is self._get_fake_palett(prepared_model.layer2)
        # MODULE_NAME ("layer2") wins over MODULE_TYPE (nn.Linear).
        assert shared_palett.n_bits == 2


class TestKMeansPalettizerCalibrationMode:
    """Test cases for KMeansPalettizer.calibration_mode context manager."""

    def test_calibration_mode_before_prepare_raises_error(
        self, simple_conv_linear_model, basic_config
    ):
        """Test that calling calibration_mode before prepare raises RuntimeError."""
        palettizer = KMeansPalettizer(simple_conv_linear_model, basic_config)

        with pytest.raises(
            RuntimeError,
            match="Model must be prepared before entering calibration mode",
        ):
            with palettizer.calibration_mode(loss_fn=nn.functional.cross_entropy):
                pass

    def test_calibration_mode_without_step_raises_error(
        self, simple_conv_linear_model, basic_config, simple_model_input
    ):
        """Test that calibration_mode raises RuntimeError if step() is never called."""
        palettizer = KMeansPalettizer(simple_conv_linear_model, basic_config)
        prepared_model = palettizer.prepare((simple_model_input,))

        with pytest.raises(
            RuntimeError,
            match="calibration_mode requires at least one call to step\\(\\)",
        ):
            with palettizer.calibration_mode(loss_fn=nn.functional.cross_entropy):
                _output = prepared_model(simple_model_input)
                # Exit without calling step()

    def test_calibration_helper_step_computes_gradients(
        self, simple_conv_linear_model, basic_config, simple_model_input
    ):
        """Test that CalibrationHelper.step() computes gradients correctly."""
        palettizer = KMeansPalettizer(simple_conv_linear_model, basic_config)
        prepared_model = palettizer.prepare((simple_model_input,))

        # Create dummy batch
        dummy_input = simple_model_input
        dummy_target = torch.randint(0, 10, (1,))

        # Zero out all gradients first
        for param in prepared_model.parameters():
            if param.grad is not None:
                param.grad = None

        with palettizer.calibration_mode(loss_fn=nn.functional.cross_entropy) as skm:
            output = prepared_model(dummy_input)
            skm.step(output, dummy_target)

            # Verify that gradients were computed
            for param in prepared_model.parameters():
                if param.requires_grad:
                    assert param.grad is not None, (
                        "Gradients should have been computed during calibration"
                    )

    def test_calibration_mode_sets_sensitivities(
        self, simple_conv_linear_model, basic_config, simple_model_input
    ):
        """Test that calibration_mode sets sensitivities in fake palettize modules."""
        palettizer = KMeansPalettizer(simple_conv_linear_model, basic_config)
        prepared_model = palettizer.prepare((simple_model_input,))

        # Create dummy batch
        dummy_input = simple_model_input
        dummy_target = torch.randint(0, 10, (1,))

        # Check that sensitivities are None before calibration
        for module in prepared_model.modules():
            if isinstance(module, _KMeansFakePalettize):
                assert module.sensitivities is None

        with palettizer.calibration_mode(loss_fn=nn.functional.cross_entropy) as skm:
            output = prepared_model(dummy_input)
            skm.step(output, dummy_target)

        # After calibration, sensitivities should be set
        for module in prepared_model.modules():
            if isinstance(module, _KMeansFakePalettize):
                assert module.sensitivities is not None

                # Verify sensitivities are positive and normalized
                assert torch.all(module.sensitivities > 0), "Sensitivities should be positive"
                assert torch.all(module.sensitivities <= 1.0), "Sensitivities should be normalized"

    def test_calibration_mode_recomputes_centroids(
        self, simple_conv_linear_model, basic_config, simple_model_input
    ):
        """Test that calibration_mode triggers centroid recomputation."""
        palettizer = KMeansPalettizer(simple_conv_linear_model, basic_config)
        prepared_model = palettizer.prepare((simple_model_input,))

        # Store initial LUTs
        initial_luts = {}
        for name, module in prepared_model.named_modules():
            if P.is_parametrized(module):
                for attr_name, parametrizations in module.parametrizations.items():
                    for p in parametrizations:
                        if isinstance(p, _KMeansFakePalettize):
                            initial_luts[f"{name}.{attr_name}"] = p.lut.clone()

        # Create dummy batch
        dummy_input = simple_model_input
        dummy_target = torch.randint(0, 10, (1,))

        with palettizer.calibration_mode(loss_fn=nn.functional.cross_entropy) as skm:
            output = prepared_model(dummy_input)
            skm.step(output, dummy_target)

        # Check that LUTs have been updated (recomputed)
        luts_changed = False
        for name, module in prepared_model.named_modules():
            if P.is_parametrized(module):
                for attr_name, parametrizations in module.parametrizations.items():
                    for p in parametrizations:
                        if isinstance(p, _KMeansFakePalettize):
                            key = f"{name}.{attr_name}"
                            if key in initial_luts:
                                # LUTs should be different after calibration
                                if not torch.equal(initial_luts[key], p.lut):
                                    luts_changed = True

        assert luts_changed, "LUTs were not recomputed after calibration"

    def test_calibration_mode_observer_states(
        self, simple_conv_linear_model, basic_config, simple_model_input
    ):
        """Test that observer and fake palettize states are managed correctly."""
        palettizer = KMeansPalettizer(simple_conv_linear_model, basic_config)
        prepared_model = palettizer.prepare((simple_model_input,))

        # After prepare, observers should be disabled and fake palettize enabled
        for module in prepared_model.modules():
            if isinstance(module, _KMeansFakePalettize):
                assert not module.observer_enabled, "Observer should be disabled after prepare"
                assert module.fake_palett_enabled, "Fake palettize should be enabled after prepare"

        # Create dummy batch
        dummy_input = simple_model_input
        dummy_target = torch.randint(0, 10, (1,))

        # Inside calibration context, observers and fake palettize are disabled
        with palettizer.calibration_mode(loss_fn=nn.functional.cross_entropy) as skm:
            for module in prepared_model.modules():
                if isinstance(module, _KMeansFakePalettize):
                    assert not module.observer_enabled, (
                        "Observer should be disabled during calibration"
                    )
                    assert not module.fake_palett_enabled, (
                        "Fake palettize should be disabled during calibration"
                    )

            output = prepared_model(dummy_input)
            skm.step(output, dummy_target)

        # After calibration, observers should be disabled and fake palettize enabled
        for module in prepared_model.modules():
            if isinstance(module, _KMeansFakePalettize):
                assert not module.observer_enabled, "Observer should be disabled after calibration"
                assert module.fake_palett_enabled, (
                    "Fake palettize should be enabled after calibration"
                )

    def test_calibration_mode_with_optional_model_parameter(
        self, simple_conv_linear_model, basic_config, simple_model_input
    ):
        """Test calibration_mode with optional model parameter."""
        palettizer = KMeansPalettizer(simple_conv_linear_model, basic_config)
        prepared_model = palettizer.prepare((simple_model_input,))

        dummy_input = simple_model_input
        dummy_target = torch.randint(0, 10, (1,))

        # Pass model explicitly to calibration_mode
        with palettizer.calibration_mode(
            model=prepared_model, loss_fn=nn.functional.cross_entropy
        ) as skm:
            output = prepared_model(dummy_input)
            skm.step(output, dummy_target)

        # Should work without errors
        for module in prepared_model.modules():
            if isinstance(module, _KMeansFakePalettize):
                assert module.sensitivities is not None

    def test_calibration_mode_handles_small_gradients(
        self, simple_conv_linear_model, basic_config, simple_model_input
    ):
        """
        Test that calibration_mode handles small gradients.
        This tests the clipping logic in _construct_sensitivities.
        """
        palettizer = KMeansPalettizer(simple_conv_linear_model, basic_config)
        prepared_model = palettizer.prepare((simple_model_input,))

        # Create a loss that only depends on specific output classes
        # This creates varying gradients: some large, some small, some zero
        def selective_loss(output, target):
            # Only penalize the first output neuron heavily
            # Other neurons will have zero or very small gradients
            first_neuron_loss = (output[:, 0] - 1.0) ** 2
            # Add a tiny contribution from other neurons to create variation
            other_neurons_loss = (output[:, 1:] ** 2).sum() * 1e-10
            return first_neuron_loss + other_neurons_loss

        dummy_input = simple_model_input
        dummy_target = torch.randint(0, 10, (1,))

        # Should not raise an error with mixed gradient magnitudes
        with palettizer.calibration_mode(loss_fn=selective_loss) as skm:
            output = prepared_model(dummy_input)
            skm.step(output, dummy_target)

        # Verify sensitivities were still set and properly clipped
        _SENSITIVITY_CLIP_THR = 1e-12
        has_clipped_values = False
        for module in prepared_model.modules():
            if isinstance(module, _KMeansFakePalettize):
                assert module.sensitivities is not None
                # Should be finite and positive
                assert torch.isfinite(module.sensitivities).all()
                assert torch.all(module.sensitivities > 0)

                # Check if any values were clipped to the threshold
                # (This tests that the clipping logic was executed)
                clipped_count = torch.sum(
                    torch.isclose(
                        module.sensitivities,
                        torch.tensor(_SENSITIVITY_CLIP_THR),
                        rtol=1e-3,
                    )
                )
                if clipped_count > 0:
                    has_clipped_values = True

        # At least some values should have been clipped
        # (verifying the edge case handling was triggered)
        assert has_clipped_values, (
            "Expected some sensitivity values to be clipped to threshold, "
            "but none were found. This may indicate the test isn't creating "
            "the right gradient pattern."
        )

    def test_calibration_mode_sensitivity_values_correctness(self, basic_config):
        """
        Test that calibration_mode computes correct sensitivity values for a simple
        model with known weights.
        """

        # Create a simple model with a single linear layer and known weights
        class SimpleLinearModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Linear layer: out_features=2, in_features=2, no bias
                self.linear = nn.Linear(2, 2, bias=False)
                # Set known weights: [[1, 2], [3, 4]]
                with torch.no_grad():
                    self.linear.weight.copy_(
                        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
                    )

            def forward(self, x):
                return self.linear(x)

        model = SimpleLinearModel()

        # Input: [[1, 2]] - chosen to produce varying gradients
        test_input = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

        # Configure palettizer
        palettizer = KMeansPalettizer(model, basic_config)
        prepared_model = palettizer.prepare((test_input,))

        # Loss function that only uses the first output element
        # This creates non-uniform gradients across the weight matrix
        def first_output_loss(output, target):
            return output[:, 0].sum()

        dummy_target = torch.zeros(1)

        # Run calibration
        with palettizer.calibration_mode(loss_fn=first_output_loss) as skm:
            output = prepared_model(test_input)
            skm.step(output, dummy_target)

        # Expected sensitivity values:
        # grad^2 = [[1, 4], [0, 0]]
        # After normalization (val / max(val)): [[0.25, 1.0], [0, 0]]
        # After replacing zeros with min non-zero: [[0.25, 1.0], [0.25, 0.25]]
        expected_sensitivities = torch.tensor([[0.25, 1.0], [0.25, 0.25]], dtype=torch.float32)

        # Find the _KMeansFakePalettize module and check sensitivities
        for module in prepared_model.modules():
            if isinstance(module, _KMeansFakePalettize):
                assert module.sensitivities is not None, (
                    "Sensitivities should be set after calibration"
                )
                torch.testing.assert_close(
                    module.sensitivities,
                    expected_sensitivities,
                    rtol=1e-5,
                    atol=1e-5,
                    msg="Sensitivity values do not match expected values",
                )

    def test_calibration_mode_saves_sensitivities_to_path(
        self, simple_conv_linear_model, basic_config, simple_model_input, tmp_path
    ):
        """
        Test that calibration_mode saves sensitivities to the specified path.
        """
        palettizer = KMeansPalettizer(simple_conv_linear_model, basic_config)
        prepared_model = palettizer.prepare((simple_model_input,))

        dummy_input = simple_model_input
        dummy_target = torch.randint(0, 10, (1,))

        # Create a temporary file path for sensitivities
        sensitivity_file = tmp_path / "sensitivities.pt"

        # Run calibration with sensitivity_path specified
        with palettizer.calibration_mode(
            loss_fn=nn.functional.cross_entropy, sensitivity_path=str(sensitivity_file)
        ) as skm:
            output = prepared_model(dummy_input)
            skm.step(output, dummy_target)

        # Verify the sensitivity file was created
        assert sensitivity_file.exists(), "Sensitivity file should be created"

        # Load and verify the saved sensitivities
        saved_sensitivities = torch.load(sensitivity_file)

        # Verify it's a dictionary
        assert isinstance(saved_sensitivities, dict), "Saved sensitivities should be a dict"

        # Verify keys match parameter names
        param_names = set()
        for name, param in prepared_model.named_parameters(remove_duplicate=True):
            if param.requires_grad:
                param_names.add(name)

        assert set(saved_sensitivities.keys()) == param_names, (
            "Saved sensitivity keys should match parameter names"
        )

        # Verify all sensitivity values are tensors with correct shapes
        for name, param in prepared_model.named_parameters(remove_duplicate=True):
            if param.requires_grad:
                assert name in saved_sensitivities
                assert isinstance(saved_sensitivities[name], torch.Tensor)
                assert saved_sensitivities[name].shape == param.shape, (
                    f"Sensitivity shape for {name} should match parameter shape"
                )

    def test_stale_centroids_warning_when_sensitivities_set(
        self, simple_conv_linear_model, basic_config, simple_model_input, caplog
    ):
        """
        Test that a warning is logged when sensitivities are set on a
        _KMeansFakePalettize module but centroids are not recomputed before use.
        """
        palettizer = KMeansPalettizer(simple_conv_linear_model, basic_config)
        prepared_model = palettizer.prepare((simple_model_input,))

        # Find a _KMeansFakePalettize module
        fake_palettize_module = None
        for module in prepared_model.modules():
            if isinstance(module, _KMeansFakePalettize):
                fake_palettize_module = module
                break

        assert fake_palettize_module is not None, "Should have a _KMeansFakePalettize module"

        # Verify centroids are not stale initially
        assert not fake_palettize_module._centroids_stale

        # Manually set sensitivities without going through calibration_mode
        # This simulates a user incorrectly setting sensitivities directly
        fake_sensitivities = torch.ones_like(prepared_model.conv.parametrizations.weight.original)
        fake_palettize_module.sensitivities = fake_sensitivities

        # Verify centroids are now stale
        assert fake_palettize_module._centroids_stale

        # Run a forward pass with observer disabled (default after prepare)
        # This should log a warning about stale centroids
        with caplog.at_level(logging.WARNING):
            _ = prepared_model(simple_model_input)

        # Check that the warning was logged
        assert any(
            "Sensitivities were updated but centroids have not been recomputed" in record.message
            for record in caplog.records
        ), "Expected warning about stale centroids was not logged"

        # Now recompute centroids by enabling observer and running a forward pass
        prepared_model.apply(_enable_observer)

        with torch.no_grad():
            _ = prepared_model(simple_model_input)

        # Disable observer for normal operation
        prepared_model.apply(_disable_observer)

        # Verify centroids are no longer stale
        assert not fake_palettize_module._centroids_stale

        # Clear the log and run another forward pass
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            _ = prepared_model(simple_model_input)

        # Verify the warning is NOT logged after recomputing centroids
        assert not any(
            "Sensitivities were updated but centroids have not been recomputed" in record.message
            for record in caplog.records
        ), "Warning should not be logged after centroids are recomputed"

    def test_save_sensitivities_before_prepare_raises_error(
        self, simple_conv_linear_model, basic_config, tmp_path
    ):
        """Test that save_sensitivities raises error if model is not prepared."""
        palettizer = KMeansPalettizer(simple_conv_linear_model, basic_config)
        sensitivity_file = tmp_path / "sensitivities.pt"

        with pytest.raises(RuntimeError, match="Model must be prepared"):
            palettizer.save_sensitivities(str(sensitivity_file))

    def test_save_sensitivities_without_calibration_raises_error(
        self, simple_conv_linear_model, basic_config, simple_model_input, tmp_path
    ):
        """Test that save_sensitivities raises error if no sensitivities exist."""
        palettizer = KMeansPalettizer(simple_conv_linear_model, basic_config)
        palettizer.prepare((simple_model_input,))
        sensitivity_file = tmp_path / "sensitivities.pt"

        # Without calibration, no sensitivities are set
        with pytest.raises(ValueError, match="No sensitivities found"):
            palettizer.save_sensitivities(str(sensitivity_file))

    def test_save_sensitivities_after_calibration(
        self, simple_conv_linear_model, basic_config, simple_model_input, tmp_path
    ):
        """Test that save_sensitivities works after calibration."""
        palettizer = KMeansPalettizer(simple_conv_linear_model, basic_config)
        prepared_model = palettizer.prepare((simple_model_input,))

        dummy_target = torch.randint(0, 10, (1,))

        # Run calibration without specifying sensitivity_path
        with palettizer.calibration_mode(loss_fn=nn.functional.cross_entropy) as skm:
            output = prepared_model(simple_model_input)
            skm.step(output, dummy_target)

        # Now save sensitivities using the new API
        sensitivity_file = tmp_path / "sensitivities.pt"
        palettizer.save_sensitivities(str(sensitivity_file))

        # Verify file was created
        assert sensitivity_file.exists(), "Sensitivity file should be created"

        # Load and verify contents
        saved_sensitivities = torch.load(sensitivity_file)
        assert isinstance(saved_sensitivities, dict)
        assert len(saved_sensitivities) > 0

        # Verify that saved sensitivities match those in the model's
        # _KMeansFakePalettize modules
        for module_name, module in prepared_model.named_modules():
            if P.is_parametrized(module):
                for attr_name, parametrizations in module.parametrizations.items():
                    for p in parametrizations:
                        if isinstance(p, _KMeansFakePalettize):
                            param_name = ".".join(
                                [module_name, "parametrizations", attr_name, "original"]
                            )
                            assert param_name in saved_sensitivities, (
                                f"Saved sensitivities should contain {param_name}"
                            )
                            torch.testing.assert_close(
                                saved_sensitivities[param_name],
                                p.sensitivities,
                                msg=f"Saved sensitivity for {param_name} should match "
                                "module sensitivity",
                            )
                            break


class TestKMeansPalettizerPrepareWithSensitivities:
    """Test cases for prepare() with precomputed sensitivities."""

    def test_prepare_with_precomputed_sensitivities(
        self, simple_conv_linear_model, basic_config, simple_model_input, tmp_path
    ):
        """
        Test that prepare() can use precomputed sensitivities for weighted k-means.
        """
        # Step 1: Prepare model and run calibration to compute sensitivities
        palettizer1 = KMeansPalettizer(copy.deepcopy(simple_conv_linear_model), basic_config)
        prepared_model1 = palettizer1.prepare((simple_model_input,))

        dummy_target = torch.randint(0, 10, (1,))
        sensitivity_file = tmp_path / "sensitivities.pt"

        with palettizer1.calibration_mode(
            loss_fn=nn.functional.cross_entropy, sensitivity_path=str(sensitivity_file)
        ) as skm:
            output = prepared_model1(simple_model_input)
            skm.step(output, dummy_target)

        # Step 2: Prepare a fresh model using the precomputed sensitivities
        palettizer2 = KMeansPalettizer(copy.deepcopy(simple_conv_linear_model), basic_config)
        prepared_model2 = palettizer2.prepare(
            (simple_model_input,), sensitivity_path=str(sensitivity_file)
        )

        # Step 3: Verify sensitivities were set in the second model
        has_sensitivities = False
        for module in prepared_model2.modules():
            if isinstance(module, _KMeansFakePalettize):
                if module.sensitivities is not None:
                    has_sensitivities = True
                    # Verify sensitivities are normalized
                    assert torch.all(module.sensitivities > 0)
                    assert torch.all(module.sensitivities <= 1.0)

        assert has_sensitivities, "Sensitivities should be set in prepared model"

    def test_precomputed_sensitivities_match_calibration_and_differ_from_vanilla(
        self, simple_conv_linear_model, basic_config, simple_model_input, tmp_path
    ):
        """
        Test that precomputed sensitivities produce consistent results with calibration
        and different results from vanilla k-means.
        """
        # Case 1: Compute and save sensitivities via calibration
        palettizer1 = KMeansPalettizer(copy.deepcopy(simple_conv_linear_model), basic_config)
        prepared_model1 = palettizer1.prepare((simple_model_input,))

        dummy_target = torch.randint(0, 10, (1,))
        sensitivity_file = tmp_path / "sensitivities.pt"

        with palettizer1.calibration_mode(
            loss_fn=nn.functional.cross_entropy, sensitivity_path=str(sensitivity_file)
        ) as skm:
            output = prepared_model1(simple_model_input)
            skm.step(output, dummy_target)

        # Original LUTs
        original_luts = {}
        for name, module in prepared_model1.named_modules():
            if isinstance(module, _KMeansFakePalettize):
                if module.lut is not None:
                    original_luts[name] = module.lut.clone()

        # Case 2: Prepare model with pre-computed sensitivities
        palettizer2 = KMeansPalettizer(copy.deepcopy(simple_conv_linear_model), basic_config)
        prepared_model2 = palettizer2.prepare(
            (simple_model_input,), sensitivity_path=str(sensitivity_file)
        )

        # Store weighted LUTs
        new_luts = {}
        for name, module in prepared_model2.named_modules():
            if isinstance(module, _KMeansFakePalettize):
                if module.lut is not None:
                    new_luts[name] = module.lut.clone()

        # Verify LUTs are the same since we are using the same sensitivity values
        for name, new_lut in new_luts.items():
            assert name in original_luts
            assert torch.equal(original_luts[name], new_lut), (
                "LUT computed from calibration should be the same as LUT computed "
                "from pre computed sensitivity values"
            )

        # Case 3: Prepare model without any sensitivity info
        palettizer3 = KMeansPalettizer(copy.deepcopy(simple_conv_linear_model), basic_config)
        prepared_model3 = palettizer3.prepare((simple_model_input,))

        vanilla_luts = {}
        for name, module in prepared_model3.named_modules():
            if isinstance(module, _KMeansFakePalettize):
                if module.lut is not None:
                    vanilla_luts[name] = module.lut.clone()

        # Verify LUTs are different from the ones computed with sensitivity values
        for name, new_lut in new_luts.items():
            assert name in vanilla_luts
            assert not torch.equal(vanilla_luts[name], new_lut), (
                "LUT computed without sensitivity should be different from the LUT "
                "computed using sensitivity values"
            )


@pytest.mark.parametrize(
    "spec_config",
    [
        # Test different n_bits
        {"n_bits": 1},
        {"n_bits": 2},
        {"n_bits": 3},
        {"n_bits": 4},
        {"n_bits": 6},
        {"n_bits": 8},
        # Test different granularities
        {"n_bits": 4, "granularity": PerTensorGranularity()},
        {
            "n_bits": 4,
            "granularity": PerGroupedChannelGranularity(axis=0, group_size=2),
        },
        {
            "n_bits": 4,
            "granularity": PerGroupedChannelGranularity(axis=0, group_size=4),
        },
        {
            "n_bits": 4,
            "granularity": PerGroupedChannelGranularity(axis=0, group_size=8),
        },
        {
            "n_bits": 4,
            "granularity": PerGroupedChannelGranularity(axis=1, group_size=2),
        },
        # Test cluster dimensions
        {"n_bits": 4, "cluster_dim": 1},
        # Test per-channel scaling
        {"n_bits": 4, "cluster_dim": 1, "enable_per_channel_scale": True},
        {"n_bits": 4, "cluster_dim": 1, "enable_per_channel_scale": False},
        # Test combined configurations
        {
            "n_bits": 2,
            "granularity": PerGroupedChannelGranularity(axis=0, group_size=4),
            "cluster_dim": 1,
            "enable_per_channel_scale": True,
        },
        # Test vector palettization
        {"n_bits": 4, "cluster_dim": 2},
        {
            "n_bits": 4,
            "cluster_dim": 2,
            "granularity": PerGroupedChannelGranularity(axis=0, group_size=4),
        },
        {
            "n_bits": 4,
            "cluster_dim": 2,
            "lut_qspec": QuantizationSpec(dtype=torch.int8, qscheme=QuantizationScheme.SYMMETRIC),
        },
        # Test vector palettization with per-channel scale
        {"n_bits": 4, "cluster_dim": 2, "enable_per_channel_scale": True},
        {
            "n_bits": 4,
            "cluster_dim": 2,
            "enable_per_channel_scale": True,
            "granularity": PerGroupedChannelGranularity(axis=0, group_size=4),
        },
        # Test quantized LUT configurations
        {
            "n_bits": 4,
            "lut_qspec": QuantizationSpec(dtype=torch.int8, qscheme=QuantizationScheme.SYMMETRIC),
        },
        {
            "n_bits": 4,
            "lut_qspec": QuantizationSpec(dtype=torch.uint8, qscheme=QuantizationScheme.ASYMMETRIC),
        },
        {
            "n_bits": 2,
            "lut_qspec": QuantizationSpec(dtype=torch.int8, qscheme=QuantizationScheme.SYMMETRIC),
            "granularity": PerGroupedChannelGranularity(axis=0, group_size=4),
        },
        {
            "n_bits": 4,
            "lut_qspec": QuantizationSpec(dtype=torch.int8, qscheme=QuantizationScheme.SYMMETRIC),
            "enable_per_channel_scale": True,
        },
    ],
)
def test_comprehensive_palettization_specs(
    spec_config, simple_conv_linear_model, simple_model_input
):
    """
    Comprehensive test with various PalettizationSpec configurations.
    """
    # Create a copy of the model for the original baseline

    original_model = copy.deepcopy(simple_conv_linear_model)
    original_model.eval()

    print(spec_config)
    # Create PalettizationSpec from test parameters
    spec = PalettizationSpec(**spec_config)

    # Setup the palettizer with the test configuration
    module_config_kwargs = {"op_state_spec": {"weight": spec}}
    if spec_config.get("cluster_dim", 1) > 1:
        module_config_kwargs["enable_fast_kmeans_mode"] = False
    config = KMeansPalettizerConfig(
        global_config=ModuleKMeansPalettizerConfig(**module_config_kwargs),
    )
    palettizer = KMeansPalettizer(simple_conv_linear_model, config)

    # Prepare the model for palettization
    prepared_model = palettizer.prepare(example_inputs=(simple_model_input,))
    prepared_model.eval()

    # Compare outputs between original and prepared models
    with torch.no_grad():
        original_output = original_model(simple_model_input)
        prepared_output = prepared_model(simple_model_input)

    # # Assertions for output similarity
    torch.all(torch.isclose(original_output, prepared_output))

    # Verify that palettization actually reduces the number of unique weight values
    n_bits = spec_config["n_bits"]
    granularity = spec_config.get("granularity", PerTensorGranularity())
    enable_pcs = spec_config.get("enable_per_channel_scale", False)
    for prep_module in [prepared_model.conv, prepared_model.linear]:
        if hasattr(prep_module, "weight"):
            try:
                num_groups = granularity.num_blocks_to_cluster(prep_module.weight)
            except _IncompatibleGranularityError:
                # layer was not palettized because of group size indivisibility
                continue

            expected_max_values = 2**n_bits * num_groups * spec_config.get("cluster_dim", 1)

            weight = prep_module.weight
            if enable_pcs:
                weight = _apply_per_channel_scale(prep_module, weight)

            unique_values = _count_unique_params(torch.unique(weight))

            assert unique_values <= expected_max_values, (
                f"Layer {prep_module} has {unique_values} unique values, "
                f"expected <= {expected_max_values} for {n_bits}-bit palettization"
            )


def _apply_per_channel_scale(module, param, param_name="weight"):
    """
    Useful when counting unique values in palettized weight tensor when per channel
    scale is applied.
    """
    fake_palett_mod = module.parametrizations[param_name][0]
    per_channel_scale = fake_palett_mod.per_channel_scale
    for _ in range(param.dim() - per_channel_scale.dim()):
        per_channel_scale = per_channel_scale.unsqueeze(-1)
    param = param / per_channel_scale

    return param


def _count_unique_params(tensor):
    """
    Returns number of unique parameters in the same tensor.
    Set a defaulted absolute tolerance, so that very close values can be treated
    as identical in palletization.
    """
    unique_set = {tensor[0]}
    for elem in tensor[1:]:
        if all(not torch.isclose(elem, uelem, atol=1e-6) for uelem in unique_set):
            unique_set.add(elem)
    return len(unique_set)


@pytest.mark.parametrize(
    "op_state_spec, expect_in_proj, expect_out_proj",
    [
        pytest.param(
            {"in_proj_weight": PalettizationSpec(n_bits=4)},
            True,
            False,
            id="in_proj_weight_only",
        ),
        pytest.param(
            {"weight": PalettizationSpec(n_bits=4)},
            False,
            True,
            id="out_proj_weight_only",
        ),
        pytest.param(
            {
                "in_proj_weight": PalettizationSpec(n_bits=4),
                "weight": PalettizationSpec(n_bits=4),
            },
            True,
            True,
            id="both_weights",
        ),
        pytest.param(
            None,
            True,
            True,
            id="default_config",
        ),
    ],
)
def test_palettize_multihead_attention(
    simple_mha_model, simple_mha_model_input, op_state_spec, expect_in_proj, expect_out_proj
):
    """Test palettization of nn.MultiheadAttention in_proj_weight and out_proj.weight."""
    module_config_kwargs = {}
    if op_state_spec is not None:
        module_config_kwargs["op_state_spec"] = op_state_spec

    config = KMeansPalettizerConfig(
        global_config=None,
        module_type_configs={
            nn.MultiheadAttention: ModuleKMeansPalettizerConfig(**module_config_kwargs),
        },
    )

    palettizer = KMeansPalettizer(simple_mha_model, config)
    prepared_model = palettizer.prepare((simple_mha_model_input,))

    assert is_parametrized(prepared_model.attn, "in_proj_weight") == expect_in_proj
    assert is_parametrized(prepared_model.attn.out_proj, "weight") == expect_out_proj
    assert not is_parametrized(prepared_model.attn, "in_proj_bias")
    assert not is_parametrized(prepared_model.attn.out_proj, "bias")

    if expect_in_proj:
        assert isinstance(
            prepared_model.attn.parametrizations["in_proj_weight"][0],
            _FakePalettizeImplBase,
        )
        assert len(torch.unique(prepared_model.attn.in_proj_weight)) <= 16

    if expect_out_proj:
        assert isinstance(
            prepared_model.attn.out_proj.parametrizations["weight"][0],
            _FakePalettizeImplBase,
        )
        assert len(torch.unique(prepared_model.attn.out_proj.weight)) <= 16

    output = prepared_model(simple_mha_model_input)
    assert output.shape == (1, 10, 64)
