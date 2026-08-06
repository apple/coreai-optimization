# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Pytest configuration file for coreai_opt tests."""

import random
import tempfile
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pytest
import torch

from coreai_opt import ExportBackend
from coreai_opt.palettization import (
    KMeansPalettizerConfig,
    ModuleKMeansPalettizerConfig,
)
from coreai_opt.palettization.spec import (
    PalettizationSpec,
    PerGroupedChannelGranularity,
    PerTensorGranularity as PalettizationPerTensorGranularity,
)
from coreai_opt.palettization.spec.spec import _SUPPORTED_LUT_DTYPES
from coreai_opt.pruning import MagnitudePrunerConfig, ModuleMagnitudePrunerConfig, PruningSpec
from coreai_opt.pruning.spec import ChannelStructured, PruningScheme, Unstructured
from coreai_opt.quantization import ModuleQuantizerConfig, QuantizerConfig
from coreai_opt.quantization.spec import (
    PerBlockGranularity,
    PerChannelGranularity,
    PerTensorGranularity,
    QuantizationScheme,
    QuantizationSpec,
    default_activation_quantization_spec,
    default_weight_quantization_spec,
)
from coreai_opt.quantization.spec.fake_quantize import _DefaultFakeQuantizeImpl
from coreai_opt.quantization.spec.qparams_calculator import StaticQParamsCalculator
from coreai_opt.quantization.spec.range_calculator import MinMaxRangeCalculator
from tests.models.composite import (  # noqa: F401
    composite_rmsnorm_input,
    composite_rmsnorm_model,
    mnist_composite_rmsnorm_example_input,
    mnist_composite_rmsnorm_pretrained_model,
    mnist_composite_rmsnorm_pretrained_state,
)
from tests.models.mnist import (  # noqa: F401
    custom_test_mnist_model,
    mnist_data,
    mnist_dataset,
    mnist_example_input,
    mnist_example_output,
)
from tests.models.resnet import (  # noqa: F401
    resnet18_model,
    resnet50_model,
    resnet_example_input,
)
from tests.models.simple import (  # noqa: F401
    gated_mlp_model,
    gated_mlp_model_input,
    shared_params_model,
    shared_params_model_input,
    simple_conv_linear_model,
    simple_linear_model,
    simple_linear_model_input,
    simple_mha_model,
    simple_mha_model_input,
    simple_model_input,
)
from tests.utils import test_artifact_path

_DEFAULT_SEED: int = 42


# Quantization dtypes that CoreML export must reject. Weight dtypes include both
# torch dtype objects and string aliases.
COREML_WEIGHT_REJECT_DTYPES = [
    pytest.param(torch.float8_e4m3fn, id="fp8-torch-e4m3fn"),
    pytest.param("float8_e4m3fn", id="fp8-str-e4m3fn"),
    pytest.param(torch.float8_e5m2, id="fp8-torch-e5m2"),
    pytest.param("float4_e2m1fn", id="fp4-str"),
    pytest.param(torch.int2, id="int2-torch"),
    pytest.param(torch.uint2, id="uint2-torch"),
]

COREML_ACT_REJECT_DTYPES = [
    pytest.param(torch.float8_e4m3fn, id="e4m3fn"),
    pytest.param(torch.float8_e5m2, id="e5m2"),
    pytest.param(torch.int4, id="int4"),
    pytest.param(torch.uint4, id="uint4"),
    pytest.param(torch.int2, id="int2"),
    pytest.param(torch.uint2, id="uint2"),
]


def make_quant_config(
    *,
    weight_dtype: torch.dtype | str | None,
    act_dtype: torch.dtype | str | None,
    execution_mode: str,
) -> QuantizerConfig:
    """Build a per-tensor symmetric QuantizerConfig for export tests.

    Args:
        weight_dtype (torch.dtype | str | None): Weight dtype, or None to disable.
        act_dtype (torch.dtype | str | None): Activation dtype, or None to disable.
        execution_mode (str): Either "eager" or "graph".

    Returns:
        QuantizerConfig: Config with the requested per-tensor symmetric specs.
    """

    def _spec(dtype: torch.dtype | str) -> QuantizationSpec:
        return QuantizationSpec(
            dtype=dtype,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerTensorGranularity(),
        )

    weight_spec = _spec(weight_dtype) if weight_dtype is not None else None
    act_spec = _spec(act_dtype) if act_dtype is not None else None
    return QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={"weight": weight_spec} if weight_spec is not None else None,
            op_input_spec={"*": act_spec},
            op_output_spec={"*": act_spec},
        ),
        execution_mode=execution_mode,
    )


def make_graph_mode_ptq_config(*, quantize_activations: bool) -> QuantizerConfig:
    """Build a graph-mode w8 (weight-only) or w8a8 PTQ QuantizerConfig.

    Args:
        quantize_activations (bool): True for w8a8, False for w8 weight-only.

    Returns:
        QuantizerConfig: Config with the default weight spec globally, plus the
            default activation spec on every op input/output when requested.
    """
    activation_spec = default_activation_quantization_spec() if quantize_activations else None
    return QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={"weight": default_weight_quantization_spec()},
            op_input_spec={"*": activation_spec} if activation_spec else None,
            op_output_spec={"*": activation_spec} if activation_spec else None,
        ),
        execution_mode="graph",
    )


@pytest.fixture(autouse=True)
def seed_every_test(request: pytest.FixtureRequest) -> None:
    """Seeding policy for test reproducibility.

    By default, tests run with nondeterministic seeding.

    Use markers to enable deterministic seeding when reproducibility is needed:
    - No marker: doesn't do anything special
    - @pytest.mark.seed: Use default seed (42) for deterministic behavior
    - @pytest.mark.seed(N): Use specific seed N for deterministic behavior
    - @pytest.mark.seed(None): Explicitly use nondeterministic seeding
    """
    marker = request.node.get_closest_marker("seed")

    if marker is None:
        # No marker: don't do anything special
        return

    # @pytest.mark.seed (no argument): use default seed
    # @pytest.mark.seed(N): use specified seed, `N` can be `None`
    seed = _DEFAULT_SEED if not marker.args else marker.args[0]

    # Validate seed type
    if seed is not None and not isinstance(seed, int):
        pytest.fail(
            f"@pytest.mark.seed expects int or None, got {type(seed).__name__}: {seed!r}",
        )

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    if seed is None:
        torch.seed()
    else:
        torch.manual_seed(seed)


@pytest.fixture(autouse=True)
def reset_dynamo() -> None:
    """Reset torch._dynamo state before each test.

    This ensures tests don't interfere with each other through cached
    dynamo compilation state.
    """
    torch._dynamo.reset()


@pytest.fixture(scope="session")
def temp_dir():
    """Fixture to provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(scope="function")
def mnist_pretrained_model(custom_test_mnist_model):  # noqa: F811
    """Load the committed 1-epoch MNIST checkpoint into a fresh model."""
    model = custom_test_mnist_model
    model.load_state_dict(
        torch.load(test_artifact_path("mnist/mnist_pretrained_1epoch_09032025.pt"))
    )
    return model


@dataclass
class ParametrizedQuantConfigs:
    """Container for parametrized Eager and PT2E quantization configs.

    Used by the parametrized_quant_config test fixture to provide both config
    types with identical quantization parameters.

    Attributes:
        eager: QuantizerConfig with eager execution mode
        pt2e: QuantizerConfig with pt2e execution mode
        model_dtype: Model dtype (float16, float32, bfloat16, or None for no conversion)

    """

    eager: QuantizerConfig
    pt2e: QuantizerConfig
    model_dtype: torch.dtype | None

    @classmethod
    def from_quant_params(
        cls,
        weight_dtype: torch.dtype,
        act_dtype: torch.dtype | None,
        qscheme: QuantizationScheme,
        w_granularity: PerTensorGranularity | PerChannelGranularity | PerBlockGranularity,
        model_dtype: torch.dtype | None,
        act_granularity: PerTensorGranularity | PerChannelGranularity | None = None,
    ) -> "ParametrizedQuantConfigs":
        """Create ParametrizedQuantConfigs from quantization parameters.

        Args:
            weight_dtype: Weight quantization dtype
            act_dtype: Activation quantization dtype (None to disable)
            qscheme: Quantization scheme
            w_granularity: Weight Quantization granularity
            model_dtype: Model dtype
            act_granularity: Activation Quantization granularity

        Returns:
            ParametrizedQuantConfigs instance

        """
        activation_qspec = None
        if act_dtype is not None:
            activation_qspec = QuantizationSpec(
                dtype=act_dtype,
                qscheme=QuantizationScheme.SYMMETRIC,
                granularity=act_granularity or PerTensorGranularity(),
                fake_quantize_cls=_DefaultFakeQuantizeImpl,
                qparam_calculator_cls=StaticQParamsCalculator,
                range_calculator_cls=MinMaxRangeCalculator,
            )

        weight_qspec = QuantizationSpec(
            dtype=weight_dtype,
            qscheme=qscheme,
            granularity=w_granularity,
            fake_quantize_cls=_DefaultFakeQuantizeImpl,
            qparam_calculator_cls=StaticQParamsCalculator,
            range_calculator_cls=MinMaxRangeCalculator,
        )

        eager_config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec={"weight": weight_qspec},
                op_input_spec={"*": activation_qspec},
                op_output_spec={"*": activation_qspec},
            ),
            execution_mode="eager",
        )

        pt2e_config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec={"weight": weight_qspec},
                op_input_spec={"*": activation_qspec},
                op_output_spec={"*": activation_qspec},
            ),
            execution_mode="graph",
        )

        return cls(
            eager=eager_config,
            pt2e=pt2e_config,
            model_dtype=model_dtype,
        )

    @property
    def has_activation_quantization(self) -> bool:
        """Check if activation quantization is enabled in this config.

        Returns:
            True if activation quantization is enabled

        """
        # Eager and pt2e configs have identical quantization settings.
        # could use self.pt2e here as well
        return (
            self.eager.global_config.op_input_spec != {"*": None}
            if self.eager.global_config
            else False
        )

    def skip_if_unsupported(
        self,
        mode: Literal["eager", "graph"],
        backend: ExportBackend,
        unsupported_configs: dict[str, Any] | list[dict[str, Any]] | None = None,
        reason: str = "",
    ) -> None:
        """Skip test if this config matches unsupported constraints.

        Args:
            mode: Quantization mode to check
            backend: Export backend to check
            unsupported_configs: Dictionary or list of dictionaries of constraints that
                make this config unsupported. Constraint keys:
                - "backend": ExportBackend value to match
                - "act_dtype": torch dtype for activation quantization (torch.int8,
                  torch.uint8, None for disabled)
                - "weight_dtype": torch dtype for weight quantization
                - "granularity_type": String name of granularity class
                  ("PerTensorGranularity", "PerChannelGranularity",
                  "PerBlockGranularity")
                - "act_granularity_axis": int axis value on activation granularity

                Example: {"backend": ExportBackend.CoreML, "act_dtype": torch.int8}
                Example: [{"granularity_type": "PerChannelGranularity"},
                         {"granularity_type": "PerBlockGranularity"}]

        Raises:
            pytest.skip: If config matches any unsupported constraints

        """
        if unsupported_configs is None:
            return

        config = self.eager if mode == "eager" else self.pt2e

        # Normalize to list
        configs_to_check = (
            unsupported_configs if isinstance(unsupported_configs, list) else [unsupported_configs]
        )

        # Check each unsupported config
        for constraints in configs_to_check:
            if "backend" in constraints and backend != constraints["backend"]:
                continue
            if self._matches_constraints(config, constraints):
                pytest.skip(
                    reason or f"{mode.upper()} + {backend.value} does not support this config",
                )

    def xfail_if_unsupported(
        self,
        mode: Literal["eager", "graph"],
        backend: ExportBackend,
        unsupported_config: dict[str, Any] | list[dict[str, Any]] | None = None,
        reason: str = "",
    ) -> None:
        """Mark test as expected failure if this config matches unsupported constraints.

        Args:
            mode: Quantization mode to check
            backend: Export backend to check
            unsupported_config: Dictionary or list of dictionaries of constraints
            reason: Reason for the expected failure

        """
        if unsupported_config is None:
            return

        config = self.eager if mode == "eager" else self.pt2e

        # Normalize to list
        configs_to_check = (
            unsupported_config if isinstance(unsupported_config, list) else [unsupported_config]
        )

        # Check each unsupported config
        for constraints in configs_to_check:
            if self._matches_constraints(config, constraints):
                pytest.xfail(
                    reason or f"{mode.upper()} + {backend.value} does not support this config",
                )

    def _matches_constraints(
        self,
        config: QuantizerConfig,
        constraints: dict[str, Any],
    ) -> bool:
        """Check if config matches all specified constraints.

        Args:
            config: Config to check
            constraints: Dictionary of constraints to match. Valid keys:
                - backend: ExportBackend value (checked by caller, ignored here)
                - act_dtype: torch dtype for activation quantization
                - weight_dtype: torch dtype for weight quantization
                - granularity_type: String name of granularity class
                - model_dtype: torch dtype for model
                - act_granularity_axis: int axis value on activation granularity

        Returns:
            True if all constraints match

        Raises:
            ValueError: If constraints contain unknown keys

        Note:
            The 'backend' key is checked by the caller before this method is called,
            so it's included in valid_keys but ignored in the constraint matching logic.

        """
        if not config.global_config:
            return False
        weight_qspec = config.global_config.op_state_spec.get("weight")
        act_qspec = config.global_config.op_input_spec.get("*")
        # Validate constraint keys to catch typos
        valid_keys = {
            "backend",
            "act_dtype",
            "weight_dtype",
            "granularity_type",
            "model_dtype",
            "act_granularity_axis",
        }
        invalid_keys = set(constraints.keys()) - valid_keys
        if invalid_keys:
            msg = f"Unknown constraint keys: {invalid_keys}. Valid keys: {valid_keys}"
            raise ValueError(msg)

        for key, value in constraints.items():
            if key == "act_dtype":
                if act_qspec is None:
                    if value is not None:
                        return False
                elif act_qspec.dtype != value:
                    return False
            elif key == "weight_dtype":
                if weight_qspec is None:
                    if value is not None:
                        return False
                elif weight_qspec.dtype != value:
                    return False
            elif key == "granularity_type":
                if weight_qspec is None:
                    if value is not None:
                        return False
                elif weight_qspec.granularity.__class__.__name__ != value:
                    return False
            elif key == "model_dtype" and self.model_dtype != value:
                return False
            elif key == "act_granularity_axis":
                if (
                    act_qspec is None
                    or not hasattr(act_qspec.granularity, "axis")
                    or act_qspec.granularity.axis != value
                ):
                    return False

        return True


@dataclass
class ParametrizedPalettConfigs:
    """Container for parametrized palettization configs.

    Used by the parametrized_palett_config test fixture to provide KMeans
    palettization configuration with parameterized settings.

    Attributes:
        config: KMeansPalettizerConfig instance
        n_bits: Number of palette bits
        granularity: Palettization granularity
        enable_per_channel_scale: Whether per-channel scaling is enabled
        cluster_dim: Cluster dimension (1 for scalar, >1 for vector palettization)
        lut_qspec: LUT quantization spec (None if LUT is not quantized)

    """

    config: KMeansPalettizerConfig
    n_bits: int
    granularity: PalettizationPerTensorGranularity | PerGroupedChannelGranularity
    enable_per_channel_scale: bool
    cluster_dim: int = 1
    lut_qspec: QuantizationSpec | None = None

    @classmethod
    def from_palett_params(
        cls,
        n_bits: int,
        granularity: PalettizationPerTensorGranularity | PerGroupedChannelGranularity,
        enable_per_channel_scale: bool,
        cluster_dim: int = 1,
        lut_qspec: QuantizationSpec | None = None,
    ) -> "ParametrizedPalettConfigs":
        """Create ParametrizedPalettConfigs from palettization parameters.

        Args:
            n_bits: Number of palette bits
            granularity: Palettization granularity
            enable_per_channel_scale: Whether to enable per-channel scaling
            cluster_dim: Cluster dimension (1 for scalar, >1 for vector)
            lut_qspec: LUT quantization spec

        Returns:
            ParametrizedPalettConfigs instance

        """
        palett_spec = PalettizationSpec(
            n_bits=n_bits,
            lut_qspec=lut_qspec,
            granularity=granularity,
            cluster_dim=cluster_dim,
            enable_per_channel_scale=enable_per_channel_scale,
        )

        config = KMeansPalettizerConfig(
            global_config=ModuleKMeansPalettizerConfig(
                op_state_spec={
                    "weight": palett_spec,
                },
                enable_fast_kmeans_mode=cluster_dim == 1,
            ),
        )

        return cls(
            config=config,
            n_bits=n_bits,
            granularity=granularity,
            enable_per_channel_scale=enable_per_channel_scale,
            cluster_dim=cluster_dim,
            lut_qspec=lut_qspec,
        )


@dataclass
class ParametrizedFP8Configs:
    """Container for parametrized FP8 quantization configs.

    Used by the parametrized_fp8_config test fixture to provide FP8 quantization
    configurations for both Eager and PT2E quantizers.

    Attributes:
        eager: QuantizerConfig instance with FP8 quantization
        pt2e: QuantizerConfig instance with FP8 quantization
        fp8_dtype: FP8 dtype (float8_e4m3fn or float8_e5m2)
        with_activation_quant: Whether activation quantization is enabled

    """

    eager: QuantizerConfig
    pt2e: QuantizerConfig
    fp8_dtype: torch.dtype
    with_activation_quant: bool
    model_dtype: torch.dtype

    @classmethod
    def from_fp8_params(
        cls,
        fp8_dtype: torch.dtype,
        with_activation_quant: bool,
        model_dtype: torch.dtype = torch.float32,
        per_channel_activations: bool = False,
        per_channel_activations_axis: int = 0,
    ) -> "ParametrizedFP8Configs":
        """Create ParametrizedFP8Configs from FP8 parameters.

        FP8 quantization requires symmetric scheme and per-tensor granularity.

        Args:
            fp8_dtype: FP8 dtype (float8_e4m3fn or float8_e5m2)
            with_activation_quant: Whether to enable activation quantization
            model_dtype: Model dtype for the test (default: float32)
            per_channel_activations: [default=False] Whether activations are to be
            quantized per-channel.
            per_channel_activations_axis: [default=0] If per_channel_activations is set,
            this value specifies the axis for per-channel quantization.

        Returns:
            ParametrizedFP8Configs instance

        """
        weight_qspec = QuantizationSpec(
            dtype=fp8_dtype,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerTensorGranularity(),
            fake_quantize_cls="default",
            qparam_calculator_cls="default",
            range_calculator_cls="minmax",
        )

        activation_qspec = None
        if with_activation_quant:
            activation_qspec = QuantizationSpec(
                dtype=fp8_dtype,
                qscheme=QuantizationScheme.SYMMETRIC,
                granularity=PerChannelGranularity(axis=per_channel_activations_axis)
                if per_channel_activations
                else PerTensorGranularity(),
                fake_quantize_cls="default",
                qparam_calculator_cls="moving_average",
                range_calculator_cls="minmax",
            )

        eager_config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec={"weight": weight_qspec},
                op_input_spec={"*": activation_qspec},
                op_output_spec={"*": activation_qspec},
            ),
            execution_mode="eager",
        )

        pt2e_config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec={"weight": weight_qspec},
                op_input_spec={"*": activation_qspec},
                op_output_spec={"*": activation_qspec},
            ),
            execution_mode="graph",
        )

        return cls(
            eager=eager_config,
            pt2e=pt2e_config,
            fp8_dtype=fp8_dtype,
            with_activation_quant=with_activation_quant,
            model_dtype=model_dtype,
        )


@dataclass
class ParametrizedFP4Configs:
    """Container for parametrized FP4 quantization configs.

    Used by the parametrized_fp4_config test fixture to provide FP4 quantization
    configurations for both Eager and PT2E quantizers.

    Attributes:
        eager: QuantizerConfig instance with FP4 quantization
        pt2e: QuantizerConfig instance with FP4 quantization
        with_activation_quant: Whether activation quantization is enabled
        model_dtype: Model dtype for the test (default: float32)
    """

    eager: QuantizerConfig
    pt2e: QuantizerConfig
    with_activation_quant: bool
    model_dtype: torch.dtype

    @classmethod
    def from_fp4_params(
        cls,
        with_activation_quant: bool,
        model_dtype: torch.dtype = torch.float32,
        weight_dtype: torch.dtype | str = "float4_e2m1fn",
        per_block_weights: bool = False,
        weight_block_size: int = 32,
        activation_dtype: torch.dtype | str = "float4_e2m1fn",
        per_block_activations: bool = False,
        activation_block_size: int = 32,
    ) -> "ParametrizedFP4Configs":
        """Create ParametrizedFP4Configs from FP4 parameters.

        FP4 quantization requires symmetric scheme and per-block granularity with block_size=32.

        Args:
            with_activation_quant: Whether to enable activation quantization.
            model_dtype: Model dtype for the test (default: float32).
            weight_dtype: Weight dtype for quantization (default: float4_e2m1fn).
            per_block_weights: Whether weights are to be quantized per-block.
            weight_block_size: Block size for weight quantization.
            activation_dtype: Activation dtype for quantization (default: float4_e2m1fn).
            per_block_activations: Whether activations are to be quantized per-block.
            activation_block_size: Block size for activation quantization.

        Returns:
            ParametrizedFP4Configs instance

        """
        weight_qspec = QuantizationSpec(
            dtype=weight_dtype,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerBlockGranularity(axis=1, block_size=weight_block_size)
            if per_block_weights
            else PerTensorGranularity(),
        )

        activation_qspec = None
        if with_activation_quant:
            activation_qspec = QuantizationSpec(
                dtype=activation_dtype,
                qscheme=QuantizationScheme.SYMMETRIC,
                granularity=PerBlockGranularity(axis=1, block_size=activation_block_size)
                if per_block_activations
                else PerTensorGranularity(),
            )

        eager_config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec={"weight": weight_qspec},
                op_input_spec={"*": activation_qspec},
                op_output_spec={"*": activation_qspec},
            ),
            execution_mode="eager",
        )

        pt2e_config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec={"weight": weight_qspec},
                op_input_spec={"*": activation_qspec},
                op_output_spec={"*": activation_qspec},
            ),
            execution_mode="graph",
        )

        return cls(
            eager=eager_config,
            pt2e=pt2e_config,
            with_activation_quant=with_activation_quant,
            model_dtype=model_dtype,
        )


@pytest.fixture(
    params=[
        (weight_dtype, act_dtype, qscheme, w_granularity, act_granularity)
        for weight_dtype in [
            torch.int8,
            torch.uint8,
            torch.int4,
            torch.uint4,
        ]
        for act_dtype in [torch.int8, torch.uint8, None]
        for qscheme in list(QuantizationScheme)
        for w_granularity in [
            PerTensorGranularity(),
            PerChannelGranularity(axis=1),
            PerBlockGranularity(axis=0, block_size=2),
        ]
        for act_granularity in [
            PerTensorGranularity(),
            PerChannelGranularity(axis=0),
            PerChannelGranularity(axis=-1),
        ]
        # Weight-only configs (act_dtype=None) produce identical results regardless of
        # act_granularity. Only include 1 combination (with PerTensorGranularity) for
        # weight-only to avoid running redundant identical tests across all
        # act_granularity values.
        if act_dtype is not None or isinstance(act_granularity, PerTensorGranularity)
    ],
    ids=lambda p: (
        f"wt:{str(p[0]).split('.')[-1]}--"
        f"act:{str(p[1]).split('.')[-1] if p[1] else 'disabled'}--"
        f"qs:{p[2].value}--"
        f"wg:{p[3].__class__.__name__.replace('Granularity', '')}--"
        f"ag:{p[4].__class__.__name__.replace('Granularity', '')}--"
        f"axis:{p[4].axis}"
    ),
)
def parametrized_quant_config_general(
    request: pytest.FixtureRequest,
) -> ParametrizedQuantConfigs:
    """Fixture for general quantization configs without model dtype conversion.

    Sets model_dtype=None to skip dtype conversion.
    Generates 252 parameter combinations.
    Weight-only configs use only PerTensorGranularity for act_granularity.

    Returns:
        ParametrizedQuantConfigs with model_dtype=None

    """
    weight_dtype, act_dtype, qscheme, w_granularity, act_granularity = request.param
    return ParametrizedQuantConfigs.from_quant_params(
        weight_dtype,
        act_dtype,
        qscheme,
        w_granularity,
        None,
        act_granularity,
    )


@pytest.fixture(
    params=[
        (weight_dtype, act_dtype, qscheme, w_granularity, model_dtype, act_granularity)
        for weight_dtype in [
            torch.int8,
            torch.uint8,
            torch.int4,
            torch.uint4,
        ]
        for act_dtype in [torch.int8, torch.uint8, None]
        for qscheme in list(QuantizationScheme)
        for w_granularity in [
            PerTensorGranularity(),
            PerChannelGranularity(axis=1),
            PerBlockGranularity(axis=0, block_size=2),
        ]
        for model_dtype in [
            torch.float16,
            torch.float32,
            torch.bfloat16,
        ]
        for act_granularity in [
            PerTensorGranularity(),
            PerChannelGranularity(axis=0),
            PerChannelGranularity(axis=-1),
        ]
        # Weight-only configs (act_dtype=None) produce identical results regardless of
        # act_granularity. Only include 1 combination (with PerTensorGranularity) for
        # weight-only to avoid running redundant identical tests across all
        # act_granularity values.
        if act_dtype is not None or isinstance(act_granularity, PerTensorGranularity)
    ],
    ids=lambda p: (
        f"wt:{str(p[0]).split('.')[-1]}--"
        f"act:{str(p[1]).split('.')[-1] if p[1] else 'disabled'}--"
        f"qs:{p[2].value}--"
        f"wg:{p[3].__class__.__name__.replace('Granularity', '')}--"
        f"m_dtype:{str(p[4]).split('.')[-1]}--"
        f"ag:{p[5].__class__.__name__.replace('Granularity', '')}--"
        f"axis:{p[5].axis}"
    ),
)
def parametrized_quant_config_mlir(
    request: pytest.FixtureRequest,
) -> ParametrizedQuantConfigs:
    """Fixture for MLIR backend quantization configs.

    MLIR backend supports multiple model dtypes.
    Generates 756 parameter combinations.
    Weight-only configs use only PerTensorGranularity for act_granularity.

    Returns:
        ParametrizedQuantConfigs with model_dtype varying across
        float16/float32/bfloat16

    """
    weight_dtype, act_dtype, qscheme, w_granularity, model_dtype, act_granularity = request.param
    return ParametrizedQuantConfigs.from_quant_params(
        weight_dtype,
        act_dtype,
        qscheme,
        w_granularity,
        model_dtype,
        act_granularity,
    )


@pytest.fixture(
    params=[
        (qscheme, act_granularity)
        for qscheme in list(QuantizationScheme)
        for act_granularity in [
            PerTensorGranularity(),
            PerChannelGranularity(axis=0),
            PerChannelGranularity(axis=1),
            PerChannelGranularity(axis=2),
            PerChannelGranularity(axis=-1),
            PerChannelGranularity(axis=-2),
            PerChannelGranularity(axis=-3),
        ]
    ],
    ids=lambda p: (
        f"qs:{p[0].value}--"
        f"ag:{p[1].__class__.__name__.replace('Granularity', '')}--"
        f"axis:{p[1].axis}"
    ),
)
def parametrized_quant_config_perchannel_act_axis_coverage(
    request: pytest.FixtureRequest,
) -> ParametrizedQuantConfigs:
    """Fixture for per-channel activation quantization axis testing.

    Uses fixed values for weight dtype (int8), activation dtype (uint8),
    weight granularity (PerTensor), and model dtype (None) to isolate
    per-channel activation axis behavior.
    Compatible with both CoreML and CoreAI backends. Intended for use with
    GatedMLPModel which has uniform rank-3 activations supporting all
    axes in [-3, 3).

    Generates 21 parameter combinations (3 qschemes x 7 act granularities).

    Returns:
        ParametrizedQuantConfigs with varied activation granularity axes

    """
    qscheme, act_granularity = request.param
    return ParametrizedQuantConfigs.from_quant_params(
        torch.int8,
        torch.uint8,
        qscheme,
        PerTensorGranularity(),
        None,
        act_granularity,
    )


@pytest.fixture(
    params=[
        (n_bits, granularity, enable_per_channel_scale, cluster_dim, lut_qspec)
        for n_bits in [1, 2, 4]
        for granularity in [
            PalettizationPerTensorGranularity(),
            PerGroupedChannelGranularity(axis=0, group_size=2),
            PerGroupedChannelGranularity(axis=1, group_size=2),
        ]
        for enable_per_channel_scale in [True, False]
        for cluster_dim in [1, 2]
        for lut_qspec in [
            None,
            *(
                QuantizationSpec(
                    dtype=dtype,
                    qscheme=QuantizationScheme.SYMMETRIC,
                )
                for dtype in sorted(_SUPPORTED_LUT_DTYPES, key=str)
            ),
        ]
        # cluster_dim=2 (vector palettization) is slow; only test with n_bits=4
        if cluster_dim == 1 or n_bits == 4
    ],
    ids=lambda p: (
        f"n_bits:{p[0]}-"
        f"granularity:{p[1].__class__.__name__.replace('Granularity', '')}"
        + (
            f"_axis{p[1].axis}_gs{p[1].group_size}"
            if isinstance(p[1], PerGroupedChannelGranularity)
            else ""
        )
        + f"-pcs:{'enabled' if p[2] else 'disabled'}"
        + (f"-cd:{p[3]}" if p[3] > 1 else "")
        + (f"-lut:{p[4].dtype}" if p[4] is not None else "")
    ),
)
def parametrized_palett_config(
    request: pytest.FixtureRequest,
) -> ParametrizedPalettConfigs:
    """Fixture for palettization configs.

    Generates parameter combinations across:
    - 3 n_bits values: [1, 2, 4]
    - 3 granularities: [PerTensor, PerGroupedChannel(axis=0), PerGroupedChannel(axis=1)]
    - 2 enable_per_channel_scale values: [True, False]
    - 2 cluster_dim values: [1, 2]
    - N+1 lut_qspec values: [None, + one symmetric spec per dtype in _SUPPORTED_LUT_DTYPES]

    cluster_dim=2 (vector palettization) is only combined with n_bits=4 to reduce
    test runtime.

    Returns:
        ParametrizedPalettConfigs instance

    """
    n_bits, granularity, enable_per_channel_scale, cluster_dim, lut_qspec = request.param
    return ParametrizedPalettConfigs.from_palett_params(
        n_bits,
        granularity,
        enable_per_channel_scale,
        cluster_dim,
        lut_qspec,
    )


@pytest.fixture(
    params=[
        pytest.param(
            (torch.float8_e4m3fn, True, torch.float32, True, -1),
            id="wt:float8_e4m3fn-act:float8_e4m3fn-qs:symmetric-wg:PerTensor-ag:PerChannel-axis:-1",
        ),
        pytest.param(
            (torch.float8_e4m3fn, False, torch.float32, False, 0),
            id="wt:float8_e4m3fn-act:disabled-qs:symmetric-wg:PerTensor",
        ),
        pytest.param(
            (torch.float8_e4m3fn, False, torch.float16, False, 0),
            id="wt:float8_e4m3fn-act:disabled-qs:symmetric-wg:PerTensor-m_dtype:float16",
        ),
        pytest.param(
            (torch.float8_e4m3fn, True, torch.float32, False, 0),
            id="wt:float8_e4m3fn-act:float8_e4m3fn-qs:symmetric-wg:PerTensor",
        ),
        pytest.param(
            (torch.float8_e5m2, False, torch.float32, False, 0),
            id="wt:float8_e5m2-act:disabled-qs:symmetric-wg:PerTensor",
        ),
        pytest.param(
            (torch.float8_e5m2, False, torch.float16, False, 0),
            id="wt:float8_e5m2-act:disabled-qs:symmetric-wg:PerTensor-m_dtype:float16",
        ),
        pytest.param(
            (torch.float8_e5m2, True, torch.float32, False, 0),
            id="wt:float8_e5m2-act:float8_e5m2-qs:symmetric-wg:PerTensor",
        ),
    ],
)
def parametrized_fp8_config(
    request: pytest.FixtureRequest,
) -> ParametrizedFP8Configs:
    """Fixture for FP8 quantization configs.

    Generates 7 parameter combinations:
    - 2 FP8 dtypes: [float8_e4m3fn, float8_e5m2]
    - 2 activation quantization modes: [False (weight-only), True (with activation)]
    - Weight-only configs also include float16 model dtype to verify scale casting
    - a per channel activation quantization configs with axis=-1

    All combinations are marked as xfail pending COREAI updates and output verification.

    Returns:
        ParametrizedFP8Configs instance

    """
    (
        fp8_dtype,
        with_activation_quant,
        model_dtype,
        per_channel_activations,
        per_channel_activations_axis,
    ) = request.param

    return ParametrizedFP8Configs.from_fp8_params(
        fp8_dtype,
        with_activation_quant,
        model_dtype,
        per_channel_activations,
        per_channel_activations_axis,
    )


@dataclass
class ParametrizedP4A8CompressionConfigs:
    """Container for parametrized P4-A8 compression (palettization + quantization) configs.

    Attributes:
        palett_config (KMeansPalettizerConfig): Palettization configuration.
        quant_config (QuantizerConfig): Activation quantization configuration.
        has_lut_quantization (bool): Whether LUT quantization is enabled.

    """

    palett_config: KMeansPalettizerConfig
    quant_config: QuantizerConfig
    has_lut_quantization: bool

    @classmethod
    def from_params(
        cls,
        lut_qspec: QuantizationSpec | None = None,
    ) -> "ParametrizedP4A8CompressionConfigs":
        """Create config pair for P4-A8 joint compression.

        Palettization: 4-bit, per-tensor granularity.
        Activation quantization: int8 symmetric per-tensor (input + output).
        Weight quantization: disabled (weights are palettized).

        Args:
            lut_qspec (QuantizationSpec | None): LUT quantization spec.
                None for unquantized LUT, or a QuantizationSpec for quantized LUT.

        Returns:
            ParametrizedP4A8CompressionConfigs: Config pair.

        """
        palett_spec = PalettizationSpec(
            n_bits=4,
            lut_qspec=lut_qspec,
        )
        palett_config = KMeansPalettizerConfig(
            global_config=ModuleKMeansPalettizerConfig(
                op_state_spec={"weight": palett_spec},
            ),
        )

        act_spec = QuantizationSpec(
            dtype=torch.int8,
            qscheme=QuantizationScheme.SYMMETRIC,
        )
        quant_config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec=None,
                op_input_spec={"*": act_spec},
                op_output_spec={"*": act_spec},
            ),
        )

        return cls(
            palett_config=palett_config,
            quant_config=quant_config,
            has_lut_quantization=lut_qspec is not None,
        )


@pytest.fixture(
    params=[
        pytest.param(
            QuantizationSpec(
                dtype=torch.int8,
                qscheme=QuantizationScheme.SYMMETRIC,
            ),
            id="P4-A8-int8lut",
        ),
        pytest.param(None, id="P4-A8-nolut"),
    ],
)
def parametrized_p4a8_compression_config(
    request: pytest.FixtureRequest,
) -> ParametrizedP4A8CompressionConfigs:
    """Fixture for P4-A8 compression (palettization + activation quantization) configs.

    Generates 2 parameter combinations:
    - P4-A8-int8lut: 4-bit palettization with int8 symmetric LUT quantization
    - P4-A8-nolut: 4-bit palettization without LUT quantization

    Both use int8 symmetric per-tensor activation quantization.

    Returns:
        ParametrizedP4A8CompressionConfigs: P4-A8 compression config pair.

    """
    return ParametrizedP4A8CompressionConfigs.from_params(lut_qspec=request.param)


@pytest.fixture(
    params=[
        pytest.param(
            ("float4_e2m1fn", False, None, torch.float16, True, 32, False, 32),
            id="wt:float4_e2m1fn-act:disabled-wg:PerBlock-wbs:32",
            # TODO: handle float4 export with torch >=2.8.
            marks=pytest.mark.xfail(
                reason="Requires fix to handle float4 export with torch >=2.8."
            ),
        ),
        pytest.param(
            ("float4_e2m1fn", True, "float8_e4m3fn", torch.float16, True, 32, False, 32),
            id="wt:float4_e2m1fn-act:float8_e4m3fn-wg:PerBlock-wbs:32-ag:PerTensor",
            # TODO: handle float4 export with torch >=2.8.
            marks=pytest.mark.xfail(
                reason="Requires fix to handle float4 export with torch >=2.8."
            ),
        ),
    ],
)
def parametrized_fp4_config(
    request: pytest.FixtureRequest,
) -> ParametrizedFP4Configs:
    """
    Fixture for FP4 quantization configs.

    Testing following combinations for weight and activation quantization:
    - Weight Quantization dtype: torch.float4_e2m1fn_x2
    - Activation Quantization dtype: {torch.float4_e2m1fn_x2, torch.float8_e4m3fn}
    - Weight quantization torch.float4_e2m1fn_x2: MLIR export only supported with
    per-block granularity and block_size=32
    - Activation quantization torch.float4_e2m1fn_x2: MLIR export not supported

    Returns:
        ParametrizedFP4Configs instance
    """
    (
        weight_dtype,
        with_activation_quant,
        activation_dtype,
        model_dtype,
        per_block_weights,
        weight_block_size,
        per_block_activations,
        activation_block_size,
    ) = request.param

    return ParametrizedFP4Configs.from_fp4_params(
        with_activation_quant,
        model_dtype,
        weight_dtype,
        per_block_weights,
        weight_block_size,
        activation_dtype,
        per_block_activations,
        activation_block_size,
    )


# ─── Pruning parametrized configs ────────────────────────────────────────────


def _get_pruning_schemes() -> list:
    return [Unstructured(), ChannelStructured(axis=0)]


@dataclass
class ParametrizedPruneConfigs:
    """Container for parametrized pruning configs.

    Attributes:
        config: MagnitudePrunerConfig instance.
        target_sparsity: Target sparsity fraction.
        pruning_scheme: PruningScheme instance (Unstructured or ChannelStructured).
        backend: Export backend (CoreML or CoreAI).
    """

    config: MagnitudePrunerConfig
    target_sparsity: float
    pruning_scheme: PruningScheme | str
    backend: ExportBackend

    @classmethod
    def from_prune_params(
        cls,
        target_sparsity: float,
        pruning_scheme: PruningScheme | str,
        backend: ExportBackend,
    ) -> "ParametrizedPruneConfigs":
        spec = PruningSpec(target_sparsity=target_sparsity, pruning_scheme=pruning_scheme)
        config = MagnitudePrunerConfig(
            global_config=ModuleMagnitudePrunerConfig(op_state_spec={"weight": spec})
        )
        return cls(
            config=config,
            target_sparsity=target_sparsity,
            pruning_scheme=pruning_scheme,
            backend=backend,
        )


@pytest.fixture(
    params=[
        (target_sparsity, pruning_scheme, backend)
        for target_sparsity in [0.25, 0.5, 0.75]
        for pruning_scheme in _get_pruning_schemes()
        for backend in [ExportBackend.CoreML, ExportBackend.CoreAI]
    ],
    ids=lambda p: f"sparsity:{p[0]}-scheme:{p[1].__class__.__name__}-backend:{p[2].value}",
)
def parametrized_prune_config(
    request: pytest.FixtureRequest,
) -> ParametrizedPruneConfigs:
    """Fixture for pruning configs parametrized across sparsity, scheme, and backend."""
    target_sparsity, pruning_scheme, backend = request.param
    return ParametrizedPruneConfigs.from_prune_params(target_sparsity, pruning_scheme, backend)
