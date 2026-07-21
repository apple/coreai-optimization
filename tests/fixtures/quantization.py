# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Quantization parametrization config and the fixtures that provide it."""

from dataclasses import dataclass
from typing import Any, Literal

import pytest
import torch

from coreai_opt import ExportBackend
from coreai_opt.quantization import ModuleQuantizerConfig, QuantizerConfig
from coreai_opt.quantization.spec import (
    PerBlockGranularity,
    PerChannelGranularity,
    PerTensorGranularity,
    QuantizationScheme,
    QuantizationSpec,
)
from coreai_opt.quantization.spec.fake_quantize import _DefaultFakeQuantizeImpl
from coreai_opt.quantization.spec.qparams_calculator import StaticQParamsCalculator
from coreai_opt.quantization.spec.range_calculator import MinMaxRangeCalculator

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

    @property
    def has_per_channel_activation_granularity(self) -> bool:
        """Check if activation quantization in this config uses PerChannelGranularity.

        Returns:
            True if activation quantization is enabled and per-channel

        """
        if not self.eager.global_config:
            return False
        act_qspec = self.eager.global_config.op_input_spec.get("*")
        return act_qspec is not None and isinstance(act_qspec.granularity, PerChannelGranularity)

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
            if "backend" in constraints and backend != constraints["backend"]:
                continue
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
