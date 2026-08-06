# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""FP8 quantization parametrization config and the fixture that provides it."""

from dataclasses import dataclass

import pytest
import torch

from coreai_opt.quantization import ModuleQuantizerConfig, QuantizerConfig
from coreai_opt.quantization.spec import (
    PerChannelGranularity,
    PerTensorGranularity,
    QuantizationScheme,
    QuantizationSpec,
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
