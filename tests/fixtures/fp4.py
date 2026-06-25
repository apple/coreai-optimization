# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""FP4 quantization parametrization config and the fixture that provides it."""

from dataclasses import dataclass

import pytest
import torch

from coreai_opt.quantization import ModuleQuantizerConfig, QuantizerConfig
from coreai_opt.quantization.spec import (
    PerBlockGranularity,
    PerTensorGranularity,
    QuantizationScheme,
    QuantizationSpec,
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
        pytest.param(
            ("float4_e2m1fn", False, None, torch.float16, True, 32, False, 32),
            id="wt:float4_e2m1fn-act:disabled-wg:PerBlock-wbs:32",
        ),
        pytest.param(
            ("float4_e2m1fn", True, "float8_e4m3fn", torch.float16, True, 32, False, 32),
            id="wt:float4_e2m1fn-act:float8_e4m3fn-wg:PerBlock-wbs:32-ag:PerTensor",
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
