# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""P4-A8 compression parametrization config and the fixture that provides it."""

from dataclasses import dataclass

import pytest
import torch

from coreai_opt.palettization import (
    KMeansPalettizerConfig,
    ModuleKMeansPalettizerConfig,
)
from coreai_opt.palettization.spec import PalettizationSpec
from coreai_opt.quantization import ModuleQuantizerConfig, QuantizerConfig
from coreai_opt.quantization.spec import QuantizationScheme, QuantizationSpec


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
