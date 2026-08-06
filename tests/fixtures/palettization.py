# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Palettization parametrization config and the fixture that provides it."""

from dataclasses import dataclass

import pytest

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
from coreai_opt.quantization.spec import QuantizationScheme, QuantizationSpec


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
