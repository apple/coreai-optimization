# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping, Set
from dataclasses import dataclass

import torch.fx
from torchao.quantization.pt2e.quantizer import (
    QuantizationSpec as TorchAOQuantizationSpec,
)

from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.quantization.config import OpQuantizerConfig
from coreai_opt.quantization.spec import QuantizationComponentFactory, QuantizationSpec


@dataclass(frozen=True)
class AnnotationContext:
    """Pass-invariant inputs an annotator may need.

    Held constant across all matches in a single annotation pass. Constructed
    once when ``_AnnotationHandler.annotate`` begins and shared by every
    annotator invocation during that pass.

    Distinct from :class:`AnnotationConfig`, which carries per-op specs that
    vary per match.

    Attributes:
        module_name_to_state_names_map (Mapping[str, Mapping[str, list[str]]]):
            For each module name, a mapping from each state target (FQN) to the
            list of local names the module uses for that state. Used during
            state-input annotation to translate a state node's target into the
            consumer module's local name(s).
        shared_observer_nodes (Set[torch.fx.Node]): Nodes whose output annotations
            are shared with their input annotations if any.
    """

    module_name_to_state_names_map: Mapping[str, Mapping[str, list[str]]]
    shared_observer_nodes: Set[torch.fx.Node]


class AnnotationConfig:
    """
    Configuration class for PT2E quantization annotations using TorchAO QuantizationSpec

    This class has the same structure as OpQuantizerConfig but uses
    torchao.quantization.pt2e.quantizer.QuantizationSpec instead of coreai_opt's
    QuantizationSpec.
    """

    def __init__(
        self,
        op_input_spec: dict[str | int, TorchAOQuantizationSpec | None],
        op_output_spec: dict[str | int, TorchAOQuantizationSpec | None],
        op_state_spec: dict[str, TorchAOQuantizationSpec | None],
    ):
        """
        Initialize AnnotationConfig.

        Args:
            op_input_spec: Quantization spec for input activations
            op_output_spec: Quantization spec for output activations
            op_state_spec: Quantization spec for states
        """
        self.op_input_spec = op_input_spec
        self.op_output_spec = op_output_spec
        self.op_state_spec = op_state_spec

    @classmethod
    def from_quantizer_config(cls, config: OpQuantizerConfig) -> AnnotationConfig:
        """
        Initialize AnnotationConfig from OpQuantizerConfig.

        Args:
            config: The OpQuantizerConfig to convert from

        Returns:
            AnnotationConfig instance with converted specifications.
        """
        return cls(
            op_input_spec={
                tensor: AnnotationConfig._convert_to_pt2e_spec(
                    spec,
                    CompressionTargetTensor.ACTIVATION,
                )
                for tensor, spec in config.op_input_spec.items()
            },
            op_output_spec={
                tensor: AnnotationConfig._convert_to_pt2e_spec(
                    spec,
                    CompressionTargetTensor.ACTIVATION,
                )
                for tensor, spec in config.op_output_spec.items()
            },
            op_state_spec={
                tensor: AnnotationConfig._convert_to_pt2e_spec(
                    spec,
                    CompressionTargetTensor.WEIGHT,
                )
                for tensor, spec in config.op_state_spec.items()
            },
        )

    @staticmethod
    def _convert_to_pt2e_spec(
        spec: QuantizationSpec | None, quantization_target: CompressionTargetTensor
    ) -> TorchAOQuantizationSpec | None:
        """
        Convert coreai_opt QuantizationSpec to TorchAO PT2E QuantizationSpec.
        """
        if spec is None:
            return None

        # Create fake quantizer partial
        fq_partial = QuantizationComponentFactory.construct_partial(
            spec=spec,
            target=quantization_target,
        )

        return TorchAOQuantizationSpec(
            dtype=spec.dtype,
            observer_or_fake_quant_ctr=fq_partial,
            # Note: quant_min and quant_max can be float values for floating-point
            # dtypes (e.g., FP4, FP8), even though TorchAO's QuantizationSpec type
            # annotation only allows int. This works at runtime since Python ignores
            # type hints during execution.
            quant_min=spec.quant_min,
            quant_max=spec.quant_max,
        )
