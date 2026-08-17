# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Type definitions and data structures for eager mode compression."""

from __future__ import annotations

from collections import namedtuple
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import torch

from coreai_opt._utils.spec_utils import PartialConstructor
from coreai_opt.config.spec import CompressionSimulatorBase


@dataclass(frozen=True)
class OpCompressionComponents:
    """
    Data structure to hold compression components for an op.

    Each field can hold a dictionary mapping an activation/state tensor to either:
    - A PartialConstructor for deferred construction
    - None if no compression is applied to that component
    """

    op_input_components: Mapping[int | str, PartialConstructor[CompressionSimulatorBase] | None] = (
        field(default_factory=dict)
    )

    op_output_components: Mapping[
        int | str, PartialConstructor[CompressionSimulatorBase] | None
    ] = field(default_factory=dict)

    op_state_components: Mapping[str, PartialConstructor[CompressionSimulatorBase] | None] = field(
        default_factory=dict
    )

    def has_any_component(self) -> bool:
        """
        Check if any compression components are set.

        Returns:
            True if any property is set, False if all properties are unset.
        """
        return bool(
            self.op_input_components or self.op_output_components or self.op_state_components
        )

    def has_activation_component(self) -> bool:
        """
        Check if any activation compression components are set.

        Returns:
            True if any activation is set, False if all activations are unset.
        """
        return bool(self.op_input_components or self.op_output_components)


@dataclass(frozen=True)
class ModuleCompressionComponents:
    """
    Data structure to hold compression components for a module.

    Each field in weight, input_activation, and output_activation can hold a dictionary
    mapping an activation/state tensor to either:
    - A PartialConstructor for deferred construction
    - None if no compression is applied to that component

    Each field in op_type_components and op_name_components can hold a dictionary mapping an op type
    or name to an OpCompressionComponents class.
    """

    weight: Mapping[str, PartialConstructor[CompressionSimulatorBase] | None] = field(
        default_factory=dict
    )

    input_activation: Mapping[int | str, PartialConstructor[CompressionSimulatorBase] | None] = (
        field(default_factory=dict)
    )

    output_activation: Mapping[int | str, PartialConstructor[CompressionSimulatorBase] | None] = (
        field(default_factory=dict)
    )

    op_type_components: Mapping[str, OpCompressionComponents] = field(default_factory=dict)

    op_name_components: Mapping[str, OpCompressionComponents] = field(default_factory=dict)

    module_input_components: Mapping[
        int | str, PartialConstructor[CompressionSimulatorBase] | None
    ] = field(default_factory=dict)

    module_output_components: Mapping[
        int | str, PartialConstructor[CompressionSimulatorBase] | None
    ] = field(default_factory=dict)

    module_state_components: Mapping[str, PartialConstructor[CompressionSimulatorBase] | None] = (
        field(default_factory=dict)
    )

    def has_any_component(self) -> bool:
        """
        Check if any compression components are set.

        A None value in a component dict is meaningful — it explicitly disables
        compression for that tensor. So a non-empty dict counts as having a component
        even if all values are None. For op_type/op_name components, we drill into
        the OpCompressionComponents to check if they have real components.

        Returns:
            True if any property or op components are set, False if all properties are unset.
        """
        return bool(
            self.weight
            or self.input_activation
            or self.output_activation
            or self.module_input_components
            or self.module_output_components
            or self.module_state_components
            or any(op_comp.has_any_component() for op_comp in self.op_type_components.values())
            or any(op_comp.has_any_component() for op_comp in self.op_name_components.values())
        )

    def has_activation_component(self) -> bool:
        """
        Check if any activation compression components are set.

        Returns:
            True if any activations or op activation components are set, False otherwise.
        """
        return bool(
            self.input_activation
            or self.output_activation
            or self.module_input_components
            or self.module_output_components
            or any(
                op_comp.has_activation_component() for op_comp in self.op_type_components.values()
            )
            or any(
                op_comp.has_activation_component() for op_comp in self.op_name_components.values()
            )
        )

    def has_module_level_component(self) -> bool:
        """
        Check if any module level compression components are set.

        Returns:
            True if any module level components are set, False otherwise.
        """
        return bool(
            self.module_input_components
            or self.module_output_components
            or self.module_state_components
        )


@dataclass(frozen=True)
class PendingOptimizerRegistration:
    """
    Represents an optimizer that needs to be registered after module boundary analysis.

    During preregistration, we know the optimizer name and possibly optimizer from op level specs,
    but the final optimizer might be overridden based on module-level input/output
    specs.
    """

    module: torch.nn.Module
    tensor_counter: int
    optimizer_name: str
    optimizer: CompressionSimulatorBase | None = None

    def with_optimizer(
        self, optimizer: CompressionSimulatorBase | None
    ) -> PendingOptimizerRegistration:
        """Create a new registration with an optimizer assigned."""
        return PendingOptimizerRegistration(
            self.module, self.tensor_counter, self.optimizer_name, optimizer
        )


@dataclass(frozen=True)
class FunctionPreregistrationRecord:
    """
    Record of pending optimizer registrations for one function call.

    This represents what we know during the initial forward pass, before
    module boundaries are fully analyzed.
    """

    function: Callable
    pending_inputs: list[PendingOptimizerRegistration]
    pending_outputs: list[PendingOptimizerRegistration]


@dataclass(frozen=True)
class FunctionRegisteredOptimizers:
    """
    Record of optimizer names registered for one function call.

    This represents what was registered after resolving module-level specs.
    Used during optimization phase to validate consistency.
    """

    input_optimizer_names: list[str]
    output_optimizer_names: list[str]


ActHandlerOutput = namedtuple(
    "ActHandlerOutput",
    ["args", "kwargs", "name_prefix", "args_to_optimize", "kwargs_to_optimize"],
)
