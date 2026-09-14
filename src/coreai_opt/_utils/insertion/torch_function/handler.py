# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Handler for eager mode compression using torch function interception."""

from collections.abc import Mapping
from typing import Any

import torch.nn as nn
import torch.nn.utils.parametrize as P

from coreai_opt._utils.torch_utils import NamedModule
from coreai_opt.config import CompressionConfig
from coreai_opt.config.spec import CompressionSimulatorBase

from .base_supported_ops_registry import BaseSupportedOpsRegistry
from .modes import (
    ActivationEagerOptimizationHandler,
    RegisterEagerOptimizationMode,
)
from .types import ModuleCompressionComponents


def _record_tensor_fqns(model: nn.Module) -> None:
    """Record the FQN of each parametrized tensor on the simulator compressing it."""
    for module_name, module in model.named_modules():
        if not P.is_parametrized(module):
            continue
        for param_name, parametrizations in module.parametrizations.items():
            fqn = f"{module_name}.{param_name}" if module_name else param_name
            for simulator in parametrizations:
                if isinstance(simulator, CompressionSimulatorBase):
                    simulator.tensor_fqn = fqn


class TorchFunctionEagerHandler:
    """
    Prepares the model for compression by inserting weight and activation
    optimizers using `__torch_function__` mode.

    This is a generic handler that works for any compression technique
    (quantization, palettization, etc.).
    """

    def __init__(
        self,
        compression_config: CompressionConfig,
        module_components_dict: Mapping[NamedModule, ModuleCompressionComponents],
        module_priority_dict: Mapping[str, int],
        supported_ops_registry: type[BaseSupportedOpsRegistry],
        optimization_type_name: str = "optimize",
    ):
        self.compression_config = compression_config
        self.module_components_dict = module_components_dict
        self.module_priority_dict = module_priority_dict
        self.supported_ops_registry = supported_ops_registry
        self.optimization_type_name = optimization_type_name
        # Will be ActivationEagerOptimizationHandler
        self.act_handler: Any | None = None

    def prepare(self, model: nn.Module, example_inputs: tuple[Any, ...]) -> nn.Module:
        """
        Performs forward pass on model to configure activation and weight optimization
        layers.
        """
        # Register optimizers on weights and activations
        with RegisterEagerOptimizationMode(
            model,
            self.compression_config,
            self.module_components_dict,
            self.module_priority_dict,
            self.supported_ops_registry,
            self.optimization_type_name,
        ) as register_optimization_mode:
            model(*example_inputs)

        # Parametrization of states is done as a separate step after the initial forward
        # pass of the model to process all inputs/outputs/states.
        # This is because when duplicate modules with states are in play, parametrizing
        # the first occurrence of the module leads to q/dq add, sub, and mul operations
        # being captured as function inputs and outputs to quantize when the module is
        # seen again later in the forward pass.
        register_optimization_mode.register_all_activations()
        register_optimization_mode.register_all_states()

        _record_tensor_fqns(model)

        if self._is_weight_only_optimization(self.module_components_dict):
            return model

        # Init manager for optimization of inputs/activations
        self.act_handler = ActivationEagerOptimizationHandler(
            model,
            compression_config=self.compression_config,
            module_components_dict=self.module_components_dict,
            supported_ops_registry=self.supported_ops_registry,
            optimization_type_name=self.optimization_type_name,
            reference_tracker=register_optimization_mode.registered_optimizers_tracker,
        )
        return model

    def remove_activation_hooks(self) -> None:
        """
        Removes the hooks installed by the activation handler.
        """
        if self.act_handler:
            self.act_handler.remove_hooks()
            self.act_handler = None

    def _is_weight_only_optimization(self, module_components_dict):
        """
        Return True if only weight optimizations have been configured
        """
        for module_compression_component in module_components_dict.values():
            if module_compression_component.has_activation_component():
                return False

        return True
