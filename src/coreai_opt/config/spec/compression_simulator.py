# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Base class for differentiable compression simulators."""

from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P

from coreai_opt._utils.registry_utils import ClassRegistryMixin as _ClassRegistryMixin


class CompressionSimulatorBase(_ClassRegistryMixin, nn.Module):
    """
    Abstract base class for compression simulators.

    This base class provides a common interface for all compression
    simulators, regardless of the specific compression technique. The
    compression simulator takes a tensor and applies the compression
    technique on the tensor, while allowing the model to be evaluated.

    Subclasses should implement the forward() method to define how the
    compression simulation is performed during training.
    """

    # FQN of the compressed tensor
    tensor_fqn: str = "<unknown>"

    @abstractmethod
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply compression simulation to the input tensor.

        This method should implement the differentiable approximation of
        the compression operation. The exact behavior depends on the
        specific compression technique.

        Args:
            tensor: Input tensor to compress

        Returns:
            Compressed tensor (or approximation thereof) with gradients
            flowing through
        """
        pass


def _record_tensor_fqns_eager(model: nn.Module) -> None:
    """Record the FQN of each parametrized tensor on the simulator compressing it."""
    for module_name, module in model.named_modules():
        if not P.is_parametrized(module):
            continue
        for param_name, parametrizations in module.parametrizations.items():
            name = f"{module_name}.{param_name}" if module_name else param_name
            for simulator in parametrizations:
                if isinstance(simulator, CompressionSimulatorBase):
                    simulator.tensor_fqn = name


def _record_tensor_fqns_graph(model: torch.fx.GraphModule) -> None:
    """Record the FQN of each compressed parameter on the simulator compressing it.

    A simulator reading from anything other than a ``get_attr`` node keeps the
    default name: an activation, or a weight arriving through a decompression op,
    has no parameter to name.
    """
    simulators = dict(model.named_modules(remove_duplicate=False))
    for node in model.graph.nodes:
        if node.op != "call_module" or not node.args:
            continue
        simulator = simulators.get(str(node.target))
        if isinstance(simulator, CompressionSimulatorBase) and node.args[0].op == "get_attr":
            simulator.tensor_fqn = str(node.args[0].target)
