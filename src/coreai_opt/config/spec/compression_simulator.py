# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Base class for differentiable compression simulators."""

from abc import abstractmethod

import torch
import torch.nn as nn

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
