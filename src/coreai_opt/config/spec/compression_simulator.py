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

    # Recorded during prepare() so that forward-time diagnostics can name the
    # tensor. Class-level defaults keep this out of __init__, and out of
    # state_dict, without touching subclass constructor chains.
    _source_module_name: str | None = None
    _source_param_name: str | None = None

    def set_source_name(self, module_name: str, param_name: str | None = None) -> None:
        """Record which model tensor this simulator compresses.

        The first name recorded wins, so simulators shared by several modules
        report a stable name.

        Args:
            module_name: Owning module's name in ``named_modules()``, or ``""``
                for a root-module parameter.
            param_name: Local parameter name (e.g. ``"weight"``), or ``None``.
        """
        if self._source_module_name is not None or self._source_param_name is not None:
            return
        self._source_module_name = module_name
        self._source_param_name = param_name

    @property
    def source_name(self) -> str:
        """FQN of the compressed tensor, or ``"<unknown>"`` if never recorded."""
        parts = [part for part in (self._source_module_name, self._source_param_name) if part]
        return ".".join(parts) if parts else "<unknown>"

    @property
    def source_module_name(self) -> str | None:
        """Owning module's FQN, usable as a ``module_name_configs`` key.

        ``None`` if never recorded, or for a root-module tensor that no module
        name addresses.
        """
        return self._source_module_name or None

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
