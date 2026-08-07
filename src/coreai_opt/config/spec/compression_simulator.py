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

    # Model-level identity of the tensor this simulator compresses, recorded
    # during prepare() so that forward-time diagnostics can name the tensor.
    # These are class-level defaults rather than __init__ assignments so that
    # subclass constructor chains stay untouched. Being plain strings, they are
    # kept out of state_dict by nn.Module.__setattr__.
    _source_module_name: str | None = None
    _source_param_name: str | None = None

    def set_source_name(self, module_name: str, param_name: str | None = None) -> None:
        """Record which model tensor this simulator compresses.

        A simulator cannot discover its own position in the model from inside
        ``forward``, so callers record it during ``prepare()`` instead. When one
        simulator is shared by several modules, the first name recorded wins, so
        that the reported name is stable across runs.

        Args:
            module_name: Fully-qualified name of the owning module as it appears
                in ``named_modules()``, or ``""`` for a root-module parameter.
            param_name: Local parameter name (e.g. ``"weight"``). ``None`` when
                the simulator does not act on a named parameter.
        """
        if self._source_module_name is not None or self._source_param_name is not None:
            return
        self._source_module_name = module_name
        self._source_param_name = param_name

    @property
    def source_name(self) -> str:
        """FQN of the compressed tensor (e.g. ``"layers.0.q_proj.weight"``).

        Returns ``"<unknown>"`` when no name was recorded, which is the case for
        activations and for any simulator created outside a ``prepare()`` pass.
        """
        parts = [part for part in (self._source_module_name, self._source_param_name) if part]
        return ".".join(parts) if parts else "<unknown>"

    @property
    def source_module_name(self) -> str | None:
        """FQN of the owning module, suitable as a ``module_name_configs`` key.

        ``None`` when no name was recorded or when the tensor belongs to the
        root module (which cannot be addressed by module name).
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
