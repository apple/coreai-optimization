# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Base model compression framework.

This module defines the abstract base class for implementing model compression
techniques such as quantization.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import contextmanager
from enum import Enum
from os import PathLike

import torch

from coreai_opt.common import ExportBackend
from coreai_opt.config import CompressionConfig

_COREAI_OPT_PREPARED_ATTR = "_coreai_opt_prepared"


class _CompressorLifecycle(Enum):
    """Lifecycle state of a model compressor."""

    IDLE = "idle"
    TRAINING = "training"
    CALIBRATING = "calibrating"


class _BaseModelCompressor(ABC):
    """
    An abstract base class for implementing model compression techniques.
    """

    _supported_modules: tuple[type[torch.nn.Module]]
    _lifecycle: _CompressorLifecycle

    def __init__(self, model: torch.nn.Module, config: CompressionConfig | None = None):
        """
        Initialize the model compressor.

        Args:
            model: The PyTorch model to compress. The model will be modified in-place
                  during the compression process.
            config: Configuration parameters for the compression
        """
        self._model = model
        self._config = config
        self._lifecycle = _CompressorLifecycle.IDLE

    @staticmethod
    def _is_model_prepared(model: torch.nn.Module) -> bool:
        """
        Check if a model has been prepared for compression.

        Args:
            model: The model to check.

        Returns:
            True if the model has _COREAI_OPT_PREPARED_ATTR set to a truthy value,
            False otherwise.
        """
        return bool(getattr(model, _COREAI_OPT_PREPARED_ATTR, False))

    @staticmethod
    def _mark_model_as_prepared(model: torch.nn.Module) -> None:
        """
        Mark a model as prepared for compression.

        Registers ``_COREAI_OPT_PREPARED_ATTR`` as a non-persistent buffer so the
        marker survives operations that rebuild the module (e.g. ``deepcopy`` of a
        ``torch.fx.GraphModule``) while staying out of ``state_dict()``.

        Args:
            model: The model to mark as prepared.
        """
        model.register_buffer(_COREAI_OPT_PREPARED_ATTR, torch.tensor(True), persistent=False)

    @classmethod
    def supported_modules(cls) -> tuple[type[torch.nn.Module]]:
        """
        Returns types of modules that are supported for compression with
        for a particular model optimization technique.

        Returns:
            Tuple of PyTorch module classes that can be compressed,
            eg. (torch.nn.Conv2d, torch.nn.Linear)
        """
        return cls._supported_modules

    @abstractmethod
    def prepare(self, *args, **kwargs) -> torch.nn.Module:
        """
        Prepare the model for compression by inserting modules, registering
        parametrizations, or adding hooks to the model. This method performs all setup
        steps common to any compression workflow (data-free, calibration-based,
        or training-based).

        Additionally, this method performs data-free model compression where applicable.
        The returned prepared model can be evaluated without affecting the compression
        settings. If data-free compression has been applied, the prepared model's
        accuracy may be lower than the baseline model's accuracy; otherwise, it should
        remain the same.

        Every compressor accepts an optional ``state_dict`` keyword argument: the state
        dict of a model previously prepared with the same config. When provided, the
        compression state (e.g. calibrated qparams, centroids, pruning masks) is loaded
        from it via ``_load_prepared_state_dict`` instead of being recomputed. The
        resulting model is in the same state as after a regular ``prepare`` call (e.g.
        observers disabled and fake quantization enabled).

        Returns:
            The prepared model, ready for evaluation or further compression steps
        """
        pass

    def _load_prepared_state_dict(
        self,
        model: torch.nn.Module,
        state_dict: Mapping[str, torch.Tensor],
    ) -> None:
        """Strict-load a previously prepared model's state into ``model`` and validate it.

        The model's structure comes from the config, never from ``state_dict``. After
        loading, every compression module must be initialized. Flags are buffers, so they
        are loaded too; ``prepare`` sets them after calling this, as it does without a
        state dict.

        Args:
            model (torch.nn.Module): The prepared model to load into.
            state_dict (Mapping[str, torch.Tensor]): State dict of a model previously
                prepared with the same config.

        Raises:
            RuntimeError: If ``state_dict`` does not match the prepared model's keys or
                shapes, or leaves a compression module uninitialized.
        """
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            raise RuntimeError(
                "State dict does not match the prepared model. It must come from a model "
                "prepared with the same config and model code (and, in graph mode, the same "
                f"example inputs, dynamic shapes and torch version): {e}"
            ) from e

        uninitialized = self._uninitialized_compression_modules(model)
        if uninitialized:
            raise RuntimeError(
                "State dict does not match the compression config. These compression "
                f"modules are uninitialized after loading: {uninitialized}"
            )

    @abstractmethod
    def _uninitialized_compression_modules(self, model: torch.nn.Module) -> list[str]:
        """Return names of compression modules in ``model`` whose state is uninitialized."""

    @abstractmethod
    def finalize(
        self,
        model: torch.nn.Module | None = None,
        backend: ExportBackend = ExportBackend.CoreAI,
        *args,
        mmap_dir: str | PathLike[str] | None = None,
        **kwargs,
    ) -> torch.nn.Module:
        """Apply the model optimizations based on the specified backend.

        Apply the model optimizations by folding the compressed weights onto the
        original weights and removing extra state. This method changes the
        representation of compression settings in the model based on
        the specified backend.

        For example, for quantization:
        - For CoreAI backend (default), custom ops will be inserted.
        - For CoreML, compression metadata will be inserted as buffers in the model.

        Args:
            model: Optional model to finalize. If None, uses self._model
            backend: Target export backend (CoreAI, CoreML)
            mmap_dir (str | None): If provided, serialize finalized compressed
                weights to safetensors files under this directory and re-load
                them via mmap, so large models can be finalized without holding
                full weights in RAM. Support is compressor-specific; unsupported
                (compressor, backend, mode) combinations raise ValueError.

        Returns:
            The finalized compressed model

        """

    @contextmanager
    def calibration_mode(self, model: torch.nn.Module | None = None, *args, **kwargs):
        """
        Context manager for calibration data-based compression workflow.

        When entering the context, compression observers are enabled to collect
        statistics during the calibration process. This method also performs any
        model setup specifically required for calibration-based compression.

        When exiting the context, observers are disabled, making the model ready
        for evaluation, and any workflow-specific temporary state is cleaned up.

        This method should be implemented by compressors that support
        calibration-based post-training compression techniques.

        Args:
            model: Optional model to setup for calibration. If None, uses self._model
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement calibration_mode(). "
            "This compressor doesn't support calibration data based post "
            "training compression."
        )

    @contextmanager
    def training_mode(self, model: torch.nn.Module | None = None, *args, **kwargs):
        """
        Context manager for training time compression workflow.

        When entering the context, compression observers are enabled to collect
        statistics during the training process. This method also performs any
        model setup specifically required for training-based compression.

        When exiting the context, observers are disabled, making the model ready
        for evaluation, and any workflow-specific temporary state is cleaned up.

        This method should be implemented by compressors that support
        training-based compression techniques.

        Args:
            model: Optional model to setup for training. If None, uses self._model
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement training_mode(). "
            "This compressor doesn't support training time compression."
        )
