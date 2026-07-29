# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Magnitude pruner implementation."""

import logging
from typing import NamedTuple

import torch

from coreai_opt._utils.config_utils import ConfigLevel as _ConfigLevel
from coreai_opt._utils.eager_utils import (
    EagerCompressionComponentBuilderMixin as _EagerCompressionComponentBuilderMixin,
)
from coreai_opt._utils.insertion.torch_function import (
    TorchFunctionEagerHandler as _TorchFunctionEagerHandler,
)
from coreai_opt._utils.spec_utils import PartialConstructor as _PartialConstructor
from coreai_opt._utils.torch_utils import (
    find_parametrization_matching_cls as _find_parametrization_matching_cls,
    move_model_to_eval as _move_model_to_eval,
)
from coreai_opt.common import ExportBackend
from coreai_opt.config.compression_config import ModuleCompressionConfig
from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.config.spec.base import CompressionSpec
from coreai_opt.pruning.spec import PruneImplBase

from ._prepare_for_export import (
    prepare_for_mil_export as _prepare_for_mil_export,
    prepare_for_mlir_export as _prepare_for_mlir_export,
)
from .base_pruner import _BasePruner
from .config import MagnitudePrunerConfig
from .supported_ops_registry import _PrunerSupportedOpsRegistry

logger = logging.getLogger(__name__)


class _ScheduledModule(NamedTuple):
    """Cached location of a scheduled PruneImplBase: the impl plus where to find its weight."""

    impl: PruneImplBase
    module: torch.nn.Module
    param_name: str


class MagnitudePruner(_BasePruner, _EagerCompressionComponentBuilderMixin):
    """Apply magnitude-based pruning to a model.

    This pruner zeros out the smallest-magnitude weight elements to reach a
    configurable sparsity target. The model is parsed in an eager fashion and
    the pruner registers parametrizations for each candidate parameter to be
    pruned. The mask is applied on every forward pass while parametrizations
    are active.

    When a ``sparsity_schedule`` is configured on a module's config, ``step()``
    advances the schedule and recomputes the mask for that module's
    parametrizations. Without a schedule, the spec's ``target_sparsity`` is
    applied statically.

    Args:
        model (torch.nn.Module): Model to prune.
        config (MagnitudePrunerConfig | None): Pruning configuration. When
            ``None``, a default config with 50 % sparsity is used.

    Example:
        >>> model = torch.nn.Linear(100, 50)
        >>> pruner = MagnitudePruner(model, MagnitudePrunerConfig())
        >>> pruner.prepare((torch.randn(1, 100),))
        >>> pruner.finalize()
    """

    _step_count: int
    _scheduled_modules: list[_ScheduledModule]

    def __init__(self, model: torch.nn.Module, config: MagnitudePrunerConfig | None = None):
        if config is None:
            config = MagnitudePrunerConfig()
        super().__init__(model, config)

        module_components_dict, module_priority_dict = (
            self._get_module_compression_components_and_priority(model, config)
        )
        self._handler = _TorchFunctionEagerHandler(
            compression_config=config,
            module_components_dict=module_components_dict,
            module_priority_dict=module_priority_dict,
            supported_ops_registry=_PrunerSupportedOpsRegistry,
            optimization_type_name="prune",
        )
        self._step_count = 0
        self._scheduled_modules = []

    def prepare(self, example_inputs: tuple[torch.Tensor]) -> torch.nn.Module:
        """Prepare the model for pruning.

        Args:
            example_inputs (tuple[torch.Tensor]): Sample inputs to trace the
                model and configure pruning parametrizations.

        Returns:
            torch.nn.Module: The prepared model with pruning parametrizations.

        Raises:
            RuntimeError: If the model has already been prepared.
        """
        if self._is_model_prepared(self._model):
            raise RuntimeError(
                "Model has already been prepared. Cannot re-prepare a prepared model."
            )

        logger.info("Preparing model for pruning")
        prepared_model = self._handler.prepare(self._model, example_inputs=example_inputs)

        self._mark_model_as_prepared(prepared_model)
        self._model = prepared_model
        # Apply schedule before we run a forward pass to initialize the parameterizations
        self._build_scheduled_modules()

        with _move_model_to_eval(prepared_model), torch.no_grad():
            prepared_model(*example_inputs)

        return self._model

    def step(self) -> None:
        """Advance the sparsity schedule by one step.

        Increments the step count, then recomputes and applies the mask for
        every parametrization with a configured ``sparsity_schedule``. Safe to
        call when no schedule is configured (no-op).
        """
        self._step_count += 1
        self._apply_schedule()

    def finalize(
        self,
        model: torch.nn.Module | None = None,
        backend: ExportBackend = ExportBackend.CoreAI,
    ) -> torch.nn.Module:
        """Finalize the model to be lowered to the target backend.

        Args:
            model (torch.nn.Module | None): Model to finalize. Uses the model
                passed at construction time when ``None``.
            backend (ExportBackend): Target export backend.

        Returns:
            torch.nn.Module: The finalized model ready for the target backend.

        Raises:
            RuntimeError: If the model has not been prepared.
        """
        update_internal = model is None
        if model is None:
            model = self._model

        if not self._is_model_prepared(model):
            raise RuntimeError("Model must be prepared before finalization. Call prepare() first.")

        match backend:
            case ExportBackend._TORCH:
                pass

            case ExportBackend.CoreAI:
                model = _prepare_for_mlir_export(model)

            case ExportBackend.CoreML:
                model = _prepare_for_mil_export(model)

            case _:
                raise ValueError(f"Unsupported backend: {backend}")

        if update_internal:
            self._model = model
        return model

    def _build_scheduled_modules(self) -> None:
        """Attach schedules to each scheduled PruneImplBase and apply the step-0 state."""
        config_dict = self._config.build_module_config_dict(self._model)
        for name, module in self._model.named_modules():
            if not hasattr(module, "parametrizations"):
                continue
            module_config = (
                config_dict[_ConfigLevel.MODULE_NAME].get(name)
                or config_dict[_ConfigLevel.MODULE_TYPE].get(name)
                or config_dict[_ConfigLevel.GLOBAL].get(name)
            )
            if module_config is None or module_config.sparsity_schedule is None:
                continue
            for param_name in module.parametrizations:
                impl = _find_parametrization_matching_cls(module, param_name, PruneImplBase)
                if impl is None:
                    continue
                impl.schedule = module_config.sparsity_schedule
                self._scheduled_modules.append(
                    _ScheduledModule(impl=impl, module=module, param_name=param_name)
                )
        self._apply_schedule()

    def _apply_schedule(self) -> None:
        """For each scheduled impl, advance its sparsity and materialize the mask."""
        for entry in self._scheduled_modules:
            entry.impl.update_sparsity(self._step_count)
            original = entry.module.parametrizations[entry.param_name].original
            with torch.no_grad():
                entry.impl(original.detach())

    @staticmethod
    def _spec_to_partial(
        spec: CompressionSpec | None,
        target: CompressionTargetTensor,
        module_config: ModuleCompressionConfig,
    ) -> _PartialConstructor | None:
        if spec is None:
            return None

        return spec.pruning_algo.with_args(
            target_sparsity=spec.target_sparsity,
            pruning_scheme=spec.pruning_scheme,
        )
