# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause


"""Priority-aware resolver for state-tensor compression optimizers."""

import itertools
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, NamedTuple

import torch
import torch.nn as nn

from coreai_opt._utils.spec_utils import PartialConstructor
from coreai_opt._utils.torch_utils import NamedModule
from coreai_opt.config.spec import CompressionSimulatorBase

from .types import ModuleCompressionComponents
from .utils import get_optimizer_from_components_dict


class _StateInventoryEntry(NamedTuple):
    """Inventory entry for one state tensor.

    Attributes:
        owners (list[NamedModule]): Modules that own this state tensor (one
            entry per owning module, even if the module aliases the tensor
            under multiple attribute names).
        local_names (list[str]): All local attribute names any owning module
            uses for this tensor. Aliases under different attributes on the
            same module each contribute an entry.
    """

    owners: list[NamedModule]
    local_names: list[str]


class StateSpecResolver:
    """Resolve and cache compression-simulator optimizers for state tensors.

    Owns the model's state inventory and the priority-aware optimizer cache.

    Responsibilities:

    - Identifying which tensors are model states and what local name(s) any
      owning module uses for them.
    - Resolving the optimizer for a given (state, current call site) pair,
      honoring ``module_state_spec`` > ``op_state_spec`` precedence and the
      module priority assigned by the quantizer.
    - Caching the resolved optimizer with a priority annotation so subsequent
      visits from lower-priority modules cannot overwrite a higher-priority
      decision.

    The cache invariant: for each state tensor, ``_optimizer_cache`` holds the
    optimizer chosen by the highest-priority module that has visited the
    tensor so far. ``module_state_spec`` matches are cached at
    ``_MODULE_STATE_PRIORITY`` (a sentinel below all ``op_state_spec``
    priorities) so they cannot be overridden by ``op_state_spec`` from any
    module.

    Lookups for both ``op_state_spec`` and ``module_state_spec`` consider all
    local names the state has across all owning modules. This matches the
    pre-refactor behavior of ``_create_state_optimizer``.
    """

    _MODULE_STATE_PRIORITY: ClassVar[int] = -1

    def __init__(
        self,
        model: nn.Module,
        module_components_dict: Mapping[NamedModule, ModuleCompressionComponents],
        module_priority_dict: Mapping[str, int],
    ) -> None:
        self._module_components_dict = module_components_dict
        self._module_priority_dict = module_priority_dict
        self._state_inventory = self._build_state_inventory(model)
        self._optimizer_cache: dict[torch.Tensor, tuple[CompressionSimulatorBase | None, int]] = {}

    @staticmethod
    def _build_state_inventory(
        model: nn.Module,
    ) -> dict[torch.Tensor, _StateInventoryEntry]:
        """Build a map from state tensor to its inventory entry.

        For each tensor reachable as a parameter or buffer of any module,
        record the owning modules and every local attribute name they use
        for it.
        """
        inventory: dict[torch.Tensor, _StateInventoryEntry] = {}
        for module_name, module in model.named_modules():
            named_module = NamedModule(module_name, module)
            for state_name, state in itertools.chain(
                module.named_parameters(recurse=False, remove_duplicate=False),
                module.named_buffers(recurse=False, remove_duplicate=False),
            ):
                if state not in inventory:
                    inventory[state] = _StateInventoryEntry(owners=[], local_names=[])
                entry = inventory[state]
                # Dedupe on insertion to maintain a stable traversal order for
                # get_all_local_names. Shared state lists are typically short so
                # the O(n) dedup cost is acceptable.
                if named_module not in entry.owners:
                    entry.owners.append(named_module)
                if state_name not in entry.local_names:
                    entry.local_names.append(state_name)
        return inventory

    def is_state_tensor(self, value: Any) -> bool:
        """Return True iff ``value`` is a Tensor reachable as a parameter or buffer of the model."""
        if not isinstance(value, torch.Tensor):
            return False
        return value in self._state_inventory

    def get_all_local_names(self, state: torch.Tensor) -> list[str]:
        """Return every local attribute name any module uses for ``state``.

        Used both for surfacing warnings when configuration mentions an
        apparent state name that the resolver cannot match, and as the
        identifier list for ``op_state_spec`` and ``module_state_spec``
        lookups. If the state is not in the inventory, returns an empty list.
        """
        entry = self._state_inventory.get(state)
        return list(entry.local_names) if entry else []

    def resolve(
        self,
        func: Callable,
        state_tensor: torch.Tensor,
        current_module: NamedModule,
        components_dict: Mapping[int | str, PartialConstructor | None],
    ) -> None:
        """Resolve and cache the optimizer for ``state_tensor`` at the current call site.

        Algorithm:

        1. Get the current module's priority.
        2. Skip if a strictly higher-priority spec is already cached for this
           tensor (strict ``>`` so equal priorities allow last-writer-wins,
           matching the pre-refactor unsorted loop behavior).
        3. On the first visit to ``state_tensor``, walk all owners in priority
           order looking for a ``module_state_spec`` match. First match wins
           and is cached at ``_MODULE_STATE_PRIORITY`` so it cannot be
           overridden by any subsequent ``op_state_spec`` resolution.
           We check for whether ``state_tensor`` is present in
           ``self._optimizer_cache`` to see if it is the first visit or not.
        4. Otherwise, look up the ``op_state_spec`` using **all** local names
           the state has across all owning modules, then cache the result at
           the current module's priority.
        """
        current_priority = self._module_priority_dict.get(current_module.name, float("inf"))
        if self._should_skip(state_tensor, current_priority):
            return

        # This block will only be run once the first time the state is encountered.
        if state_tensor not in self._optimizer_cache:
            optimizer, found = self._resolve_module_state(func, state_tensor)
            if found:
                self._optimizer_cache[state_tensor] = (optimizer, self._MODULE_STATE_PRIORITY)
                return

        optimizer = self._resolve_op_state(func, state_tensor, components_dict)
        self._optimizer_cache[state_tensor] = (optimizer, current_priority)

    def get_optimizer(self, state: torch.Tensor) -> CompressionSimulatorBase | None:
        """Return the cached optimizer for ``state``, or None if absent or resolved to None."""
        optimizer, _ = self._optimizer_cache.get(state, (None, None))
        return optimizer

    def _should_skip(self, state_tensor: torch.Tensor, current_priority: int) -> bool:
        """Return True if a strictly higher-priority result is already cached for
        ``state_tensor``.
        """
        _, cached_priority = self._optimizer_cache.get(state_tensor, (None, None))
        return cached_priority is not None and current_priority > cached_priority

    def _resolve_module_state(
        self, func: Callable, state_tensor: torch.Tensor
    ) -> tuple[CompressionSimulatorBase | None, bool]:
        """Walk owner modules in priority order, returning the first ``module_state_spec`` match
        and a found flag.
        """
        local_state_names = self.get_all_local_names(state_tensor)
        for named_module in self._priority_sorted_owner_modules(state_tensor):
            module_components = self._module_components_dict.get(named_module)
            if not module_components or not module_components.module_state_components:
                continue
            optimizer, found = get_optimizer_from_components_dict(
                func, local_state_names, module_components.module_state_components
            )
            if found:
                return optimizer, True
        return None, False

    def _resolve_op_state(
        self,
        func: Callable,
        state_tensor: torch.Tensor,
        components_dict: Mapping[int | str, PartialConstructor | None],
    ) -> CompressionSimulatorBase | None:
        """Look up the ``op_state_spec`` optimizer using all local names for ``state_tensor``."""
        local_state_names = self.get_all_local_names(state_tensor)
        optimizer, _ = get_optimizer_from_components_dict(func, local_state_names, components_dict)
        return optimizer

    def _priority_sorted_owner_modules(self, state_tensor: torch.Tensor) -> list[NamedModule]:
        """Return the owners of ``state_tensor`` sorted by ascending priority"""
        return sorted(
            self._state_inventory[state_tensor].owners,
            key=lambda nm: self._module_priority_dict.get(nm.name, float("inf")),
        )
