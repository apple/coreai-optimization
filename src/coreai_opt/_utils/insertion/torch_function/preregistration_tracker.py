# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tracker for pending optimizer registrations during eager mode compression preparation."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from .types import FunctionPreregistrationRecord, PendingOptimizerRegistration


class PreregistrationTracker:
    """
    Tracks pending optimizer registrations during the preregistration phase.

    This class collects information about which optimizers need to be created
    for each function call, before module boundaries are fully analyzed and
    before module-level input/output specs can be resolved.
    """

    def __init__(self) -> None:
        """Initialize an empty preregistration tracker."""
        # module_name -> func_name -> list of FunctionPreregistrationRecord
        # For a given module, each call to a function would append a FunctionPreregistrationRecord
        # to the corresponding function's func_name entry in the dictionary.
        self._pending: dict[str, dict[str, list[FunctionPreregistrationRecord]]] = {}

    def initialize_module(self, module_name: str) -> None:
        """
        Initialize tracking for a module.

        Args:
            module_name: The fully qualified name of the module to track
        """
        if module_name not in self._pending:
            self._pending[module_name] = {}

    def record_function_call(
        self,
        module_name: str,
        func_name: str,
        function: Callable,
        pending_inputs: list[PendingOptimizerRegistration],
        pending_outputs: list[PendingOptimizerRegistration],
    ) -> None:
        """
        Record a function invocation with pending optimizer registrations.

        Args:
            module_name: The fully qualified name of the module
            func_name: The base name of the function (e.g., "add", "mul")
            function: The function being called
            pending_inputs: Pending input optimizer registrations
            pending_outputs: Pending output optimizer registrations

        Raises:
            RuntimeError: If the module was not initialized
        """
        if module_name not in self._pending:
            error_msg = (
                f"Attempting to record function {func_name} for module {module_name} "
                "but the module was not initialized in PreregistrationTracker."
            )
            raise RuntimeError(error_msg)

        if func_name not in self._pending[module_name]:
            self._pending[module_name][func_name] = []
        self._pending[module_name][func_name].append(
            FunctionPreregistrationRecord(function, pending_inputs, pending_outputs)
        )

    def get_function_call_count(self, module_name: str, func_name: str) -> int:
        """
        Get the number of times a function has been called in a module.

        Args:
            module_name: The fully qualified name of the module
            func_name: The base name of the function (e.g., "add", "mul")

        Returns:
            The number of times the function has been invoked in this module

        Raises:
            RuntimeError: If the module was not initialized
        """
        if module_name not in self._pending:
            error_msg = f"PreregistrationTracker has no module with name {module_name}."
            raise RuntimeError(error_msg)

        if func_name not in self._pending[module_name]:
            return 0

        return len(self._pending[module_name][func_name])

    def get_pending_for_module(
        self, module_name: str
    ) -> Mapping[str, list[FunctionPreregistrationRecord]]:
        """
        Get all pending registrations for a module.

        Args:
            module_name: The fully qualified name of the module

        Returns:
            Dictionary mapping function names to their pending registrations
        """
        if module_name not in self._pending:
            error_msg = f"PreregistrationTracker has no module with name {module_name}."
            raise RuntimeError(error_msg)
        return self._pending[module_name]

    def get_all_pending_registrations(
        self,
    ) -> dict[str, dict[str, list[FunctionPreregistrationRecord]]]:
        """
        Return self._pending dictionary containing all pending registrations.

        Returns:
            Dictionary containing all pending registrations.
        """
        return self._pending
