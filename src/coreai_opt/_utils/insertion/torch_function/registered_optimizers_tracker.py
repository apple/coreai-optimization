# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tracker for registered optimizers during eager mode compression."""

from __future__ import annotations

from collections import defaultdict
from typing import TypeAlias

from .types import FunctionRegisteredOptimizers

RegisteredOptimizersDict: TypeAlias = dict[
    str, defaultdict[str, list[FunctionRegisteredOptimizers]]
]


class RegisteredOptimizersTracker:
    """
    Tracks which optimizers were actually registered for each function call.

    This is used during the optimization phase to validate that the same
    optimizers are applied in the same order as during registration.
    """

    def __init__(self) -> None:
        """Initialize an empty registered optimizers tracker."""
        self._registered: RegisteredOptimizersDict = {}

    def initialize_module(self, module_name: str) -> None:
        """
        Initialize tracking for a module.

        Args:
            module_name: The fully qualified name of the module to track
        """
        if module_name not in self._registered:
            self._registered[module_name] = defaultdict(list)

    def record_registration(
        self,
        module_name: str,
        func_name: str,
        input_optimizer_names: list[str],
        output_optimizer_names: list[str],
    ) -> None:
        """
        Record that optimizers were registered for a function invocation.

        Args:
            module_name: The fully qualified name of the module
            func_name: The base name of the function (e.g., "add", "mul")
            input_optimizer_names: Names of input optimizers registered
            output_optimizer_names: Names of output optimizers registered

        Raises:
            RuntimeError: If the module was not initialized
        """
        if module_name not in self._registered:
            error_msg = (
                f"Attempting to record function {func_name} for module {module_name} "
                "but the module was not initialized in RegisteredOptimizersTracker."
            )
            raise RuntimeError(error_msg)

        self._registered[module_name][func_name].append(
            FunctionRegisteredOptimizers(input_optimizer_names, output_optimizer_names)
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
        if module_name not in self._registered:
            error_msg = f"RegisteredOptimizersTracker has no module with name {module_name}."
            raise RuntimeError(error_msg)

        if func_name not in self._registered[module_name]:
            return 0

        return len(self._registered[module_name][func_name])

    def has_module(self, module_name: str) -> bool:
        """
        Check if a module has been initialized in the tracker.

        Args:
            module_name: The fully qualified name of the module

        Returns:
            True if the module is being tracked, False otherwise
        """
        return module_name in self._registered

    def has_function(self, module_name: str, func_name: str) -> bool:
        """
        Check if a function has been called in a module.

        Args:
            module_name: The fully qualified name of the module
            func_name: The base name of the function (e.g., "add", "mul")

        Returns:
            True if the function has been called in this module, False otherwise
        """
        return module_name in self._registered and func_name in self._registered[module_name]

    def get_module_registrations(
        self, module_name: str
    ) -> dict[str, list[FunctionRegisteredOptimizers]]:
        """
        Get all registered optimizers for a module.

        Args:
            module_name: The fully qualified name of the module

        Returns:
            Dictionary mapping function names to their registered optimizers
        """
        if not self.has_module(module_name):
            error_msg = f"RegisteredOptimizersTracker has no module with name {module_name}."
            raise RuntimeError(error_msg)
        return self._registered[module_name]

    def reset_module(self, module_name: str) -> None:
        """
        Clear all tracking data for a module.

        Args:
            module_name: The fully qualified name of the module to reset
        """
        self._registered.pop(module_name, None)

    def reset_all(self) -> None:
        """Clear all tracking data for all modules."""
        self._registered.clear()

    def get_registry_dict(self) -> RegisteredOptimizersDict:
        """
        Get the underlying registry dictionary.

        Returns:
            The full registered optimizers dictionary
        """
        return self._registered

    def remove_optimizer_names(self, module_name: str, names_to_remove: set[str]) -> None:
        """Remove specified optimizer names from a module's registrations.

        Args:
            module_name (str): Fully qualified name of the module whose
                registrations should be updated.
            names_to_remove (set[str]): Set of optimizer names to remove.

        """
        if module_name not in self._registered:
            return
        for func_name in self._registered[module_name]:
            self._registered[module_name][func_name] = [
                FunctionRegisteredOptimizers(
                    [n for n in record.input_optimizer_names if n not in names_to_remove],
                    [n for n in record.output_optimizer_names if n not in names_to_remove],
                )
                for record in self._registered[module_name][func_name]
            ]

    def validate_against_reference(
        self, reference_tracker: RegisteredOptimizersTracker, module_name: str
    ) -> None:
        """
        Validate that registrations match a reference tracker.

        This is used during optimization to ensure the same functions are called
        the same number of times with the same input/output optimizers as during
        registration.

        Args:
            reference_tracker: The reference tracker to validate against
            module_name: The module to validate

        Raises:
            RuntimeError: If validation fails
        """
        current_registrations = self.get_module_registrations(module_name)
        reference_registrations = reference_tracker.get_module_registrations(module_name)

        extra_funcs_in_reference = reference_registrations.keys() - current_registrations.keys()
        for extra_func in extra_funcs_in_reference:
            for record in reference_registrations[extra_func]:
                if record.input_optimizer_names or record.output_optimizer_names:
                    error_msg = (
                        f"Function {extra_func} contains registered optimizers which were not "
                        "seen during optimization.\n"
                        f"Registered optimizers: {reference_registrations[extra_func]}\n"
                    )
                    raise RuntimeError(error_msg)

        for func_name in current_registrations:
            current_records = current_registrations[func_name]
            reference_records = reference_registrations[func_name]

            if len(current_records) != len(reference_records):
                error_msg = (
                    f"Function {func_name} in module {module_name} was seen a "
                    "different number of times during registration "
                    f"({len(reference_records)}) "
                    f"as compared to optimization ({len(current_records)})."
                )
                raise RuntimeError(error_msg)

            for idx, (current, reference) in enumerate(
                zip(current_records, reference_records, strict=True)
            ):
                if reference != current:
                    error_msg = (
                        f"Function {func_name} index {idx} in module {module_name} "
                        "has a different set of input and output optimizers "
                        f"used: {current} compared to registered in reference {reference}."
                    )
                    raise RuntimeError(error_msg)
