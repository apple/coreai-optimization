# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""TorchFunctionMode classes for eager mode compression."""

import abc
import contextlib
import itertools
import logging
import re
import types
from collections.abc import Callable, Generator, Mapping
from typing import Any, Literal, NamedTuple

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
from torch.overrides import TorchFunctionMode, _get_current_function_mode
from torch.utils._pytree import tree_map

from coreai_opt._utils.spec_utils import PartialConstructor as _PartialConstructor
from coreai_opt._utils.torch_utils import NamedModule
from coreai_opt.config import CompressionConfig
from coreai_opt.config.spec import CompressionSimulatorBase

from .base_supported_ops_registry import BaseSupportedOpsRegistry
from .module_boundary_tracker import ModuleBoundaryInfo, ModuleBoundaryTracker
from .preregistration_tracker import PreregistrationTracker
from .registered_optimizers_tracker import RegisteredOptimizersTracker
from .state_spec_resolver import StateSpecResolver
from .types import (
    ModuleCompressionComponents,
    OpCompressionComponents,
    PendingOptimizerRegistration,
)
from .utils import (
    any_tensor_optimizable,
    get_func_base_name,
    get_func_name,
    get_optimizer_from_components_dict,
    is_optimizable_tensor,
    normalize_args_kwargs,
)

logger = logging.getLogger(__name__)


def _is_interceptable_func(func: Callable) -> bool:
    """
    Return True if func is a torch operation that should be intercepted for optimization.
    """
    # torch.overrides.get_overridable_functions() is a functools cached object so repeated calls
    # are not costly.
    return any(
        func in func_list
        for torch_entity, func_list in torch.overrides.get_overridable_functions().items()
        if isinstance(torch_entity, (type, types.ModuleType))
    )


class StateParametrizationInfo(NamedTuple):
    """
    Records information for parametrizing a state including module to parametrize, state
    name to parametrize, and the optimizer to use.
    """
    module: torch.nn.Module
    state_name: str
    optimizer: torch.nn.Module


class ScopedEagerOptimizationModeBase(TorchFunctionMode, abc.ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        compression_config: CompressionConfig,
        module_components_dict: Mapping[NamedModule, ModuleCompressionComponents],
        supported_ops_registry: type[BaseSupportedOpsRegistry],
        optimization_type_name: str = "optimize",
        hooks_allow_list: list | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.named_modules = dict(model.named_modules(remove_duplicate=False))
        self.compression_config = compression_config
        self.module_components_dict = module_components_dict
        self.supported_ops_registry = supported_ops_registry
        self.optimization_type_name = optimization_type_name
        self.registered_optimizers_tracker = RegisteredOptimizersTracker()
        self.parents: list[NamedModule] = list()
        self.hooks = list()
        self.is_entered = False
        for name, module in model.named_modules(remove_duplicate=True):
            # If there is no allow list or if the module is in the allow list,
            # register enter/exit hooks
            if (hooks_allow_list is None or module in hooks_allow_list):
                pre_hook = module.register_forward_pre_hook(self.enter_module(name))
                hook = module.register_forward_hook(
                    self.exit_module(name), always_call=True
                )

            # If the module is not in the allow list, register fallback enter/exit hooks
            else:
                pre_hook = module.register_forward_pre_hook(
                    self.fallback_enter_module(name)
                )
                hook = module.register_forward_hook(
                    self.fallback_exit_module(name), always_call=True
                )
            self.hooks.append(pre_hook)
            self.hooks.append(hook)

    def __enter__(self) -> Any:
        self.is_entered = True
        return super().__enter__()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        self.is_entered = False
        # XXX: note that _get_current_function_mode() doesn't return the
        # current function mode if called from its __torch_function__ , so
        # entering/exiting logic should be avoided when calling a module from
        # __torch_function__ itself.
        if not isinstance(_get_current_function_mode(), self.__class__):
            raise RuntimeError(
                f"{self.__class__.__name__} is not at the top of the stack."
            )
        return super().__exit__(exc_type, exc_val, exc_tb)

    @abc.abstractmethod
    def fallback_enter_module(self, name: str) -> Any:
        pass

    @abc.abstractmethod
    def fallback_exit_module(self, name: str) -> Any:
        pass

    @abc.abstractmethod
    def enter_module(self, name: str) -> Any:
        pass

    @abc.abstractmethod
    def exit_module(self, name: str) -> Any:
        pass

    @property
    def current_module(self) -> nn.Module:
        if self.parents:
            return self.parents[-1].module
        else:
            # This can be a call from outside of a model
            return self.model

    @property
    def current_module_name(self) -> str:
        if self.parents:
            return str(self.parents[-1].name)
        else:
            return ""

    def remove_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()

    @staticmethod
    def _build_optimizer_name(*parts: str | int) -> str:
        """Build optimizer module name by joining parts with underscores."""
        return "_".join(str(part) for part in parts)


class RegisterEagerOptimizationMode(ScopedEagerOptimizationModeBase):
    def __init__(
        self,
        model: torch.nn.Module,
        compression_config: CompressionConfig,
        module_components_dict: Mapping[NamedModule, ModuleCompressionComponents],
        module_priority_dict: Mapping[str, int],
        supported_ops_registry: type[BaseSupportedOpsRegistry],
        optimization_type_name: str = "optimize",
    ) -> None:
        super().__init__(
            model=model,
            compression_config=compression_config,
            module_components_dict=module_components_dict,
            supported_ops_registry=supported_ops_registry,
            optimization_type_name=optimization_type_name,
        )
        self.traversed_modules: set[nn.Module] = set()
        self.state_resolver = StateSpecResolver(
            model=model,
            module_components_dict=module_components_dict,
            module_priority_dict=module_priority_dict,
        )
        self.module_boundary_tracker = ModuleBoundaryTracker()
        self.preregistration_tracker = PreregistrationTracker()
        # Set default value to True for
        self._module_has_module_spec_dict = self._get_module_has_module_spec_dict()

    def _get_module_has_module_spec_dict(self) -> Mapping[str, bool]:
        """
        Build a dict mapping each module name to whether a module-level spec applies to it.

        A module has a module-level spec if it (or any ancestor) has module_input_components,
        module_output_components, or module_state_components set in the module_components_dict.
        """
        module_has_module_spec_dict = {}
        self._fill_module_has_module_spec_dict(
            module_name="",
            module=self.model,
            parent_has_module_spec=False,
            module_components_dict=self.module_components_dict,
            module_has_module_spec_dict=module_has_module_spec_dict,
        )

        return module_has_module_spec_dict

    @staticmethod
    def _fill_module_has_module_spec_dict(
        module_name: str,
        module: torch.nn.Module,
        parent_has_module_spec: bool,
        module_components_dict: Mapping[NamedModule, ModuleCompressionComponents],
        module_has_module_spec_dict: dict[str, bool],
    ) -> None:
        """
        Recursively populate module_has_module_spec_dict for a module and its children.

        If a parent has a module-level spec, all descendants inherit True. Otherwise, the module
        is checked directly against module_components_dict.
        """
        if parent_has_module_spec:
            has_module_spec = True
        else:
            module_component = module_components_dict.get(NamedModule(module_name, module))
            has_module_spec = (
                False if not module_component else module_component.has_module_level_component()
            )

        module_has_module_spec_dict[module_name] = has_module_spec
        for child_name, child_module in module.named_children():
            RegisterEagerOptimizationMode._fill_module_has_module_spec_dict(
                child_name if module_name == "" else ".".join([module_name, child_name]),
                child_module,
                has_module_spec,
                module_components_dict,
                module_has_module_spec_dict,
            )

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
        self.remove_hooks()
        return super().__exit__(exc_type, exc_val, exc_tb)

    def fallback_enter_module(self, name: str) -> Any:
        return self.enter_module(name)

    def fallback_exit_module(self, name: str) -> Any:
        return self.exit_module(name)

    def enter_module(self, name: str) -> Callable:
        def f(module: nn.Module, inputs: Any) -> None:
            current_named_module = NamedModule(name, module)
            self.parents.append(current_named_module)
            if self.current_module_name not in self._module_has_module_spec_dict:
                error_msg = (
                    f"{self.current_module_name} not found in module_has_module_spec_dict.\n"
                    f"Dictionary contents: {self._module_has_module_spec_dict}"
                )
                raise RuntimeError(error_msg)
            if self.current_module in self.traversed_modules:
                return

            # Below steps only need to be done the first time we enter each module.
            # No point in keeping track of module boundary tensors if it is an already traversed
            # module, since repeated traversals will also be skipped during torch function
            # processing.
            self.preregistration_tracker.initialize_module(self.current_module_name)
            self.registered_optimizers_tracker.initialize_module(self.current_module_name)
            self.module_boundary_tracker.record_module_boundary_tensors(
                current_named_module, inputs, "input"
            )
        return f

    def exit_module(self, name: str) -> Callable:
        def f(module: nn.Module, inputs: Any, outputs: Any) -> None:
            assert (
                self.parents[-1].name == name
            ), f"{self.parents[-1].name} is not {name}."
            if self.current_module not in self.traversed_modules:
                # No point in keeping track of module boundary tensors if it is an already traversed
                # module, since repeated traversals will also be skipped during torch function
                # processing.
                self.module_boundary_tracker.record_module_boundary_tensors(
                    self.parents[-1], outputs, "output"
                )
            self.parents.pop()
            # Keep track of which modules have been fully executed at least once
            self.traversed_modules.add(module)

        return f

    def register_all_activations(self) -> None:
        """
        Resolve pending activation optimizers and register them.

        This method:
        1. Gets all pending registrations for the current module
        2. Resolves optimizers based on module and op level input/output specs
        3. Registers the optimizers on the module
        4. Records what was registered for validation during optimization
        """
        # Iterate over all pending registrations, structured as:
        #   module_name -> module_funcs: dict[func_name -> list[FunctionPreregistrationRecord]]
        #
        # module_funcs maps each function name (e.g., "linear") to the list of
        # FunctionPreregistrationRecords collected for that function within the module
        # (pending_records). A function can have multiple records if it was called more than once.
        #
        # The inner generator flattens this into (func_name, record) pairs so each
        # individual FunctionPreregistrationRecord is processed independently.
        for module_name, module_funcs in (
            self.preregistration_tracker.get_all_pending_registrations().items()
        ):
            for f_name, pending_record in (
                (func_name, record)
                for func_name, pending_records in module_funcs.items()
                for record in pending_records
            ):
                # Resolve pending input registrations
                resolved_inputs = self._resolve_pending_registrations(
                    pending_record.function,
                    pending_record.pending_inputs,
                    boundary_type="input"
                )

                # Resolve pending output registrations
                resolved_outputs = self._resolve_pending_registrations(
                    pending_record.function,
                    pending_record.pending_outputs,
                    boundary_type="output"
                )

                self._register_resolved_optimizations(
                    resolved_inputs, resolved_outputs, module_name, f_name
                )

    def _register_resolved_optimizations(
        self,
        input_registrations: list[PendingOptimizerRegistration],
        output_registrations: list[PendingOptimizerRegistration],
        module_name: str,
        func_name: str
    ):
        """
        Given resolved input and output registrations for a function within a module, register
        the optimizations and record them in self.registered_optimizers_tracker.
        """
        # Register the optimizers on the current module
        registered_input_names: list[str] = []
        registered_output_names: list[str] = []

        for registration in input_registrations:
            self._register_optimizer(registration, registered_input_names, module_name)

        for registration in output_registrations:
            self._register_optimizer(registration, registered_output_names, module_name)

        # Record what was registered
        self.registered_optimizers_tracker.record_registration(
            module_name,
            func_name,
            registered_input_names,
            registered_output_names,
        )

    def _register_optimizer(
        self,
        registration: PendingOptimizerRegistration,
        registered_names_list: list[str],
        module_name: str,
    ) -> None:
        """
        Given a PendingOptimizerRegistration, register the optimizer and add the registered name to
        registered_names_list.
        """
        if registration.optimizer is not None:
            if hasattr(registration.module, registration.optimizer_name):
                error_msg = (
                    f"Optimizer {registration.optimizer_name} already exists in module "
                    f"{module_name}"
                )
                raise RuntimeError(error_msg)
            registration.module.register_module(registration.optimizer_name, registration.optimizer)
            registered_names_list.append(registration.optimizer_name)

    def _resolve_pending_registrations(
        self,
        func: Callable,
        pending_registrations: list[PendingOptimizerRegistration],
        boundary_type: Literal["input", "output"],
    ) -> list[PendingOptimizerRegistration]:
        """
        Resolve pending registrations by checking module-level components to see if any op-level
        components should be overridden.

        Args:
            func: The function being optimized
            pending_registrations: List of pending registrations to resolve
            boundary_type: Either "input" or "output" depending on which boundary to resolve for

        Returns:
            List of registrations with optimizers resolved
        """
        resolved_registrations: list[PendingOptimizerRegistration] = []

        for pending in pending_registrations:
            pending = self._get_module_component_override_if_applicable(
                func,
                pending,
                boundary_type
            )
            resolved_registrations.append(pending)
        return resolved_registrations

    def _get_module_component_override_if_applicable(
        self,
        func: Callable,
        pending: PendingOptimizerRegistration,
        boundary_type: Literal["input", "output"]
    ) -> PendingOptimizerRegistration:
        """
        For a specific pending optimizer registration, check if the optimizer should be overridden
        by any module level specifications.

        Return a new pending optimizer registration with the override if so, otherwise return the
        original pending registration.
        """
        module_boundaries_for_tensor = (
            self.module_boundary_tracker.get_module_boundaries_for_tensor(
                pending.tensor_counter, boundary_type
            )
        )

        module_boundaries_by_module = self._split_module_boundary_info_list_by_module(
            module_boundaries_for_tensor
        )

        # Module boundaries by module is ordered such that innermost nested modules are processed
        # before higher level modules. By virtue of how the config is parsed via config precedence
        # rules, inner module boundary specs will always have higher priority than outer module
        # boundary specs.
        for module_boundaries in module_boundaries_by_module:
            module_components = self.module_components_dict.get(
                module_boundaries[0].named_module
            )
            if module_components is None:
                continue

            if boundary_type == "input":
                components_dict = module_components.module_input_components
            else:
                components_dict = module_components.module_output_components

            self._warn_on_multiple_specifications_for_same_tensor(
                module_boundaries,
                components_dict
            )

            # Check for module-level spec override for the corresponding input/output
            # components dict
            override_optimizer, found = get_optimizer_from_components_dict(
                func,
                [module_boundary.index for module_boundary in module_boundaries],
                components_dict
            )

            if found:
                return pending.with_optimizer(override_optimizer)
        return pending

    @staticmethod
    def _warn_on_multiple_specifications_for_same_tensor(
        module_boundaries: list[ModuleBoundaryInfo],
        components_dict: Mapping[int | str, CompressionSimulatorBase | None]
    ):
        """
        Given a list of module boundaries for a tensor, check whether multiple indices match
        specifications in components_dict. If so, raise a warning to inform the user which index
        will be used to configure quantization for the tensor.
        """
        if len(module_boundaries) < 2:
            return

        indices_to_check = [module_boundary.index for module_boundary in module_boundaries]
        previous_matched_index = None

        for index in indices_to_check:
            if index in components_dict:
                if previous_matched_index is None:
                    previous_matched_index = index
                else:
                    warning_msg = (
                        "Components dict contains multiple specifications for the same tensor. "
                        f"The specification for index {previous_matched_index} will be used to "
                        "configure quantizers for the tensor.\n"
                        f"Components dict: {components_dict}"
                    )
                    logger.warning(warning_msg)
                    break


    @staticmethod
    def _split_module_boundary_info_list_by_module(
        module_boundaries_for_tensor: list[ModuleBoundaryInfo]
    ) -> list[list[ModuleBoundaryInfo]]:
        """
        Helper function to split the list of module_boundaries_for_tensor into a list of lists where
        each element in the top level list corresponds to a list of module boundary info for the
        same module.

        Each top level list will only have more than one element if the same tensor is fed into
        the module as input multiple times.
        """
        return [
            list(group)
            for _, group in itertools.groupby(
                module_boundaries_for_tensor, key=lambda b: b.named_module.name
            )
        ]

    def register_all_states(self) -> None:
        """
        Parametrize every state for which self.state_resolver has a cached optimizer.
        """
        states_to_parametrize: list[StateParametrizationInfo] = []
        # Go through all modules to identify modules to parametrize. We cannot
        # parametrize as we go since this updates the model in place during the for
        # loops.
        for module in self.model.modules():
            for state_name, state in itertools.chain(
                module.named_parameters(recurse=False, remove_duplicate=False),
                module.named_buffers(recurse=False, remove_duplicate=False)
            ):
                optimizer = self.state_resolver.get_optimizer(state)
                if optimizer:
                    states_to_parametrize.append(StateParametrizationInfo(
                        module,
                        state_name,
                        optimizer
                    ))

        # Register all parametrizations
        for state_to_parametrize in states_to_parametrize:
            P.register_parametrization(
                state_to_parametrize.module,
                state_to_parametrize.state_name,
                state_to_parametrize.optimizer,
                unsafe=True
            )

    def _preregister_input_and_state_optimization(
        self,
        func: Callable,
        curr_func_count: int,
        args: tuple,
        kwargs: dict,
        op_compression_components: OpCompressionComponents,
    ) -> list[PendingOptimizerRegistration]:
        """
        Handle pre-registration of optimizers for inputs and states of a function.

        Returns list of pending optimizer registrations.
        """
        args, kwargs = normalize_args_kwargs(func, args, kwargs)
        pending_registrations: list[PendingOptimizerRegistration] = []
        for tensor_idx, (kwarg_name, tensor) in enumerate(
            itertools.chain(((None, arg) for arg in args), kwargs.items()
            )
        ):
            components_dict = (
                op_compression_components.op_state_components
                if self.state_resolver.is_state_tensor(tensor)
                else op_compression_components.op_input_components
            )
            if not is_optimizable_tensor(tensor):
                self._warn_non_quantizable_tensor_setting(
                    func, curr_func_count, tensor, tensor_idx, components_dict, False
                )
                continue

            if self.state_resolver.is_state_tensor(tensor):
                self.state_resolver.resolve(
                    func,
                    tensor,
                    NamedModule(self.current_module_name, self.current_module),
                    components_dict,
                )
            else:
                pending_registration = self._create_pending_activation_registration(
                    func=func,
                    func_counter=curr_func_count,
                    activation_type="input",
                    tensor=tensor,
                    tensor_idx=tensor_idx,
                    components_dict=components_dict,
                    kwarg_name=kwarg_name,
                )
                pending_registrations.append(pending_registration)
        return pending_registrations

    def _warn_non_quantizable_tensor_setting(
        self,
        func: Callable,
        func_counter: int,
        tensor: torch.nn.Parameter | torch.Tensor,
        idx: int,
        components_dict: Mapping[int | str, _PartialConstructor | None],
        is_output: bool
    ) -> None:
        """
        Check whether module_components is attempting to set a specific non-quantizable
        tensor and raise a warning if so.

        Only specific matches will raise a warning. If the user has set "*" to quantize
        all inputs/outputs/states, no warning will be raised.
        """
        if self.state_resolver.is_state_tensor(tensor):
            setting_type = "state"
            tensor_identifiers = self.state_resolver.get_all_local_names(tensor)
        elif is_output:
            setting_type = "output"
            tensor_identifiers = [idx]
        else:
            setting_type = "input"
            tensor_identifiers = [idx]

        for tensor_identifier in tensor_identifiers:
            if tensor_identifier in components_dict:
                warning_msg = (
                    f"Config is attempting to set {setting_type} tensor "
                    f"{tensor_identifier} for function "
                    f"{get_func_name(func, func_counter)} in module "
                    f"{self.current_module_name} but the tensor is not a "
                    "quantizable floating point tensor. "
                    "No quantization will be performed on the tensor. "
                    f"Remove the {setting_type} setting from the config to disable "
                    "this warning.\n"
                    f"{setting_type} setting: {components_dict}"
                )
                logger.warning(warning_msg)

                # Just return if any tensor identifiers for a tensor was found, skip
                # looking for other identifiers for the same tensor.
                return

    def _get_op_compression_components(
        self, func_name: str, func_type: str, module_component: ModuleCompressionComponents
    ) -> OpCompressionComponents:
        """
        Return the appropriate op_compression_components from module_component using func
        and func_counter.

        If no op_name or op_type specific configs match, return a new
        OpCompressionComponent built from module_component's default weight/input/output
        activation fields.

        When multiple op_name or op_type keys match, the last key in insertion order
        takes precedence (i.e., later entries override earlier ones).
        """
        if module_component is None:
            return OpCompressionComponents(
                op_input_components={},
                op_output_components={},
                op_state_components={},
            )

        # Check for op_name match
        for op_name, op_name_config in reversed(module_component.op_name_components.items()):
            try:
                if re.fullmatch(op_name, func_name):
                    return op_name_config
            except re.error as e:
                error_msg = (
                    f"Invalid regex pattern '{op_name}' in op_name_config: "
                    f"{op_name_config}"
                )
                raise ValueError(error_msg) from e

        # Check for op_type match
        for op_type, op_type_config in reversed(module_component.op_type_components.items()):
            if op_type == func_type:
                return op_type_config

        # If no matching op_name or op_type found, use default op input/output/state
        # settings as found in module_component.
        op_compression_components = OpCompressionComponents(
            op_input_components=module_component.input_activation,
            op_output_components=module_component.output_activation,
            op_state_components=module_component.weight,
        )
        return op_compression_components

    def _create_pending_activation_registration(
        self,
        func: Callable,
        func_counter: int,
        activation_type: Literal["input", "output"],
        tensor: torch.Tensor,
        tensor_idx: int,
        components_dict: Mapping[int | str, _PartialConstructor | None],
        kwarg_name: str | None = None,
    ) -> PendingOptimizerRegistration:
        """
        Create a pending optimizer registration for an activation tensor.

        Args:
            func: Function being optimized
            func_counter: Global function counter for naming
            activation_type: "input" or "output"
            tensor: The tensor to create registration for
            tensor_idx: Index of the tensor
            components_dict: Component dictionary to get optimizer from
            kwarg_name: Optional kwarg name for inputs (None for positional args)

        Returns:
            PendingOptimizerRegistration
        """
        # Get optimizer from components
        act_optimizer, _ = get_optimizer_from_components_dict(func, tensor_idx, components_dict)

        # Build optimizer name based on activation type
        func_name = get_func_name(func, func_counter)
        if activation_type == "input":
            optimizer_name = self._build_optimizer_name(
                func_name,
                self.optimization_type_name,
                tensor_idx if kwarg_name is None else kwarg_name
            )
        else:
            optimizer_name = self._build_optimizer_name(
                func_name,
                self.optimization_type_name,
                "output",
                tensor_idx
            )

        return PendingOptimizerRegistration(
            self.current_module,
            tensor_counter=self.module_boundary_tracker.get_or_assign_counter(tensor),
            optimizer_name=optimizer_name,
            optimizer=act_optimizer,
        )

    def _preregister_output_optimization(
        self,
        func: Callable,
        func_counter: int,
        outputs: tuple,
        op_compression_components: OpCompressionComponents,
    ) -> list[PendingOptimizerRegistration]:
        """
        Handle pre-registration for outputs of a function.

        Returns:
            List of pending optimizer registrations for outputs.
        """
        pending_registrations: list[PendingOptimizerRegistration] = []
        if not isinstance(outputs, tuple):
            outputs = (outputs,)

        for idx, output in enumerate(outputs):
            if not is_optimizable_tensor(output):
                self._warn_non_quantizable_tensor_setting(
                    func,
                    func_counter,
                    output,
                    idx,
                    op_compression_components.op_output_components,
                    True
                )
                continue

            pending_registration = self._create_pending_activation_registration(
                func=func,
                func_counter=func_counter,
                activation_type="output",
                tensor=output,
                tensor_idx=idx,
                components_dict=op_compression_components.op_output_components,
            )
            pending_registrations.append(pending_registration)
        return pending_registrations

    def __torch_function__(
        self,
        func: Callable,
        types: list,
        args: tuple = (),
        kwargs: dict | None = None
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        already_traversed_module = self.current_module in self.traversed_modules
        module_components = self.module_components_dict.get(
            NamedModule(self.current_module_name, self.current_module)
        )
        out = func(*args, **kwargs)

        # We should process the function if
        # 1. There is a module level spec in play and the function is any which we can intercept, or
        # 2. There is no module level spec and the function is a registered operation.
        # This is an optimization that allows us to skip unregistered functions if we don't have any
        # potential module level specifications to apply.
        should_process_function = (
            _is_interceptable_func(func)
            if self._module_has_module_spec_dict.get(self.current_module_name)
            else self.supported_ops_registry.supports_operation(func)
        )

        # Even if the func is not a registered function in the registry, allow it to continue in
        # case any module level specs must be applied to the function inputs or outputs.
        # This does not apply to non-interceptable functions, which are always ignored.
        if (
            already_traversed_module
            or not should_process_function
            or not any_tensor_optimizable(args, kwargs)
        ):
            return out

        func_base_name = get_func_base_name(func)

        curr_func_count = self.preregistration_tracker.get_function_call_count(
            self.current_module_name, func_base_name
        )

        # Get op-level compression components.
        # For functions not registered as supported functions, disable all op level quantization
        # for them by using empty op_compression_components. Module level specifications will
        # still be checked separately.
        op_compression_components = OpCompressionComponents()
        if self.supported_ops_registry.supports_operation(func):
            # Get op-level compression components
            local_func_name = get_func_name(func, curr_func_count)
            func_name_with_module_name = (
                local_func_name
                if self.current_module_name == ""
                else f"{self.current_module_name}.{local_func_name}"
            )
            op_compression_components = self._get_op_compression_components(
                func_name_with_module_name,
                self.supported_ops_registry.get_func_type(func),
                module_components,
            )

        # Collect pending input registrations
        pending_inputs = self._preregister_input_and_state_optimization(
            func, curr_func_count, args, kwargs, op_compression_components
        )

        # Collect pending output registrations
        pending_outputs = self._preregister_output_optimization(
            func, curr_func_count, out, op_compression_components
        )

        # Record pending registrations
        self.preregistration_tracker.record_function_call(
            self.current_module_name, func_base_name, func, pending_inputs, pending_outputs
        )
        return out


class ActivationEagerOptimizationHandler(ScopedEagerOptimizationModeBase):
    def __init__(
        self,
        model: torch.nn.Module,
        compression_config: CompressionConfig,
        module_components_dict: Mapping[NamedModule, ModuleCompressionComponents],
        supported_ops_registry: type[BaseSupportedOpsRegistry],
        optimization_type_name: str,
        reference_tracker: RegisteredOptimizersTracker,
    ) -> None:
        hooks_allow_list = [
            module for name, module in model.named_modules()
            if reference_tracker.has_module(name)
        ]

        super().__init__(
            model=model,
            compression_config=compression_config,
            module_components_dict=module_components_dict,
            supported_ops_registry=supported_ops_registry,
            optimization_type_name=optimization_type_name,
            hooks_allow_list=hooks_allow_list,
        )
        self.reference_tracker = reference_tracker
        self.reenter_ctx_on_module_exit: set[nn.Module] = set()
        self._locally_disabled_hooks = False

    @contextlib.contextmanager
    def locally_disable_hooks(self) -> Generator:
        self._locally_disabled_hooks = True
        try:
            yield
        finally:
            self._locally_disabled_hooks = False

    def _maybe_force_cleanup_on_error(self) -> None:
        """
        Force cleanup of TorchFunctionMode state when an error occurs.

        This method is called from exception handlers in exit hooks to ensure
        TorchFunctionMode is properly disabled even when exceptions occur during
        forward pass, preventing contamination of subsequent operations.

        This is a best-effort cleanup that swallows any errors to avoid masking
        the original exception.
        """
        if self.is_entered:
            try:
                self.is_entered = False
                # Clear tracking state
                self.parents.clear()
                self.reenter_ctx_on_module_exit.clear()
                self.registered_optimizers_tracker = RegisteredOptimizersTracker()
                # Force exit TorchFunctionMode regardless of parent stack state
                if isinstance(_get_current_function_mode(), self.__class__):
                    super().__exit__(None, None, None)
            except Exception:
                error_msg = (
                    "An error occurred during TorchFunctionMode cleanup. TorchFunctionMode may "
                    "still be active as a result."
                )
                logger.error(error_msg, exc_info=True)

    def fallback_enter_module(self, name: str) -> Callable:
        def f(module: nn.Module, inputs: Any) -> None:
            if self._locally_disabled_hooks:
                return
            self.parents.append(NamedModule(name, module))
            if self.is_entered:
                if self.reference_tracker.has_module(name):
                    error_msg = (
                        f"Module '{name}' found in reference tracker but not hooked for "
                        "optimization."
                    )
                    raise RuntimeError(error_msg)
                self.__exit__(None, None, None)
                self.reenter_ctx_on_module_exit.add(module)

        return f

    def fallback_exit_module(self, name: str) -> Callable:
        def f(module: nn.Module, inputs: Any, outputs: Any) -> None:
            if self._locally_disabled_hooks:
                return

            try:
                assert self.current_module is module, (
                    f"Current module ({self.current_module}) is not the module being exited "
                    f"({module})."
                )
                self.parents.pop()
                if module in self.reenter_ctx_on_module_exit:
                    self.__enter__()
                    self.reenter_ctx_on_module_exit.remove(module)
                elif self.is_entered and not self.parents:
                    self.__exit__(None, None, None)
            except Exception:
                # Force cleanup and re-raise the original exception
                self._maybe_force_cleanup_on_error()
                raise

        return f

    def enter_module(self, name: str) -> Callable:
        def f(module: nn.Module, inputs: Any) -> None:
            if self._locally_disabled_hooks:
                return
            self.parents.append(NamedModule(name, module))
            self.registered_optimizers_tracker.initialize_module(name)
            # If the ctx manager is not entered and the module is among those with
            # optimization registered on, we enter the ctx
            if not self.is_entered:
                if not self.reference_tracker.has_module(name):
                    error_msg = f"Module '{name}' not found in reference tracker."
                    raise RuntimeError(error_msg)
                self.__enter__()

        return f

    def exit_module(self, name: str) -> Callable:
        def f(module: nn.Module, inputs: Any, outputs: Any) -> None:
            if self._locally_disabled_hooks:
                return

            try:
                assert self.current_module is module, (
                    f"Current module ({self.current_module}) is not the module being exited "
                    f"({module})."
                )
                if not self.reference_tracker.has_module(self.current_module_name):
                    error_msg = (
                        f"Module '{self.current_module_name}' not found in reference tracker."
                    )
                    raise RuntimeError(error_msg)

                # Validate and reset function counts for this module
                self.registered_optimizers_tracker.validate_against_reference(
                    self.reference_tracker, self.current_module_name
                )
                # Reset the function counts for this module after validation
                self.registered_optimizers_tracker.reset_module(self.current_module_name)

                self.parents.pop()
                if self.is_entered and not self.parents:
                    # Reset registered_optimizers_tracker upon exiting uppermost module
                    self.registered_optimizers_tracker.reset_all()
                    self.__exit__(None, None, None)
            except Exception:
                # Force cleanup and re-raise the original exception
                self._maybe_force_cleanup_on_error()
                raise

        return f

    def _apply_activation_optimizer_if_exists(
        self,
        func: Callable,
        func_counter: int,
        value: Any,
        *name_parts: str | int
    ) -> tuple[torch.Tensor, str | None]:
        """Apply activation optimizer to a single tensor.

        Args:
            func: Function being optimized
            func_counter: Function counter for naming
            value: Value to potentially optimize
            *name_parts: Variable parts for building optimizer name

        Returns:
            Tuple of (optimized_value, optimizer_name) if optimizer exists,
            else (value, None)
        """
        optimizer_name = self._build_optimizer_name(
            get_func_name(func, func_counter),
            self.optimization_type_name,
            *name_parts
        )
        if hasattr(self.current_module, optimizer_name):
            optimizer = self.current_module.get_submodule(optimizer_name)
            return tree_map(optimizer, value), optimizer_name
        return value, None

    def _input_optimization(
        self, func: Callable, func_counter, args: tuple, kwargs: dict
    ) -> tuple[tuple, dict, list[str]]:
        """ Handle optimizing input activations of func. """
        inputs_quantized: list[str] = []
        normalized_args, normalized_kwargs = normalize_args_kwargs(func, args, kwargs)
        mutable_args = list(normalized_args)

        # Process positional arguments
        for idx in range(len(mutable_args)):
            mutable_args[idx], optimizer_name = (
                self._apply_activation_optimizer_if_exists(
                    func, func_counter, mutable_args[idx], idx
                )
            )
            if optimizer_name is not None:
                inputs_quantized.append(optimizer_name)

        # Process keyword arguments
        for kwarg_name in list(normalized_kwargs.keys()):
            normalized_kwargs[kwarg_name], optimizer_name = (
                self._apply_activation_optimizer_if_exists(
                    func, func_counter, normalized_kwargs[kwarg_name], kwarg_name
                )
            )
            if optimizer_name is not None:
                inputs_quantized.append(optimizer_name)

        args_to_return, kwargs_to_return = self._get_args_kwargs_to_return(
            args,
            mutable_args,
            normalized_kwargs
        )
        return args_to_return, kwargs_to_return, inputs_quantized

    @staticmethod
    def _get_args_kwargs_to_return(
        args: tuple[Any],
        mutable_args: list[Any],
        normalized_kwargs: dict[str, Any],
    ) -> tuple[tuple[Any], dict[str, Any]]:
        """
        Given the original args, move an appropriate number of kwargs from
        normalized_kwargs to mutable_args to match the same number of args/kwargs as
        was originally given.
        """
        num_args_in_normalized_kwargs = len(args) - len(mutable_args)
        normalized_kwargs_list = list(normalized_kwargs.items())
        return_args = (
            mutable_args +
            [v for (_, v) in normalized_kwargs_list][:num_args_in_normalized_kwargs]
        )
        return_kwargs = dict(normalized_kwargs_list[num_args_in_normalized_kwargs:])
        return tuple(return_args), return_kwargs

    def _output_optimization(
        self, func: Callable, func_counter: int, outputs: torch.Tensor | tuple
    ) -> tuple[torch.Tensor | tuple, list[str]]:
        """ Handle optimizing output activations of func. """
        # Handle both single tensor and tuple inputs
        outputs_quantized: list[str] = []
        was_single_tensor = not isinstance(outputs, tuple)
        if was_single_tensor:
            outputs = (outputs,)

        mutable_outputs = list(outputs)

        # Process all outputs
        for idx in range(len(mutable_outputs)):
            mutable_outputs[idx], optimizer_name = (
                self._apply_activation_optimizer_if_exists(
                    func, func_counter, mutable_outputs[idx], "output", idx
                )
            )
            if optimizer_name is not None:
                outputs_quantized.append(optimizer_name)

        outputs = tuple(mutable_outputs)
        # Return in the same format as input
        return (
            (outputs[0], outputs_quantized)
            if was_single_tensor
            else (outputs, outputs_quantized)
        )

    def __torch_function__(
        self,
        func: Callable,
        types: list,
        args: tuple = (),
        kwargs: dict | None = None
    ) -> Any:
        if kwargs is None:
            kwargs = {}

        func_base_name = get_func_base_name(func)
        if not self.reference_tracker.has_function(
            self.current_module_name,
            func_base_name
        ):
            # Functions like detach() can appear here when not seen earlier, if the
            # model was run with no_grad() for instance. If a function was not seen
            # earlier, simply shortcut and return without quantizing.
            return func(*args, **kwargs)

        func_counter = self.registered_optimizers_tracker.get_function_call_count(
            self.current_module_name, func_base_name
        )

        # We don't want the call to the input to trigger any hook
        # when calling a module from inside __torch_function__
        with self.locally_disable_hooks():
            # Apply input optimization
            args, kwargs, inputs_quantized = self._input_optimization(
                func, func_counter, args, kwargs
            )
            out = func(*args, **kwargs)
            # Apply output optimization
            out, outputs_quantized = self._output_optimization(
                func, func_counter, out
            )
            self.registered_optimizers_tracker.record_registration(
                self.current_module_name,
                func_base_name,
                inputs_quantized,
                outputs_quantized
            )
            return out
