# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import copy
import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import IO, Annotated, Any, Generic, Self, TypeAlias, TypeVar, overload

import torch
import torch.nn as nn
import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from coreai_opt._utils.config_utils import ConfigLevel as _ConfigLevel
from coreai_opt._utils.python_utils import (
    fqn as _fqn,
    get_generic_type_arg as _get_generic_type_arg,
)
from coreai_opt.config.spec import CompressionSpec


def _convert_none_to_empty_dict(v: Any):
    """If v is None, return an empty dict. Return v otherwise."""
    if v is None:
        return {}
    return v


def _convert_digit_str_keys_to_int(v: Any):
    """Coerce digit-string dict keys to int for fields that accept int keys."""
    if not isinstance(v, dict):
        return v

    def _try_int_key(k: Any) -> Any:
        if isinstance(k, str) and k.isascii() and k.isdigit():
            return int(k)
        return k

    result: dict[Any, Any] = {}
    original_keys: dict[Any, Any] = {}
    for k, val in v.items():
        new_key = _try_int_key(k)
        if new_key in result:
            msg = (
                f"Key collision detected: keys {original_keys[new_key]!r} and {k!r} "
                f"both convert to {new_key!r}"
            )
            raise ValueError(msg)
        result[new_key] = val
        original_keys[new_key] = k
    return result


_SpecT = TypeVar("_SpecT", bound=CompressionSpec)


class OpCompressionConfig(BaseModel, ABC, Generic[_SpecT]):
    """
    Abstract base configuration class for op-level compression settings.

    This generic class defines the structure for configuring compression at the
    operation level. Subclasses must implement the default spec providers to define
    compression-specific default values. Parameterized by ``_SpecT``, the compression
    spec type (e.g., QuantizationSpec, PalettizationSpec).

    Attributes:
        op_input_spec (dict[str | int, _SpecT | None] | None): Compression specifications
            for operation inputs. Keys can be either all indices or all string names,
            but not a mix of both. The special key "*" can be used in both cases to
            refer to all inputs. Values are compression spec objects or None to
            disable compression.

        op_output_spec (dict[str | int, _SpecT | None] | None): Compression
            specifications for operation outputs. Keys can be either all indices or all
            string names, but not a mix of both. The special key "*" can be used in both
            cases to refer to all outputs. Values are compression spec objects or None
            to disable compression.

        op_state_spec (dict[str, _SpecT | None] | None): Compression specifications for
            operation state tensors (parameters, buffers, constants). Keys are string
            names (e.g., "weight", "bias") or "*" to refer to all state inputs.
            Values are compression spec objects or None to disable compression.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid", validate_assignment=True
    )

    op_input_spec: Annotated[
        dict[str | int, _SpecT | None] | None,
        BeforeValidator(_convert_none_to_empty_dict),
        BeforeValidator(_convert_digit_str_keys_to_int),
    ]
    op_output_spec: Annotated[
        dict[str | int, _SpecT | None] | None,
        BeforeValidator(_convert_none_to_empty_dict),
        BeforeValidator(_convert_digit_str_keys_to_int),
    ]
    op_state_spec: Annotated[
        dict[str, _SpecT | None] | None, BeforeValidator(_convert_none_to_empty_dict)
    ]

    @classmethod
    @abstractmethod
    def get_default_input_spec(cls) -> dict[str | int, _SpecT | None]:
        """
        Provide default input spec for this compression type.

        Override in subclasses to define compression-specific defaults.
        Return empty dict if this compression type doesn't apply to inputs.

        Returns:
            Dictionary mapping input identifiers to compression specs
        """
        pass

    @classmethod
    @abstractmethod
    def get_default_output_spec(cls) -> dict[str | int, _SpecT | None]:
        """
        Provide default output spec for this compression type.

        Override in subclasses to define compression-specific defaults.
        Return empty dict if this compression type doesn't apply to outputs.

        Returns:
            Dictionary mapping output identifiers to compression specs
        """
        pass

    @classmethod
    @abstractmethod
    def get_default_state_spec(cls) -> dict[str, _SpecT | None]:
        """
        Provide default state spec for this compression type.

        Override in subclasses to define compression-specific defaults.
        Return empty dict if this compression type doesn't apply to state.

        Returns:
            Dictionary mapping state tensor names to compression specs
        """
        pass

    @model_validator(mode="before")
    @classmethod
    def apply_defaults(cls, data: Any) -> Any:
        """
        Apply class-specific defaults before validation.

        This validator runs before field validation and applies the default specs
        defined by subclasses if the specs are not provided in the input data.

        Note: Explicitly passing None will result in an empty dict (via BeforeValidator)
        while omitting the field entirely will apply defaults.
        """
        if not isinstance(data, dict):
            return data

        if "op_input_spec" not in data:
            data["op_input_spec"] = cls.get_default_input_spec()
        if "op_output_spec" not in data:
            data["op_output_spec"] = cls.get_default_output_spec()
        if "op_state_spec" not in data:
            data["op_state_spec"] = cls.get_default_state_spec()
        return data


class WeightOnlyOpValidationMixin:
    """
    Mixin that adds weight-only validation to OpCompressionConfig subclasses.

    This mixin is for compression types that only apply to weights/state tensors
    and don't compress activations (inputs/outputs). Examples include palettization,
    pruning, and low-rank decomposition.

    This mixin:

    1. Provides default empty implementations for get_default_input_spec and
       get_default_output_spec (satisfying abstract methods)
    2. Adds a model_validator that rejects any op_input_spec or op_output_spec

    Note:
        The mixin MUST come first in the inheritance list due to Python's
        MRO and abstract method resolution.

    Example:
        >>> class MyOpConfig(WeightOnlyOpValidationMixin, OpCompressionConfig[MySpec]):
        ...     @classmethod
        ...     def get_default_state_spec(cls):
        ...         return {"weight": MySpec()}
    """

    @classmethod
    def get_default_input_spec(cls) -> dict:
        """Return empty dict as weight-only compression doesn't apply to inputs."""
        return {}

    @classmethod
    def get_default_output_spec(cls) -> dict:
        """Return empty dict as weight-only compression doesn't apply to outputs."""
        return {}

    @model_validator(mode="after")
    def validate_weight_only_op_constraint(self):
        """Ensure no input/output specs are set (weight-only compression)."""
        error_msg = (
            f"{self.__class__.__name__} does not support {{key}}. "
            f"This is a weight-only compression type that only supports op_state_spec."
        )
        if self.op_input_spec:
            raise ValueError(error_msg.format(key="op_input_spec"))
        if self.op_output_spec:
            raise ValueError(error_msg.format(key="op_output_spec"))
        return self


_OpConfigT = TypeVar("_OpConfigT", bound=OpCompressionConfig)


class ModuleCompressionConfig(BaseModel, Generic[_OpConfigT, _SpecT]):
    """
    Abstract base configuration class for module-level compression settings.

    This generic class defines the structure for configuring compression at the
    module level. Subclasses must implement the default spec providers to define
    compression-specific default values. Parameterized by ``_OpConfigT`` (the op-level
    config type, e.g., OpQuantizerConfig) and ``_SpecT`` (the compression spec type,
    e.g., QuantizationSpec).

    Attributes:
        op_input_spec (dict[str | int, _SpecT | None] | None): Compression specifications
            for operation inputs applied to all registered operations / patterns within
            this module that don't have a more specific configuration.

        op_output_spec (dict[str | int, _SpecT | None] | None): Compression
            specifications for operation outputs applied to all registered operations /
            patterns within this module that don't have a more specific configuration.

        op_state_spec (dict[str, _SpecT | None] | None): Compression specifications for
            operation state tensors applied to all registered operations / patterns
            within this module that don't have a more specific configuration.

        op_type_config (dict[str, _OpConfigT | None] | None): Operation type-specific
            configurations. Keys are operation type names, values are op-level config
            objects or None to disable compression for that type.

        op_name_config (dict[str, _OpConfigT | None] | None): Operation name-specific
            configurations. Keys are operation name patterns (supports regex), values
            are op-level config objects or None to disable compression.

        module_input_spec (dict[str | int, _SpecT | None] | None): Compression
            specifications for module inputs. Module input settings treat the module
            as an opaque entity.

        module_output_spec (dict[str | int, _SpecT | None] | None): Compression
            specifications for module outputs. Module output settings treat the module
            as an opaque entity.

        module_state_spec (dict[str, _SpecT | None] | None): Compression specifications
            for module state tensors (parameters, buffers, constants). Module state
            settings will override op state settings for the same state tensors.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid", validate_assignment=True
    )

    op_input_spec: Annotated[
        dict[str | int, _SpecT | None] | None,
        BeforeValidator(_convert_none_to_empty_dict),
        BeforeValidator(_convert_digit_str_keys_to_int),
    ]
    op_output_spec: Annotated[
        dict[str | int, _SpecT | None] | None,
        BeforeValidator(_convert_none_to_empty_dict),
        BeforeValidator(_convert_digit_str_keys_to_int),
    ]
    op_state_spec: Annotated[
        dict[str, _SpecT | None] | None, BeforeValidator(_convert_none_to_empty_dict)
    ]
    op_type_config: Annotated[
        dict[str, _OpConfigT | None] | None, BeforeValidator(_convert_none_to_empty_dict)
    ] = Field(default_factory=dict)
    op_name_config: Annotated[
        dict[str, _OpConfigT | None] | None, BeforeValidator(_convert_none_to_empty_dict)
    ] = Field(default_factory=dict)
    module_input_spec: Annotated[
        dict[str | int, _SpecT | None] | None,
        BeforeValidator(_convert_none_to_empty_dict),
        BeforeValidator(_convert_digit_str_keys_to_int),
    ] = Field(default_factory=dict)
    module_output_spec: Annotated[
        dict[str | int, _SpecT | None] | None,
        BeforeValidator(_convert_none_to_empty_dict),
        BeforeValidator(_convert_digit_str_keys_to_int),
    ] = Field(default_factory=dict)
    module_state_spec: Annotated[
        dict[str, _SpecT | None] | None, BeforeValidator(_convert_none_to_empty_dict)
    ] = Field(default_factory=dict)

    @classmethod
    def _get_op_config_class(cls) -> type[_OpConfigT] | None:
        """
        Get the op config class from generic type parameters.

        Assumes the class is a direct subclass of
        ModuleCompressionConfig[_OpConfigT, _SpecT].
        """
        return _get_generic_type_arg(cls, ModuleCompressionConfig, arg_index=0)

    @model_validator(mode="before")
    @classmethod
    def apply_defaults(cls, data: Any) -> Any:
        """
        Apply class-specific defaults before validation.

        Gets default specs from the _OpConfigT type parameter's default methods.

        Note: Explicitly passing None will result in an empty dict (via BeforeValidator)
        while omitting the field entirely will apply defaults.
        """
        if not isinstance(data, dict):
            return data

        op_config_cls = cls._get_op_config_class()
        if op_config_cls is not None:
            if "op_input_spec" not in data:
                data["op_input_spec"] = op_config_cls.get_default_input_spec()
            if "op_output_spec" not in data:
                data["op_output_spec"] = op_config_cls.get_default_output_spec()
            if "op_state_spec" not in data:
                data["op_state_spec"] = op_config_cls.get_default_state_spec()

        return data

    @model_validator(mode="after")
    def _normalize_none_op_configs(self) -> ModuleCompressionConfig:
        """Replace None values in op_type_config/op_name_config with disabled configs.

        A None value means "disable compression for this op type/name". This
        validator normalises it to an explicit OpConfig with empty specs.
        """
        op_config_cls = self._get_op_config_class()
        if op_config_cls is None:
            raise RuntimeError(
                f"Unable to generate empty op_config_cls for config type {type(self).__name__}."
            )

        disabled_op_cfg = op_config_cls(
            op_input_spec=None,
            op_output_spec=None,
            op_state_spec=None,
        )

        for cfg_name in ("op_type_config", "op_name_config"):
            cfg_dict = getattr(self, cfg_name)
            if any(v is None for v in cfg_dict.values()):
                setattr(
                    self,
                    cfg_name,
                    {k: disabled_op_cfg if v is None else v for k, v in cfg_dict.items()},
                )

        return self

    def _get_fake_module_kwargs(self) -> dict[str, Any]:
        """Get the settings to forward to the compression simulator module's constructor.

        Returns:
            Dictionary of field names to their values.
        """
        base_field_names = set(ModuleCompressionConfig.model_fields.keys())
        return {k: v for k, v in self.model_dump().items() if k not in base_field_names}


class WeightOnlyModuleValidationMixin:
    """
    Mixin that adds weight-only validation to ModuleCompressionConfig subclasses.

    This mixin is for compression types that only apply to weights/state tensors
    and don't compress activations (inputs/outputs).

    This mixin adds a model_validator that rejects activation specs:
    - op_input_spec, op_output_spec (op-level activations)
    - module_input_spec, module_output_spec (module-level activations)

    Only op_state_spec and module_state_spec are allowed.

    Note:
        The mixin should come first in the inheritance list for proper MRO resolution.

    Example:
        >>> class MyModuleConfig(
        ...     WeightOnlyModuleValidationMixin,
        ...     ModuleCompressionConfig[MyOpConfig, MySpec]
        ... ):
        ...     pass
    """

    @model_validator(mode="after")
    def validate_weight_only_module_constraint(self):
        """Ensure no activation specs are set (weight-only compression)."""
        error_msg = (
            f"{self.__class__.__name__} does not support {{key}}. "
            f"This is a weight-only compression type that only supports "
            f"op_state_spec and module_state_spec."
        )
        if self.op_input_spec:
            raise ValueError(error_msg.format(key="op_input_spec"))
        if self.op_output_spec:
            raise ValueError(error_msg.format(key="op_output_spec"))
        if self.module_input_spec:
            raise ValueError(error_msg.format(key="module_input_spec"))
        if self.module_output_spec:
            raise ValueError(error_msg.format(key="module_output_spec"))
        return self


_T = TypeVar("_T", bound=ModuleCompressionConfig)

ModuleConfigDict: TypeAlias = dict[_ConfigLevel, dict[str, _T]]


def _build_module_alias_map(
    model: torch.nn.Module,
) -> tuple[dict[str, list[str]], dict[int, str]]:
    """Return a mapping from each module's canonical name to all its registered names,
    and a reverse mapping from module id to canonical name.

    A "canonical" name is the first name returned by ``named_modules()`` (which
    deduplicates by object identity).  An "alias" is any additional name under
    which the same module object is reachable, arising when a model registers the
    same submodule under more than one attribute (e.g. HuggingFace wrappers that
    hoist backbone children to the top level).

    The first returned dict has one entry per canonical name; the value is a list
    containing the canonical name followed by any alias names, in registration
    order.  Modules with no aliases have a single-element list.

    The second returned dict maps id(module) → canonical name for all modules.

    Args:
        model: The model to inspect.

    Returns:
        Tuple of (canonical_to_aliases, id_to_canonical) where canonical_to_aliases
        maps canonical module name → list of all names, and id_to_canonical maps
        id(module) → canonical name.
    """
    id_to_canonical: dict[int, str] = {}
    canonical_to_aliases: dict[str, list[str]] = {}

    for name, module in model.named_modules(remove_duplicate=False):
        m_id = id(module)
        if m_id not in id_to_canonical:
            id_to_canonical[m_id] = name
            canonical_to_aliases[name] = [name]
        else:
            canonical_to_aliases[id_to_canonical[m_id]].append(name)

    return canonical_to_aliases, id_to_canonical


class CompressionConfig(BaseModel, Generic[_T]):
    """
    Top level configuration class for model compression.

    This class manages compression configurations at different scopes:
    - Global configuration (applies to all modules by default)
    - Module type configurations (applies to all modules of specific type)
    - Module name configurations (applies to module instances identified by name)

    The configuration lookup follows a hierarchical precedence, where more specific
    configurations override more general ones.

    Generic type _T must be a subclass of ModuleCompressionConfig.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, extra="forbid", validate_assignment=True
    )

    global_config: _T | None = None
    module_type_configs: Annotated[
        dict[str | type[nn.Module], _T | None], BeforeValidator(_convert_none_to_empty_dict)
    ] = {}
    module_name_configs: Annotated[
        dict[str, _T | None], BeforeValidator(_convert_none_to_empty_dict)
    ] = {}

    # Guards only_for() against double-redistribution.
    _global_config_disabled: bool = PrivateAttr(default=False)

    @model_validator(mode="before")
    @classmethod
    def _apply_default_global_config(cls, data: Any) -> Any:
        """Apply default global_config based on the generic type parameter _T."""
        if "global_config" not in data:
            module_config_cls = cls._get_module_config_class()
            if module_config_cls is not None:
                data["global_config"] = module_config_cls()

        return data

    @classmethod
    def _get_module_config_class(cls) -> type[_T] | None:
        """
        Get the module config class from generic type parameters.

        Assumes the class is a direct subclass of CompressionConfig[_T].
        """
        return _get_generic_type_arg(cls, CompressionConfig, arg_index=0)

    @model_validator(mode="after")
    def _normalize_none_module_configs(self) -> CompressionConfig:
        """Replace None values in global/module_type/module_name configs with disabled configs.

        A None value means "disable compression for this scope". This validator
        normalises it to an explicit ModuleCompressionConfig with empty specs, so
        downstream consumers never need to handle None.
        """
        # Use __dict__ to bypass validate_assignment which would re-trigger
        # model validators and cause infinite recursion.
        if self.global_config is None:
            self._global_config_disabled = True
        self.__dict__["global_config"] = self._normalize_none_config(self.global_config)

        for cfg_name in ("module_type_configs", "module_name_configs"):
            cfg_dict = getattr(self, cfg_name)
            if any(v is None for v in cfg_dict.values()):
                self.__dict__[cfg_name] = {
                    k: self._normalize_none_config(v) for k, v in cfg_dict.items()
                }

        return self

    @model_validator(mode="after")
    def _validate_global_config_restrictions(self) -> CompressionConfig:
        """Ensure global_config doesn't have module-level specs"""
        if self.global_config is not None and (
            self.global_config.module_input_spec
            or self.global_config.module_output_spec
            or self.global_config.module_state_spec
        ):
            error_msg = (
                "global_config cannot have module_input_spec, "
                "module_output_spec, or module_state_spec. "
                "These are only allowed in module_type_configs and "
                "module_name_configs."
            )
            raise ValueError(error_msg)
        return self

    @field_validator("module_type_configs", mode="after")
    @classmethod
    def _normalize_module_type_configs(cls, value):
        normalized_value = {}
        for module_type, config in value.items():
            module_type_str = cls._normalize_module_type(module_type)
            normalized_value[module_type_str] = config

        return normalized_value

    @staticmethod
    def _normalize_module_type(module_type: str | type[nn.Module]) -> str:
        """
        Normalize module type to a fully-qualified name string.

        Args:
            module_type: Either a fully-qualified name string or a PyTorch module type

        Returns:
            str: Normalized fully-qualified name of the module type
        """
        if isinstance(module_type, str):
            # module_type specified as a string
            if "." not in module_type:
                raise ValueError(f"Expected fully-qualified name, got {module_type}")
            return module_type

        if isinstance(module_type, type) and issubclass(module_type, nn.Module):
            # module_type specified as a type
            return _fqn(module_type)

        error_msg = (
            f"Keys in module_type_configs must be either fully-qualified "
            f"strings or nn.Module types, got {type(module_type)}"
        )
        raise TypeError(error_msg)

    def _normalize_none_config(self, config: _T | None) -> _T:
        """Return a disabled ModuleCompressionConfig if config is None, otherwise return as-is."""
        if config is not None:
            return config
        module_config_cls = self._get_module_config_class()
        if module_config_cls is None:
            error_msg = (
                f"Unable to generate empty module_config_cls for config type {type(self).__name__}."
            )
            raise RuntimeError(error_msg)

        return module_config_cls(
            op_input_spec=None,
            op_output_spec=None,
            op_state_spec=None,
            module_input_spec=None,
            module_output_spec=None,
            module_state_spec=None,
        )

    def set_global(self, config: _T | None) -> Self:
        """Set the global config.

        Accepts a ``ModuleCompressionConfig`` (the canonical form) or ``None``
        to disable compression globally.
        """
        self._global_config_disabled = config is None
        self.global_config = self._normalize_none_config(config)
        return self

    def set_module_type(
        self,
        module_type: str | type[nn.Module],
        config: _T | None,
    ) -> Self:
        """Set the module level compression config for a given module type.

        If the module level compression config for an existing module type was
        already set, the new config will override the old one.
        """
        module_type_str = self._normalize_module_type(module_type)
        self.module_type_configs[module_type_str] = self._normalize_none_config(config)
        return self

    def set_module_name(self, module_name: str, config: _T | None) -> Self:
        """Set the module level compression config for a given module instance.

        If the module level compression config for an existing module was
        already set, the new config will override the old one.
        """
        self.module_name_configs[module_name] = self._normalize_none_config(config)
        return self

    @overload
    def only_for(
        self,
        targets: list[str | type[nn.Module]] | tuple[str | type[nn.Module], ...],
        /,
    ) -> Self: ...

    @overload
    def only_for(self, *targets: str | type[nn.Module]) -> Self: ...

    def only_for(self, *targets: Any) -> Self:
        """Restrict this config to apply only to the given module types/names.

        Disables ``global_config`` and re-applies it as a deep-copied per-module
        override on each listed target. Targets may be ``nn.Module`` subclasses
        or module-name strings, mixed in the same call and passed either as
        varargs or as a single list/tuple. All targets are validated before any
        mutation happens.

        Args:
            *targets: One or more ``nn.Module`` subclasses or module name
                strings, passed as varargs or a single list/tuple.

        Returns:
            Self: ``self``, for chaining.

        Raises:
            ValueError: If no targets are provided, or if ``global_config`` is
                already disabled. Pass all targets in one call instead of
                chaining ``only_for``.
            TypeError: If a target is neither an ``nn.Module`` subclass nor a
                string.

        Note:
            If a target already has an explicit override (via ``set_module_type``
            or ``set_module_name``), ``only_for`` overwrites it with the former
            global config. To keep per-target customizations, call ``only_for``
            first and ``set_module_type`` / ``set_module_name`` after.

            The ``ValueError`` raised when ``only_for`` is called twice (see
            ``Raises``) uses a private attribute that is excluded from
            ``model_dump`` / ``to_yaml``, so a round-tripped config will
            accept ``only_for`` again (functionally a no-op).

        Example:
            >>> config = QuantizerConfig.presets.w8().only_for(nn.Linear, nn.Conv2d)
            >>> config = QuantizerConfig.presets.w8().only_for([nn.Linear, "lm_head"])
        """
        targets = self._unpack_targets(targets)
        if not targets:
            msg = "only_for requires at least one target"
            raise ValueError(msg)
        if self._is_global_config_disabled():
            msg = (
                "only_for requires a non-disabled global_config to "
                "redistribute as per-module overrides. If you've already "
                "called only_for or set_global(None), pass all targets in "
                "one only_for(...) call instead of chaining."
            )
            raise ValueError(msg)
        for target in targets:
            self._validate_target(target)

        spec = self.global_config
        self._global_config_disabled = True
        self.global_config = self._normalize_none_config(None)
        for target in targets:
            self._apply_target_config(target, copy.deepcopy(spec))
        return self

    @overload
    def without(
        self,
        targets: list[str | type[nn.Module]] | tuple[str | type[nn.Module], ...],
        /,
    ) -> Self: ...

    @overload
    def without(self, *targets: str | type[nn.Module]) -> Self: ...

    def without(self, *targets: Any) -> Self:
        """Exclude the given module types/names from this config.

        Each target gets a per-module override of ``None`` (disabled). The
        global config and other overrides are unchanged. Targets may be
        ``nn.Module`` subclasses or module-name strings, mixed in the same call
        and passed either as varargs or as a single list/tuple. All targets are
        validated before any mutation happens. Passing no targets (or an empty
        list) is a no-op.

        Args:
            *targets: Zero or more ``nn.Module`` subclasses or module name
                strings, passed as varargs or a single list/tuple.

        Returns:
            Self: ``self``, for chaining.

        Raises:
            TypeError: If a target is neither an ``nn.Module`` subclass nor a
                string.

        Example:
            >>> config = QuantizerConfig.presets.w4().without(nn.LayerNorm)
            >>> config = QuantizerConfig.presets.w4().without([nn.LayerNorm, "lm_head"])
        """
        targets = self._unpack_targets(targets)
        for target in targets:
            self._validate_target(target)
        for target in targets:
            self._apply_target_config(target, None)
        return self

    def _apply_target_config(self, target: str | type[nn.Module], config: _T | None) -> None:
        """Route ``target`` to ``set_module_type`` or ``set_module_name`` based on its kind."""
        if isinstance(target, type):
            self.set_module_type(target, config)
        else:
            self.set_module_name(target, config)

    @staticmethod
    def _unpack_targets(targets: tuple[Any, ...]) -> tuple[Any, ...]:
        """Unwrap a single list/tuple argument into its elements.

        Strings are not unpacked even though they are iterable — a bare string
        is a valid single target.
        """
        if len(targets) == 1 and isinstance(targets[0], (list, tuple)):
            return tuple(targets[0])
        return targets

    @staticmethod
    def _validate_target(target: Any) -> None:
        """Raise ``TypeError`` if ``target`` is not a string or ``nn.Module`` subclass."""
        if isinstance(target, str):
            return
        if isinstance(target, type):
            if issubclass(target, nn.Module):
                return
            msg = f"targets must be nn.Module subclasses or name strings, got {target.__name__}"
            raise TypeError(msg)
        msg = f"targets must be module types or name strings, got {type(target).__name__}"
        raise TypeError(msg)

    def _is_global_config_disabled(self) -> bool:
        """Return True if ``global_config`` was explicitly disabled."""
        return self._global_config_disabled

    def get_module_config(self, name: str, module: torch.nn.Module) -> _T:
        """
        Get the compression config for a module with priority.

        1. Module name match (supports regex)
        2. Module type match
        3. Global config

        Args:
            name (str): Name of module to get config for
            module(torch.nn.Module): Module to get config for

        Returns:
            _T: Module config for the given module.
        """
        # module name match
        for mod_name in self.module_name_configs:
            if re.fullmatch(mod_name, name):
                return self.module_name_configs[mod_name]

        # module type match
        module_fqn = _fqn(type(module))
        if module_fqn in self.module_type_configs:
            return self.module_type_configs[module_fqn]

        # fallback to global config
        return self.global_config

    def build_module_config_dict(self, model: torch.nn.Module) -> ModuleConfigDict[_T]:
        """
        Build a mapping of module names to their quantization configurations,
        separating modules by config level.

        The modules are associated with configs according to the following rules:

        - If a module is already associated with a config of a higher priority, the
          lower priority config is ignored (module_name > module_type > global)

        - A module may match with more than one config within the same config level.
          For example, a nested module named model.outer.inner could be associated
          with module name level configs for "model.outer.inner" as well as
          "model.outer.*".
          Whichever config is defined later in the config list is
          the one which ends up being used.

        - When a module is matched with a config, the config is applied to all of
          its child modules recursively, subject to the above priority constraint.
          For example, assume a nested module model.outer.inner.module and the
          following config::

              module_name: {"model.outer.inner": config1},
              module_type: {type(model): config2}

          Then we would have the following associations:

          - "model": config2 (set with module_type config)
          - "model.outer": config2 (recursively set when setting "model")
          - "model.outer.inner": config1 (higher priority module_name match)
          - "model.outer.inner.module": config1 (set as part of recursively
            processing "model.outer.inner"'s child modules)

        Args:
            model: The model with modules to get configs for

        Returns:
            Dictionary with nested dictionary mapping modules to configs for each config
            level
        """
        module_config_dict: ModuleConfigDict[_T] = {
            _ConfigLevel.MODULE_NAME: {},
            _ConfigLevel.MODULE_TYPE: {},
            _ConfigLevel.GLOBAL: {},
        }
        name_to_modules_dict = dict(model.named_modules())

        # Build alias map and id→canonical lookup in one pass.
        canonical_to_aliases, id_to_canonical = _build_module_alias_map(model)

        _visited: set[int] = set()

        def _apply_configs(
            identifier_and_config: Iterable[tuple[str, _T]],
            config_level: _ConfigLevel,
            matcher_fn: Callable[[str, str, nn.Module], bool],
        ) -> None:
            """Apply configs to modules matching the given criteria."""
            for identifier, config in identifier_and_config:
                # Use reversed for name_to_modules_dict in order to process nested
                # child modules before parents. This is only relevant for module level
                # settings since they do not pass on to child module settings. For op
                # level settings, because they are inherited by child modules, the order
                # of processing would not matter.
                for name, module in reversed(name_to_modules_dict.items()):
                    # Match against all known names (canonical + aliases) so that a
                    # user-supplied regex or type spec targeting an alias path still
                    # applies the config under the canonical name.
                    all_names = canonical_to_aliases.get(name, [name])
                    if any(matcher_fn(identifier, n, module) for n in all_names):
                        self._set_config_for_module(
                            name,
                            module,
                            module_config_dict,
                            config,
                            config_level,
                            _visited,
                            id_to_canonical,
                        )

        # Apply in priority order
        _apply_configs(
            reversed(self.module_name_configs.items()),
            _ConfigLevel.MODULE_NAME,
            lambda identifier, name, _: re.fullmatch(identifier, name),
        )
        _apply_configs(
            reversed(self.module_type_configs.items()),
            _ConfigLevel.MODULE_TYPE,
            lambda identifier, _, module: _fqn(type(module)) == identifier,
        )
        # Global config matches all remaining modules, no special id/matcher_fn needed
        _apply_configs([("", self.global_config)], _ConfigLevel.GLOBAL, lambda *_: True)
        return module_config_dict

    def _prepare_config_for_child(self, parent_config: _T) -> _T:
        """
        Prepare config for child modules by filtering out module-level specs.

        When recursively applying configs to child modules, all properties of the
        parent config are propagated EXCEPT module-level settings. This includes:
        - Op-level settings: op_input_spec, op_output_spec, op_state_spec,
          op_type_config, and op_name_config
        - Compression-specific properties

        Module-level settings (module_input_spec, module_output_spec,
        module_state_spec) are NOT inherited by children as they are specific
        to the parent module boundary.

        Args:
            parent_config: The config from the parent module

        Returns:
            Config with all properties except module-level settings.
        """
        # Create new config with only op-level settings
        child_config = copy.deepcopy(parent_config)
        # Omit module_input/output/state_spec
        child_config.module_input_spec = None
        child_config.module_output_spec = None
        child_config.module_state_spec = None

        return child_config

    def _set_config_for_module(
        self,
        module_name: str,
        module: torch.nn.Module,
        module_config_dict: ModuleConfigDict[_T],
        config: _T,
        config_level: _ConfigLevel,
        visited: set[int],
        id_to_canonical: dict[int, str],
    ):
        """
        Set a config for module_name in module_config_dict. Skip if the module_name
        exists in module_config_dict within an equal or higher config level.

        This function also recursively sets the same config for all child modules of
        the given module in accordance to the rule above.

        visited tracks module object ids already processed to prevent infinite recursion
        and duplicate entries. id_to_canonical maps id(module) to its canonical name
        (as yielded by named_modules()), ensuring module_config_dict keys are always
        canonical even when a module is reached via an alias path through named_children().
        """
        # Always use the canonical name so module_config_dict keys stay consistent
        # with module_name_to_state_names_map (which is also built from named_modules()).
        # This handles the case where named_children() reaches a shared module via a
        # non-canonical path (e.g. a shared ReLU at body.15.body.1 whose canonical
        # name is body.0.body.1).
        module_id = id(module)
        canonical_name = id_to_canonical.get(module_id, module_name)

        if module_id in visited:
            return
        visited.add(module_id)

        # Since configs within a config level are processed in reversed order, if a
        # module name already exists for the current config level, we can skip.
        # We also skip if the name exists in a higher config level since that should
        # always take priority.
        for level in _ConfigLevel.priority_order():
            if canonical_name in module_config_dict[level]:
                return
            if level == config_level:
                break

        module_config_dict[config_level][canonical_name] = config
        child_config = self._prepare_config_for_child(config)

        for child_name, child_module in module.named_children():
            # Construct the path from the canonical name so all entries remain
            # rooted at canonical paths.
            child_full_name = f"{canonical_name}.{child_name}" if canonical_name else child_name
            self._set_config_for_module(
                child_full_name,
                child_module,
                module_config_dict,
                child_config,
                config_level,
                visited,
                id_to_canonical,
            )

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> CompressionConfig | None:
        """
        Create configuration from a dictionary.

        The dictionary must contain _CONFIG_KEY key whose value
        defines the hierarchical configuration structure specifying which
        compression specs to apply at different scopes (global, module type,
        module name) and levels (op-level or module-level).
        All compression specifications should be inline dictionaries (not
        references). If this dictionary was created from YAML using from_yaml(),
        any YAML anchor/alias references have already been resolved by the YAML
        parser and substituted with the actual spec dictionaries.

        This method requires subclasses to define _CONFIG_KEY and _SPEC_KEY class
        attributes. It will:
        1. Validate that only expected keys are present
        2. Extract the nested config from config_dict[_CONFIG_KEY]
        3. Create the config from the extracted data

        Args:
            config_dict (:obj:`dict` of :obj:`str` and values): A nested dictionary
                of strings and values. Must have the key specified by _CONFIG_KEY
                with the actual config nested inside.

        Returns:
            CompressionConfig: The created configuration object, or None if
                config_dict is None.

        Raises:
            RuntimeError: If:
                - Subclass doesn't define _CONFIG_KEY or _SPEC_KEY
                - Unexpected keys are found in config_dict
                - The required _CONFIG_KEY is not present in config_dict

        Example:
            >>> # Define a sample spec class for this compression type
            >>> class MySpec(CompressionSpec):
            ...     _compression_type = CompressionType.QUANTIZATION
            ...     foo: str = "default"  # String property
            ...     bar: float = 1.0          # Float property
            >>>
            >>> # Subclass with config key pattern
            >>> class MyConfig(CompressionConfig):
            ...     _CONFIG_KEY: ClassVar[str] = "my_config"
            ...     _SPEC_KEY: ClassVar[str] = "my_spec"
            >>>
            >>> # Create configuration dictionary
            >>> # (Equivalent to a parsed YAML file with all anchors/aliases resolved)
            >>> # Note: Specs are inlined as dictionaries, not as MySpec objects
            >>> config_dict = {
            ...     'my_config': {
            ...         'global_config': {
            ...             'op_input_spec': {
            ...                 "*": {
            ...                     'foo': 'custom_1',
            ...                     'bar': 1.0
            ...                 }
            ...             },
            ...             'op_state_spec': {
            ...                 'weight': {
            ...                     'foo': 'custom_2',
            ...                     'bar': 0.5
            ...                 }
            ...             }
            ...         },
            ...         'module_type_configs': {
            ...             'torch.nn.modules.linear.Linear': {
            ...                 'op_state_spec': {
            ...                     'weight': {
            ...                         'foo': 'custom_3',
            ...                         'bar': 0.25
            ...                     }
            ...                 }
            ...             }
            ...         },
            ...         'module_name_configs': {
            ...             'model.encoder.layer1': {
            ...                 'module_output_spec': {
            ...                     "*": {
            ...                         'foo': 'default',
            ...                         'bar': 2.0
            ...                     }
            ...                 }
            ...             }
            ...         }
            ...     }
            ... }
            >>>
            >>> # Load the configuration
            >>> config = MyConfig.from_dict(config_dict)
            >>> # Access global config and its specs
            >>> print(config.global_config.op_state_spec["weight"].bar)
            >>> # 0.5

        """
        # Check if subclass defines required class attributes
        config_key = getattr(cls, "_CONFIG_KEY", None)
        spec_key = getattr(cls, "_SPEC_KEY", None)

        if config_key is None:
            raise RuntimeError(f"{cls.__name__} must define _CONFIG_KEY class attribute. ")

        if spec_key is None:
            raise RuntimeError(f"{cls.__name__} must define _SPEC_KEY class attribute. ")

        # Build expected keys
        expected_keys = {config_key, spec_key}

        # Validate keys
        for key in config_dict:
            if key not in expected_keys:
                error_msg = (
                    f"Found unexpected key '{key}' in config dict. "
                    f"Supported keys are {expected_keys}."
                )
                raise RuntimeError(error_msg)

        # Check required config key exists
        if config_key not in config_dict:
            error_msg = (
                f"Did not find '{config_key}' in config dict. Expected keys: {expected_keys}."
            )
            raise RuntimeError(error_msg)

        # Unwrap and create from nested config
        return cls(**config_dict[config_key])

    @classmethod
    def from_yaml(cls, yml: IO | str | Path) -> CompressionConfig | None:
        """
        Create configuration from a YAML file or stream.

        Args:
            yml: File path or IO stream containing YAML data

        Returns:
            A CompressionConfig instance or None if the YAML content was empty

        Raises:
            ValueError: If the YAML content is not a dictionary
        """
        # Handle different input types
        if isinstance(yml, str | Path):
            path = Path(yml)
            config_data = yaml.safe_load(path.read_bytes())
        else:
            # Handle IO stream directly
            config_data = yaml.safe_load(yml)

        # Handle empty content
        if config_data is None:
            warnings.warn(
                "Empty YAML content detected, returning None instead of a configuration object",
                stacklevel=2,
            )
            return None

        # Validate data type
        if not isinstance(config_data, dict):
            raise ValueError(f"Invalid YAML: expected dict, got {type(config_data)}.")

        # Create configuration from dictionary
        return cls.from_dict(config_data)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to dictionary.
        """
        config_key = getattr(self, "_CONFIG_KEY", None)

        if config_key is None:
            raise RuntimeError(
                f"{self.__class__.__name__} must define _CONFIG_KEY class attribute. "
            )
        return {config_key: self.model_dump()}
