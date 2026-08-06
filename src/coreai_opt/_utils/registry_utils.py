# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import logging
from abc import ABC
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, ClassVar, TypeVar, cast

logger = logging.getLogger(__name__)

__all__: list[str] = [
    "ClassRegistryMixin",
    "ConfigRegistryMixin",
    "FunctionRegistryMixin",
    "RegistryMixin",
]

T = TypeVar("T")
TYPE_ = "type"


class RegistryMixin(ABC):
    REGISTRY: ClassVar["OrderedDict[Any, Any]"]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.REGISTRY = OrderedDict()

    @classmethod
    def register(cls, key: Any) -> Callable[[T], T]:
        def inner_wrapper(wrapped_obj: T) -> T:
            # Ensure this class has its own registry before registering
            # This is a safety net for cases where Pydantic BaseModel might interfere
            # with the registry creation in __init_subclass__
            if not hasattr(cls, "REGISTRY") or "REGISTRY" not in cls.__dict__:
                cls.REGISTRY = OrderedDict()

            obj_name = getattr(wrapped_obj, "__name__", "UnknownObject")
            if key in cls.REGISTRY:
                logger.warning(
                    f"Key: {key} is already registered with object: "
                    f"{cls.REGISTRY[key].__name__} "
                    f"in registry: {cls.__name__}. "
                    f"Over-writing the key with new object: {obj_name}"
                )
            cls.REGISTRY[key] = wrapped_obj
            return wrapped_obj

        return inner_wrapper

    @classmethod
    def _get_object(cls, key: Any) -> Any:
        if key in cls.REGISTRY:
            return cls.REGISTRY[key]
        raise KeyError(f"No object is registered with key: {key} in registry {cls.__name__}.")

    @classmethod
    def list_registry_keys(cls) -> set[str]:
        return set(cls.REGISTRY.keys())

    @classmethod
    def list_registry_values(cls) -> set[Any]:
        return set(cls.REGISTRY.values())


class ClassRegistryMixin(RegistryMixin):
    @classmethod
    def get_class(cls, key: Any) -> Any:
        return cls._get_object(key)

    @classmethod
    def resolve(cls, data: str | type) -> type:
        """Resolve a string key or class type against this registry.

        Args:
            data (str | type): Either a string key registered in the registry, or a
                class type. If a class is given, it is returned unchanged provided
                it is one of the registered values.

        Returns:
            type: The registered class corresponding to ``data``.

        Raises:
            ValueError: If ``data`` is a string not registered as a key, or a class
                that is not registered as a value.
        """
        if isinstance(data, str):
            if data in cls.REGISTRY:
                return cls.get_class(data)  # type: ignore[no-any-return]
            raise ValueError(
                f"No class is registered with key: '{data}' "
                f"in registry {cls.__name__}. "
                f"Available keys: {sorted(cls.list_registry_keys())}"
            )
        if data in cls.list_registry_values():
            return data
        name = getattr(data, "__name__", data)
        available_classes = sorted(c.__name__ for c in cls.list_registry_values())
        raise ValueError(
            f"{name} is not a registered class in {cls.__name__}. "
            f"Available classes: {available_classes}"
        )


class FunctionRegistryMixin(RegistryMixin):
    @classmethod
    def get_function(cls, key: Any) -> Any:
        return cls._get_object(key)


ConfigType = TypeVar("ConfigType", bound="ConfigRegistryMixin")


class ConfigRegistryMixin(ClassRegistryMixin):
    @classmethod
    def maybe_build_from_dict(
        cls: type[ConfigType], data: ConfigType | dict[str, Any]
    ) -> ConfigType:
        """
        Construct class from a dictionary object that has been registered with the
        name pointed by the `type` key.
        """

        # TODO: See if there is a way to do this recurisvely
        if isinstance(data, dict):
            if "type" not in data:
                raise ValueError(f"Missing {TYPE_} key in dict for {cls.__name__}")
            class_type = data[TYPE_]
            # Type cast is safe here because we expect registered classes to be
            # subclasses of ConfigRegistryMixin, and the registry pattern ensures
            # that only compatible classes are registered
            data_without_type = {k: v for k, v in data.items() if k != TYPE_}
            return cast(ConfigType, cls.get_class(class_type)(**data_without_type))
        return data
