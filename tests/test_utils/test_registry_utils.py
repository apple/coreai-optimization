# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import pytest
from pydantic import BaseModel, ConfigDict

from coreai_opt._utils.registry_utils import (
    ClassRegistryMixin,
    ConfigRegistryMixin,
    FunctionRegistryMixin,
)


class TestRegistryMixins:
    def test_class_registry(self):
        class MyClassRegistry(ClassRegistryMixin):
            pass

        @MyClassRegistry.register("class_a")
        class ClassA:
            def __init__(self, val=1):
                self.val = val

        @MyClassRegistry.register("class_b")
        class ClassB:
            pass

        assert len(MyClassRegistry.REGISTRY) == 2
        assert MyClassRegistry.get_class("class_a") == ClassA
        assert MyClassRegistry.get_class("class_b") == ClassB

        assert MyClassRegistry.list_registry_keys() == {"class_a", "class_b"}
        assert MyClassRegistry.list_registry_values() == {ClassA, ClassB}

        # resolve() returns the registered class for both string keys and class types.
        assert MyClassRegistry.resolve("class_a") is ClassA
        assert MyClassRegistry.resolve(ClassB) is ClassB

        # Unknown string key surfaces a ValueError that lists registered keys.
        with pytest.raises(ValueError, match="class_c"):
            MyClassRegistry.resolve("class_c")

        # An unregistered class also raises ValueError.
        class UnregisteredClass:
            pass

        with pytest.raises(ValueError, match="UnregisteredClass"):
            MyClassRegistry.resolve(UnregisteredClass)

        # Test overwriting registry name
        @MyClassRegistry.register("class_a")
        class NewClassA:
            pass

        assert MyClassRegistry.get_class("class_a") == NewClassA

        with pytest.raises(KeyError):
            MyClassRegistry.get_class("class_c")

    def test_function_registry(self):
        class MyFunctionRegistry(FunctionRegistryMixin):
            pass

        @MyFunctionRegistry.register("func_a")
        def func_a():
            return "a"

        @MyFunctionRegistry.register("func_b")
        def func_b():
            return "b"

        assert MyFunctionRegistry.get_function("func_a") is func_a
        assert MyFunctionRegistry.get_function("func_b") is func_b
        assert MyFunctionRegistry.get_function("func_a")() == "a"

        assert MyFunctionRegistry.list_registry_keys() == {"func_a", "func_b"}
        assert MyFunctionRegistry.list_registry_values() == {func_a, func_b}

    def test_config_registry(self):
        class MyConfigRegistry(ConfigRegistryMixin):
            pass

        @MyConfigRegistry.register("class_a")
        class ClassA:
            def __init__(self, arg1=1, arg2="foo"):
                self.arg1 = arg1
                self.arg2 = arg2

        config1 = {"type": "class_a"}
        instance1 = MyConfigRegistry.maybe_build_from_dict(config1)
        assert isinstance(instance1, ClassA)
        assert instance1.arg1 == 1
        assert instance1.arg2 == "foo"

        instance2 = MyConfigRegistry.maybe_build_from_dict(ClassA())
        assert isinstance(instance1, ClassA)
        assert instance1.arg1 == 1
        assert instance1.arg2 == "foo"

        config2 = {"type": "class_a", "arg1": 10, "arg2": "bar"}
        instance2 = MyConfigRegistry.maybe_build_from_dict(config2)
        assert isinstance(instance2, ClassA)
        assert instance2.arg1 == 10
        assert instance2.arg2 == "bar"

        with pytest.raises(ValueError):
            MyConfigRegistry.maybe_build_from_dict({"val": 5})

    def test_pydantic_class_registry(self):
        """Ensure pydantic registry classes don't have shared registries"""

        class RegistryClassA(BaseModel, ConfigRegistryMixin):
            model_config = ConfigDict(frozen=True, extra="forbid")

        class RegistryClassB(BaseModel, ConfigRegistryMixin):
            model_config = ConfigDict(frozen=True, extra="forbid")

        @RegistryClassA.register("type_1")
        class RegistryClassAType1(RegistryClassA):
            pass

        @RegistryClassA.register("type_2")
        class RegistryClassAType2(RegistryClassA):
            pass

        @RegistryClassB.register("type_x")
        class RegistryClassBTypeX(RegistryClassB):
            pass

        @RegistryClassB.register("type_y")
        class RegistryClassBTypeY(RegistryClassB):
            pass

        assert len(RegistryClassA.REGISTRY) == 2
        assert len(RegistryClassB.REGISTRY) == 2

        assert "type_1" in RegistryClassA.REGISTRY
        assert "type_2" in RegistryClassA.REGISTRY
        assert "type_x" in RegistryClassB.REGISTRY
        assert "type_y" in RegistryClassB.REGISTRY
