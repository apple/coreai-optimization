# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for eager-mode compression component types."""

import pytest

from coreai_opt._utils.insertion.torch_function.types import (
    ModuleCompressionComponents,
    OpCompressionComponents,
)
from coreai_opt.palettization.kmeans.kmeans_fake_palettize import _KMeansFakePalettize


def test_op_components_disabled_entries_are_not_activation_components():
    """``{"*": None}`` disables the tensor, so no activation machinery is needed."""
    components = OpCompressionComponents(
        op_input_components={"*": None},
        op_output_components={"*": None},
        op_state_components={"*": None},
    )

    assert not components.has_activation_component()


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("op_input_components", True),
        ("op_output_components", True),
        ("op_state_components", False),
    ],
)
def test_op_components_live_entries_are_activation_components(field, expected):
    components = OpCompressionComponents(**{field: {"*": _KMeansFakePalettize.with_args()}})

    assert components.has_activation_component() is expected


def test_module_components_disabled_entries_are_not_activation_components():
    """The module-level mirror of the op-level disabled case."""
    components = ModuleCompressionComponents(
        weight={"*": None},
        input_activation={"*": None},
        output_activation={"*": None},
        module_input_components={"*": None},
        module_output_components={"*": None},
        module_state_components={"*": None},
    )

    assert not components.has_activation_component()


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("input_activation", True),
        ("output_activation", True),
        ("module_input_components", True),
        ("module_output_components", True),
        ("weight", False),
        ("module_state_components", False),
    ],
)
def test_module_components_live_entries_are_activation_components(field, expected):
    components = ModuleCompressionComponents(**{field: {"*": _KMeansFakePalettize.with_args()}})

    assert components.has_activation_component() is expected


def test_module_components_mixed_entries_are_activation_components():
    """One live entry among disabled ones still needs the activation machinery."""
    components = ModuleCompressionComponents(
        input_activation={0: None, 1: _KMeansFakePalettize.with_args()}
    )

    assert components.has_activation_component()


def test_module_components_disabled_op_components_are_not_activation_components():
    """Disabled entries nested under op_type/op_name are not activation components either."""
    disabled = OpCompressionComponents(op_input_components={"*": None})
    live = OpCompressionComponents(op_input_components={"*": _KMeansFakePalettize.with_args()})

    assert not ModuleCompressionComponents(
        op_type_components={"linear": disabled}
    ).has_activation_component()
    assert not ModuleCompressionComponents(
        op_name_components={"foo": disabled}
    ).has_activation_component()
    assert ModuleCompressionComponents(
        op_type_components={"linear": live}
    ).has_activation_component()
    assert ModuleCompressionComponents(op_name_components={"foo": live}).has_activation_component()
