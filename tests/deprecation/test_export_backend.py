# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for ExportBackend, including deprecated alias resolution."""

from __future__ import annotations

import warnings

import pytest

from coreai_opt import ExportBackend


class TestExportBackend:
    """Construction and deprecation behavior for ExportBackend."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("torch", ExportBackend.Torch),
            ("coreml", ExportBackend.CoreML),
            ("coreai", ExportBackend.CoreAI),
        ],
    )
    def test_construct_from_current_value_does_not_warn(
        self,
        value: str,
        expected: ExportBackend,
    ) -> None:
        """Resolve current values without emitting a DeprecationWarning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assert ExportBackend(value) is expected

    @pytest.mark.parametrize(
        ("old_name", "expected"),
        [
            ("MIL", ExportBackend.CoreML),
            ("MLIR", ExportBackend.CoreAI),
        ],
    )
    def test_deprecated_alias_warns_and_resolves(
        self,
        old_name: str,
        expected: ExportBackend,
    ) -> None:
        """Resolve deprecated names via attribute access and value lookup."""
        new_name = expected.name
        old_value = old_name.lower()

        with pytest.warns(
            DeprecationWarning,
            match=rf"ExportBackend\.{old_name} is deprecated",
        ) as record:
            assert getattr(ExportBackend, old_name) is expected
        assert f"ExportBackend.{new_name}" in str(record[0].message)

        with pytest.warns(
            DeprecationWarning,
            match=rf"ExportBackend\('{old_value}'\) is deprecated",
        ) as record:
            assert ExportBackend(old_value) is expected
        assert f"ExportBackend.{new_name}" in str(record[0].message)

    def test_deprecated_literal_attribute_access_resolves_for_clients(self) -> None:
        """Verify literal `ExportBackend.MIL` / `.MLIR` (what client code writes) still works."""
        with pytest.warns(
            DeprecationWarning,
            match=r"MIL is deprecated, use ExportBackend\.CoreML instead",
        ):
            assert ExportBackend.MIL is ExportBackend.CoreML

        with pytest.warns(
            DeprecationWarning,
            match=r"MLIR is deprecated, use ExportBackend\.CoreAI instead",
        ):
            assert ExportBackend.MLIR is ExportBackend.CoreAI
