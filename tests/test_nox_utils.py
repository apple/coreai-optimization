# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for ci/nox/utils.py."""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from packaging.specifiers import SpecifierSet

from ci.nox.utils import (
    _get_minimum_python_minor_version,
    get_supported_python_versions,
)


@contextmanager
def mock_pyproject(tmp_path: Path, content: str, patch_target: str = "ci.nox.utils.REPO_ROOT"):
    """Context manager for creating mock pyproject.toml and patching REPO_ROOT.

    Args:
        tmp_path: Temporary directory path
        content: Content to write to pyproject.toml
        patch_target: Module path to patch (default: ci.nox.utils.REPO_ROOT)
    """
    (tmp_path / "pyproject.toml").write_text(content)
    with patch(patch_target, tmp_path):
        yield


class TestGetMinimumPythonMinorVersion:
    """Tests for _get_minimum_python_minor_version."""

    @pytest.mark.parametrize(
        ("specifier_str", "expected"),
        [
            (">=3.10", 10),
            (">=3.11", 11),
            (">=3.10, <3.13", 10),
            (">=3.9, <3.12", 9),
            (">3.10", 10),
            (">3.10, <=3.12", 10),
        ],
    )
    def test_extracts_minimum_version(self, specifier_str: str, expected: int) -> None:
        """Test that the minimum minor version is correctly extracted."""
        specifier = SpecifierSet(specifier_str)
        assert _get_minimum_python_minor_version(specifier) == expected

    def test_raises_error_when_no_lower_bound(self) -> None:
        """Test that ValueError is raised when no lower bound is specified."""
        specifier = SpecifierSet("<3.13")
        with pytest.raises(ValueError, match="No lower bound found in specifier"):
            _get_minimum_python_minor_version(specifier)


class TestGetSupportedPythonVersions:
    """Tests for get_supported_python_versions."""

    def test_returns_list_of_version_strings(self) -> None:
        """Test that the function returns a list of version strings."""
        versions = get_supported_python_versions()
        assert isinstance(versions, list)
        assert len(versions) > 0
        for version in versions:
            assert isinstance(version, str)
            assert version.startswith("3.")

    @pytest.mark.parametrize(
        ("requires_python", "expected_versions"),
        [
            (">=3.10, <3.13", ["3.10", "3.11", "3.12"]),
            (">=3.11, <3.14", ["3.11", "3.12", "3.13"]),
            (">=3.9, <3.11", ["3.9", "3.10"]),
            (">=3.10, <3.11", ["3.10"]),
        ],
    )
    def test_returns_correct_versions_for_specifier(
        self, tmp_path: Path, requires_python: str, expected_versions: list[str]
    ) -> None:
        """Test that correct versions are returned for a given specifier."""
        content = f"""
[project]
name = "test-project"
requires-python = "{requires_python}"
"""
        with mock_pyproject(tmp_path, content):
            versions = get_supported_python_versions()
        assert versions == expected_versions
