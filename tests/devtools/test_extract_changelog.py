# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for scripts/pre_commit/extract_changelog.py."""

import subprocess
import sys
from pathlib import Path

import pytest

from scripts._utils import find_repo_root


@pytest.fixture
def tmp_changelog(tmp_path):
    """Create a temporary changelog file with standard content."""
    changelog = tmp_path / "CHANGELOG.md"
    content = """# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
### Added
- Upcoming feature

## [1.0.0] - 2026-01-27
### Added
- New feature
### Fixed
- Bug fix

## [0.9.0] - 2026-01-15
### Changed
- Minor update

[Unreleased]: https://github.com/owner/repo/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/owner/repo/compare/v0.9.0...v1.0.0
[0.9.0]: https://github.com/owner/repo/releases/tag/v0.9.0
"""
    changelog.write_text(content)
    return changelog


@pytest.fixture
def script_path():
    """Path to the extract_changelog.py script."""
    return find_repo_root(Path(__file__)) / "scripts" / "pre_commit" / "extract_changelog.py"


"""Test running extract_changelog.py."""


def test_extract_changelog(script_path, tmp_changelog):
    """Run script as subprocess and extract version."""
    result = subprocess.run(
        [sys.executable, str(script_path), "1.0.0", str(tmp_changelog)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    assert "New feature" in result.stdout
    assert "Bug fix" in result.stdout


def test_extract_unreleased(script_path, tmp_changelog):
    """Extract Unreleased section"""
    result = subprocess.run(
        [sys.executable, str(script_path), "Unreleased", str(tmp_changelog)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    assert "Upcoming feature" in result.stdout


def test_version_not_found(script_path, tmp_changelog):
    """Should exit with code 1 when version not found."""
    result = subprocess.run(
        [sys.executable, str(script_path), "99.99.99", str(tmp_changelog)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Error:" in result.stderr


def test_file_not_found(script_path, tmp_path):
    """Should exit with code 1 when file doesn't exist."""
    missing = tmp_path / "MISSING.md"
    result = subprocess.run(
        [sys.executable, str(script_path), "1.0.0", str(missing)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Error:" in result.stderr


def test_missing_version_argument(script_path):
    """Should exit with code 1 when version argument missing."""
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "Usage:" in result.stderr


def test_default_changelog_path(script_path, tmp_path, monkeypatch):
    """Should use CHANGELOG.md by default if no path specified."""
    # Change to temp directory
    monkeypatch.chdir(tmp_path)

    # Create CHANGELOG.md in current directory
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text("""# Changelog

## [1.0.0] - 2026-01-27
### Added
- Default path test
""")

    result = subprocess.run(
        [sys.executable, str(script_path), "1.0.0"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    assert "Default path test" in result.stdout


def test_empty_version_section(script_path, tmp_path):
    """Handle version with no content."""
    changelog = tmp_path / "CHANGELOG.md"
    content = """# Changelog

## [1.0.0] - 2026-01-27

## [0.9.0] - 2026-01-15
### Added
- Feature
"""
    changelog.write_text(content)

    result = subprocess.run(
        [sys.executable, str(script_path), "1.0.0", str(changelog)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    # Should return empty or minimal content
    assert result.stdout.strip() == ""


def test_version_with_special_characters(script_path, tmp_path):
    """Handle version with special characters."""
    changelog = tmp_path / "CHANGELOG.md"
    content = """# Changelog

## [1.0.0-beta.1] - 2026-01-27
### Added
- Beta feature
"""
    changelog.write_text(content)

    result = subprocess.run(
        [sys.executable, str(script_path), "1.0.0-beta.1", str(changelog)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    assert "Beta feature" in result.stdout


def test_multiple_categories(script_path, tmp_path):
    """Extract version with multiple change categories."""
    changelog = tmp_path / "CHANGELOG.md"
    content = """# Changelog

## [1.0.0] - 2026-01-27
### Added
- New feature A
- New feature B

### Changed
- Updated X
- Updated Y

### Deprecated
- Old API

### Removed
- Legacy code

### Fixed
- Bug 1
- Bug 2

### Security
- CVE fix
"""
    changelog.write_text(content)

    result = subprocess.run(
        [sys.executable, str(script_path), "1.0.0", str(changelog)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    assert all(
        section in result.stdout
        for section in ["Added", "Changed", "Deprecated", "Removed", "Fixed", "Security"]
    )
    assert "New feature A" in result.stdout
    assert "CVE fix" in result.stdout


def test_blank_lines_preserved(script_path, tmp_path):
    """Blank lines within a section should be preserved."""
    changelog = tmp_path / "CHANGELOG.md"
    content = """# Changelog

## [1.0.0] - 2026-01-27
### Added
- Feature A

- Feature B (with blank line above)
"""
    changelog.write_text(content)

    result = subprocess.run(
        [sys.executable, str(script_path), "1.0.0", str(changelog)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0
    lines = result.stdout.split("\n")
    # Check that blank line is preserved
    assert "" in lines  # Empty string represents blank line
