# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from pathlib import Path

import pytest

from scripts._utils import find_repo_root


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path


def test_nearest_pyproject_toml_wins(tmp_path: Path) -> None:
    _touch(tmp_path / "pyproject.toml")
    inner = tmp_path / "inner"
    _touch(inner / "pyproject.toml")
    assert find_repo_root(inner) == inner


def test_file_start_is_searched_from_its_directory(tmp_path: Path) -> None:
    _touch(tmp_path / "pyproject.toml")
    assert find_repo_root(_touch(tmp_path / "pkg" / "mod.py")) == tmp_path


def test_marker_overrides_pyproject_toml(tmp_path: Path) -> None:
    _touch(tmp_path / "root.marker")
    inner = tmp_path / "inner"
    _touch(inner / "pyproject.toml")
    assert find_repo_root(inner, marker="root.marker") == tmp_path


def test_git_metadata_does_not_end_the_search(tmp_path: Path) -> None:
    _touch(tmp_path / "pyproject.toml")
    (tmp_path / "checkout" / ".git").mkdir(parents=True)
    assert find_repo_root(tmp_path / "checkout") == tmp_path


def test_raises_when_no_ancestor_has_the_marker(tmp_path: Path) -> None:
    # tmp_path's ancestors are outside the test's control and may hold a real
    # pyproject.toml, so search for a name no tree contains.
    marker = "no-such-marker"
    with pytest.raises(RuntimeError, match=marker):
        find_repo_root(tmp_path, marker=marker)
