# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Shared pytest fixtures for docs tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from coreai_opt._utils.repo_utils import find_repo_root

# Add docs scripts to path so tests can import them
_repo_root = find_repo_root(__file__)
_scripts_dir = _repo_root / "docs" / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Return the repository root path."""
    return _repo_root
