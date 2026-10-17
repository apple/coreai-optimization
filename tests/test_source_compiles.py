# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Check that the package source compiles without warnings."""

import warnings
from pathlib import Path

import pytest

import coreai_opt

_SRC_ROOT = Path(coreai_opt.__file__).parent
_SOURCE_FILES = sorted(_SRC_ROOT.rglob("*.py"))


@pytest.mark.parametrize("path", _SOURCE_FILES, ids=lambda p: str(p.relative_to(_SRC_ROOT.parent)))
def test_source_compiles_without_warnings(path: Path) -> None:
    # Things like invalid escape sequences only warn at compile time, and the
    # warning is lost once a .pyc is cached, so compile from source directly.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
