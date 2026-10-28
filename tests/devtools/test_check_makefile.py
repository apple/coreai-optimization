# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the check-makefile pre-commit hook.

Runs the script via subprocess to match how pre-commit invokes it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts._utils import find_repo_root

SCRIPT = find_repo_root(Path(__file__)) / "scripts" / "pre_commit" / "check_makefile.py"


def _run_checker(cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


class TestCheckMakefile:
    def test_passes_when_phony_matches_targets(self, tmp_path: Path) -> None:
        (tmp_path / "Makefile").write_text(
            ".PHONY: all build\n\nall:\n\t@echo\n\nbuild:\n\t@echo\n",
        )
        result = _run_checker(tmp_path)
        assert result.returncode == 0

    def test_fails_when_target_missing_from_phony(self, tmp_path: Path) -> None:
        (tmp_path / "Makefile").write_text(
            ".PHONY: all\n\nall:\n\t@echo\n\nbuild:\n\t@echo\n",
        )
        result = _run_checker(tmp_path)
        assert result.returncode == 1
        assert "Target 'build' is defined but missing from .PHONY" in result.stdout

    def test_fails_when_phony_has_stale_entry(self, tmp_path: Path) -> None:
        (tmp_path / "Makefile").write_text(
            ".PHONY: all ghost\n\nall:\n\t@echo\n",
        )
        result = _run_checker(tmp_path)
        assert result.returncode == 1
        assert "'ghost' is listed in .PHONY but has no target definition" in result.stdout

    def test_fails_when_phony_unsorted(self, tmp_path: Path) -> None:
        (tmp_path / "Makefile").write_text(
            ".PHONY: build all\n\nall:\n\t@echo\n\nbuild:\n\t@echo\n",
        )
        result = _run_checker(tmp_path)
        assert result.returncode == 1
        assert "not in alphabetical order" in result.stdout

    def test_finds_lowercase_makefile(self, tmp_path: Path) -> None:
        (tmp_path / "makefile").write_text(
            ".PHONY: all\n\nall:\n\t@echo\n",
        )
        result = _run_checker(tmp_path)
        assert result.returncode == 0

    def test_fails_when_no_makefile_exists(self, tmp_path: Path) -> None:
        result = _run_checker(tmp_path)
        assert result.returncode == 1
        assert "no Makefile found" in result.stdout
