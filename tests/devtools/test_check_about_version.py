# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the check-about-version pre-commit hook.

Runs the script via subprocess, the same way pre-commit invokes it, against
a throwaway git repo built under ``tmp_path``, so tagging never touches this
repo's own history.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts._utils import find_repo_root

SCRIPT = find_repo_root(Path(__file__)) / "scripts" / "pre_commit" / "check_about_version.py"


def _write_about(repo: Path, *, latest_released: str, version: str) -> None:
    about_dir = repo / "src" / "coreai_opt"
    about_dir.mkdir(parents=True, exist_ok=True)
    (about_dir / "_about.py").write_text(
        f'latest_released_version = "{latest_released}"\n__version__ = "{version}"\n',
        encoding="utf-8",
    )


def _init_repo(repo: Path) -> None:
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    # This throwaway repo has no signing key configured, so override the
    # user's global git config in case it defaults commit/tag signing to on.
    subprocess.run(["git", "config", "commit.gpgSign", "false"], cwd=repo, check=True)
    subprocess.run(["git", "config", "tag.gpgSign", "false"], cwd=repo, check=True)


def _tag(repo: Path, tag: str) -> None:
    (repo / "README.md").write_text(f"# {tag}\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "--quiet", "-m", tag], cwd=repo, check=True)
    subprocess.run(["git", "tag", tag], cwd=repo, check=True)


def _run_checker(cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


class TestCheckAboutVersion:
    def test_passes_with_no_prior_release(self, tmp_path: Path) -> None:
        # No tag yet, so the tag check is skipped. Only the
        # __version__-matches-latest_released_version check runs.
        repo = tmp_path / "repo"
        _init_repo(repo)
        _write_about(repo, latest_released="0.2.1", version="0.2.2.dev0")
        result = _run_checker(repo)
        assert result.returncode == 0, result.stdout

    def test_fails_when_version_is_not_the_bumped_candidate(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo)
        _write_about(repo, latest_released="0.2.1", version="0.2.5.dev0")
        result = _run_checker(repo)
        assert result.returncode == 1
        assert "latest_released_version '0.2.1' implies '0.2.2.dev0'" in result.stdout

    def test_passes_when_latest_released_version_matches_the_latest_tag(
        self, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo)
        _write_about(repo, latest_released="0.2.1", version="0.2.2.dev0")
        _tag(repo, "v0.2.1")
        result = _run_checker(repo)
        assert result.returncode == 0, result.stdout

    def test_fails_when_latest_released_version_is_behind_the_latest_tag(
        self, tmp_path: Path
    ) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo)
        # A release was tagged (0.2.1) but latest_released_version was never
        # updated from the prior release (0.2.0) — this is exactly the
        # mismatch the check exists to catch.
        _write_about(repo, latest_released="0.2.0", version="0.2.1.dev0")
        _tag(repo, "v0.2.1")
        result = _run_checker(repo)
        assert result.returncode == 1
        assert "latest release tag is v0.2.1" in result.stdout

    def test_ignores_non_release_tags(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo)
        _write_about(repo, latest_released="0.2.1", version="0.2.2.dev0")
        _tag(repo, "not-a-release")
        result = _run_checker(repo)
        assert result.returncode == 0, result.stdout
