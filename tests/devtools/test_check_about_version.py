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

import pytest

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


def _track_branch(repo: Path, name: str) -> None:
    # The hook reads refs/remotes/origin/, which is what a fetch would populate.
    subprocess.run(
        ["git", "update-ref", f"refs/remotes/origin/{name}", "HEAD"], cwd=repo, check=True
    )


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
        assert "latest_released_version '0.2.1' allows only" in result.stdout

    @pytest.mark.parametrize("version", ["1.1.0.dev0"])
    def test_accepts_a_patch_minor_or_major_as_the_next_release(
        self, tmp_path: Path, version: str
    ) -> None:
        # The next release is declared in __version__, so a minor or major is
        # chosen by a PR editing it rather than being inferred from the last
        # released version.
        repo = tmp_path / "repo"
        _init_repo(repo)
        _write_about(repo, latest_released="1.0.1", version=version)
        result = _run_checker(repo)
        assert result.returncode == 0, result.stdout

    @pytest.mark.parametrize("version", ["1.1.1.dev0"])
    def test_rejects_a_skipped_number_or_partial_reset(self, tmp_path: Path, version: str) -> None:
        # 1.0.3 / 1.2.0 / 3.0.0 skip a number; 1.1.1 bumps the minor without
        # resetting the patch.
        repo = tmp_path / "repo"
        _init_repo(repo)
        _write_about(repo, latest_released="1.0.1", version=version)
        result = _run_checker(repo)
        assert result.returncode == 1

    def test_rejects_a_version_without_the_dev0_suffix(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo)
        _write_about(repo, latest_released="1.0.1", version="1.1.0")
        result = _run_checker(repo)
        assert result.returncode == 1

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

    def test_passes_while_a_release_is_branched_but_not_yet_tagged(self, tmp_path: Path) -> None:
        # The stabilization window: release/0.2.2 is cut and main has already
        # moved on to 0.2.3.dev0, but v0.2.2 won't exist until the branch is
        # stabilized. latest_released_version is ahead of the newest tag on
        # purpose, and the branch is what says so.
        repo = tmp_path / "repo"
        _init_repo(repo)
        _write_about(repo, latest_released="0.2.2", version="0.2.3.dev0")
        _tag(repo, "v0.2.1")
        _track_branch(repo, "release/0.2.2")
        result = _run_checker(repo)
        assert result.returncode == 0, result.stdout

    def test_accepts_a_release_branch_naming_its_own_release(self, tmp_path: Path) -> None:
        # A release branch sets latest_released_version to the version it
        # produces, so __version__ is that same version rather than a next one.
        repo = tmp_path / "repo"
        _init_repo(repo)
        _write_about(repo, latest_released="1.1.0", version="1.1.0.dev0")
        _tag(repo, "v1.0.0")
        _track_branch(repo, "release/1.1.0")
        result = _run_checker(repo)
        assert result.returncode == 0, result.stdout

    def test_fails_when_the_branch_is_for_a_different_version(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        _init_repo(repo)
        _write_about(repo, latest_released="0.2.2", version="0.2.3.dev0")
        _tag(repo, "v0.2.1")
        _track_branch(repo, "release/0.9.9")
        result = _run_checker(repo)
        assert result.returncode == 1
        assert "there is no release/0.2.2 branch" in result.stdout
