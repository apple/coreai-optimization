#!/usr/bin/env python3

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Verify ``_about.py``'s ``latest_released_version`` and ``__version__``."""

from __future__ import annotations

import sys
from pathlib import Path

# pre-commit runs this as `entry: python scripts/pre_commit/...`, which puts
# the script's own directory on sys.path[0], not the repo root — so
# `scripts.release.release_utils` can't be imported without help. Walk up to
# the project root (the directory holding pyproject.toml, alongside `scripts/`)
# so this keeps working if the script moves. It can't call
# scripts._utils.find_repo_root: that lives in the very package being
# bootstrapped, so the bootstrap has to be stdlib-only.
_repo_root = Path(__file__).resolve().parent
while not (_repo_root / "pyproject.toml").is_file():
    if _repo_root == _repo_root.parent:
        _msg = "Could not locate the project root (no pyproject.toml in any parent)"
        raise RuntimeError(_msg)
    _repo_root = _repo_root.parent
sys.path.insert(0, str(_repo_root))

from scripts.release.release_utils import (  # noqa: E402
    RELEASE_BRANCH_PREFIX,
    current_branch,
    latest_release_tag,
    read_about,
    release_branch_exists,
    valid_next_versions,
)


def main() -> int:
    """Check ``latest_released_version`` and the ``__version__`` computed from it."""
    repo_root = Path.cwd()  # pre-commit runs hooks with cwd set to the repo root
    about = read_about(repo_root)
    latest_released = about.latest_released_version

    errors = []

    latest_tag = latest_release_tag(repo_root)
    if (
        latest_tag is not None
        and latest_tag != latest_released
        and not release_branch_exists(repo_root, latest_released)
    ):
        errors.append(
            f"latest_released_version is {latest_released!r} in {about.path}, but the "
            f"latest release tag is v{latest_tag} and there is no "
            f"{RELEASE_BRANCH_PREFIX}{latest_released} branch. Update latest_released_version "
            f"to {latest_tag!r}, or cut the release branch before bumping it."
        )

    # On `release/<version>`, both fields must name that release. Anywhere else — `main`,
    # a fix branch off a release branch, a detached HEAD — accept either shape: a next
    # release plus `.dev0`, or `latest_released_version` itself.
    branch = current_branch(repo_root) or ""
    allowed = [f"{candidate}.dev0" for candidate in valid_next_versions(latest_released)]
    allowed.append(latest_released)
    if branch.startswith(RELEASE_BRANCH_PREFIX):
        branch_version = branch.removeprefix(RELEASE_BRANCH_PREFIX)
        if not latest_released == about.version == branch_version:
            errors.append(
                f"{branch} must name its own release: set both latest_released_version and "
                f"__version__ to {branch_version!r} in {about.path} (they are "
                f"{latest_released!r} and {about.version!r})."
            )
    elif about.version not in allowed:
        errors.append(
            f"__version__ is {about.version!r} in {about.path}, but latest_released_version "
            f"{latest_released!r} allows only {', '.join(repr(a) for a in allowed)}. "
            "__version__ must take latest_released_version, add one to exactly one of its "
            "numbers, reset every number after it to zero, and end in '.dev0' — or, on a "
            "release branch, be latest_released_version itself with no '.dev0'.",
        )

    if errors:
        for error in errors:
            sys.stdout.write(f"_about.py: {error}\n")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
