#!/usr/bin/env python3

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Verify ``_about.py``'s ``latest_released_version`` and ``__version__``.

Two checks:

1. ``latest_released_version`` matches the repo's latest ``vX.Y.Z`` release
   tag (fetched fresh from ``origin`` so an out-of-date local tag can't hide a
   mismatch). Skipped if there's no such tag yet (e.g. before the first
   release).
2. ``__version__`` equals the ``.dev0`` version computed from
   ``latest_released_version`` (``next_candidate_version``) — its last number
   plus one. This stops a release candidate from looking like it already
   shipped a release that hasn't happened yet; see
   ``scripts/release/release_utils.next_candidate_version``.
"""

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
    latest_release_tag,
    next_candidate_version,
    read_about,
)


def main() -> int:
    """Check ``latest_released_version`` and the ``__version__`` computed from it."""
    repo_root = Path.cwd()  # pre-commit runs hooks with cwd set to the repo root
    about = read_about(repo_root)
    latest_released = about.latest_released_version

    errors = []

    latest_tag = latest_release_tag(repo_root)
    if latest_tag is not None and latest_tag != latest_released:
        errors.append(
            f"latest_released_version is {latest_released!r} in {about.path}, but the "
            f"latest release tag is v{latest_tag}. Update latest_released_version "
            f"to {latest_tag!r}."
        )

    expected_version = next_candidate_version(latest_released)
    if about.version != expected_version:
        errors.append(
            f"__version__ is {about.version!r} in {about.path}, but latest_released_version "
            f"{latest_released!r} implies {expected_version!r}. __version__ must be "
            "latest_released_version with its last segment incremented by one, "
            "plus '.dev0'."
        )

    if errors:
        for error in errors:
            sys.stdout.write(f"_about.py: {error}\n")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
