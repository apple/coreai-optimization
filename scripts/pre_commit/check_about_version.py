#!/usr/bin/env python3

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Verify ``_about.py``'s ``latest_released_version`` and ``__version__``.

Two checks:

1. ``latest_released_version`` names a release that has either been tagged or
   been branched. The tag is fetched fresh from ``origin`` so an out-of-date
   local tag can't hide a mismatch. The branch alternative covers the
   stabilization window: a release branch is cut before its tag exists, and
   ``main`` moves to the next candidate at the cut, so between those two points
   ``latest_released_version`` legitimately names an untagged release. Skipped
   if there's no release tag yet (e.g. before the first release).
2. ``__version__`` names a release that may directly follow
   ``latest_released_version`` — exactly one number up by one, everything
   after it reset to zero — plus a ``.dev0`` suffix. From ``"1.0.1"`` that
   admits ``"2.0.0.dev0"``, ``"1.1.0.dev0"``, and ``"1.0.2.dev0"``, so a
   minor or major is chosen by editing ``__version__``, while a skipped
   number or a move backwards is rejected. This also stops a release
   candidate from looking like it already shipped; see
   ``scripts/release/release_utils.valid_next_versions``.

   A release branch is the exception: it sets ``latest_released_version`` to
   the version it produces, so ``__version__`` is that same version plus
   ``.dev0``. See ``release_utils.release_branch_version``.
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
    RELEASE_BRANCH_PREFIX,
    latest_release_tag,
    read_about,
    release_branch_exists,
    release_branch_version,
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

    # On `main`, `__version__` names a release after `latest_released_version`.
    # On a release branch it names that same release, because the branch sets
    # `latest_released_version` to the version it produces.
    allowed = [f"{candidate}.dev0" for candidate in valid_next_versions(latest_released)]
    allowed.append(release_branch_version(latest_released))
    if about.version not in allowed:
        errors.append(
            f"__version__ is {about.version!r} in {about.path}, but latest_released_version "
            f"{latest_released!r} allows only {', '.join(repr(a) for a in allowed)}. "
            "__version__ must take latest_released_version, add one to exactly one of its "
            "numbers, reset every number after it to zero, and end in '.dev0' — or, on a "
            "release branch, be latest_released_version itself plus '.dev0'."
        )

    if errors:
        for error in errors:
            sys.stdout.write(f"_about.py: {error}\n")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
