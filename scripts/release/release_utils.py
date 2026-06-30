# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Generic version-management helpers used by the build pipeline.

This module is OSS-clean and free of internal-only references. The PyPI URL
helpers that target Apple's internal index live in
``scripts/release/release.py`` and stay internal-only.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

# coreai_opt's _about.py lives under `external/src` in the internal repo but at
# the top-level `src` in the exported OSS tree, so resolve against both layouts.
_VERSION_FILE_CANDIDATES = (
    Path("src") / "coreai_opt" / "_about.py",
    Path("external") / "src" / "coreai_opt" / "_about.py",
)


def _resolve_version_file(repo_root: Path) -> Path:
    """Return the path to coreai_opt's ``_about.py`` for either repo layout."""
    for rel in _VERSION_FILE_CANDIDATES:
        candidate = repo_root / rel
        if candidate.is_file():
            return candidate
    checked = ", ".join(str(c) for c in _VERSION_FILE_CANDIDATES)
    msg = f"Could not locate coreai_opt/_about.py under {repo_root} (checked {checked})"
    raise FileNotFoundError(msg)


def get_short_sha() -> str:
    """Return the short commit SHA of HEAD."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def get_dev_release_version(base_version: str) -> str:
    """Compute the nightly version by bumping patch and appending a dev suffix.

    Bumps the patch component so the nightly sorts above the current stable
    release per PEP 440 (e.g. ``0.0.3`` -> ``0.0.4.dev202603031430+abc1234``).
    The dev suffix encodes the build's UTC timestamp down to the minute so
    multiple builds on the same day get distinct versions.

    Args:
        base_version (str): Semantic version from ``_about.py`` (e.g. ``"0.0.3"``).

    Returns:
        str: PEP 440 dev version, e.g. ``"0.0.4.dev202603031430+abc1234"``.
    """
    major, minor, patch = base_version.split(".")
    bumped_patch = int(patch) + 1
    dev_timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M")
    short_sha = get_short_sha()
    return f"{major}.{minor}.{bumped_patch}.dev{dev_timestamp}+{short_sha}"


def get_package_version(repo_root: Path) -> str:
    """Read package version from _about.py."""
    about_file = _resolve_version_file(repo_root)
    spec = importlib.util.spec_from_file_location("_about", about_file)
    if spec is None or spec.loader is None:
        msg = f"Could not load module spec from {about_file}"
        raise ImportError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__


def write_version(repo_root: Path, version: str) -> None:
    """Replace the ``__version__`` value in ``_about.py``.

    Args:
        repo_root: Repository root directory.
        version: Version string to write (e.g. ``"0.0.3.dev202603031430+abc1234"``).

    Raises:
        RuntimeError: If ``_about.py`` does not contain a ``__version__`` assignment.
    """
    about = _resolve_version_file(repo_root)
    content = about.read_text(encoding="utf-8")
    updated = re.sub(
        r'(__version__\s*=\s*)["\'].*?["\']',
        rf'\1"{version}"',
        content,
    )
    if updated == content:
        msg = f"Could not find __version__ assignment in {about}"
        raise RuntimeError(msg)
    about.write_text(updated, encoding="utf-8", newline="\n")


def get_dist_files(dist_dir: Path = Path("dist")) -> list[Path]:
    """Get distribution files (wheels and tarballs) from a dist directory.

    Args:
        dist_dir (Path): Directory containing the build artifacts. Defaults to
            ``Path("dist")``, the conventional location relative to the cwd.

    Returns:
        list[Path]: All ``.whl`` and ``.tar.gz`` files under ``dist_dir``.
    """
    return list(dist_dir.glob("*.whl")) + list(dist_dir.glob("*.tar.gz"))
