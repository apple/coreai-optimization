# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Generic version-management helpers used by the build pipeline.

This module is OSS-clean and free of internal-only references. The PyPI URL
helpers that target Apple's internal index live in
``scripts/release/release.py`` and stay internal-only.

Version scheme
--------------
``main`` carries the next planned release with a ``.dev0`` suffix in
``_about.py`` (e.g. ``0.2.2.dev0``). ``.dev0`` is only ever an on-tree
marker — it never appears in a built wheel. ``build.py`` always resolves it to
one of:

* ``build.py --no-sources`` -> the release with the ``.dev`` suffix stripped,
  e.g. ``0.2.2`` (``make build``)
* ``build.py --dev``      -> the release plus a unique dev suffix,
  e.g. ``0.2.2.dev<ts>+<sha>`` (``timestamped_dev_version``; ``make build-dev``)
* ``DEV_VERSION=...``     -> the given version, used exactly as given
  (dev builds only)

A release is cut on a release branch by dropping the ``.dev0`` suffix in
``_about.py`` and tagging; ``main`` is then bumped to the next planned
release + ``.dev0``.

Extending the scheme downstream
--------------------------------
A repo that vendors this one (e.g. under ``external/``, building a single
combined wheel from both trees) can insert one extra release segment by
setting ``COREAI_OPT_VERSION_EXTENSION`` to that segment, e.g. ``"1"``:

* on-tree ``0.2.2.dev0``  + extension ``"1"`` -> ``0.2.2.1.dev0``
* ``make build``   -> ``0.2.2.1``
* ``make build-dev`` -> ``0.2.2.1.dev<ts>+<sha>``

There is still exactly one ``_about.py`` (this package's own); the extension
is a plain string composed entirely by this module — see
``apply_version_extension``. No downstream file, package, or logic is
involved.
"""

from __future__ import annotations

import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

# coreai_opt's _about.py lives under `external/src` in a vendoring repo but at
# the top-level `src` in this repo, so resolve against both layouts.
_ABOUT_PATH_CANDIDATES = (
    Path("src") / "coreai_opt" / "_about.py",
    Path("external") / "src" / "coreai_opt" / "_about.py",
)

# Environment variable a vendoring repo sets to insert one extra release
# segment (e.g. "1"), composed by this module — see `apply_version_extension`.
ENV_VERSION_EXTENSION = "COREAI_OPT_VERSION_EXTENSION"

# Matches the `__version__ = "..."` assignment line in `_about.py`; used by
# both `read_version` (extract) and `write_version` (replace) so the two stay
# in sync.
_VERSION_LINE_RE = re.compile(r'^(__version__\s*=\s*)["\'](?P<version>.*?)["\']', re.MULTILINE)


def resolve_about_path(repo_root: Path) -> Path:
    """Return the path to coreai_opt's ``_about.py`` for either repo layout."""
    for rel in _ABOUT_PATH_CANDIDATES:
        candidate = repo_root / rel
        if candidate.is_file():
            return candidate
    checked = ", ".join(str(repo_root / rel) for rel in _ABOUT_PATH_CANDIDATES)
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


def strip_dev_suffix(version: str) -> str:
    """Strip a PEP 440 ``.dev`` suffix, returning the release segment.

    ``"0.2.2.dev0"`` -> ``"0.2.2"``; ``"0.2.2.1.dev202607231430+abc1234"`` ->
    ``"0.2.2.1"``; a version without ``.dev`` is returned unchanged. Works for
    any number of release segments (three, or four with an extension applied).
    """
    base, _sep, _rest = version.partition(".dev")
    return base


def apply_version_extension(version: str, extension: str | None) -> str:
    """Insert ``extension`` as an extra release segment before any ``.dev`` suffix.

    ``extension`` is a plain string segment (e.g. ``"1"``), not a package or
    file — this is the entirety of the "extend the scheme downstream" seam.

    ``"0.2.2.dev0"`` + ``"1"`` -> ``"0.2.2.1.dev0"``; ``"0.2.2"`` + ``"1"`` ->
    ``"0.2.2.1"``; a falsy ``extension`` returns ``version`` unchanged.
    """
    if not extension:
        return version
    base, sep, rest = version.partition(".dev")
    return f"{base}.{extension}{sep}{rest}"


def timestamped_dev_version(
    release_base: str,
    *,
    now: datetime | None = None,
    sha: str | None = None,
) -> str:
    """Append a unique, PEP 440-sortable dev suffix to a clean release base.

    The suffix encodes the build's UTC timestamp down to the minute plus the
    short commit SHA, so repeated builds get distinct versions that all sort
    below the eventual release (``0.2.2.dev202607231430+abc1234 < 0.2.2``). The
    release base is used verbatim — no segment is bumped — so ``main`` must
    already carry the target release (this repo keeps it as ``<release>.dev0``).
    Segment-count-agnostic, so it also handles an extended base
    (``"0.2.2.1"`` -> ``"0.2.2.1.dev...+..."``).

    Args:
        release_base: Clean release version, e.g. ``"0.2.2"`` or ``"0.2.2.1"``.
        now: UTC timestamp to encode; defaults to the current time. Injectable
            for deterministic tests.
        sha: Short commit SHA to append; defaults to ``get_short_sha()``.
            Injectable for deterministic tests.

    Returns:
        str: PEP 440 dev version, e.g. ``"0.2.2.dev202607231430+abc1234"``.
    """
    dev_timestamp = (now or datetime.now(UTC)).strftime("%Y%m%d%H%M")
    short_sha = sha if sha is not None else get_short_sha()
    return f"{release_base}.dev{dev_timestamp}+{short_sha}"


def resolve_build_version(
    release_base: str,
    *,
    dev: bool,
    dev_version_override: str | None = None,
) -> str:
    """Decide the version to write into ``_about.py`` for a build.

    Precedence:

    1. ``dev_version_override`` (the ``DEV_VERSION`` env var) — used exactly as
       given. Only honored for dev builds.
    2. ``dev`` build — a generated, unique dev version (``build.py --dev``).
    3. otherwise — ``release_base`` as-is (``build.py``, a plain release build).

    Callers pass a ``release_base`` with any ``.dev`` suffix already stripped
    (``strip_dev_suffix``) and any extension already applied
    (``apply_version_extension``) — this function only decides dev vs. release,
    it doesn't shape the release segments itself.

    Args:
        release_base: The clean release to build, e.g. ``"0.2.2"`` or, with an
            extension applied, ``"0.2.2.1"``.
        dev: Whether this is a dev build (``build.py --dev``).
        dev_version_override: Explicit version to use as-is, or ``None``.

    Returns:
        str: The version string to write into ``_about.py`` before building.
    """
    if dev:
        return dev_version_override or timestamped_dev_version(release_base)
    return release_base


def read_version(about_text: str) -> str:
    """Extract the ``__version__`` string literal from ``_about.py`` source text.

    Args:
        about_text: The full source text of an ``_about.py`` file.

    Returns:
        str: The version string, e.g. ``"0.2.2.dev0"``.

    Raises:
        RuntimeError: If the text has no ``__version__`` assignment.
    """
    match = _VERSION_LINE_RE.search(about_text)
    if match is None:
        msg = "Could not find a __version__ string literal"
        raise RuntimeError(msg)
    return match.group("version")


def write_version(about_path: Path, version: str) -> None:
    """Overwrite the ``__version__`` assignment in an ``_about.py`` file.

    Args:
        about_path: Path to the ``_about.py`` file to rewrite.
        version: Version string to write (e.g. ``"0.2.2.dev202607231430+abc1234"``).

    Raises:
        RuntimeError: If the file has no ``__version__`` assignment.
    """
    content = about_path.read_text(encoding="utf-8")
    updated, count = _VERSION_LINE_RE.subn(rf'\1"{version}"', content, count=1)
    if count == 0:
        msg = f"Could not find __version__ assignment in {about_path}"
        raise RuntimeError(msg)
    about_path.write_text(updated, encoding="utf-8", newline="\n")


def get_dist_files(dist_dir: Path = Path("dist")) -> list[Path]:
    """Get distribution files (wheels and tarballs) from a dist directory.

    Args:
        dist_dir (Path): Directory containing the build artifacts. Defaults to
            ``Path("dist")``, the conventional location relative to the cwd.

    Returns:
        list[Path]: All ``.whl`` and ``.tar.gz`` files under ``dist_dir``.
    """
    return list(dist_dir.glob("*.whl")) + list(dist_dir.glob("*.tar.gz"))
