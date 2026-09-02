# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Generic version-management helpers used by the build pipeline.

This module is OSS-clean and free of internal-only references. The PyPI URL
helpers that target Apple's internal index live in
``scripts/release/release.py`` and stay internal-only.

``_about.py`` hard-codes ``latest_released_version`` (the last tagged release,
e.g. ``"0.2.1"``) and ``__version__``, the release the tree is working toward
(e.g. ``"0.2.2.dev0"``). A build takes its version from ``__version__``, which
is what lets a minor or major release be declared rather than only a
last-digit bump.

A repo that vendors this one as a submodule can add its own extra release
number via ``COREAI_OPT_VERSION_EXTENSION``; see ``next_release_base``. See
``RELEASE.md`` for the full scheme, the release workflow, and worked examples.
"""

from __future__ import annotations

import ast
import os
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# coreai_opt's _about.py lives under `external/src` in a vendoring repo but at
# the top-level `src` in this repo, so resolve against both layouts.
_ABOUT_PATH_CANDIDATES = (
    Path("src") / "coreai_opt" / "_about.py",
    Path("external") / "src" / "coreai_opt" / "_about.py",
)

# Environment variables the build reads. Callers use the getters below rather
# than reaching into os.environ, so there's one place that knows both the names
# and how each value is interpreted.
VERSION_EXTENSION_ENV_VAR = "COREAI_OPT_VERSION_EXTENSION"
DEV_VERSION_ENV_VAR = "DEV_VERSION"

# Branch a release is stabilized on before it is tagged, e.g. `release/0.2.2`.
RELEASE_BRANCH_PREFIX = "release/"

# Cap on the `origin` fetch in `latest_release_tag`. It runs from a pre-commit
# hook, so an unreachable `origin` (VPN down, laptop offline mid-handshake) must
# not stall the commit for the OS connect timeout.
_FETCH_TIMEOUT_SECONDS = 5


# =============================================================================
# Environment inputs
# =============================================================================


def get_version_extension() -> str | None:
    """Return the extra release number a downstream repo asked for, if any.

    Read from ``COREAI_OPT_VERSION_EXTENSION``. The value is the number that
    repo is about to release next, used as-is — see ``next_release_base``.

    Returns:
        str | None: The extra number, or ``None`` when the variable is unset.
    """
    return os.environ.get(VERSION_EXTENSION_ENV_VAR)


def get_dev_version_override() -> str | None:
    """Return an exact dev version to build, if one was pinned.

    Read from ``DEV_VERSION``. Only honored for dev builds, where it replaces
    the generated timestamped version — see ``resolve_build_version``.

    Returns:
        str | None: The pinned version, or ``None`` when the variable is unset.
    """
    return os.environ.get(DEV_VERSION_ENV_VAR)


# =============================================================================
# Repository inputs
# =============================================================================


def get_short_sha() -> str:
    """Return the short commit SHA of HEAD."""
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def latest_release_tag(repo_root: Path) -> str | None:
    """Return the latest ``vX.Y.Z`` release tag, or ``None`` if there isn't one yet.

    Fetches tags from ``origin`` first, so an out-of-date local clone can't
    report a tag that's since changed or been replaced upstream. The fetch is
    best-effort and time-boxed: if ``origin`` is missing or unreachable, this
    falls back to whatever tags the clone already has rather than blocking.

    Args:
        repo_root: Repository root to run ``git`` in.

    Returns:
        str | None: The tag's version (e.g. ``"0.2.1"`` for tag ``v0.2.1"``),
            or ``None`` if the repo has no ``vX.Y.Z`` tag yet.
    """
    try:
        subprocess.run(
            ["git", "fetch", "--tags", "--prune", "origin"],
            cwd=repo_root,
            capture_output=True,
            check=False,  # offline / no `origin` remote: fall back to local tags
            timeout=_FETCH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        pass  # unreachable remote: fall back to local tags rather than hang
    result = subprocess.run(
        ["git", "tag", "--list", "v[0-9]*.[0-9]*.[0-9]*", "--sort=-v:refname"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    tags = result.stdout.split()
    return tags[0].removeprefix("v") if tags else None


def release_branch_exists(repo_root: Path, version: str) -> bool:
    """Return whether ``origin`` has a ``release/<version>`` branch.

    A release branch is cut before its tag exists, and ``main`` moves to the
    next candidate as soon as the cut happens — so for the length of the
    stabilization window ``latest_released_version`` names a release that is
    branched but not yet tagged. This reports whether that window is open; see
    ``scripts/pre_commit/check_about_version.py``.

    Only the remote-tracking ref is consulted, so a local branch of the same
    name cannot satisfy the check. ``latest_release_tag`` fetches from
    ``origin`` before this runs, which is what keeps that ref current.

    Args:
        repo_root: Repository root to run ``git`` in.
        version: Release the branch is named for, e.g. ``"0.2.2"``.

    Returns:
        bool: ``True`` if ``origin`` has ``release/<version>``.
    """
    result = subprocess.run(
        [
            "git",
            "for-each-ref",
            "--format=%(refname)",
            f"refs/remotes/origin/{RELEASE_BRANCH_PREFIX}{version}",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return bool(result.stdout.strip())


# =============================================================================
# Version arithmetic
# =============================================================================


def strip_dev_suffix(version: str) -> str:
    """Strip a PEP 440 ``.dev`` suffix, returning the release segment.

    ``"0.2.2.dev0"`` -> ``"0.2.2"``; ``"0.2.2.1.dev202607231430+abc1234"`` ->
    ``"0.2.2.1"``; a version without ``.dev`` is returned unchanged. Works for
    any number of release segments (three, or four with an extension applied).
    """
    base, _sep, _rest = version.partition(".dev")
    return base


def apply_version_extension(version: str, extension: str | None) -> str:
    """Add ``extension`` as one more number at the end of ``version``.

    ``extension`` is the number a downstream repo is about to release next
    (e.g. ``"1"`` for its first release off ``version``, ``"2"`` for the one
    after that), used exactly as given.

    ``"0.2.1"`` + ``"1"`` -> ``"0.2.1.1"``. If ``extension`` is empty or
    ``None``, ``version`` is returned unchanged. Only call this with a plain
    release (no ``.dev`` suffix) — see ``next_release_base``.
    """
    return f"{version}.{extension}" if extension else version


def valid_next_versions(version: str) -> list[str]:
    """Return every release that may directly follow ``version``.

    Exactly one number goes up by one and everything after it resets to zero,
    so ``"1.0.1"`` -> ``["2.0.0", "1.1.0", "1.0.2"]`` — a major, a minor, and a
    patch. Ordered most-significant first. Anything else is either a skipped
    number or a move backwards.

    Args:
        version: A plain release (no ``.dev`` suffix), e.g. ``"1.0.1"``.

    Returns:
        list[str]: The releases that may follow, most-significant bump first.
    """
    numbers = [int(part) for part in version.split(".")]
    return [
        ".".join(str(n) for n in numbers[:i] + [numbers[i] + 1] + [0] * (len(numbers) - i - 1))
        for i in range(len(numbers))
    ]


def release_branch_version(latest_released_version: str) -> str:
    """Return the ``__version__`` a ``release/<version>`` branch carries.

    A release branch names its own release: ``latest_released_version`` is set
    to the version the branch produces, so ``__version__`` is that same version
    plus ``.dev0`` rather than a next candidate. On ``main`` the two always
    differ, which is what tells a release branch apart from ``main``.

    Making the branch self-describing is what lets a downstream repo pin
    ``external/`` to a release branch and still resolve the right baseline —
    ``latest_released_version`` then means the same thing on every commit.

    Args:
        latest_released_version: The release the branch produces, e.g. ``"1.1.0"``.

    Returns:
        str: The ``__version__`` that branch carries, e.g. ``"1.1.0.dev0"``.
    """
    return f"{latest_released_version}.dev0"


# =============================================================================
# Choosing the version to build
# =============================================================================


def next_release_base(
    latest_released_version: str, version: str, extension: str | None = None
) -> str:
    """Compute the release ``build.py`` should build next.

    With no extension, this is ``version`` (``_about.py``'s ``__version__``)
    with its ``.dev`` suffix removed — the tree states the release it is
    working toward, so a minor or major is expressed by editing ``__version__``
    rather than being inferred. With an extension, the downstream repo has
    already picked the number it is about to release, and it is appended to the
    last *published* release instead:

    * no extension: ``__version__`` ``"0.2.2.dev0"`` -> ``"0.2.2"``
    * no extension: ``__version__`` ``"1.1.0.dev0"`` -> ``"1.1.0"``
    * extension ``"1"``: ``latest_released_version`` ``"0.2.1"`` ->
      ``"0.2.1.1"``

    Args:
        latest_released_version: The last tagged release, e.g. ``"0.2.1"``.
        version: ``_about.py``'s ``__version__``, e.g. ``"0.2.2.dev0"``.
        extension: The extra number to release next, or ``None``.

    Returns:
        str: The release ``build.py`` builds toward, e.g. ``"0.2.2"``.
    """
    if extension:
        return apply_version_extension(latest_released_version, extension)
    return strip_dev_suffix(version)


def next_candidate_version(
    latest_released_version: str, version: str, extension: str | None = None
) -> str:
    """Compute the ``.dev0`` version the tree is working toward.

    Same as ``next_release_base`` with ``.dev0`` added at the end, e.g.
    ``"0.2.2.dev0"`` with extension ``"1"`` -> ``"0.2.1.1.dev0"``.

    Args:
        latest_released_version: The last tagged release, e.g. ``"0.2.1"``.
        version: ``_about.py``'s ``__version__``, e.g. ``"0.2.2.dev0"``.
        extension: The extra number to release next, or ``None``.

    Returns:
        str: The ``.dev0`` version this tree is working toward.
    """
    return f"{next_release_base(latest_released_version, version, extension)}.dev0"


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

    Callers pass the ``release_base`` produced by ``next_release_base``, which
    already has any extension applied — this function only decides dev vs.
    release, it doesn't shape the release segments itself.

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


# =============================================================================
# Reading and writing _about.py
# =============================================================================


def resolve_about_path(repo_root: Path) -> Path:
    """Return the path to coreai_opt's ``_about.py`` for either repo layout."""
    for rel in _ABOUT_PATH_CANDIDATES:
        candidate = repo_root / rel
        if candidate.is_file():
            return candidate
    checked = ", ".join(str(repo_root / rel) for rel in _ABOUT_PATH_CANDIDATES)
    msg = f"Could not locate coreai_opt/_about.py under {repo_root} (checked {checked})"
    raise FileNotFoundError(msg)


def read_version(about_text: str) -> str:
    """Extract the ``__version__`` string literal from ``_about.py`` source text.

    Args:
        about_text: The full source text of an ``_about.py`` file.

    Returns:
        str: The version string, e.g. ``"0.2.2.dev0"``.

    Raises:
        RuntimeError: If the text has no ``__version__`` string literal.
    """
    return _read_literal(about_text, "__version__")


def read_latest_released_version(about_text: str) -> str:
    """Extract the ``latest_released_version`` string literal from ``_about.py``.

    Args:
        about_text: The full source text of an ``_about.py`` file.

    Returns:
        str: The last tagged release, e.g. ``"0.2.1"``.

    Raises:
        RuntimeError: If the text has no ``latest_released_version`` string literal.
    """
    return _read_literal(about_text, "latest_released_version")


def _find_assignment(about_text: str, name: str) -> ast.expr | None:
    """Return the value node assigned to ``name`` at module level, or ``None``."""
    for node in ast.parse(about_text).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return node.value
    return None


def _read_literal(about_text: str, name: str) -> str:
    """Return the string literal assigned to ``name`` in ``_about.py`` source text.

    Parses the source instead of running it, so reading a version never
    executes ``_about.py``. This accepts exactly what setuptools' ``attr:``
    reader accepts — a literal, however it's quoted, parenthesized, or split
    across adjacent string pieces — and rejects computed values such as
    ``_base + ".dev0"``, which would fail the build for the same reason.
    """
    value_node = _find_assignment(about_text, name)
    if value_node is None:
        msg = f"Could not find a {name} string literal"
        raise RuntimeError(msg)
    try:
        value = ast.literal_eval(value_node)
    except ValueError:
        msg = f"{name} in _about.py must be a plain string literal"
        raise RuntimeError(msg) from None
    if not isinstance(value, str):
        msg = f"{name} in _about.py must be a string, got {type(value).__name__}"
        raise RuntimeError(msg)
    return value


@dataclass(frozen=True)
class AboutFile:
    """``_about.py``'s location, source text, and the versions it declares.

    Attributes:
        path: Where the file was found — needed to write to it, and to name it
            in error messages.
        text: The file's exact source, which ``build.py`` keeps so it can
            restore the file after temporarily writing a build version into it.
        latest_released_version: The last tagged release, e.g. ``"0.2.1"``.
        version: The on-tree ``__version__``, e.g. ``"0.2.2.dev0"``.
    """

    path: Path
    text: str
    latest_released_version: str
    version: str


def read_about(repo_root: Path) -> AboutFile:
    """Locate ``_about.py``, read it once, and parse both versions from it.

    Every caller needs some mix of the path, the raw text, and the two
    versions, so they all come from a single read — no caller has to repeat
    the resolve/read/parse steps or the file's encoding.

    Args:
        repo_root: Repository root to resolve ``_about.py`` under.

    Returns:
        AboutFile: The file's path, source text, and declared versions.

    Raises:
        FileNotFoundError: If ``_about.py`` isn't found under ``repo_root``.
        RuntimeError: If either version is missing or isn't a string literal.
    """
    about_path = resolve_about_path(repo_root)
    text = about_path.read_text(encoding="utf-8")
    return AboutFile(
        path=about_path,
        text=text,
        latest_released_version=read_latest_released_version(text),
        version=read_version(text),
    )


def write_version(about: AboutFile, version: str) -> None:
    """Rewrite ``__version__`` in ``_about.py``, leaving every other byte intact.

    Locates the assignment with ``ast`` and splices the new literal over exactly
    the span the old value occupied, so writes accept precisely what
    ``_read_literal`` accepts — including a parenthesized or implicitly
    concatenated literal, which a line-oriented pattern cannot rewrite. The
    replacement is computed from ``about.text``, so the file isn't re-read.

    Args:
        about: The file to rewrite, as returned by ``read_about``.
        version: Version string to write (e.g. ``"0.2.2.dev202607231430+abc1234"``).

    Raises:
        RuntimeError: If the text has no ``__version__`` assignment.
    """
    value_node = _find_assignment(about.text, "__version__")
    if value_node is None:
        msg = f"Could not find __version__ assignment in {about.path}"
        raise RuntimeError(msg)
    # ast reports positions as (1-based line, 0-based *byte* column), so splice
    # on the encoded source rather than the str to stay correct for non-ASCII.
    data = about.text.encode("utf-8")
    line_starts = [0]
    for line in data.splitlines(keepends=True):
        line_starts.append(line_starts[-1] + len(line))
    start = line_starts[value_node.lineno - 1] + value_node.col_offset
    end = line_starts[value_node.end_lineno - 1] + value_node.end_col_offset
    updated = data[:start] + f'"{version}"'.encode() + data[end:]
    about.path.write_text(updated.decode("utf-8"), encoding="utf-8", newline="\n")


def restore_about(about: AboutFile) -> None:
    """Write ``_about.py`` back exactly as ``read_about`` found it.

    Used by ``build.py`` to undo the temporary build version. Lives here so the
    encoding and newline contract stays in the module that owns the file.
    """
    about.path.write_text(about.text, encoding="utf-8", newline="\n")


# =============================================================================
# Build artifacts
# =============================================================================


def get_dist_files(dist_dir: Path = Path("dist")) -> list[Path]:
    """Get distribution files (wheels and tarballs) from a dist directory.

    Args:
        dist_dir (Path): Directory containing the build artifacts. Defaults to
            ``Path("dist")``, the conventional location relative to the cwd.

    Returns:
        list[Path]: All ``.whl`` and ``.tar.gz`` files under ``dist_dir``.
    """
    return list(dist_dir.glob("*.whl")) + list(dist_dir.glob("*.tar.gz"))
