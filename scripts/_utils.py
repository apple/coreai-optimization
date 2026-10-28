# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Cross-cutting utilities for scripts/ tooling."""

from __future__ import annotations

import functools
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def find_repo_root(start: Path, *, marker: str = "pyproject.toml") -> Path:
    """Return the closest directory at or above ``start`` that contains ``marker``.

    The default marker, ``pyproject.toml``, needs no git metadata, so the lookup
    also works in an unpacked source archive. Files that run before this module
    is importable inline the same walk.

    Args:
        start (Path): File or directory to search from. A file is searched
            from its parent directory.
        marker (str): Name of the file that marks the root.

    Returns:
        Path: Absolute path to the closest directory that contains ``marker``.

    Raises:
        RuntimeError: If neither ``start`` nor any of its ancestors contains
            ``marker``.
    """
    base = start.resolve()
    if base.is_file():
        base = base.parent
    for candidate in [base, *base.parents]:
        if (candidate / marker).is_file():
            return candidate
    msg = f"No {marker} found in {base} or any of its parents"
    raise RuntimeError(msg)


@functools.cache
def _default_scratch_dir() -> Path:
    """Return ``<repo_root>/.local/``."""
    return find_repo_root(Path(__file__)) / ".local"


@contextmanager
def scratch_file(*, suffix: str = "", parent_dir: Path | None = None) -> Iterator[Path]:
    """Yield a path to an empty temp file under parent_dir; remove it on exit.

    Args:
        suffix (str): Filename suffix including any leading dot, e.g., ".toml".
            Required when downstream tooling filters by extension (pre-commit's
            ``types:``, mime detection, etc.).
        parent_dir (Path | None): Directory to create the file in. Defaults to
            ``<repo_root>/.local/`` when ``None``; the repo root is resolved
            lazily on first use.

    Yields:
        Path: Absolute path to the empty file. The file is removed when the
            context exits, regardless of how it exits.
    """
    if parent_dir is None:
        parent_dir = _default_scratch_dir()
    parent_dir.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(suffix=suffix, dir=parent_dir)
    os.close(fd)
    path = Path(name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


@contextmanager
def scratch_dir(*, prefix: str | None = None, parent_dir: Path | None = None) -> Iterator[Path]:
    """Yield a path to a fresh temp directory under parent_dir; remove it (and contents) on exit.

    Args:
        prefix (str | None): Directory-name prefix. Defaults to tempfile's
            ``"tmp"`` prefix when ``None``.
        parent_dir (Path | None): Parent directory under which to create the
            temp directory. Defaults to ``<repo_root>/.local/`` when ``None``;
            the repo root is resolved lazily on first use.

    Yields:
        Path: Absolute path to the fresh temp directory. The directory and any
            contents are removed when the context exits.
    """
    if parent_dir is None:
        parent_dir = _default_scratch_dir()
    parent_dir.mkdir(parents=True, exist_ok=True)
    name = tempfile.mkdtemp(prefix=prefix, dir=parent_dir)
    path = Path(name)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
