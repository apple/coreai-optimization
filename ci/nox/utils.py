# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Utility functions for nox sessions."""

import os
import tomllib
from pathlib import Path

from nox import Session
from packaging.specifiers import SpecifierSet

from coreai_opt._utils.repo_utils import find_repo_root

# Find repository root (where pyproject.toml is located)
REPO_ROOT = find_repo_root(__file__)


def change_dir_to_project_root(session: Session) -> None:
    """Change to the project root used for builds and ``uv``.

    ``UV_PROJECT``, when set, points at the directory that holds ``uv.lock`` and
    the build-ready ``pyproject.toml``; otherwise the package root fills both
    roles.
    """
    session.chdir(os.environ.get("UV_PROJECT") or str(REPO_ROOT))


def get_pytest_executable(session: Session) -> str:
    """Get the command to run pytest using the nox session's Python executable.

    Args:
        session: Nox session

    Returns:
        Command string to run pytest via python -m pytest
    """
    python_path = Path(session.bin) / "python"
    return f"{python_path} -m pytest"


def _get_minimum_python_minor_version(specifier: SpecifierSet) -> int:
    """Extract the minimum Python minor version from a specifier set.

    Args:
        specifier: A SpecifierSet parsed from requires-python.

    Returns:
        The minimum minor version (e.g., 10 for ">=3.10").

    Raises:
        ValueError: If no lower bound is specified in the specifier.
    """
    for spec in specifier:
        if spec.operator in (">=", ">"):
            return int(spec.version.split(".")[1])
    raise ValueError(f"No lower bound found in specifier: {specifier}")


def get_supported_python_versions() -> list[str]:
    """Parse requires-python from pyproject.toml and return supported versions."""
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)

    specifier = SpecifierSet(pyproject["project"]["requires-python"])
    min_minor = _get_minimum_python_minor_version(specifier)

    # Generate candidate versions from lower bound up to a reasonable upper limit
    all_versions = [f"3.{minor}" for minor in range(min_minor, min_minor + 10)]
    return [v for v in all_versions if specifier.contains(v)]


def build_pytest_args(
    default_args: list[str],
    posargs: list[str] | None = None,
    python_version: str | None = None,
) -> list[str]:
    """Build pytest arguments from posargs with optional session-specific handling.

    Args:
        posargs: Optional pytest arguments passed from command line (session.posargs)
        default_args: Default arguments to use when posargs is empty.
                     If None, defaults to ["-v"]
        python_version: Optional Python version for session-specific junitxml filename.
                       When provided and --junit is in posargs, adds
                       --junitxml=test-results/pytest-results-{python_version}.xml
                       and --cov-append if --cov is also present.

    Returns:
        List of pytest arguments
    """
    if not posargs:
        return list(default_args)

    pytest_args = list(posargs)

    # Handle session-specific junitxml for multi-python-version sessions
    if python_version and "--junit" in posargs:
        pytest_args.append(f"--junitxml=test-results/pytest-results-{python_version}.xml")
        # Use --cov-append to accumulate coverage across sessions
        if "--cov" in posargs or any(arg.startswith("--cov=") for arg in posargs):
            pytest_args.append("--cov-append")

    return pytest_args
