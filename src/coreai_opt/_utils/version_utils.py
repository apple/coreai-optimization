# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from types import ModuleType

from packaging import version


def version_ge(module: ModuleType, target_version: str) -> bool:
    return version.parse(module.__version__) >= version.parse(target_version)


_MIN_TORCHAO_REQUIRING_TORCH_2_11 = "0.18.0"
_MIN_TORCH_FOR_NEW_TORCHAO = "2.11.0.dev0"
_TORCHAO_RELEASE_NOTES_URL = "https://github.com/pytorch/ao/releases/tag/v0.18.0"


def torchao_torch_incompatibility(torchao_version: str, torch_version: str) -> str | None:
    """Describe why the installed torchao and torch versions are incompatible.

    Args:
        torchao_version: The installed torchao version.
        torch_version: The installed torch version.

    Returns:
        A message explaining the incompatibility, or ``None`` if the pair is supported.
    """
    if version.parse(torchao_version) < version.parse(_MIN_TORCHAO_REQUIRING_TORCH_2_11):
        return None
    if version.parse(torch_version) >= version.parse(_MIN_TORCH_FOR_NEW_TORCHAO):
        return None
    return (
        f"torchao {torchao_version} does not support torch<2.11 "
        f"(found torch {torch_version}). See the torchao "
        f"{_MIN_TORCHAO_REQUIRING_TORCH_2_11} release notes for more information: "
        f"{_TORCHAO_RELEASE_NOTES_URL}"
    )


# The highest torch minor version exercised by CI (the `torch_2_*` dependency
# group named by HIGHEST_TORCH_GROUP in the Makefile). The runtime dependency
# range is floor-only, so torch above this installs fine but is untested --
# `untested_torch_version` warns at import instead of pip refusing to resolve.
# Keep in sync with pyproject.toml; `test_max_tested_torch_matches_pyproject`
# fails if this drifts.
_MAX_TESTED_TORCH = "2.11"


def untested_torch_version(torch_version: str) -> str | None:
    """Describe why the installed torch version has not been tested.

    Compares on (major, minor) only, so a patch release of a tested minor
    (``2.11.5``) is treated as tested while a new minor (``2.12.0``) is not.

    Args:
        torch_version: The installed torch version.

    Returns:
        A message naming the untested version, or ``None`` if it is at or below
        the highest CI-tested version.
    """
    installed = version.parse(torch_version)
    max_tested = version.parse(_MAX_TESTED_TORCH)
    if (installed.major, installed.minor) <= (max_tested.major, max_tested.minor):
        return None
    return (
        f"coreai-opt has not been tested with torch {torch_version}. The highest "
        f"tested version is torch {_MAX_TESTED_TORCH}. This is not a known "
        f"incompatibility -- coreai-opt intentionally allows newer torch so "
        f"installs are not blocked -- but if you hit unexpected behavior, try "
        f"torch {_MAX_TESTED_TORCH} before filing a bug. Silence this with: "
        f"warnings.filterwarnings('ignore', message='coreai-opt has not been tested')"
    )
