# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from types import ModuleType

from packaging import version
from packaging.specifiers import SpecifierSet


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


_MAX_TESTED_TORCH = "2.14"
parsed_max_tested_torch_version = version.parse(_MAX_TESTED_TORCH)
_TESTED_TORCH_RANGE = SpecifierSet(
    f"<{parsed_max_tested_torch_version.major}.{parsed_max_tested_torch_version.minor + 1}",
    prereleases=True,
)


def untested_torch_version(torch_version: str) -> str | None:
    """Describe why the installed torch version has not been tested.

    Args:
        torch_version: The installed torch version.

    Returns:
        A message naming the untested version, or ``None`` if it is at or below
        the highest CI-tested version.
    """
    if version.parse(torch_version) in _TESTED_TORCH_RANGE:
        return None
    return (
        f"coreai-opt has not been tested with torch {torch_version}. The highest "
        f"tested torch version is {_MAX_TESTED_TORCH}. If you want to silence this "
        f"message please run: "
        f"warnings.filterwarnings('ignore', category=coreai_opt.UntestedTorchVersionWarning)"
    )
