#!/usr/bin/env python3

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Log package versions and Python executable information."""

import sys
from importlib.metadata import PackageNotFoundError, distributions, version

TORCH_PACKAGES: list[str] = ["torch", "torchvision", "torchao"]

COREAI_NAME_SUBSTRING = "coreai"


def _get_version(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "not installed"


def _find_coreai_versions() -> dict[str, str]:
    """Return {name: version} for every installed distribution whose name contains 'coreai'."""
    found: dict[str, str] = {}
    for dist in distributions():
        name = dist.name
        if name and COREAI_NAME_SUBSTRING in name.lower():
            found[name] = dist.version
    return dict(sorted(found.items()))


def main() -> None:
    print("=== Python ===")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")

    print("=== Torch ===")
    for pkg in TORCH_PACKAGES:
        print(f"{pkg}: {_get_version(pkg)}")

    print("=== CoreAI ===")
    coreai_versions = _find_coreai_versions()
    if coreai_versions:
        for name, pkg_version in coreai_versions.items():
            print(f"{name}: {pkg_version}")
    else:
        print("no coreai packages installed")


if __name__ == "__main__":
    main()
