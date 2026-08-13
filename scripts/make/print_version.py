#!/usr/bin/env python3

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Print the version this tree is working toward.

Usage:
    print_version.py             the ``.dev0`` candidate, e.g. ``0.2.2.dev0``
                                  (``make version-dev``)
    print_version.py --release   the release itself, e.g. ``0.2.2`` — the
                                  version ``make build`` publishes

Both are computed from ``_about.py``'s ``latest_released_version`` using the
same helpers ``build.py`` uses, so what this prints and what a build produces
can't drift apart. ``_about.py`` is read as source text, never executed. A repo
that uses this one as a submodule can add its own extra number via
``COREAI_OPT_VERSION_EXTENSION``; see
``scripts/release/release_utils.next_release_base``.
"""

import argparse
import sys
from pathlib import Path

# `make version-dev` exports PYTHONPATH, but the release workflow runs this script
# directly, which puts only `scripts/make/` on sys.path. Walk up to the project
# root (the directory holding pyproject.toml, alongside `scripts/`) so this
# keeps working if the script moves. It can't call
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
    get_version_extension,
    next_candidate_version,
    next_release_base,
    read_about,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the version this tree is working toward.")
    parser.add_argument(
        "--release",
        action="store_true",
        help="Print the release version (no .dev0 suffix), as `make build` publishes it",
    )
    args = parser.parse_args()

    about = read_about(_repo_root)
    compute = next_release_base if args.release else next_candidate_version
    sys.stdout.write(
        f"{compute(about.latest_released_version, about.version, get_version_extension())}\n"
    )


if __name__ == "__main__":
    main()
