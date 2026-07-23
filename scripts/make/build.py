# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Build the coreai-opt package.

Usage:
    build.py --no-sources    Build a release: the `.dev` suffix stripped, via
                              `uv build --no-sources` (ignores [tool.uv.sources];
                              the recommended way to build the publishable
                              artifact). Called by `make build`.
    build.py --dev            Build a dev wheel with a timestamped PEP 440 .dev
                              version. Called by `make build-dev`.

``main`` carries the next planned release with a ``.dev0`` suffix in ``_about.py``
(e.g. ``0.2.2.dev0``); that suffix is only ever an on-tree marker and must never
end up in a built wheel. This script computes the version to build, writes it
into ``_about.py``, builds, then restores the file. A repo that vendors this one
(building a single combined wheel from both trees) can insert one extra release
segment via ``COREAI_OPT_VERSION_EXTENSION``; see
``scripts/release/release_utils.apply_version_extension``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# build.py ships to OSS, where `external/` is the repo root, so it imports the
# helpers via the plain `scripts.*` namespace — that resolves to `external/scripts/*`
# internally (PEP 420 merging) and to the OSS-root `scripts/*` post-export. The
# internal-only scripts use the explicit `external.scripts.*` form instead.
from scripts._utils import find_repo_root as _find_repo_root
from scripts.release.release_utils import (
    ENV_VERSION_EXTENSION,
    apply_version_extension,
    read_version,
    resolve_about_path,
    resolve_build_version,
    strip_dev_suffix,
    write_version,
)


def run_build(*, no_sources: bool) -> None:
    """Run ``uv build`` to produce the wheel and sdist.

    Args:
        no_sources: Pass ``--no-sources`` to ``uv build``, ignoring
            ``[tool.uv.sources]`` so the artifact doesn't depend on uv-specific
            index overrides — used for the publishable release build.
    """
    print(f"Building package with python (version: {sys.version})...")
    command = ["uv", "build", *(["--no-sources"] if no_sources else [])]
    subprocess.run(command, check=True)
    print("Build complete! Check dist/ directory")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the coreai-opt package.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dev",
        action="store_true",
        help="Build a dev wheel with a timestamped PEP 440 .dev version",
    )
    mode.add_argument(
        "--no-sources",
        action="store_true",
        help="Build a release via `uv build --no-sources`",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    repo_root = _find_repo_root(Path(__file__))

    about = resolve_about_path(repo_root)
    original = about.read_text(encoding="utf-8")  # exact bytes to restore afterwards
    on_tree_version = read_version(original)  # e.g. "0.2.2.dev0"
    extended = apply_version_extension(on_tree_version, os.environ.get(ENV_VERSION_EXTENSION))
    release_base = strip_dev_suffix(extended)
    build_version = resolve_build_version(
        release_base,
        dev=args.dev,
        dev_version_override=os.environ.get("DEV_VERSION"),
    )
    try:
        write_version(about, build_version)
        print(f"Version: {build_version}")
        run_build(no_sources=args.no_sources)
    finally:
        about.write_text(original, encoding="utf-8", newline="\n")  # restore on-tree version


if __name__ == "__main__":
    main()
