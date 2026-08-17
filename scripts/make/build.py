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

``_about.py`` stores ``latest_released_version`` (the last tagged release) by
hand; ``__version__`` names the release the tree is working toward. This
script takes the version to build from ``__version__``, writes it into
``_about.py``, builds, then restores the file. A repo that uses this one as a submodule (building
one combined wheel) can add its own extra number to the version with
``COREAI_OPT_VERSION_EXTENSION``; see
``scripts/release/release_utils.next_release_base``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# build.py ships to OSS, where `external/` is the repo root, so it imports the
# helpers via the plain `scripts.*` namespace — that resolves to `external/scripts/*`
# internally (PEP 420 merging) and to the OSS-root `scripts/*` post-export. The
# internal-only scripts use the explicit `external.scripts.*` form instead.
from scripts._utils import find_repo_root as _find_repo_root
from scripts.release.release_utils import (
    get_dev_version_override,
    get_version_extension,
    next_release_base,
    read_about,
    resolve_build_version,
    restore_about,
    write_version,
)


def run_build(*, no_sources: bool) -> None:
    """Run ``uv build`` to produce the wheel and sdist.

    Args:
        no_sources: Pass ``--no-sources`` to ``uv build``, ignoring
            ``[tool.uv.sources]`` so the artifact doesn't depend on uv-specific
            index overrides — used for the publishable release build.
    """
    sys.stdout.write(f"Building package with python (version: {sys.version})...\n")
    command = ["uv", "build"]
    if no_sources:
        command.append("--no-sources")
    # `uv build` writes to the same stdout, but Python block-buffers when stdout
    # isn't a terminal (CI logs, `make build > file`). Flush first so our lines
    # don't land after the child's. This also flushes anything buffered earlier,
    # so it's the only flush the script needs.
    sys.stdout.flush()
    subprocess.run(command, check=True)
    sys.stdout.write("Build complete! Check dist/ directory\n")


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

    about = read_about(repo_root)
    release_base = next_release_base(
        about.latest_released_version, about.version, get_version_extension()
    )
    build_version = resolve_build_version(
        release_base,
        dev=args.dev,
        dev_version_override=get_dev_version_override(),
    )
    try:
        write_version(about, build_version)
        sys.stdout.write(f"Version: {build_version}\n")
        run_build(no_sources=args.no_sources)
    finally:
        # Restore the exact bytes read before the build rewrote the version.
        restore_about(about)


if __name__ == "__main__":
    main()
