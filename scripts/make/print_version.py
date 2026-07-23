#!/usr/bin/env python3

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Print the development version carried on the tree (e.g. 0.2.2.dev0).

Reads ``_about.py`` as plain text (no import, so no torch dependency). A repo
that vendors this one can insert one extra release segment via
``COREAI_OPT_VERSION_EXTENSION`` (e.g. printing ``0.2.2.1.dev0``); see
``scripts/release/release_utils.apply_version_extension``.
"""

import os
from pathlib import Path

from scripts._utils import find_repo_root
from scripts.release.release_utils import (
    ENV_VERSION_EXTENSION,
    apply_version_extension,
    read_version,
    resolve_about_path,
)


def main() -> None:
    about = resolve_about_path(find_repo_root(Path(__file__)))
    on_tree_version = read_version(about.read_text(encoding="utf-8"))
    print(apply_version_extension(on_tree_version, os.environ.get(ENV_VERSION_EXTENSION)))


if __name__ == "__main__":
    main()
