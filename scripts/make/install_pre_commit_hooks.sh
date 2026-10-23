#!/usr/bin/env bash

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

set -euo pipefail

# Install pre-commit hooks.
# Idempotent — safe to call from setup_env.sh, make check, or manually.
#
# Requires: an activated venv with pre-commit installed (via uv sync).

# Project root (outermost pyproject.toml ancestor), computed by setup_env.sh's repo_root().
PROJECT_ROOT="${1:-}"
if [[ -z "$PROJECT_ROOT" ]]; then
    echo "Usage: $0 <project-root>" >&2
    exit 1
fi

# Skip if the project root is not a git repo root (e.g., .oss-export/coreai-opt/
# has no .git of its own — installing hooks would corrupt the parent repo's
# hook configuration). A git repo root always has a `.git` entry (directory for
# main checkouts, file for worktrees).
if [[ ! -e "$PROJECT_ROOT/.git" ]]; then
    echo "Not at git repository root — skipping hook installation"
    exit 0
fi

# Configure git hooks. Force-install so the shebang points to the current venv.
if [[ "${INSTALL_PRECOMMIT:-true}" == "false" ]]; then
    echo "Skipping git hook configuration (INSTALL_PRECOMMIT=$INSTALL_PRECOMMIT)"
else
    uv run --no-sync --active pre-commit install -f
    echo "✓ Pre-commit hooks configured to use $VIRTUAL_ENV"
fi
