#!/usr/bin/env bash

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

set -euo pipefail

# Install pre-commit hooks and their non-uv-managed dependencies.
# Idempotent — safe to call from setup_env.sh, make check, or manually.
#
# Requires: an activated venv with pre-commit installed (via uv sync).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Project root (the outermost pyproject.toml ancestor) must be passed as $1.
# setup_env.sh computes it via its repo_root() helper. The internal layout has
# a nested external/pyproject.toml so $REPO_ROOT (above) isn't sufficient on its
# own, and the OSS export tree (.oss-export/coreai-opt/) is itself a project
# root that must NOT install hooks into the parent worktree's .git.
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

# shellcheck source=../utils.sh
source "$REPO_ROOT/scripts/utils.sh"
# shellcheck source=../run_quietly.sh
source "$REPO_ROOT/scripts/run_quietly.sh"

# Install lychee link checker.
# Prefer the OS package manager (brew on macOS, dnf/apt on Linux).
# Fall back to cargo build for distros that don't package lychee (e.g., RHEL).
echo "Installing non-uv-managed pre-commit hook dependencies..."
if ! run_quietly ensure_package "lychee"; then
    echo "Package manager install failed. Falling back to cargo install..."
    if run_quietly ensure_package "cargo" && run_quietly cargo install lychee --locked; then
        # cargo install puts binaries in ~/.cargo/bin, which may not be on PATH.
        export PATH="$HOME/.cargo/bin:$PATH"
        echo "lychee installed via cargo: $(lychee --version 2>&1 | head -n1)"
    fi
fi

if ! command -v lychee &>/dev/null; then
    echo "Error: Could not install lychee via package manager or cargo."
    echo "  Install lychee manually: https://lychee.cli.rs/"
    exit 1
fi

# Configure git hooks. Force-install so the shebang points to the current venv.
uv run --no-sync --active pre-commit install -f
echo "✓ Pre-commit hooks configured to use $VIRTUAL_ENV"
