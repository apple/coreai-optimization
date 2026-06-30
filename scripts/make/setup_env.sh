#!/usr/bin/env bash

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

set -euo pipefail

# Find repo root by walking up to the outermost `pyproject.toml` in a
# consecutive ancestor chain. A stop-at-first-match walk can land in a nested
# sub-package whose `pyproject.toml` shadows the actual root; walking past
# the first match while consecutive ancestors also have `pyproject.toml`
# finds the outermost one.
# TODO: Centralize repo-root detection in a shared coreai-opt-utils
# package so setup_env.sh and Python code use the same logic.
repo_root() {
    local dir="${1:-$PWD}"
    # Resolve to absolute path
    dir="$(cd "$dir" && pwd)"
    local found=""
    while [ "$dir" != "/" ]; do
        if [ -f "$dir/pyproject.toml" ]; then
            found="$dir"
        elif [ -n "$found" ]; then
            # Chain broke: previous match is the outermost in the consecutive run.
            break
        fi
        dir="$(dirname "$dir")"
    done
    if [ -n "$found" ]; then
        echo "$found"
        return 0
    fi
    return 1
}

# Get repo root
COREAI_OPT_HOME="$(repo_root "$(dirname "$0")")" || {
    echo "Could not determine coreai_opt home" >&2
    exit 1
}

# Directory containing this script's siblings (e.g., install_pre_commit_hooks.sh).
# Distinct from $COREAI_OPT_HOME because scripts may live in a subdirectory.
if [ -d "$COREAI_OPT_HOME/external/scripts" ]; then
    SCRIPTS_DIR="$COREAI_OPT_HOME/external/scripts"
else
    SCRIPTS_DIR="$COREAI_OPT_HOME/scripts"
fi

# Extract available dependency groups from pyproject.toml
# This pipeline does the following:
# 1. awk: Extract lines between [dependency-groups] section and next section
#    - BEGIN {in_section=0}: Initialize flag to track if we're in the target section
#    - /^\[dependency-groups\]/: When we find the [dependency-groups] header
#      * in_section=1: Set flag to true (we're now inside the section)
#      * next: Skip to next line (don't print the header itself)
#    - in_section && /^\[/: If we're in the section AND hit another section header (line starting with [)
#      * exit: Stop processing (we've left the dependency-groups section)
#    - in_section {print}: If we're in the section, print the line
# 2. grep -E '^[a-z_-]+ = \[': Filter to lines that define groups
#    - ^[a-z_-]+: Group name at start of line (lowercase letters, hyphens, underscores)
#    - = \[: Followed by space, equals sign, space, opening bracket
# 3. cut -d' ' -f1: Extract just the group name
#    - -d' ': Use space as delimiter
#    - -f1: Take first field (the group name before the space)
# 4. tr '\n' ' ': Convert newlines to spaces
#    - Creates space-separated list: "dev torch benchmark rio turi "
PYPROJECT_TOML="$COREAI_OPT_HOME/pyproject.toml"
AVAILABLE_GROUPS=$(
    awk '
        BEGIN { in_section=0 }
        /^\[dependency-groups\]/ { in_section=1; next }
        in_section && /^\[/ { exit }
        in_section { print }
    ' "$PYPROJECT_TOML" |
        grep -E '^[a-z_-]+ = \[' |
        cut -d' ' -f1 |
        tr '\n' ' '
)

if [[ -z "$AVAILABLE_GROUPS" ]]; then
    echo "Error: Could not read dependency groups from $PYPROJECT_TOML"
    exit 1
fi

group_torch_pin() {
    local group="$1"
    awk -v group="$group" '
        $0 ~ "^" group "[[:space:]]*=[[:space:]]*\\[" { in_group = 1; next }
        in_group && /^\]/ { exit }
        in_group && match($0, /"torch[[:space:]]*==[[:space:]]*[0-9][0-9.]*/) {
            version = substr($0, RSTART, RLENGTH)
            sub(/"torch[[:space:]]*==[[:space:]]*/, "", version)
            print version
            exit
        }
    ' "$PYPROJECT_TOML"
}

# Parse command line arguments
VENV=".venv"
PYTHON_VERSION=""
EXTRA_GROUPS=()
EXCLUDE_GROUPS=()
ALL_GROUPS=false
ENSURE_MODE=false

# Groups excluded from --all-groups due to mutual conflicts in pyproject.toml.
# tamm-export is omitted because it's opt-in only (never in default-groups or --all-groups).
CONFLICTING_GROUPS=("highest_tested_torch" "lowest_tested_torch")

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Setup development environment with uv"
    echo ""
    echo "Options:"
    echo "  --venv <name>          Virtual environment name (default: .venv)"
    echo "  --python-version <ver> Python version (required)"
    echo "  --with-<group>         Install additional dependency group (e.g., --with-docs, --with-turi)"
    echo "  --without-<group>      Exclude a default dependency group (e.g., --without-coreai)"
    echo "  --all-groups           Install all non-conflicting dependency groups"
    echo "  --ensure               Quick check mode: skip setup if venv exists and deps are present"
    echo "  --help                 Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  VENV                Virtual environment name, overrides --venv (default: .venv)"
    echo ""
    echo "Available dependency groups: $AVAILABLE_GROUPS"
    echo "Conflicting groups (excluded from --all-groups): ${CONFLICTING_GROUPS[*]}"
    echo ""
    echo "Examples:"
    echo "  $0 --python-version 3.11                               # Setup with dev group only"
    echo "  $0 --python-version 3.11 --with-docs                   # Setup with dev and docs groups"
    echo "  $0 --python-version 3.11 --all-groups                                    # Setup with all non-conflicting groups"
    echo "  $0 --python-version 3.11 --all-groups --with-highest_tested_torch        # Setup with all groups and highest torch"
    echo "  $0 --python-version 3.11 --all-groups --with-lowest_tested_torch         # Setup with all groups and lowest torch"
    echo "  $0 --python-version 3.11 --venv .venv-exp              # Setup with custom venv name"
    echo "  $0 --python-version 3.11 --with-docs --venv .venv-exp  # Setup with docs group and custom venv name"
    echo "  $0 --python-version 3.12                               # Setup with Python 3.12"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
    --venv)
        if [[ $# -lt 2 ]]; then
            echo "Error: --venv requires a value"
            show_help
            exit 1
        fi
        VENV="$2"
        shift 2
        ;;
    --python-version)
        if [[ $# -lt 2 ]]; then
            echo "Error: --python-version requires a value"
            show_help
            exit 1
        fi
        PYTHON_VERSION="$2"
        shift 2
        ;;
    --all-groups)
        ALL_GROUPS=true
        shift
        ;;
    --ensure)
        ENSURE_MODE=true
        shift
        ;;
    --with-*)
        # Extract group name after --with-
        GROUP_NAME="${1#--with-}"
        EXTRA_GROUPS+=("$GROUP_NAME")
        shift
        ;;
    --without-*)
        # Extract group name after --without-
        GROUP_NAME="${1#--without-}"
        EXCLUDE_GROUPS+=("$GROUP_NAME")
        shift
        ;;
    --help)
        show_help
        exit 0
        ;;
    *)
        echo "Error: Unknown argument $1"
        show_help
        exit 1
        ;;
    esac
done

if [[ "$ENSURE_MODE" == true ]]; then
    INSTALL_PRECOMMIT="${INSTALL_PRECOMMIT:-false}"
else
    INSTALL_PRECOMMIT="${INSTALL_PRECOMMIT:-true}"
fi

# Validate required arguments
if [[ -z "$PYTHON_VERSION" ]]; then
    echo "Error: --python-version is required"
    show_help
    exit 1
fi

# Validate that each group name in the argument list exists in AVAILABLE_GROUPS
validate_groups() {
    for GROUP in "$@"; do
        if [[ ! " $AVAILABLE_GROUPS " == *" $GROUP "* ]]; then
            echo "Error: Unknown dependency group '$GROUP'"
            echo "Available groups: $AVAILABLE_GROUPS"
            exit 1
        fi
    done
}

# Validate dependency group names early
[[ ${#EXTRA_GROUPS[@]} -gt 0 ]] && validate_groups "${EXTRA_GROUPS[@]}"
[[ ${#EXCLUDE_GROUPS[@]} -gt 0 ]] && validate_groups "${EXCLUDE_GROUPS[@]}"

# Check for conflicts between --with-<group> and --without-<group>
if [[ ${#EXTRA_GROUPS[@]} -gt 0 && ${#EXCLUDE_GROUPS[@]} -gt 0 ]]; then
    for GROUP in "${EXTRA_GROUPS[@]}"; do
        if [[ " ${EXCLUDE_GROUPS[*]} " == *" ${GROUP} "* ]]; then
            echo "Error: Group '$GROUP' cannot be both included (--with-$GROUP) and excluded (--without-$GROUP)"
            exit 1
        fi
    done
fi

# Validate venv name contains only allowed characters
if [[ ! "$VENV" =~ ^[a-zA-Z0-9._/-]+$ ]]; then
    echo "Error: Virtual environment name contains invalid characters"
    echo "Allowed: letters, numbers, '.', '-', '_', '/'"
    echo "Examples: .venv, .venv-py312, .venv-testing, .venv-feature-x, .venv-exp"
    exit 1
fi

# Validate venv name (must start with .venv and be local to repo)
if [[ "$VENV" != .venv* ]]; then
    echo "Error: Virtual environment name must start with '.venv'"
    echo "Examples: .venv, .venv-py312, .venv-testing, .venv-feature-x, .venv-exp"
    exit 1
fi

if [[ "$VENV" == /* ]]; then
    echo "Error: Virtual environment name cannot be an absolute path"
    exit 1
fi

# Check if uv is installed
if ! command -v uv &>/dev/null; then
    echo "Error: uv is not installed. See README.md for installation instructions." >&2
    exit 1
fi

# --ensure mode: skip setup if venv exists and required deps are already installed.
# This is the fast path called by Make targets to avoid re-running full setup.
if [[ "$ENSURE_MODE" == "true" ]] && [ -f "$VENV/bin/python" ]; then
    IMPORT_STMTS="import pytest"
    if [[ ${#EXTRA_GROUPS[@]} -gt 0 ]]; then
        for GROUP in "${EXTRA_GROUPS[@]}"; do
            case "$GROUP" in
            docs) IMPORT_STMTS+="; import sphinx" ;;
            highest_tested_torch | lowest_tested_torch)
                IMPORT_STMTS+="; import torchao"
                EXPECTED_TORCH="$(group_torch_pin "$GROUP")"
                # These groups always pin torch, so an empty result means the
                # pyproject parse regressed — fail loudly instead of silently
                # skipping the version check (which would reintroduce the bug).
                if [[ -z "$EXPECTED_TORCH" ]]; then
                    echo "Error: could not parse a torch pin for group '$GROUP' in $PYPROJECT_TOML" >&2
                    exit 1
                fi
                IMPORT_STMTS+="; import torch; assert torch.__version__.split('+')[0] == '$EXPECTED_TORCH'"
                ;;
            rio) IMPORT_STMTS+="; import turi_lightning" ;;
            tamm-export) IMPORT_STMTS+="; import tamm_export" ;;
            esac
        done
    fi

    if "$VENV/bin/python" -c "$IMPORT_STMTS" 2>/dev/null; then
        exit 0
    fi

    # Deps missing or pinned torch mismatch — fall through to full setup.
    # If a pinned-torch group expected a specific version, surface the mismatch
    # so the rebuild isn't silent.
    if [[ -n "${EXPECTED_TORCH:-}" ]]; then
        ACTUAL_TORCH="$("$VENV/bin/python" -c \
            "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || true)"
        if [[ -n "$ACTUAL_TORCH" && "$ACTUAL_TORCH" != "$EXPECTED_TORCH" ]]; then
            echo "Note: $VENV has torch $ACTUAL_TORCH, expected $EXPECTED_TORCH; rebuilding." >&2
        fi
    fi
fi

echo "=========================================="
echo "Setting up development environment with uv"
echo "=========================================="
echo ""

echo "[1/3] Setting up virtual environment..."
VENV_PREEXISTED=false
if [ -d "$VENV" ] && [ -f "$VENV/bin/activate" ]; then
    echo "Virtual environment '$VENV' already exists, reusing it"
    VENV_PREEXISTED=true
else
    echo "Creating new virtual environment: $VENV with Python $PYTHON_VERSION"
    uv venv --python "$PYTHON_VERSION" "$VENV"
fi

source "$VENV/bin/activate"

echo ""
echo "[2/3] Installing dependencies..."
# Build uv sync command with optional dependency groups
SYNC_CMD=(uv sync --active)
if [[ "$ALL_GROUPS" == "true" ]]; then
    SYNC_CMD+=(--all-groups)
    # Exclude conflicting groups unless explicitly requested via --with-*
    for GROUP in "${CONFLICTING_GROUPS[@]}"; do
        if [[ ! " ${EXTRA_GROUPS[*]:-} " == *" ${GROUP} "* ]]; then
            SYNC_CMD+=(--no-group "$GROUP")
        fi
    done
elif [[ ${#EXTRA_GROUPS[@]} -gt 0 ]]; then
    for GROUP in "${EXTRA_GROUPS[@]}"; do
        SYNC_CMD+=(--group "$GROUP")
    done
fi
# Apply explicit group exclusions (e.g., --without-coreai)
if [[ ${#EXCLUDE_GROUPS[@]} -gt 0 ]]; then
    for GROUP in "${EXCLUDE_GROUPS[@]}"; do
        SYNC_CMD+=(--no-group "$GROUP")
    done
fi
echo "Running: ${SYNC_CMD[*]}"
"${SYNC_CMD[@]}"

echo ""
echo "[3/3] Installing hook dependencies and configuring pre-commit hooks..."
if [[ "$VENV_PREEXISTED" == "true" ]]; then
    echo "Skipped (venv already exists, hooks already configured)"
else
    INSTALL_PRECOMMIT="$INSTALL_PRECOMMIT" "$SCRIPTS_DIR/make/install_pre_commit_hooks.sh" "$COREAI_OPT_HOME"
fi

echo ""
echo "=========================================="
echo "✅ Development environment setup complete!"
echo "=========================================="
echo ""
echo "To activate the virtual environment, run:"
echo "  source $VENV/bin/activate"
echo ""
