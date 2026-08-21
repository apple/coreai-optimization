#!/usr/bin/env bash

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

#
# Build a local copy of the published multi-version documentation site.
#
# `make docs-open` serves one doc set as the web root, but the real site nests each
# version a level down (/main/, /v0.2.1/) with a redirect and versions.json at the
# root — so it cannot exercise the version picker, the root redirect, or the
# switcher script's versions.json fetch. This builds that layout locally:
#
#   <site>/main/      built from the working tree
#   <site>/vX.Y.Z/    built from that tag's documented source
#
# A tag supplies *source only* — its docstrings, prose pages, notebooks and
# tutorials. The build itself always uses the current tree's tooling: this
# conf.py, these docs/scripts/, this Makefile, this docs environment. So a tag
# cut before versioned docs existed still renders with the version picker,
# because the picker comes from the current conf.py rather than the tag's.
#
# That is what the layout below buys. Each tag's `src/` and `docs/src/` are staged
# under .local/docs/<version>/ and the current `conf.py` is pointed at them, so no
# tag ever contributes build logic:
#
#   .local/docs/vX.Y.Z/
#   ├── src/         <- from the tag (autodoc reads docstrings here)
#   ├── docs/src/    <- from the tag (prose, notebooks, images)
#   └── docs/scripts/<- from the CURRENT tree (extensions, api index generator)
#
# Usage:
#   build_versioned_preview.sh --site <dir> [--tags "v0.2.1 v0.3.0"] [--all-tags]
#
# With neither --tags nor --all-tags, only `main` is built.
#
# Environment:
#   MAKE_FLAGS   extra flags for the `main` build's `make docs` (e.g. QUIET=0)

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
ASSEMBLER="$SCRIPT_DIR/assemble_versioned_site.py"
STAGE_ROOT="$REPO_ROOT/.local/docs"

SITE=""
TAGS=()
ALL_TAGS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
    --site)
        SITE="$2"
        shift 2
        ;;
    --tags)
        # shellcheck disable=SC2206  # deliberate word-split of a space-separated list
        TAGS=($2)
        shift 2
        ;;
    --all-tags)
        ALL_TAGS=true
        shift
        ;;
    *)
        echo "Error: unknown argument '$1'" >&2
        exit 1
        ;;
    esac
done

if [[ -z "$SITE" ]]; then
    echo "Error: --site is required" >&2
    exit 1
fi

# Absolute, because the staged builds run from other directories.
mkdir -p "$SITE"
SITE="$(cd -- "$SITE" && pwd)"

if [[ "$ALL_TAGS" == true ]]; then
    # Sorted oldest-first only for readable progress output; the assembler owns the
    # order the dropdown actually uses.
    mapfile -t TAGS < <(git -C "$REPO_ROOT" tag --list 'v[0-9]*.[0-9]*.[0-9]*' --sort=v:refname)
    if [[ ${#TAGS[@]} -eq 0 ]]; then
        echo "==> No vX.Y.Z tags found locally. Fetch them first:"
        echo "        git fetch --tags origin"
        echo "==> Building 'main' only."
    fi
fi

# A tag can only be previewed if it carries the two source trees the current
# conf.py reads: docs/src/ for prose and src/coreai_opt/ for docstrings. Checked
# with `git cat-file` so an unusable tag is skipped without any checkout.
tag_has_docs_source() {
    git -C "$REPO_ROOT" cat-file -e "$1:docs/src/index.md" 2>/dev/null &&
        git -C "$REPO_ROOT" cat-file -e "$1:src/coreai_opt/__init__.py" 2>/dev/null
}

BUILDABLE=()
for tag in ${TAGS[@]+"${TAGS[@]}"}; do
    if tag_has_docs_source "$tag"; then
        BUILDABLE+=("$tag")
    else
        echo "==> Skipping $tag: no docs/src + src/coreai_opt to build from"
    fi
done

# Stand up the version directories before planning. The dropdown is rendered into
# every page at build time, so the full list has to exist before any Sphinx run.
for tag in ${BUILDABLE[@]+"${BUILDABLE[@]}"}; do
    mkdir -p "$SITE/$tag"
done

# Locally the site root *is* the server root, so no URL prefix — on GitHub Pages
# the site is served under /<repo-name>/ and the workflow passes that instead.
VERSIONS="$(python3 "$ASSEMBLER" --site "$SITE" --version main --url-prefix '' plan)"
echo "==> Previewing versions: $VERSIONS"

# Stage one tag's source under .local/docs/<tag>/ and build it with the current
# tooling. Layout mirrors the repo (src/ beside docs/) because conf.py locates the
# package by walking up from itself looking for src/coreai_opt — the same search
# that makes it work in both the OSS and internal trees.
stage_tag() {
    local tag="$1"
    local stage="$STAGE_ROOT/$tag"

    rm -rf "$stage"
    mkdir -p "$stage/docs"

    # Source from the tag: docstrings and prose.
    git -C "$REPO_ROOT" archive "$tag" src | tar -x -C "$stage"
    git -C "$REPO_ROOT" archive "$tag" docs/src | tar -x -C "$stage"

    # Tooling from the current tree: conf.py, the templates and static assets it
    # references, and the api-index generator. Applied after the tag's docs/src so
    # the current versions win.
    #
    # The trailing `/.` matters. The tag already ships _templates/, _static/ and
    # docs/scripts/, and `cp -a src dest` copies *into* an existing dest — which
    # would produce _templates/_templates/ and leave the tag's own templates in
    # place, so the version picker would render from the tag's markup instead of
    # ours. Removing the destination first makes the copy a replacement.
    for dir in docs/scripts docs/src/_templates docs/src/_static; do
        rm -rf "$stage/$dir"
        cp -a "$REPO_ROOT/$dir" "$stage/$dir"
    done
    cp -a "$REPO_ROOT/docs/src/conf.py" "$stage/docs/src/conf.py"
}

build_version() {
    local version="$1" src_dir="$2"

    echo ""
    echo "==> Building $version"
    # PYTHONPATH is what makes this a build of the *tag's* source. The docs venv
    # installs coreai_opt editable, pointing at the working tree's src/, so without
    # this autodoc would read current docstrings while the prose came from the tag.
    # PYTHONPATH precedes site-packages, so the staged copy wins.
    #
    # Reuses the repo's .venv-docs rather than syncing a per-version env: the
    # staged tree is built by the current conf.py, so the current environment is
    # the correct one by definition. `sphinx-build` directly rather than
    # `make docs` because the staged tree has no Makefile — and the mermaid-cli /
    # Chrome / pandoc setup `make docs` performs has already run for `main`.
    PYTHONPATH="$src_dir/src" DOCS_VERSION="$version" DOCS_VERSIONS="$VERSIONS" \
        uv run --no-sync --active sphinx-build -E -b html \
        "$src_dir/docs/src" "$src_dir/docs/build/html"

    cp -a "$src_dir/docs/build/html/." "$SITE/$version/"
}

# Build `main` first, through `make docs`, so its mermaid-cli / Chrome / pandoc /
# .venv-docs setup runs once and the staged tag builds can reuse all of it.
echo ""
echo "==> Building main from the working tree"
DOCS_VERSION=main DOCS_VERSIONS="$VERSIONS" \
    make -C "$REPO_ROOT" docs _QUIET_HEADER=1 _DOCS_ALL=1 ${MAKE_FLAGS:-}

# shellcheck source=/dev/null
source "$REPO_ROOT/.venv-docs/bin/activate"

for tag in ${BUILDABLE[@]+"${BUILDABLE[@]}"}; do
    stage_tag "$tag"
    build_version "$tag" "$STAGE_ROOT/$tag"
done

python3 "$ASSEMBLER" --site "$SITE" --version main --url-prefix '' \
    assemble --html "$REPO_ROOT/docs/build/html"
