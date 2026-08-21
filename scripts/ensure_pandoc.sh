#!/usr/bin/env bash

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

#
# Ensure the `pandoc` binary is available, installing it if needed.
#
# nbsphinx shells out to `pandoc` to convert notebook markdown cells during the
# docs build. On macOS it comes from Homebrew. On Linux the dnf/apt repos do not
# ship pandoc, so we download the upstream static binary into the repo's own
# `.local/bin/` (gitignored) rather than a system prefix — CI runners and any
# other unprivileged environment cannot write to /usr/local, and a docs build
# should not need root. Override the version with PANDOC_VERSION.
#
# Prints nothing but diagnostics on stdout, so callers can capture the install
# location from `pandoc_bin_dir` below if they need to extend PATH themselves.
#
# Usage: ensure_pandoc.sh

set -euo pipefail

# shellcheck source=utils.sh
source "$(dirname -- "${BASH_SOURCE[0]}")/utils.sh"

# Repo-local install prefix. Resolved from this script's location rather than the
# caller's working directory, so it lands in the same place wherever it runs from.
_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
pandoc_prefix="${PANDOC_PREFIX:-$_repo_root/.local}"
pandoc_bin_dir="$pandoc_prefix/bin"

# An install from a previous run is on disk but not necessarily on PATH.
if [[ -x "$pandoc_bin_dir/pandoc" ]]; then
    export PATH="$pandoc_bin_dir:$PATH"
fi

if command -v pandoc &>/dev/null; then
    echo "pandoc already installed: $(pandoc --version | head -n1)"
    exit 0
fi

echo "pandoc not found. Attempting installation..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    ensure_package pandoc
    exit
fi

if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "Error: unsupported OS ($OSTYPE). Install pandoc manually." >&2
    exit 1
fi

# Linux: dnf/apt repos do not provide pandoc — fetch the upstream binary.
pandoc_version="${PANDOC_VERSION:-3.9.0.2}"
case "$(uname -m)" in
x86_64) arch="amd64" ;;
aarch64 | arm64) arch="arm64" ;;
*)
    echo "Error: unsupported architecture $(uname -m) for pandoc." >&2
    exit 1
    ;;
esac

# SHA-256 digests keyed by "<version>-<arch>".
# Source: https://api.github.com/repos/jgm/pandoc/releases/tags/<VERSION>
# When bumping PANDOC_VERSION, add the new digests here.
declare -A PANDOC_SHA256=(
    ["3.9.0.2-amd64"]="a69abfababda8a56969a254b09f9553a7be89ddec00d4e0fe9fd585d71a67508"
    ["3.9.0.2-arm64"]="b6d21e8f9c3b15744f5a7ab40248019157ed7793875dbe0383d4c82ff572b528"
)
sha256_key="${pandoc_version}-${arch}"
expected_sha256="${PANDOC_SHA256[$sha256_key]:-}"
if [[ -z "$expected_sha256" ]]; then
    echo "Error: no pinned SHA-256 for pandoc ${pandoc_version} (${arch})." >&2
    echo "Update PANDOC_SHA256 in $(basename "$0") using the asset digests from:" >&2
    echo "  https://api.github.com/repos/jgm/pandoc/releases/tags/${pandoc_version}" >&2
    exit 1
fi

base_url="https://github.com/jgm/pandoc/releases/download/${pandoc_version}"
tarball="pandoc-${pandoc_version}-linux-${arch}.tar.gz"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

echo "Downloading pandoc ${pandoc_version} (linux-${arch}) from GitHub..."
if ! curl -fsSL "${base_url}/${tarball}" -o "${tmpdir}/${tarball}"; then
    echo "Error: failed to download pandoc from ${base_url}/${tarball}." >&2
    exit 1
fi

echo "Verifying SHA-256 checksum..."
if ! echo "${expected_sha256}  ${tarball}" | (cd "$tmpdir" && sha256sum --check); then
    echo "Error: SHA-256 verification failed for ${tarball}." >&2
    exit 1
fi

echo "Extracting to ${pandoc_prefix}..."
mkdir -p "$pandoc_prefix"
# `--no-same-owner --no-same-permissions` because the upstream tarball records
# root-owned entries with fixed modes; replaying those as an unprivileged user is
# what made extraction into a system prefix fail.
#
# The whole archive is extracted, not just bin/. Selecting a subset would need a
# glob, and the two tars disagree about those: GNU tar ignores patterns unless
# given --wildcards, while BSD tar rejects that flag outright. Extracting
# everything sidesteps the incompatibility, and the extra man pages are harmless
# in a prefix we own.
if ! tar xz --strip-components=1 --no-same-owner --no-same-permissions \
    -C "$pandoc_prefix" -f "${tmpdir}/${tarball}"; then
    echo "Error: failed to extract ${tarball}." >&2
    exit 1
fi

export PATH="$pandoc_bin_dir:$PATH"

if ! command -v pandoc &>/dev/null; then
    echo "Error: pandoc installation failed. Please install pandoc manually." >&2
    exit 1
fi
echo "pandoc installed successfully: $(pandoc --version | head -n1)"
echo "Installed to ${pandoc_bin_dir} — add it to PATH to use pandoc directly."
