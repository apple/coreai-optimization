#!/usr/bin/env python3

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Assemble the multi-version documentation site published to GitHub Pages.

The published site is a plain directory tree, one independent Sphinx build per
version, with a redirect at the root::

    /                 -> redirects to main/
    /main/            docs built from the main branch
    /v0.2.1/          docs built from the v0.2.1 release
    /versions.json    the manifest that drives the version dropdown

Sphinx never needs to know about the other versions: each directory is a separate
build, so publishing one version cannot disturb another. That is what lets a
release's docs stay frozen after it ships while ``main`` keeps moving.

This script runs twice per publish, because the version list has to exist
*before* Sphinx runs (the dropdown is rendered into every page at build time) but
the built HTML only exists after:

``plan``
    Read the versions already present in the site tree, add the one being
    published, and print the manifest. The workflow feeds this to ``sphinx-build``
    via ``DOCS_VERSIONS`` and, on success, writes it to ``versions.json``.

``assemble``
    Copy the freshly built HTML into its version directory, write
    ``versions.json``, the root redirect, and ``.nojekyll``, after checking that
    every version the previous publish recorded is still present.

That last check is the reason this is a script rather than a few shell lines.
``actions/deploy-pages`` republishes the *entire* artifact every run, so it has no
notion of touching one version: if the site tree handed to it were missing a
version — a truncated checkout, a wrong ref — that version would be deleted from
the live site even though the ``gh-pages`` branch still holds it. Comparing the
tree against the ``versions.json`` the last publish wrote catches that before
anything is deployed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path, PurePosixPath

# Directory name of the doc set built from the default branch. Sorts first in the
# dropdown and is what the site root redirects to.
MAIN_VERSION = "main"

# A published release's directory name: `v` + the release tag's version numbers.
# Two or more segments are accepted, not exactly three, so the 4-segment scheme a
# downstream repo can opt into via COREAI_OPT_VERSION_EXTENSION (see RELEASE.md)
# is listed rather than silently omitted from the dropdown.
_RELEASE_DIR = re.compile(r"^v(\d+(?:\.\d+)+)$")

# Top-level entries that are part of the site shell rather than a doc set. Any
# other unexpected directory is reported instead of ignored — see _is_version_dir.
_SITE_SHELL = frozenset({".git", ".github"})

_REDIRECT_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Redirecting to the CoreAI-Opt documentation</title>
    <meta http-equiv="refresh" content="0; url={target}/">
    <link rel="canonical" href="{target}/">
  </head>
  <body>
    <p>Redirecting to <a href="{target}/">{target}/</a>&hellip;</p>
  </body>
</html>
"""


def _is_version_name(name: str) -> bool:
    """Return True if ``name`` is a published doc set's directory name.

    Takes the name rather than the path so the same rule validates ``--version``
    (a string, before any directory exists) and filters the site listing.
    """
    return name == MAIN_VERSION or bool(_RELEASE_DIR.match(name))


def _release_sort_key(name: str) -> tuple[int, ...]:
    """Return the numeric sort key for a release directory name.

    Compares segment by segment as integers, so v0.10.0 outranks v0.9.0 — which a
    lexicographic sort gets backwards. Shorter versions sort below longer ones with
    the same prefix (v0.2.1 below v0.2.1.1), matching how the extension scheme in
    RELEASE.md layers an extra segment onto a published release.
    """
    return tuple(int(part) for part in _RELEASE_DIR.match(name).group(1).split("."))


def sort_versions(versions: set[str] | list[str]) -> list[str]:
    """Return ``versions`` ordered newest first.

    ``main`` sorts ahead of every release, since it documents unreleased work and
    is therefore the most current thing on the site.
    """
    releases = sorted(
        (v for v in versions if v != MAIN_VERSION), key=_release_sort_key, reverse=True
    )
    return ([MAIN_VERSION] if MAIN_VERSION in versions else []) + releases


def discover_versions(site: Path) -> list[str]:
    """Return the version directories present in ``site``, newest first."""
    return sort_versions(
        [child.name for child in site.iterdir() if child.is_dir() and _is_version_name(child.name)]
    )


def versions_after_publish(site: Path, version: str) -> list[str]:
    """Return the site's versions as they will be once ``version`` is published.

    Both subcommands go through this so they cannot disagree: ``plan`` bakes this
    list into every page's dropdown, while ``assemble`` writes it to versions.json
    for the switcher script to re-read. A divergence would leave a page whose
    dropdown contradicts the manifest.
    """
    return sort_versions({*discover_versions(site), version})


def unrecognized_dirs(site: Path) -> list[str]:
    """Return top-level directories that are neither a doc set nor the site shell.

    Reported rather than ignored: a directory this script does not recognize is
    still served by Pages but is absent from the dropdown, so silently skipping it
    would hide a published doc set from every reader.
    """
    return sorted(
        child.name
        for child in site.iterdir()
        if child.is_dir() and not _is_version_name(child.name) and child.name not in _SITE_SHELL
    )


def build_manifest(versions: list[str], url_prefix: str) -> list[list[str]]:
    """Return ``[label, href]`` pairs for the version dropdown.

    Hrefs are absolute from the server root rather than relative, because a
    relative ``../main/`` would resolve against the current page's directory and
    so would break on any nested page (e.g. ``/v0.2.1/quantization/api.html``).
    """
    prefix = url_prefix.strip("/")
    prefix = f"/{prefix}" if prefix else ""
    return [[version_label(v), f"{prefix}/{v}/"] for v in versions]


def version_label(version: str) -> str:
    """Return the human-facing label for ``version`` in the version dropdown.

    ``conf.py`` imports this so the label on the dropdown button matches the one
    in the list the workflow generates; the two are rendered from separate
    template variables, so a divergence would show a button that disagrees with
    its own menu.
    """
    return f"{version} (development)" if version == MAIN_VERSION else version


def _write_manifest(site: Path, manifest: list[list[str]]) -> None:
    (site / "versions.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _published_versions(site: Path) -> set[str]:
    """Return the versions the last publish recorded in ``versions.json``.

    Read back from the manifest rather than the directory listing, so it reflects
    what the previous run *said* it published. Comparing the two is what catches a
    site tree that arrived incomplete.
    """
    manifest = site / "versions.json"
    if not manifest.is_file():
        return set()
    try:
        entries = json.loads(manifest.read_text())
    except json.JSONDecodeError:
        return set()
    # Recover the version from each href's trailing path segment ("/prefix/v0.2.1/").
    return {PurePosixPath(href).name for _label, href in entries}


def _check_site_is_complete(site: Path, present: set[str]) -> str | None:
    """Return an error message if the site tree looks wrong to publish.

    ``deploy-pages`` replaces the entire site with whatever this script assembles,
    so a tree that arrived incomplete — a truncated or wrong-ref checkout — would
    silently take live documentation offline. Fail before that happens.
    """
    missing = _published_versions(site) - present
    if missing:
        return (
            f"Site tree is missing version(s) that versions.json says are "
            f"published: {', '.join(sorted(missing))}. Refusing to publish, because "
            f"deploy-pages replaces the whole site and these would go offline. "
            f"Check that the gh-pages checkout completed."
        )

    unknown = unrecognized_dirs(site)
    if unknown:
        return (
            f"Unrecognized top-level director(ies) on the site: "
            f"{', '.join(unknown)}. These are served but cannot appear in the "
            f"version dropdown, so readers would have no way to reach them. Either "
            f"rename them to 'main' or 'vX.Y.Z', or remove them from the gh-pages "
            f"branch."
        )

    return None


def cmd_plan(args: argparse.Namespace) -> int:
    """Print the version manifest for the site as it will look after publishing.

    Includes the version being published even though its directory does not exist
    yet, so the dropdown Sphinx bakes into the pages lists the doc set the reader
    is currently looking at.
    """
    versions = versions_after_publish(args.site, args.version)
    print(json.dumps(build_manifest(versions, args.url_prefix)))
    return 0


def cmd_assemble(args: argparse.Namespace) -> int:
    """Install the built HTML as ``<site>/<version>/`` and refresh the site shell."""
    if not (args.html / "index.html").is_file():
        print(f"::error::No index.html under {args.html} — refusing to publish an empty doc set.")
        return 1

    versions = versions_after_publish(args.site, args.version)
    error = _check_site_is_complete(args.site, set(versions))
    if error:
        print(f"::error::{error}")
        return 1

    target = args.site / args.version
    # Replace rather than merge: a stale file from the previous build of this same
    # version (a page that was renamed or removed) would otherwise linger forever.
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(args.html, target)

    _write_manifest(args.site, build_manifest(versions, args.url_prefix))

    # The root redirect stands in for a symlink, which GitHub Pages artifacts
    # reject. MAIN_VERSION is the default view: it documents the current source.
    (args.site / "index.html").write_text(_REDIRECT_HTML.format(target=MAIN_VERSION))

    # sphinx.ext.githubpages writes .nojekyll into each build, so every version
    # directory has one — but not the assembled root, which needs its own or
    # Jekyll strips the underscore-prefixed directories (_static, _images).
    (args.site / ".nojekyll").touch()

    print(f"Assembled site with versions: {', '.join(versions)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", type=Path, required=True, help="checkout of the gh-pages branch")
    parser.add_argument(
        "--version", required=True, help=f"version being published ('{MAIN_VERSION}' or 'vX.Y.Z')"
    )
    parser.add_argument(
        "--url-prefix",
        default="",
        help="path the site is served under (e.g. /coreai-optimization)",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan", help="print the post-publish version manifest as JSON")
    assemble = sub.add_parser("assemble", help="install built HTML and refresh the site shell")
    assemble.add_argument("--html", type=Path, required=True, help="Sphinx HTML output directory")

    args = parser.parse_args(argv)
    if not _is_version_name(args.version):
        parser.error(
            f"--version must be '{MAIN_VERSION}' or a version tag like 'v1.2.3', "
            f"got {args.version!r}"
        )
    if not args.site.is_dir():
        parser.error(f"--site is not a directory: {args.site}")

    return cmd_plan(args) if args.command == "plan" else cmd_assemble(args)


if __name__ == "__main__":
    sys.exit(main())
