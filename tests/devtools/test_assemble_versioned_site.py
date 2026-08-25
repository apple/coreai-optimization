# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for docs/scripts/assemble_versioned_site.py.

The site assembler decides what the published documentation site contains, and
``actions/deploy-pages`` replaces the whole site with whatever it produces. A bug
here does not fail loudly — it takes previously published release docs offline. So
these tests focus on the properties that protect against that: version ordering
(which a lexicographic sort gets wrong), which directory names count as a doc set,
and the refusal to publish a tree that lost a version.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from _test_helpers import load_script

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "docs" / "scripts" / "assemble_versioned_site.py"
_URL_PREFIX = "/coreai-optimization"

asm = load_script(_SCRIPT)


@pytest.fixture
def site(tmp_path: Path) -> Path:
    """A site tree holding several published versions plus a site-shell directory."""
    root = tmp_path / "site"
    for name in ("main", "v0.2.1", "v0.9.0", "v0.10.0", ".git"):
        (root / name).mkdir(parents=True)
    return root


@pytest.fixture
def html(tmp_path: Path) -> Path:
    """A minimal Sphinx HTML output directory."""
    out = tmp_path / "html"
    out.mkdir()
    (out / "index.html").write_text("<html>docs</html>")
    return out


def run_plan(site: Path, version: str) -> int:
    """Invoke the ``plan`` subcommand the way the workflow does."""
    argv = ["--site", str(site), "--version", version, "--url-prefix", _URL_PREFIX]
    return asm.main([*argv, "plan"])


def run_assemble(site: Path, version: str, html: Path) -> int:
    """Invoke the ``assemble`` subcommand the way the workflow does."""
    argv = ["--site", str(site), "--version", version, "--url-prefix", _URL_PREFIX]
    return asm.main([*argv, "assemble", "--html", str(html)])


def test_discover_versions_orders_main_first_then_by_number(site: Path) -> None:
    """`main` leads, then releases descend numerically — not lexicographically."""
    # v0.10.0 must outrank v0.9.0; sorted() on the strings would invert them.
    assert asm.discover_versions(site) == ["main", "v0.10.0", "v0.9.0", "v0.2.1"]


def test_discover_versions_ignores_non_version_directories(site: Path) -> None:
    """Site-shell directories must not be mistaken for published doc sets."""
    assert ".git" not in asm.discover_versions(site)


def test_discover_versions_accepts_a_four_segment_version(site: Path, html: Path) -> None:
    """A version carrying the RELEASE.md extension segment is a real doc set.

    A downstream repo can add a 4th number via COREAI_OPT_VERSION_EXTENSION, so
    `v0.2.1.1` must appear in the dropdown rather than being silently skipped.
    """
    (site / "v0.2.1.1").mkdir()

    assert asm.discover_versions(site) == ["main", "v0.10.0", "v0.9.0", "v0.2.1.1", "v0.2.1"]

    assert run_assemble(site, "main", html) == 0
    manifest = json.loads((site / "versions.json").read_text())
    assert ["v0.2.1.1", "/coreai-optimization/v0.2.1.1/"] in manifest


def test_build_manifest_uses_root_absolute_hrefs() -> None:
    """Hrefs must be absolute, or the dropdown breaks on nested pages."""
    manifest = asm.build_manifest(["main", "v0.3.0"], _URL_PREFIX)
    assert manifest == [
        ["main (development)", "/coreai-optimization/main/"],
        ["v0.3.0", "/coreai-optimization/v0.3.0/"],
    ]


def test_build_manifest_without_url_prefix() -> None:
    """A site served from the domain root still yields absolute hrefs."""
    assert asm.build_manifest(["main"], "") == [["main (development)", "/main/"]]


def test_version_label_marks_main_as_development() -> None:
    """conf.py imports this, so the dropdown button matches its own menu entry."""
    assert asm.version_label("main") == "main (development)"
    assert asm.version_label("v0.3.0") == "v0.3.0"


def test_assemble_publishes_only_the_named_version(site: Path, html: Path) -> None:
    """Publishing one version must leave every other version untouched."""
    (site / "v0.2.1" / "index.html").write_text("<html>old release</html>")

    assert run_assemble(site, "v0.3.0", html) == 0

    assert (site / "v0.3.0" / "index.html").read_text() == "<html>docs</html>"
    # The frozen release is byte-for-byte as it was.
    assert (site / "v0.2.1" / "index.html").read_text() == "<html>old release</html>"


def test_assemble_writes_the_site_shell(site: Path, html: Path) -> None:
    """The root redirect, .nojekyll and manifest are all refreshed."""
    assert run_assemble(site, "main", html) == 0

    assert (site / ".nojekyll").exists()
    assert "url=main/" in (site / "index.html").read_text()
    manifest = json.loads((site / "versions.json").read_text())
    assert manifest[0] == ["main (development)", "/coreai-optimization/main/"]
    assert ["v0.2.1", "/coreai-optimization/v0.2.1/"] in manifest


def test_assemble_replaces_rather_than_merges(site: Path, html: Path) -> None:
    """A page removed since the last build must not survive in the new one."""
    (site / "main" / "removed.html").write_text("<html>stale</html>")

    assert run_assemble(site, "main", html) == 0

    assert not (site / "main" / "removed.html").exists()


def test_assemble_rejects_an_empty_build(site: Path, tmp_path: Path) -> None:
    """A build that produced no index.html must not replace a live doc set."""
    empty = tmp_path / "empty"
    empty.mkdir()

    assert run_assemble(site, "main", empty) == 1


def test_assemble_fails_closed_when_a_published_version_is_missing(site: Path, html: Path) -> None:
    """A tree missing a version its own manifest lists aborts the publish.

    This is the real hazard — a truncated or wrong-ref gh-pages checkout. Because
    deploy-pages replaces the whole site, publishing such a tree would take the
    missing release offline even though the branch still holds it.
    """
    # A manifest from a previous publish that shipped a version now absent on disk.
    assert run_assemble(site, "main", html) == 0
    shutil.rmtree(site / "v0.2.1")

    assert run_assemble(site, "main", html) == 1


def test_assemble_fails_closed_on_an_unrecognized_directory(site: Path, html: Path) -> None:
    """A directory that can't appear in the dropdown must not be silently ignored.

    Pages would still serve it, but no reader could navigate to it — so treat it as
    a mistake to fix rather than skipping past it.
    """
    (site / "latest").mkdir()

    assert run_assemble(site, "main", html) == 1


def test_plan_includes_the_version_being_published(
    site: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A brand-new version appears in the dropdown of its own first build."""
    assert run_plan(site, "v0.3.0") == 0

    manifest = json.loads(capsys.readouterr().out)
    assert ["v0.3.0", "/coreai-optimization/v0.3.0/"] in manifest


@pytest.mark.parametrize("bad", ["1.2.3", "latest", "v1.2.3rc1", "main-dev", "v1", "v1.2."])
def test_rejects_version_names_that_are_not_doc_sets(site: Path, bad: str) -> None:
    """Only `main` and a `v` + dotted-numbers tag may name a directory on the site."""
    with pytest.raises(SystemExit) as excinfo:
        run_plan(site, bad)
    assert excinfo.value.code != 0


@pytest.mark.parametrize("good", ["main", "v1.2", "v1.2.3", "v1.2.3.4"])
def test_accepts_version_names_with_any_segment_count(site: Path, good: str) -> None:
    """Two or more numeric segments are all valid doc-set names.

    The release scheme is three segments today and four when a downstream repo
    applies COREAI_OPT_VERSION_EXTENSION, so the count is not fixed.
    """
    assert run_plan(site, good) == 0
