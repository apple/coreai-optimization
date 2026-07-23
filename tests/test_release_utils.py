# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for scripts/release/release_utils.py version helpers and the extension seam."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from packaging.version import Version

from scripts.release.release_utils import (
    apply_version_extension,
    read_version,
    resolve_about_path,
    resolve_build_version,
    strip_dev_suffix,
    timestamped_dev_version,
    write_version,
)

_FIXED_NOW = datetime(2026, 7, 23, 1, 2, tzinfo=UTC)
_FIXED_TS = "202607230102"
_FIXED_SHA = "abc1234"


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.2.2.dev0", "0.2.2"),
        ("0.2.2", "0.2.2"),
        ("0.2.2.1.dev0", "0.2.2.1"),
        ("0.2.2.1", "0.2.2.1"),
        (f"0.2.2.dev{_FIXED_TS}+{_FIXED_SHA}", "0.2.2"),
        (f"0.2.2.1.dev{_FIXED_TS}+{_FIXED_SHA}", "0.2.2.1"),
    ],
)
def test_strip_dev_suffix(version: str, expected: str) -> None:
    assert strip_dev_suffix(version) == expected


@pytest.mark.parametrize(
    ("version", "extension", "expected"),
    [
        ("0.2.2.dev0", "1", "0.2.2.1.dev0"),
        ("0.2.2", "1", "0.2.2.1"),
        (f"0.2.2.dev{_FIXED_TS}+{_FIXED_SHA}", "1", f"0.2.2.1.dev{_FIXED_TS}+{_FIXED_SHA}"),
        ("0.2.2.dev0", None, "0.2.2.dev0"),
        ("0.2.2.dev0", "", "0.2.2.dev0"),
    ],
)
def test_apply_version_extension(version: str, extension: str | None, expected: str) -> None:
    assert apply_version_extension(version, extension) == expected


@pytest.mark.parametrize("release_base", ["0.2.2", "0.2.2.1"])
def test_timestamped_dev_version_does_not_bump_and_uses_base(release_base: str) -> None:
    result = timestamped_dev_version(release_base, now=_FIXED_NOW, sha=_FIXED_SHA)
    assert result == f"{release_base}.dev{_FIXED_TS}+{_FIXED_SHA}"
    # The release segment is preserved verbatim (no patch/segment bump).
    assert strip_dev_suffix(result) == release_base


def test_resolve_build_version_override_wins() -> None:
    version = resolve_build_version("0.2.2", dev=True, dev_version_override="9.9.9.dev1+deadbee")
    assert version == "9.9.9.dev1+deadbee"


def test_resolve_build_version_dev_generates_timestamped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.release.release_utils.timestamped_dev_version",
        lambda base: f"{base}.dev{_FIXED_TS}+{_FIXED_SHA}",
    )
    version = resolve_build_version("0.2.2", dev=True)
    assert version == f"0.2.2.dev{_FIXED_TS}+{_FIXED_SHA}"


def test_resolve_build_version_release_is_used_as_is() -> None:
    # resolve_build_version doesn't strip .dev itself — callers pass a
    # release_base with any suffix/extension already resolved.
    assert resolve_build_version("0.2.2", dev=False) == "0.2.2"


def test_build_pipeline_strips_dev_suffix_for_a_plain_release() -> None:
    # The pipeline build.py runs: strip .dev0 from the on-tree version, then
    # resolve_build_version. A plain `make build` must never ship `.dev0`.
    release_base = strip_dev_suffix("0.2.2.dev0")
    assert resolve_build_version(release_base, dev=False) == "0.2.2"


def test_resolve_build_version_applies_extension() -> None:
    # COREAI_OPT_VERSION_EXTENSION inserts an extra release segment; callers
    # apply it (and strip .dev) before calling resolve_build_version.
    release_base = strip_dev_suffix(apply_version_extension("0.2.2.dev0", "1"))
    assert resolve_build_version(release_base, dev=False) == "0.2.2.1"
    version = resolve_build_version(release_base, dev=True)
    assert version.startswith("0.2.2.1.dev")


def test_pep440_sort_order() -> None:
    ordered = [
        "0.2.2.1.dev0",
        f"0.2.2.1.dev{_FIXED_TS}+{_FIXED_SHA}",
        "0.2.2.1",
        "0.2.3",
        "0.2.3.1.dev0",
        "0.2.3.1",
    ]
    versions = [Version(v) for v in ordered]
    assert versions == sorted(versions)


def test_resolve_about_path_resolves_both_layouts(tmp_path: Path) -> None:
    for layout in (Path("src"), Path("external") / "src"):
        pkg_dir = tmp_path / layout / "coreai_opt"
        pkg_dir.mkdir(parents=True)
        about = pkg_dir / "_about.py"
        about.write_text('__version__ = "1.2.3"\n', encoding="utf-8")
        assert resolve_about_path(tmp_path) == about
        about.unlink()  # remove so the next layout is the only match


def test_resolve_about_path_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_about_path(tmp_path)


def test_read_version() -> None:
    assert read_version('__version__ = "0.2.2.dev0"\n') == "0.2.2.dev0"
    assert read_version("# header\n__version__ = '0.2.2.dev0'\n_other = 1\n") == "0.2.2.dev0"


def test_read_version_missing_raises() -> None:
    with pytest.raises(RuntimeError):
        read_version("x = 1\n")


def test_write_version_rewrites_literal(tmp_path: Path) -> None:
    about = tmp_path / "_about.py"
    about.write_text('# header\n__version__ = "0.2.2.dev0"\n_other = 1\n', encoding="utf-8")
    write_version(about, "0.2.2.1.dev202607230102+abc1234")
    text = about.read_text(encoding="utf-8")
    assert '__version__ = "0.2.2.1.dev202607230102+abc1234"' in text
    assert "_other = 1" in text  # only the __version__ line is rewritten


def test_write_version_missing_assignment_raises(tmp_path: Path) -> None:
    about = tmp_path / "_about.py"
    about.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        write_version(about, "1.2.3")


def test_write_version_noop_when_value_unchanged(tmp_path: Path) -> None:
    # Writing the same value back (e.g. re-running a release build) is a no-op
    # rewrite when __version__ is already that literal. The line was still
    # matched and rewritten, so this must not raise "missing assignment".
    about = tmp_path / "_about.py"
    about.write_text('__version__ = "0.2.2"\n', encoding="utf-8")
    write_version(about, "0.2.2")
    assert about.read_text(encoding="utf-8") == '__version__ = "0.2.2"\n'
