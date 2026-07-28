# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for scripts/release/release_utils.py version helpers and the extension seam."""

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from packaging.version import Version

from scripts.release.release_utils import (
    DEV_VERSION_ENV_VAR,
    VERSION_EXTENSION_ENV_VAR,
    AboutFile,
    apply_version_extension,
    bump_last_segment,
    get_dev_version_override,
    get_version_extension,
    latest_release_tag,
    next_candidate_version,
    next_release_base,
    read_about,
    read_latest_released_version,
    read_version,
    resolve_about_path,
    resolve_build_version,
    restore_about,
    strip_dev_suffix,
    timestamped_dev_version,
    write_version,
)

_FIXED_NOW = datetime(2026, 7, 23, 1, 2, tzinfo=UTC)
_FIXED_TS = "202607230102"
_FIXED_SHA = "abc1234"


class TestVersionArithmetic:
    """Tests for strip_dev_suffix, bump_last_segment, and apply_version_extension."""

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
    def test_strip_dev_suffix(self, version: str, expected: str) -> None:
        assert strip_dev_suffix(version) == expected

    @pytest.mark.parametrize(
        ("version", "expected"),
        [
            ("0.2.1", "0.2.2"),
            ("0.2.1.1", "0.2.1.2"),
            ("1.0.9", "1.0.10"),
        ],
    )
    def test_bump_last_segment(self, version: str, expected: str) -> None:
        assert bump_last_segment(version) == expected

    @pytest.mark.parametrize(
        ("version", "extension", "expected"),
        [
            ("0.2.1", "1", "0.2.1.1"),
            ("0.2.1", "0", "0.2.1.0"),
            ("0.2.1", None, "0.2.1"),
            ("0.2.1", "", "0.2.1"),
        ],
    )
    def test_apply_version_extension(
        self, version: str, extension: str | None, expected: str
    ) -> None:
        assert apply_version_extension(version, extension) == expected


class TestNextVersion:
    """Tests for next_release_base and next_candidate_version."""

    @pytest.mark.parametrize(
        ("latest_released", "extension", "expected"),
        [
            # No extension: add one to the external release's own last number.
            ("0.2.1", None, "0.2.2"),
            # extension is the number the downstream repo is about to release
            # next, used exactly as given (see the next_release_base docstring).
            ("0.2.1", "1", "0.2.1.1"),  # first release off 0.2.1
            ("0.2.1", "2", "0.2.1.2"),  # second release off 0.2.1
        ],
    )
    def test_next_release_base(
        self, latest_released: str, extension: str | None, expected: str
    ) -> None:
        assert next_release_base(latest_released, extension) == expected

    @pytest.mark.parametrize(
        ("latest_released", "extension", "expected"),
        [
            ("0.2.1", None, "0.2.2.dev0"),
            ("0.2.1", "1", "0.2.1.1.dev0"),
        ],
    )
    def test_next_candidate_version_is_next_release_base_plus_dev0(
        self, latest_released: str, extension: str | None, expected: str
    ) -> None:
        assert next_candidate_version(latest_released, extension) == expected

    def test_pep440_sort_order(self) -> None:
        ordered = [
            "0.2.1.1.dev0",
            f"0.2.1.1.dev{_FIXED_TS}+{_FIXED_SHA}",
            "0.2.1.1",
            "0.2.1.2.dev0",
            "0.2.1.2",
            "0.2.2.dev0",
            "0.2.2",
        ]
        versions = [Version(v) for v in ordered]
        assert versions == sorted(versions)


class TestResolveBuildVersion:
    """Tests for resolve_build_version and the timestamped_dev_version it generates."""

    @pytest.mark.parametrize("release_base", ["0.2.2", "0.2.2.1"])
    def test_timestamped_dev_version_does_not_bump_and_uses_base(self, release_base: str) -> None:
        result = timestamped_dev_version(release_base, now=_FIXED_NOW, sha=_FIXED_SHA)
        assert result == f"{release_base}.dev{_FIXED_TS}+{_FIXED_SHA}"
        # The release segment is preserved verbatim (no patch/segment bump).
        assert strip_dev_suffix(result) == release_base

    def test_override_wins(self) -> None:
        version = resolve_build_version(
            "0.2.2", dev=True, dev_version_override="9.9.9.dev1+deadbee"
        )
        assert version == "9.9.9.dev1+deadbee"

    def test_dev_generates_timestamped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "scripts.release.release_utils.timestamped_dev_version",
            lambda base: f"{base}.dev{_FIXED_TS}+{_FIXED_SHA}",
        )
        version = resolve_build_version("0.2.2", dev=True)
        assert version == f"0.2.2.dev{_FIXED_TS}+{_FIXED_SHA}"

    def test_release_is_used_as_is(self) -> None:
        # resolve_build_version doesn't build the version number itself — the
        # caller passes it a release_base already computed by next_release_base.
        assert resolve_build_version("0.2.2", dev=False) == "0.2.2"

    def test_build_pipeline_computes_release_base_from_latest_released(self) -> None:
        # This is what build.py does: next_release_base(latest_released, extension),
        # then resolve_build_version. It never reads the on-tree __version__.
        release_base = next_release_base("0.2.1", "1")
        assert resolve_build_version(release_base, dev=False) == "0.2.1.1"
        version = resolve_build_version(release_base, dev=True)
        assert version.startswith("0.2.1.1.dev")


class TestResolveAboutPath:
    """Tests for resolve_about_path."""

    @pytest.mark.parametrize("layout", [Path("src"), Path("external") / "src"])
    def test_resolves_both_layouts(self, tmp_path: Path, layout: Path) -> None:
        pkg_dir = tmp_path / layout / "coreai_opt"
        pkg_dir.mkdir(parents=True)
        about = pkg_dir / "_about.py"
        about.write_text('__version__ = "1.2.3"\n', encoding="utf-8")
        assert resolve_about_path(tmp_path) == about

    def test_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            resolve_about_path(tmp_path)


class TestReadVersion:
    """Tests for read_version and read_latest_released_version."""

    @pytest.mark.parametrize(
        "source",
        [
            '__version__ = "0.2.2.dev0"\n',  # double-quoted
            "__version__ = '0.2.2.dev0'\n",  # single-quoted
            '__version__ = ("0.2.2.dev0")\n',  # parenthesized
            '__version__ = ("0.2.2" ".dev0")\n',  # adjacent string pieces
            '__version__   =    "0.2.2.dev0"\n',  # extra whitespace
            '__version__ = "0.2.2.dev0"  # next release\n',  # trailing comment
            "# header\n__version__ = '0.2.2.dev0'\n_other = 1\n",  # surrounded by other lines
        ],
    )
    def test_accepts_any_literal_spelling(self, source: str) -> None:
        # Parsing the source (rather than pattern-matching a line) accepts exactly
        # what setuptools' `attr:` reader accepts, so anything that reads here also
        # builds.
        assert read_version(source) == "0.2.2.dev0"

    def test_missing_raises(self) -> None:
        with pytest.raises(RuntimeError):
            read_version("x = 1\n")

    def test_rejects_a_computed_value(self) -> None:
        # setuptools reads `__version__` statically and would fail the build on a
        # computed value, so reading rejects it too rather than deferring the error.
        with pytest.raises(RuntimeError, match="plain string literal"):
            read_version('_base = "0.2.2"\n__version__ = _base + ".dev0"\n')

    def test_rejects_a_non_string(self) -> None:
        with pytest.raises(RuntimeError, match="must be a string"):
            read_version("__version__ = 42\n")

    def test_reads_latest_released_version(self) -> None:
        text = 'latest_released_version = "0.2.1"\n__version__ = "0.2.2.dev0"\n'
        assert read_latest_released_version(text) == "0.2.1"

    def test_latest_released_version_missing_raises(self) -> None:
        with pytest.raises(RuntimeError):
            read_latest_released_version('__version__ = "0.2.2.dev0"\n')


class TestReadAbout:
    """Tests for read_about."""

    def test_returns_path_text_and_both_versions(self, tmp_path: Path) -> None:
        about_dir = tmp_path / "src" / "coreai_opt"
        about_dir.mkdir(parents=True)
        source = '# header\nlatest_released_version = "0.2.1"\n__version__ = "0.2.2.dev0"\n'
        (about_dir / "_about.py").write_text(source, encoding="utf-8")

        about = read_about(tmp_path)

        assert about.path == about_dir / "_about.py"
        # The text is kept verbatim so build.py can restore the file byte-for-byte.
        assert about.text == source
        assert about.latest_released_version == "0.2.1"
        assert about.version == "0.2.2.dev0"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_about(tmp_path)

    def test_missing_field_raises(self, tmp_path: Path) -> None:
        about_dir = tmp_path / "src" / "coreai_opt"
        about_dir.mkdir(parents=True)
        # `__version__` present but `latest_released_version` absent.
        (about_dir / "_about.py").write_text('__version__ = "0.2.2.dev0"\n', encoding="utf-8")
        with pytest.raises(RuntimeError, match="latest_released_version"):
            read_about(tmp_path)


class TestWriteVersion:
    """Tests for write_version and restore_about."""

    @staticmethod
    def _about(tmp_path: Path, source: str) -> AboutFile:
        path = tmp_path / "_about.py"
        path.write_text(source, encoding="utf-8")
        return AboutFile(
            path=path, text=source, latest_released_version="0.2.1", version="0.2.2.dev0"
        )

    def test_rewrites_literal(self, tmp_path: Path) -> None:
        about = self._about(tmp_path, '# header\n__version__ = "0.2.2.dev0"\n_other = 1\n')
        write_version(about, "0.2.2.1.dev202607230102+abc1234")
        text = about.path.read_text(encoding="utf-8")
        assert '__version__ = "0.2.2.1.dev202607230102+abc1234"' in text
        assert "_other = 1" in text  # only the __version__ value is rewritten

    @pytest.mark.parametrize(
        "source",
        [
            '__version__ = "0.2.2.dev0"\n',  # double-quoted
            "__version__ = '0.2.2.dev0'\n",  # single-quoted
            '__version__ = ("0.2.2.dev0")\n',  # parenthesized
            '__version__ = ("0.2.2" ".dev0")\n',  # adjacent string pieces
            '__version__ = (\n    "0.2.2.dev0"\n)\n',  # split across lines
            '__version__ = "0.2.2.dev0"  # next release\n',  # trailing comment
        ],
    )
    def test_rewrites_every_spelling_read_version_accepts(
        self, tmp_path: Path, source: str
    ) -> None:
        # Writes locate the assignment with ast, the same way reads do, so any
        # spelling that reads here can also be rewritten. A line-oriented
        # pattern would accept only the first two and fail the release build on
        # the rest, long after `make version` and the pre-commit hook passed.
        about = self._about(tmp_path, source)
        write_version(about, "0.2.2")
        assert read_version(about.path.read_text(encoding="utf-8")) == "0.2.2"

    def test_missing_assignment_raises(self, tmp_path: Path) -> None:
        about = self._about(tmp_path, "x = 1\n")
        with pytest.raises(RuntimeError):
            write_version(about, "1.2.3")

    def test_noop_when_value_unchanged(self, tmp_path: Path) -> None:
        # Writing the same value back (e.g. re-running a release build) must
        # leave the file untouched rather than raise "missing assignment".
        about = self._about(tmp_path, '__version__ = "0.2.2"\n')
        write_version(about, "0.2.2")
        assert about.path.read_text(encoding="utf-8") == '__version__ = "0.2.2"\n'

    def test_restore_about_puts_back_the_exact_bytes(self, tmp_path: Path) -> None:
        source = '# header\n__version__ = "0.2.2.dev0"\n_other = 1\n'
        about = self._about(tmp_path, source)
        write_version(about, "9.9.9")
        restore_about(about)
        assert about.path.read_text(encoding="utf-8") == source


class TestRepoAndEnvInputs:
    """Tests for the env-var getters and latest_release_tag."""

    def test_get_version_extension(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(VERSION_EXTENSION_ENV_VAR, raising=False)
        assert get_version_extension() is None

        monkeypatch.setenv(VERSION_EXTENSION_ENV_VAR, "1")
        assert get_version_extension() == "1"

    def test_get_dev_version_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(DEV_VERSION_ENV_VAR, raising=False)
        assert get_dev_version_override() is None

        monkeypatch.setenv(DEV_VERSION_ENV_VAR, "9.9.9.dev1+deadbee")
        assert get_dev_version_override() == "9.9.9.dev1+deadbee"

    def test_latest_release_tag_reads_the_highest_version_tag(self, tmp_path: Path) -> None:
        subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "commit.gpgSign", "false"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "tag.gpgSign", "false"], cwd=tmp_path, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True
        )
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
        (tmp_path / "README.md").write_text("hi\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "--quiet", "-m", "init"], cwd=tmp_path, check=True)
        subprocess.run(["git", "tag", "v0.2.0"], cwd=tmp_path, check=True)
        subprocess.run(["git", "tag", "v0.2.1"], cwd=tmp_path, check=True)
        subprocess.run(["git", "tag", "not-a-release"], cwd=tmp_path, check=True)

        assert latest_release_tag(tmp_path) == "0.2.1"

    def test_latest_release_tag_returns_none_when_no_release_tag_exists(
        self, tmp_path: Path
    ) -> None:
        subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
        assert latest_release_tag(tmp_path) is None

    def test_latest_release_tag_falls_back_to_local_tags_when_the_fetch_hangs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An unreachable `origin` must not stall the pre-commit hook: the fetch
        # is time-boxed and the check proceeds on whatever tags the clone has.
        subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "tag.gpgSign", "false"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=tmp_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
        subprocess.run(["git", "commit", "--quiet", "--allow-empty", "-m", "x"], cwd=tmp_path)
        subprocess.run(["git", "tag", "v0.3.0"], cwd=tmp_path, check=True)

        real_run = subprocess.run

        def fake_run(command: list[str], **kwargs: object) -> object:
            if "fetch" in command:
                raise subprocess.TimeoutExpired(command, timeout=5)
            return real_run(command, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr("scripts.release.release_utils.subprocess.run", fake_run)
        assert latest_release_tag(tmp_path) == "0.3.0"
