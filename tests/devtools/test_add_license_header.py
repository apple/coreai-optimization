# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from _test_helpers import load_script

_EXTERNAL_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _EXTERNAL_ROOT / "scripts" / "pre_commit" / "add_license_header.py"
_TEMPLATE = _EXTERNAL_ROOT / "configs" / "BSD-3-LICENSE-HEADER-TEMPLATE"

add_license_header = load_script(_SCRIPT)

# The exact rendered block (with current-year 2026) used to assemble expected output.
_HEADER_2026 = (
    "# Copyright 2026 Apple Inc.\n"
    "#\n"
    "# Use of this source code is governed by a BSD-3-Clause license that can\n"
    "# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause\n"
)

# Pre-parsed Jinja template (matches how ``main`` hoists the parse) — share
# across helpers so each test doesn't pay the parse cost.
_TEMPLATE_OBJ = add_license_header.Template(_TEMPLATE.read_text(encoding="utf-8"))

# The template's expression-free lines, used to recognise our own header.
_INVARIANT = add_license_header._template_invariant_lines(_TEMPLATE.read_text(encoding="utf-8"))

# A leading block that looks like a license but is not ours (drives the conflict tests).
_FOREIGN_HEADER = (
    "# Copyright 2020 Example Corp.\n"
    "#\n"
    '# Licensed under the Apache License, Version 2.0 (the "License").\n'
    "\n"
    "import os\n"
)


def _block(year_string: str = "2026", style=None) -> list[str]:
    kwargs = {"year_string": year_string}
    if style is not None:
        kwargs["style"] = style
    return add_license_header.render_header_block(_TEMPLATE_OBJ, **kwargs)


def _build(original: str, block_lines: list[str] | None = None) -> str:
    return add_license_header.build_new_content(
        original, block_lines if block_lines is not None else _block(), _INVARIANT
    )


def _git(repo: Path, *args: str, year: int | None = None) -> None:
    """Run a git command in ``repo`` isolated from global config, optionally dated."""
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    if year is not None:
        stamp = f"{year}-06-15T12:00:00"
        env["GIT_AUTHOR_DATE"] = stamp
        env["GIT_COMMITTER_DATE"] = stamp
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
    )


# --- normalize -------------------------------------------------------------


def test_normalize_single_year() -> None:
    assert add_license_header.normalize({2026}) == "2026"


def test_normalize_two_consecutive_years_are_listed() -> None:
    assert add_license_header.normalize({2026, 2027}) == "2026, 2027"


def test_normalize_three_consecutive_years_collapse() -> None:
    assert add_license_header.normalize({2025, 2026, 2027}) == "2025-2027"


def test_normalize_mixed_runs() -> None:
    years = {2015, 2016, 2017, 2019, 2025, 2026, 2027}
    assert add_license_header.normalize(years) == "2015-2017, 2019, 2025-2027"


def test_normalize_gap_after_range() -> None:
    assert add_license_header.normalize({2016, 2017, 2018, 2020}) == "2016-2018, 2020"


def test_normalize_non_consecutive_years() -> None:
    assert add_license_header.normalize({2020, 2022, 2024}) == "2020, 2022, 2024"


# --- render_header_block ---------------------------------------------------


def test_render_produces_exact_marker_free_block() -> None:
    assert _block("2026") == [
        "# Copyright 2026 Apple Inc.",
        "#",
        "# Use of this source code is governed by a BSD-3-Clause license that can",
        "# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause",
    ]


def test_render_preserves_internal_blank_and_strips_edges() -> None:
    template = add_license_header.Template("\n\nCopyright {{ year }} Apple Inc.\n\nSecond line\n\n")
    rendered = add_license_header.render_header_block(template, year_string="2026")
    assert rendered == ["# Copyright 2026 Apple Inc.", "#", "# Second line"]


# --- build_new_content -----------------------------------------------------


def test_insert_into_plain_file() -> None:
    assert _build("import os\n") == _HEADER_2026 + "\n\nimport os\n"


def test_build_is_idempotent() -> None:
    once = _build("import os\n")
    twice = _build(once)
    assert once == twice


def test_replaces_legacy_marker_block() -> None:
    legacy = (
        "# LICENSE HEADER MANAGED BY add-license-header\n"
        "#\n"
        "# Copyright 2026 Apple Inc.\n"
        "#\n"
        "# Use of this source code is governed by a BSD-3-Clause license that can\n"
        "# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause\n"
        "#\n"
        "\n"
        '"""Module docstring."""\n'
    )
    result = _build(legacy)
    assert result == _HEADER_2026 + '\n"""Module docstring."""\n'
    assert "LICENSE HEADER MANAGED BY" not in result


def test_already_correct_header_is_noop() -> None:
    content = _HEADER_2026 + "\nimport os\n"
    assert _build(content) == content


def test_refreshes_our_header_when_year_differs() -> None:
    stale = (
        "# Copyright 2020 Apple Inc.\n"
        "#\n"
        "# Use of this source code is governed by a BSD-3-Clause license that can\n"
        "# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause\n"
        "\n"
        "import os\n"
    )
    assert _build(stale, _block("2026")) == _HEADER_2026 + "\nimport os\n"


def test_refresh_preserves_pep8_two_blanks_after_header() -> None:
    # Files with `class`/`def` directly after the header conventionally use two
    # blank lines (PEP 8 E302). A refresh must replace only the `#` lines and
    # leave the surrounding spacing alone, otherwise the hook fights ruff.
    content = _HEADER_2026 + "\n\nclass Foo:\n    pass\n"
    assert _build(content) == content


def test_refresh_preserves_zero_blanks_after_header() -> None:
    # If the user wrote no blank line between the header and the body, the
    # refresh leaves it that way — the script touches only the header lines.
    content = _HEADER_2026 + "import os\n"
    assert _build(content) == content


def test_foreign_license_header_raises_conflict() -> None:
    with pytest.raises(add_license_header.HeaderConflictError):
        _build(_FOREIGN_HEADER)


def test_drifted_casing_raises_conflict() -> None:
    drifted = (
        "# Copyright 2026 Apple Inc.\n"
        "#\n"
        "# Use of this source code is governed by a BSD-3-clause license that can\n"
        "# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause\n"
        "\n"
        "import os\n"
    )
    with pytest.raises(add_license_header.HeaderConflictError):
        _build(drifted)


def test_leading_license_prose_is_not_a_conflict() -> None:
    # A first-line comment that merely mentions "license" (no copyright year or
    # SPDX tag) is prose, not a header: insert above it rather than fail.
    result = _build("# Helpers for license-key parsing\nimport os\n")
    assert result == _HEADER_2026 + "\n\n# Helpers for license-key parsing\nimport os\n"


def test_supports_non_bsd_template_idempotently() -> None:
    template_text = "Copyright {{ year }} Example.\n\nMIT License: permission is hereby granted.\n"
    template = add_license_header.Template(template_text)
    block = add_license_header.render_header_block(template, year_string="2026")
    invariant = add_license_header._template_invariant_lines(template_text)
    once = add_license_header.build_new_content("import os\n", block, invariant)
    twice = add_license_header.build_new_content(once, block, invariant)
    assert once == twice
    assert once.startswith("# Copyright 2026 Example.\n")


# --- comment style dispatch (.js, .css, .html) -----------------------------


def _roundtrip(original: str, style) -> str:
    block = add_license_header.render_header_block(_TEMPLATE_OBJ, year_string="2026", style=style)
    return add_license_header.build_new_content(original, block, _INVARIANT, style)


def test_pick_style_dispatch_by_extension() -> None:
    assert add_license_header._pick_style(Path("a.py")).name == "hash"
    assert add_license_header._pick_style(Path("a.sh")).name == "hash"
    assert add_license_header._pick_style(Path("Makefile")).name == "hash"
    assert add_license_header._pick_style(Path("Dockerfile")).name == "hash"
    assert add_license_header._pick_style(Path("a.js")).name == "slash"
    assert add_license_header._pick_style(Path("a.css")).name == "block-c"
    assert add_license_header._pick_style(Path("a.html")).name == "block-html"
    assert add_license_header._pick_style(Path("a.HTML")).name == "block-html"  # case-insensitive


def test_javascript_header_uses_slash_comments() -> None:
    style = add_license_header._SLASH
    expected_header = (
        "// Copyright 2026 Apple Inc.\n"
        "//\n"
        "// Use of this source code is governed by a BSD-3-Clause license that can\n"
        "// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause\n"
    )
    once = _roundtrip("export const x = 1;\n", style)
    twice = _roundtrip(once, style)
    assert once == expected_header + "\n\nexport const x = 1;\n"
    assert once == twice  # idempotent


def test_css_header_uses_c_block_comments() -> None:
    style = add_license_header._BLOCK_C
    expected_header = (
        "/*\n"
        " * Copyright 2026 Apple Inc.\n"
        " *\n"
        " * Use of this source code is governed by a BSD-3-Clause license that can\n"
        " * be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause\n"
        " */\n"
    )
    once = _roundtrip("article h2 { padding: 0; }\n", style)
    twice = _roundtrip(once, style)
    assert once == expected_header + "\n\narticle h2 { padding: 0; }\n"
    assert once == twice  # idempotent


def test_css_refresh_recognises_existing_block_header() -> None:
    style = add_license_header._BLOCK_C
    stale = (
        "/*\n"
        " * Copyright 2020 Apple Inc.\n"
        " *\n"
        " * Use of this source code is governed by a BSD-3-Clause license that can\n"
        " * be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause\n"
        " */\n"
        "\n"
        "article h2 { padding: 0; }\n"
    )
    refreshed = _roundtrip(stale, style)
    assert "Copyright 2020" not in refreshed
    assert "Copyright 2026" in refreshed


def test_jinja_header_uses_block_comment() -> None:
    style = add_license_header._BLOCK_HTML
    expected_header = (
        "{#-\n"
        "  Copyright 2026 Apple Inc.\n"
        "\n"
        "  Use of this source code is governed by a BSD-3-Clause license that can\n"
        "  be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause\n"
        "-#}\n"
    )
    once = _roundtrip("{% extends 'base.html' %}\n", style)
    twice = _roundtrip(once, style)
    assert once == expected_header + "\n\n{% extends 'base.html' %}\n"
    assert once == twice  # idempotent


def test_jinja_refresh_recognises_existing_block_header() -> None:
    style = add_license_header._BLOCK_HTML
    stale = (
        "{#-\n"
        "  Copyright 2020 Apple Inc.\n"
        "\n"
        "  Use of this source code is governed by a BSD-3-Clause license that can\n"
        "  be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause\n"
        "-#}\n"
        "\n"
        "{% extends 'base.html' %}\n"
    )
    refreshed = _roundtrip(stale, style)
    assert "Copyright 2020" not in refreshed
    assert "Copyright 2026" in refreshed


def test_main_dispatches_style_by_extension(tmp_path: Path, monkeypatch) -> None:
    js_file = tmp_path / "x.js"
    js_file.write_text("export const x = 1;\n")
    monkeypatch.chdir(tmp_path)
    add_license_header.main(["--license-file", str(_TEMPLATE), str(js_file)])
    text = js_file.read_text(encoding="utf-8")
    assert text.startswith("// Copyright")
    assert "// Use of this source code" in text


# --- shebang and edge cases ------------------------------------------------


def test_shebang_gets_blank_line_then_header() -> None:
    result = _build("#!/usr/bin/env bash\nset -e\n")
    assert result == "#!/usr/bin/env bash\n\n" + _HEADER_2026 + "\n\nset -e\n"


def test_shebang_only_file() -> None:
    result = _build("#!/usr/bin/env bash\n")
    assert result == "#!/usr/bin/env bash\n\n" + _HEADER_2026


def test_empty_file_gets_header() -> None:
    assert _build("") == _HEADER_2026


def test_whitespace_only_file_gets_header() -> None:
    assert _build("\n\n   \n") == _HEADER_2026


def test_missing_trailing_newline_gets_single_newline() -> None:
    assert _build("import os") == _HEADER_2026 + "\n\nimport os\n"


def test_leading_plain_comment_is_preserved() -> None:
    result = _build("# TODO: refactor\nimport os\n")
    assert result == _HEADER_2026 + "\n\n# TODO: refactor\nimport os\n"


def test_invariant_phrase_in_code_body_stays_idempotent() -> None:
    content = "import os\n\n# Use of this source code is governed by a BSD-3-Clause license\n"
    once = _build(content)
    twice = _build(once)
    assert once == twice
    assert once.startswith(_HEADER_2026)


# --- _git_commit_years -----------------------------------------------------


def test_git_commit_years_two_non_consecutive(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    target = repo / "mod.py"
    target.write_text("x = 1\n")
    _git(repo, "add", "mod.py")
    _git(repo, "commit", "-m", "first", year=2020)
    target.write_text("x = 2\n")
    _git(repo, "add", "mod.py")
    _git(repo, "commit", "-m", "second", year=2024)

    monkeypatch.chdir(repo)
    assert add_license_header._git_commit_years(Path("mod.py")) == {2020, 2024}


def test_git_commit_years_three_consecutive_collapse(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    target = repo / "mod.py"
    for index, year in enumerate((2020, 2021, 2022)):
        target.write_text(f"x = {index}\n")
        _git(repo, "add", "mod.py")
        _git(repo, "commit", "-m", f"c{index}", year=year)

    monkeypatch.chdir(repo)
    years = add_license_header._git_commit_years(Path("mod.py"))
    assert add_license_header.normalize(years) == "2020-2022"


def test_git_commit_years_untracked_returns_empty(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert add_license_header._git_commit_years(Path("nope.py")) == set()


# --- main / executable contract --------------------------------------------


def test_main_exit_codes(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "sample.py"
    target.write_text("import os\n")
    monkeypatch.chdir(tmp_path)
    argv = ["--license-file", str(_TEMPLATE), str(target)]
    assert add_license_header.main(argv) == 1  # header inserted
    assert add_license_header.main(argv) == 0  # already correct


def test_main_fails_on_conflicting_header(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "sample.py"
    target.write_text(_FOREIGN_HEADER)
    monkeypatch.chdir(tmp_path)
    argv = ["--license-file", str(_TEMPLATE), str(target)]
    assert add_license_header.main(argv) == 1
    assert target.read_text(encoding="utf-8") == _FOREIGN_HEADER  # left untouched for manual fix


@pytest.mark.parametrize(
    ("extra_argv", "expected"),
    [
        (["--start-year", "2026"], "# Copyright 2026 Apple Inc."),
        ([], "# Copyright 2024, 2026 Apple Inc."),
    ],
)
def test_start_year_floor(
    tmp_path: Path, monkeypatch, extra_argv: list[str], expected: str
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    target = repo / "mod.py"
    target.write_text("x = 1\n")
    _git(repo, "add", "mod.py")
    _git(repo, "commit", "-m", "old", year=2024)

    monkeypatch.chdir(repo)
    add_license_header.main(["--license-file", str(_TEMPLATE), *extra_argv, "mod.py"])
    assert expected in Path("mod.py").read_text(encoding="utf-8")


def test_script_runs_as_executable_and_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("import os\n")
    cmd = [sys.executable, str(_SCRIPT), "--license-file", str(_TEMPLATE), str(target)]
    first = subprocess.run(cmd, capture_output=True, text=True)
    assert first.returncode == 1
    assert "Apple Inc." in target.read_text(encoding="utf-8")
    second = subprocess.run(cmd, capture_output=True, text=True)
    assert second.returncode == 0
