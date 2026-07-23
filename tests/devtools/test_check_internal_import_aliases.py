# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Golden (before, after) cases for the internal-import-alias pre-commit hook.

Each case maps a short label to a ``(before, after)`` pair: the module source before the hook
runs, and the source after ``--fix`` is applied. ``after == before`` means the hook makes no
change — either the import already complies, it is out of scope, or the violation is real but
cannot be auto-fixed (reported for a manual fix instead). The label doubles as the pytest id,
so the two dicts read as a quick catalogue of everything the hook covers.

The rule is direction-sensitive, so the cases are split by the *importing* module:
- public modules (no ``_`` path segment) must alias private-module imports with a ``_`` prefix;
- private modules (a ``_`` path segment) must not carry that redundant alias.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _test_helpers import load_script

_SCRIPT = (
    Path(__file__).parent.parent.parent
    / "scripts"
    / "pre_commit"
    / "check_internal_import_aliases.py"
)
_hook = load_script(_SCRIPT)


def _apply_fix(tmp_path: Path, before: str, *, private: bool) -> str:
    """Write ``before`` into a public/private ``coreai_opt`` module; return it after ``--fix``."""
    package_root = tmp_path / "coreai_opt"
    module = (package_root / "_internal" / "mod.py") if private else (package_root / "mod.py")
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text(before)
    _hook.check_file(module, package_root, fix=True)
    return module.read_text()


# label -> (before, after) for a PUBLIC module importing from a private module.
PUBLIC_MODULE_CASES: dict[str, tuple[str, str]] = {
    "absolute-missing-alias": (
        "from coreai_opt._utils import helper\n\nhelper()\n",
        "from coreai_opt._utils import helper as _helper\n\n_helper()\n",
    ),
    "absolute-already-aliased-noop": (
        "from coreai_opt._utils import helper as _helper\n\n_helper()\n",
        "from coreai_opt._utils import helper as _helper\n\n_helper()\n",
    ),
    # A custom alias is prefixed, not replaced (the author's chosen name is kept).
    "custom-alias-gets-prefixed": (
        "from coreai_opt._utils import helper as h\n\nh()\n",
        "from coreai_opt._utils import helper as _h\n\n_h()\n",
    ),
    "custom-alias-already-prefixed-noop": (
        "from coreai_opt._utils import helper as _h\n\n_h()\n",
        "from coreai_opt._utils import helper as _h\n\n_h()\n",
    ),
    "two-names-single-line": (
        "from coreai_opt._utils import helper, other\n\nhelper()\nother()\n",
        "from coreai_opt._utils import helper as _helper, other as _other\n\n_helper()\n_other()\n",
    ),
    # Adding the aliases pushes past the line limit, so the statement is paren-wrapped.
    "long-import-wraps": (
        "from coreai_opt._utils import clear_parametrization_original, prepare_mmap_dir\n\n"
        "clear_parametrization_original()\nprepare_mmap_dir()\n",
        "from coreai_opt._utils import (\n"
        "    clear_parametrization_original as _clear_parametrization_original,\n"
        "    prepare_mmap_dir as _prepare_mmap_dir,\n"
        ")\n\n"
        "_clear_parametrization_original()\n_prepare_mmap_dir()\n",
    ),
    # Relative import of a private module -> aliased, leading dots preserved.
    "relative-import": (
        "from ._utils import helper\n\nhelper()\n",
        "from ._utils import helper as _helper\n\n_helper()\n",
    ),
    "relative-deeper-private": (
        "from .sub._priv import helper\n\nhelper()\n",
        "from .sub._priv import helper as _helper\n\n_helper()\n",
    ),
    # Relative import of a public sibling -> unchanged (target is not private).
    "relative-public-sibling-noop": (
        "from .utils import helper\n\nhelper()\n",
        "from .utils import helper\n\nhelper()\n",
    ),
    # Plain `import ... as name` binding a private module to a public name -> prefixed.
    "plain-import-aliased": (
        "import coreai_opt._utils.helpers as helpers\n\nhelpers.run()\n",
        "import coreai_opt._utils.helpers as _helpers\n\n_helpers.run()\n",
    ),
    # Bare `import ...` binds the public top-level package `coreai_opt` -> unchanged.
    "plain-import-bare-noop": (
        "import coreai_opt._utils.helpers\n\ncoreai_opt._utils.helpers.run()\n",
        "import coreai_opt._utils.helpers\n\ncoreai_opt._utils.helpers.run()\n",
    ),
    "public-module-import-noop": (
        "from coreai_opt.utils import helper\n\nhelper()\n",
        "from coreai_opt.utils import helper\n\nhelper()\n",
    ),
    # A different top-level package (prefix look-alike) is not our package -> unchanged.
    "sibling-top-level-package-noop": (
        "from coreai_opt_benchmarking._bench import helper\n\nhelper()\n",
        "from coreai_opt_benchmarking._bench import helper\n\nhelper()\n",
    ),
    "imported-symbol-already-private-noop": (
        "from coreai_opt._utils import _internal_helper\n\n_internal_helper()\n",
        "from coreai_opt._utils import _internal_helper\n\n_internal_helper()\n",
    ),
    # Trailing comment sits outside the statement span, so it survives the rewrite.
    "single-line-trailing-comment": (
        "from coreai_opt._utils import helper  # noqa: F401\n\nhelper()\n",
        "from coreai_opt._utils import helper as _helper  # noqa: F401\n\n_helper()\n",
    ),
    # Comment inside a multi-line import -> aliased in place so the comment is preserved.
    "multiline-inner-comment": (
        "from coreai_opt._utils import (\n"
        "    helper,  # the sole public entry point\n"
        "    other,\n"
        ")\n\n"
        "helper()\nother()\n",
        "from coreai_opt._utils import (\n"
        "    helper as _helper,  # the sole public entry point\n"
        "    other as _other,\n"
        ")\n\n"
        "_helper()\n_other()\n",
    ),
    # Bound name shadowed by a function parameter -> reported but NOT auto-fixed.
    "shadowed-param-not-fixed": (
        "from coreai_opt._utils import helper\n\n\ndef run(helper):\n    return helper\n",
        "from coreai_opt._utils import helper\n\n\ndef run(helper):\n    return helper\n",
    ),
    # Bound name rebound at module scope -> unsafe to rename, so reported but NOT auto-fixed.
    "module-level-rebind-not-fixed": (
        "from coreai_opt._utils import helper\n\nhelper = helper or None\n",
        "from coreai_opt._utils import helper\n\nhelper = helper or None\n",
    ),
    # Multibyte character earlier on the line -> rename lands at the correct byte offset.
    "multibyte-offset": (
        "from coreai_opt._utils import helper\n\n\n"
        "def run():\n    café = 1\n    return café + helper()\n",
        "from coreai_opt._utils import helper as _helper\n\n\n"
        "def run():\n    café = 1\n    return café + _helper()\n",
    ),
}

# label -> (before, after) for a PRIVATE module importing from a private module.
PRIVATE_MODULE_CASES: dict[str, tuple[str, str]] = {
    # Redundant canonical `_` alias -> stripped, references renamed back.
    "strip-redundant-alias": (
        "from coreai_opt._utils import helper as _helper\n\n_helper()\n",
        "from coreai_opt._utils import helper\n\nhelper()\n",
    ),
    "no-alias-noop": (
        "from coreai_opt._utils import helper\n\nhelper()\n",
        "from coreai_opt._utils import helper\n\nhelper()\n",
    ),
    # A deliberate custom name is left alone in a private module.
    "custom-alias-noop": (
        "from coreai_opt._utils import helper as h\n\nh()\n",
        "from coreai_opt._utils import helper as h\n\nh()\n",
    ),
    "relative-strip-alias": (
        "from ._sibling import helper as _helper\n\n_helper()\n",
        "from ._sibling import helper\n\nhelper()\n",
    ),
    # Alias stripped in place inside a multi-line import so the comment is preserved.
    "multiline-inner-comment-strip": (
        "from coreai_opt._utils import (\n"
        "    helper as _helper,  # note\n"
        "    other as _other,\n"
        ")\n\n"
        "_helper()\n_other()\n",
        "from coreai_opt._utils import (\n"
        "    helper,  # note\n"
        "    other,\n"
        ")\n\n"
        "helper()\nother()\n",
    ),
}


@pytest.mark.parametrize(
    ("before", "after"),
    PUBLIC_MODULE_CASES.values(),
    ids=PUBLIC_MODULE_CASES.keys(),
)
def test_public_module_cases(tmp_path: Path, before: str, after: str) -> None:
    assert _apply_fix(tmp_path, before, private=False) == after


@pytest.mark.parametrize(
    ("before", "after"),
    PRIVATE_MODULE_CASES.values(),
    ids=PRIVATE_MODULE_CASES.keys(),
)
def test_private_module_cases(tmp_path: Path, before: str, after: str) -> None:
    assert _apply_fix(tmp_path, before, private=True) == after


def _write_module(tmp_path: Path, source: str) -> Path:
    """Write ``source`` into a public ``coreai_opt`` module under ``tmp_path``."""
    package_root = tmp_path / "coreai_opt"
    package_root.mkdir(parents=True, exist_ok=True)
    (package_root / "mod.py").write_text(source)
    return package_root


def test_detection_fails_and_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Detection (no --fix) exits non-zero on a violation so the commit is blocked."""
    root = _write_module(tmp_path, "from coreai_opt._utils import helper\n\nhelper()\n")
    exit_code = _hook.main(["--root", str(root)])
    assert exit_code == 1
    assert "Missing `_` alias" in capsys.readouterr().out


def test_detection_passes_when_clean(tmp_path: Path) -> None:
    """Detection exits 0 when there is nothing to flag."""
    root = _write_module(tmp_path, "value = 1\n")
    assert _hook.main(["--root", str(root)]) == 0


def test_fix_resolves_and_then_passes(tmp_path: Path) -> None:
    """--fix applies the fix (exit 0); a following detection run is then clean."""
    root = _write_module(tmp_path, "from coreai_opt._utils import helper\n\nhelper()\n")
    assert _hook.main(["--root", str(root)]) == 1  # blocks before fixing
    assert _hook.main(["--fix", "--root", str(root)]) == 0  # applies the fix
    assert _hook.main(["--root", str(root)]) == 0  # clean afterward
    assert (root / "mod.py").read_text() == (
        "from coreai_opt._utils import helper as _helper\n\n_helper()\n"
    )
