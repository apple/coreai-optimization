# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for Sphinx API reference coverage and rendering.

Complements ``tests/test_api_visibility.py`` (which enforces ``__all__``
declarations) to ensure every public symbol is both declared AND documented.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

from coreai_opt._utils.api_visibility_utils import find_public_packages

_ROOT_PACKAGE = "coreai_opt"


def _load_module_from_path(name: str, path: Path):  # noqa: ANN202
    """Import a Python file as a module by filesystem path."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_api_doc_coverage(repo_root: Path) -> None:
    """Committed api/index.md must match the auto-generated version."""
    gen = _load_module_from_path(
        "generate_api_index", repo_root / "docs" / "scripts" / "generate_api_index.py"
    )
    api_index = repo_root / "docs" / "src" / "api" / "index.md"
    assert api_index.exists(), (
        f"API index not found at {api_index}.\nRun `make render-api-index` to generate it."
    )

    assert api_index.read_text() == gen.generate_api_index(), (
        "docs/src/api/index.md is out of sync with the package tree.\n"
        "Run `make render-api-index` to regenerate it, then stage the result."
    )


def test_autodoc_skip_filters_external_methods(repo_root: Path) -> None:
    """No inherited external methods (torch, pydantic, etc.) survive the autodoc filter.

    Calls the real ``_autodoc_skip_member`` from ``docs/src/conf.py`` against
    every member of every public API class. Any member that passes the filter
    must originate from coreai_opt.
    """
    conf = _load_module_from_path("docs_conf", repo_root / "docs" / "src" / "conf.py")
    skip_fn = conf._autodoc_skip_member

    leaks: list[str] = []
    for pkg_name in find_public_packages(_ROOT_PACKAGE):
        mod = importlib.import_module(pkg_name)
        for sym in getattr(mod, "__all__", []):
            cls = getattr(mod, sym, None)
            if not isinstance(cls, type):
                continue
            for name in dir(cls):
                member = getattr(cls, name, None)
                if member is None:
                    continue
                if skip_fn(app=None, what="class", name=name, obj=member, skip=False, options=None):
                    continue
                origin = getattr(member, "__module__", None) or ""
                if origin and not origin.startswith(_ROOT_PACKAGE):
                    leaks.append(f"{pkg_name}.{sym}.{name} (from {origin})")

    assert not leaks, (
        "External methods leak through _autodoc_skip_member in docs/src/conf.py.\n"
        "These would appear in the Sphinx API reference:\n"
        + "\n".join(f"  - {s}" for s in sorted(set(leaks)))
    )
