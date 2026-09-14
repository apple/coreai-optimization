# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Generate docs/src/api/index.md from one or more package trees.

Walks public packages and public modules that declare ``__all__``, reads the
``__all__`` and ``__doc__`` of each, and emits a MyST Markdown file with
autosummary directives grouped by package or module.

A module-level ``__all__`` is treated as the signal that the module is a
self-contained public API surface — the same pattern used by ``numpy.fft``,
``torch.nn``, ``sklearn.linear_model``, etc. Modules without ``__all__`` are
treated as implementation files; their symbols are documented only via the
package re-export (if any).

``generate_api_index`` and ``main`` take the roots to document, defaulting to
``coreai_opt`` alone. A distribution that ships companion packages alongside
``coreai_opt`` passes them all in, so a single walk resolves each symbol to one
owning root.

Called automatically by ``docs/src/conf.py`` during the Sphinx build. Can also
be run standalone::

    python docs/scripts/generate_api_index.py
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import Path

from coreai_opt._utils.api_visibility_utils import (
    accessible_public_names,
    collect_declared_obj_id_map,
    find_public_modules,
    find_public_packages,
)
from coreai_opt._utils.repo_utils import find_repo_root

_ROOT_PACKAGE = "coreai_opt"

_OUTPUT_PATH = Path("docs/src/api/index.md")


def _root_index(scope_name: str, root_packages: Sequence[str]) -> int:
    """Return the position in ``root_packages`` of the root a scope belongs to."""
    return next(
        i
        for i, root in enumerate(root_packages)
        if scope_name == root or scope_name.startswith(f"{root}.")
    )


def _build_ownership_map(scopes: list[str], root_packages: Sequence[str]) -> dict[int, str]:
    """Map each object ID to the scope that owns it: earliest root, then shallowest.

    A "scope" is either a package (``__init__.py``) or a module that defines
    its own ``__all__``. Sorting shallowest-first within a root means a package
    re-export wins over a module-local declaration when both list the same
    symbol, documenting each symbol at the canonical user-facing import path —
    the topmost place the user can reach it.

    Root order is the primary key because depth is not comparable across roots:
    a symbol that a later root re-exports at its top level would otherwise
    outrank the earlier root's nested — but canonical — path for it. Ordering by
    root first keeps a symbol documented under the root that owns it, whatever
    depth each root exposes it at.
    """
    ordered = sorted(scopes, key=lambda s: (_root_index(s, root_packages), s.count(".")))
    return collect_declared_obj_id_map(ordered)


def _collect_owned_symbols(
    scope_name: str,
    obj_id_map: dict[int, str],
) -> list[str]:
    """Return sorted symbol names owned by this scope (package or module).

    Uses the same ``accessible_public_names`` filter as ``test_api_visibility``
    so the doc generator and the visibility tests agree on what is public.
    """
    mod = importlib.import_module(scope_name)
    return sorted(
        name
        for name in accessible_public_names(mod)
        if obj_id_map.get(id(getattr(mod, name))) == scope_name
    )


def _heading_level(scope_name: str) -> str:
    """Return the markdown heading prefix for a scope based on its depth."""
    depth = scope_name.count(".")
    # depth 0 (root) and depth 1 get ##, depth 2+ get ###
    return "##" if depth <= 1 else "###"


def _module_summary(scope_name: str) -> str:
    """Return the first line of a module's docstring, or an empty string."""
    mod = importlib.import_module(scope_name)
    doc = (mod.__doc__ or "").strip()
    if not doc:
        return ""
    return doc.split("\n")[0].strip()


def _generate_autosummary_block(scope_name: str, symbols: list[str]) -> str:
    """Generate a MyST eval-rst block with an autosummary directive."""
    entries = "\n".join(f"   {scope_name}.{name}" for name in symbols)
    return f"""```{{eval-rst}}
.. autosummary::
   :toctree: generated

{entries}
```"""


def _has_children_with_symbols(
    scope_name: str,
    scopes: list[str],
    ownership: dict[str, list[str]],
) -> bool:
    """Check if any child scope of scope_name has owned symbols."""
    prefix = scope_name + "."
    return any(ownership.get(s) for s in scopes if s.startswith(prefix))


def _is_excluded(scope_name: str, exclude: Sequence[str]) -> bool:
    """Return True if a scope is, or lives under, any excluded dotted package path."""
    return any(scope_name == e or scope_name.startswith(f"{e}.") for e in exclude)


def _find_self_documenting_modules(root_package: str, exclude: Sequence[str] = ()) -> list[str]:
    """Return public modules that declare their own ``__all__``.

    Modules without ``__all__`` are treated as implementation files and
    documented only through their parent package's re-exports.

    Excluded modules are filtered before they are imported, so a subtree whose
    modules import optional dependencies can be skipped without installing them.
    """
    return [
        name
        for name in find_public_modules(root_package)
        if not _is_excluded(name, exclude) and hasattr(importlib.import_module(name), "__all__")
    ]


def _collect_scopes(root_package: str, exclude: Sequence[str] = ()) -> list[str]:
    """Return every documentable scope under a root: its public packages and modules."""
    packages = [p for p in find_public_packages(root_package) if not _is_excluded(p, exclude)]
    return packages + _find_self_documenting_modules(root_package, exclude)


def generate_api_index(
    root_packages: Sequence[str] = (_ROOT_PACKAGE,),
    exclude: Sequence[str] = (),
) -> str:
    """Generate the full api/index.md content from one or more package trees.

    Roots are documented in the order given, each as its own group of sections;
    within a group, scopes are sorted alphabetically.

    Each symbol appears exactly once, at the shallowest scope (package or
    self-documenting module) that exports it via ``__all__``. This matches
    the canonical user-facing import path — the topmost place the symbol
    is reachable. Ownership is resolved across all roots together, and an
    earlier root outranks a later one, so a symbol that a later root
    re-exports is still documented under the root that owns it.

    Args:
        root_packages (Sequence[str]): Dotted root package names to document,
            in precedence order — a symbol reachable from more than one root
            is documented under the earliest one listed. Defaults to
            ``coreai_opt`` alone.
        exclude (Sequence[str]): Dotted package paths to leave undocumented,
            along with everything beneath them. Excluded modules are never
            imported, so a subtree requiring optional dependencies can be
            skipped without installing them.

    Returns:
        str: Full MyST Markdown content for ``docs/src/api/index.md``.
    """
    # Sort within each root rather than globally, so roots stay grouped in the
    # order given instead of interleaving alphabetically.
    scopes = [s for root in root_packages for s in sorted(_collect_scopes(root, exclude))]
    obj_id_map = _build_ownership_map(scopes, root_packages)

    # Pre-compute owned symbols for every scope
    ownership = {scope: _collect_owned_symbols(scope, obj_id_map) for scope in scopes}

    lines: list[str] = ["# API Reference", ""]

    for scope_name in scopes:
        symbols = ownership[scope_name]
        has_children = _has_children_with_symbols(scope_name, scopes, ownership)

        # Skip scopes that have no owned symbols AND no children with symbols
        if not symbols and not has_children:
            continue

        heading = _heading_level(scope_name)
        summary = _module_summary(scope_name)

        lines.append(f"{heading} {scope_name}")
        lines.append("")
        if summary:
            lines.append(summary)
            lines.append("")

        if symbols:
            lines.append(_generate_autosummary_block(scope_name, symbols))
            lines.append("")

    return "\n".join(lines)


def main(
    root_packages: Sequence[str] = (_ROOT_PACKAGE,),
    exclude: Sequence[str] = (),
) -> None:
    """Write the generated API index to ``docs/src/api/index.md``, skipping no-op writes.

    Args:
        root_packages (Sequence[str]): Dotted root package names to document.
        exclude (Sequence[str]): Dotted package paths to leave undocumented.
    """
    # Anchor on the cwd so `make` writes the invoking repo's index.
    output = find_repo_root() / _OUTPUT_PATH

    content = generate_api_index(root_packages, exclude)
    if output.exists() and output.read_text() == content:
        print(f"Up to date: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content)
        print(f"Generated {output}")


if __name__ == "__main__":
    main()
