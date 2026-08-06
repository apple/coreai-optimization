# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Print the public API surface of all public packages and modules.

For each public package and module (no path segment starting with '_'), prints every symbol
declared in __all__ along with its type information:
  - Classes: inheritance chain and own (non-inherited) public methods
  - Modules: labeled "(module)"
  - Other values: labeled with their type name

Also scans for public symbols that are defined but not listed in __all__, so developers can
decide whether to add them or prefix them with '_'.

Usage:
    python scripts/make/print_api_list.py
    make api-list

    # Inspect a single module (dotted name or file path):
    python scripts/make/print_api_list.py coreai_opt.quantization.spec.spec
    python scripts/make/print_api_list.py path/to/module.py
    make api-list MODULE=coreai_opt.quantization.spec.spec
"""

import importlib
import inspect
import sys
import types
from pathlib import Path
from typing import NamedTuple

from coreai_opt._utils.api_visibility_utils import (
    collect_declared_obj_id_map,
    find_names_missing_from_all,
    find_public_modules,
    find_public_packages,
    originating_public_names,
)

_ROOT_PACKAGE = "coreai_opt"

_MISSING = object()

_UNDECLARED_TITLE = "Undeclared Public Symbols (not in __all__)"


def _reexport_label(pkg_name: str) -> str:
    """Format the re-export annotation label for a given package name."""
    return f"[re-exported via {pkg_name}]"


def _type_desc(obj: type) -> str:
    """Return a type description string showing the direct parent class."""
    # MRO is [TheClass, ..., object]. When len == 2, the only parent is object, which is not
    # worth showing. Otherwise mro[1] is the direct parent.
    direct_parent = obj.__mro__[1] if len(obj.__mro__) > 2 else None
    return f"-> {direct_parent.__name__}" if direct_parent else ""


class _SymbolEntry(NamedTuple):
    """A single symbol from __all__ with its type description and own methods."""

    name: str
    type_desc: str  # e.g. "-> BaseClass -> ABC", "(module)", "(str)"
    own_methods: list[str]
    re_exported_via: str  # declaring package name, or "" if declared in own __all__


def _pydantic_hook_names(cls: type) -> set[str]:
    """Return names of Pydantic-registered validator/lifecycle hooks on a class.

    These are implementation details called by Pydantic's validation machinery and should not
    appear as user-facing public API methods.
    """
    names: set[str] = set()
    # model_post_init is a Pydantic lifecycle hook always called internally
    if "model_post_init" in cls.__dict__:
        names.add("model_post_init")
    # __pydantic_decorators__ stores field_validators and model_validators
    decorators = getattr(cls, "__pydantic_decorators__", None)
    if decorators is not None:
        names.update(decorators.field_validators)
        names.update(decorators.model_validators)
    return names


def _collect_symbol_entry(name: str, obj: object, re_exported_via: str = "") -> _SymbolEntry:
    """Build a _SymbolEntry for a single symbol from __all__."""
    if obj is _MISSING:
        return _SymbolEntry(name, "(WARNING: missing from module)", [], re_exported_via)

    if isinstance(obj, types.ModuleType):
        return _SymbolEntry(name, "(module)", [], re_exported_via)

    if isinstance(obj, type):
        type_desc = _type_desc(obj)
        pydantic_hooks = _pydantic_hook_names(obj)
        # Only methods defined in the class's own __dict__, not inherited ones.
        # inspect.getmembers resolves descriptors, so regular methods and static methods appear
        # as functions, while classmethods appear as bound methods.
        own_methods = sorted(
            member_name
            for member_name, member in inspect.getmembers(obj)
            if not member_name.startswith("_")
            and member_name in obj.__dict__
            and member_name not in pydantic_hooks
            and (inspect.isfunction(member) or inspect.ismethod(member))
        )
        return _SymbolEntry(name, type_desc, own_methods, re_exported_via)

    return _SymbolEntry(name, f"({type(obj).__name__})", [], re_exported_via)


def _make_section_header(text: str, bar_width: int) -> str:
    """Format a section title as a three-line banner of '=' characters."""
    bar = "=" * bar_width
    # Center the title between "=== " and " ===" markers.
    inner = f" {text} "
    middle = f"==={inner.center(bar_width - 6, '=')}==="
    return f"\n{bar}\n{middle}\n{bar}"


def _compute_column_widths(
    all_entries: list[tuple[str, list[_SymbolEntry]]],
    all_undeclared: dict[str, list[str]],
) -> tuple[int, int]:
    """Compute column widths for aligned output.

    Returns:
        A tuple of (name_col, reexport_col) where name_col is the width for the symbol/method
        name column and reexport_col is the width for the re-export annotation column (0 if no
        re-exports).

    """
    max_symbol_len = 0
    max_reexport_len = 0
    max_method_len = 0
    for _, entries in all_entries:
        for entry in entries:
            max_symbol_len = max(max_symbol_len, len(entry.name))
            if entry.re_exported_via:
                label_len = len(_reexport_label(entry.re_exported_via))
                max_reexport_len = max(max_reexport_len, label_len)
            for method in entry.own_methods:
                # +3 accounts for the "." prefix and "()" suffix in ".method()".
                max_method_len = max(max_method_len, len(method) + 3)

    max_undeclared_len = 0
    for names in all_undeclared.values():
        for name in names:
            max_undeclared_len = max(max_undeclared_len, len(name))

    # Symbol lines are indented 4 spaces; method lines are indented 6 (2 extra).
    name_col = max(max_symbol_len + 4, max_undeclared_len + 4, max_method_len + 8)
    reexport_col = max_reexport_len + 4 if max_reexport_len else 0
    return name_col, reexport_col


def _collect_module_entries(
    target_modules: list[str],
    declared_obj_id_map: dict[int, str],
) -> tuple[list[tuple[str, list[_SymbolEntry]]], dict[str, list[str]]]:
    """Collect symbol entries and undeclared names for each dotted module name.

    Args:
        target_modules: Module or package dotted names to inspect.
        declared_obj_id_map: Map from object ID to the package that declares it, used to identify
           re-exported symbols and suppress already-declared ones from the undeclared section.

    Returns:
        A tuple of (all_entries, all_undeclared) where all_entries pairs each dotted name with
        its list of symbol entries, and all_undeclared maps dotted names to their undeclared
        public symbol names.

    """
    declared_obj_ids = set(declared_obj_id_map.keys())
    all_entries: list[tuple[str, list[_SymbolEntry]]] = []
    all_undeclared: dict[str, list[str]] = {}

    for dotted_name in target_modules:
        mod = importlib.import_module(dotted_name)
        mod_file = getattr(mod, "__file__", None) or ""
        is_package = mod_file.endswith("__init__.py")
        declared_all = list(getattr(mod, "__all__", []))

        if declared_all:
            # Module has its own __all__ -- show those symbols directly.
            entries = [
                _collect_symbol_entry(name, getattr(mod, name, _MISSING))
                for name in sorted(declared_all)
            ]
        elif not is_package:
            # Non-package module with no __all__ -- show symbols that originate here and are
            # re-exported via a parent package's __all__.
            entries = [
                _collect_symbol_entry(
                    name,
                    getattr(mod, name, _MISSING),
                    re_exported_via=declared_obj_id_map.get(id(getattr(mod, name)), ""),
                )
                for name in sorted(originating_public_names(mod))
                if id(getattr(mod, name)) in declared_obj_ids
            ]
        else:
            entries = []

        all_entries.append((dotted_name, entries))
        truly_undeclared = [
            name
            for name in find_names_missing_from_all(mod)
            if id(getattr(mod, name)) not in declared_obj_ids
        ]
        if truly_undeclared:
            all_undeclared[dotted_name] = truly_undeclared

    return all_entries, all_undeclared


def _format_declared_lines(
    all_entries: list[tuple[str, list[_SymbolEntry]]],
    packages_for_declared_ids: list[str],
    name_col: int,
    reexport_col: int,
) -> list[str]:
    """Format lines for the declared API surface section."""
    lines: list[str] = []
    for dotted_name, entries in all_entries:
        if not entries and packages_for_declared_ids:
            continue
        lines.append(f"\n  {dotted_name}:")
        if entries:
            for entry in entries:
                name_pad = max(name_col - len(entry.name), 1)
                if entry.re_exported_via:
                    reexport_label = _reexport_label(entry.re_exported_via)
                    reexport_pad = max(reexport_col - len(reexport_label), 1)
                else:
                    reexport_label = ""
                    # No re-export label: pad by the full reexport_col so that the type_desc
                    # column aligns with entries that do have one.
                    reexport_pad = reexport_col
                lines.append(
                    f"    {entry.name}{' ' * name_pad}{reexport_label}"
                    f"{' ' * reexport_pad}{entry.type_desc}",
                )
                for method in entry.own_methods:
                    method_str = f".{method}()"
                    # Align (method) with the type_desc column, which starts at name_col +
                    # reexport_col from the 4-space symbol indent. Method lines are indented 6
                    # spaces (2 extra), so subtract 2.
                    pad = max(name_col + reexport_col - 2 - len(method_str), 1)
                    lines.append(f"      {method_str}{' ' * pad}(method)")
        else:
            lines.append("    (no __all__ defined)")
    return lines


def _format_undeclared_lines(
    all_undeclared: dict[str, list[str]],
    name_col: int,
) -> list[str]:
    """Format lines for the undeclared public symbols section."""
    lines: list[str] = []
    for dotted_name, truly_undeclared in all_undeclared.items():
        mod = importlib.import_module(dotted_name)
        lines.append(f"\n  {dotted_name}:")
        for name in truly_undeclared:
            obj = getattr(mod, name)
            type_label = _type_desc(obj) if isinstance(obj, type) else f"({type(obj).__name__})"
            lines.append(f"    {name:<{name_col}}{type_label}")
    return lines


def _print_api(
    target_modules: list[str],
    packages_for_declared_ids: list[str],
    title: str,
) -> None:
    """Print the public API surface and undeclared public symbols.

    Args:
        target_modules: Module or package dotted names to display.
        packages_for_declared_ids: Packages whose __all__ entries are used to suppress already-
            re-exported symbols from the undeclared section and to annotate re-exported symbols
            when inspecting a non-package module.
        title: Header label for the public API surface block.

    """
    declared_obj_id_map = collect_declared_obj_id_map(packages_for_declared_ids)
    all_entries, all_undeclared = _collect_module_entries(target_modules, declared_obj_id_map)

    name_col, reexport_col = _compute_column_widths(all_entries, all_undeclared)
    bar_width = max(len(title), len(_UNDECLARED_TITLE)) + 8

    lines = [_make_section_header(title, bar_width)]
    lines.extend(
        _format_declared_lines(all_entries, packages_for_declared_ids, name_col, reexport_col),
    )

    undeclared_lines = _format_undeclared_lines(all_undeclared, name_col)
    if undeclared_lines:
        lines.append(_make_section_header(_UNDECLARED_TITLE, bar_width))
        lines.extend(undeclared_lines)
    else:
        lines.append("\nNo undeclared public symbols found.")

    print("\n".join(lines))


def _path_to_dotted_name(path_str: str) -> str:
    """Convert a file path to a dotted module name.

    Returns the shortest dotted name by using the longest matching sys.path entry as the base.
    """
    path = Path(path_str).resolve()
    path = path.parent if path.name == "__init__.py" else path.with_suffix("")

    best: str | None = None
    for sys_path_entry in sys.path:
        base = Path(sys_path_entry).resolve()
        if not path.is_relative_to(base):
            continue
        candidate = ".".join(path.relative_to(base).parts)
        if best is None or len(candidate) < len(best):
            best = candidate
    if best is None:
        msg = f"Cannot resolve {path_str!r} to a dotted module name via sys.path"
        raise ValueError(msg)
    return best


if __name__ == "__main__":
    if len(sys.argv) > 1:
        module_arg = sys.argv[1]
        if module_arg.endswith(".py") or "/" in module_arg or "\\" in module_arg:
            _dotted_name = _path_to_dotted_name(module_arg)
        else:
            _dotted_name = module_arg
        _print_api(
            target_modules=[_dotted_name],
            packages_for_declared_ids=find_public_packages(_ROOT_PACKAGE),
            title=f"Public API Surface: {_dotted_name}",
        )
    else:
        _public_packages = find_public_packages(_ROOT_PACKAGE)
        _public_modules = find_public_modules(_ROOT_PACKAGE)
        _print_api(
            target_modules=sorted(set(_public_packages + _public_modules)),
            packages_for_declared_ids=_public_packages,
            title="Public API Surface",
        )
