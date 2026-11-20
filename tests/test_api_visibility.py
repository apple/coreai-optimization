# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Test that public API visibility is enforced via __all__ declarations.

For each public package (no path segment starting with '_'), verifies:
  - __all__ is defined
  - Every non-submodule, non-dunder name in __all__ exists on the module
  - Every accessible public non-submodule name is listed in __all__
  - Dunder names (e.g. __version__) listed in __all__ exist on the module

For each public module (non-package .py file on a public import path), verifies:
  - Every public symbol defined in the module is re-exported via __all__ in some package
    __init__.py

Submodule names are excluded from both sides of the package check -- they appear in dir() as
side-effects of Python's import machinery and are not enforced.

The full public API surface can be printed via `make api-list`.
"""

import importlib
import types

import pytest

from coreai_opt._utils.api_visibility_utils import (
    accessible_public_names,
    collect_declared_obj_ids,
    find_public_modules,
    find_public_packages,
    originating_public_names,
)
from tests.test_utils.general import COREAI_AVAILABLE

_PUBLIC_PACKAGES = find_public_packages("coreai_opt")
_PUBLIC_MODULES = find_public_modules("coreai_opt")
if not COREAI_AVAILABLE:
    # coreai_utils imports coreai when it loads, so walk_packages can't list its modules either.
    _PUBLIC_PACKAGES.remove("coreai_opt.coreai_utils")


def _has_private_origin(obj: object) -> bool:
    """Return True if obj's defining module path contains a '_'-prefixed segment."""
    module_name = getattr(obj, "__module__", None)
    if not isinstance(module_name, str):
        return False
    parts = module_name.split(".")
    return any(part.startswith("_") for part in parts)


# --------------------------------------------------------------------------------------
# Package-level __all__ tests
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("pkg_name", _PUBLIC_PACKAGES)
def test_all_is_defined(pkg_name: str) -> None:
    """Every public package must define __all__."""
    mod = importlib.import_module(pkg_name)
    assert hasattr(mod, "__all__"), (
        f"{pkg_name} does not define __all__. "
        "All public packages must explicitly declare their public API via __all__."
    )


@pytest.mark.parametrize("pkg_name", _PUBLIC_PACKAGES)
def test_all_contains_no_submodules(pkg_name: str) -> None:
    """__all__ must not list subpackage or submodule names."""
    mod = importlib.import_module(pkg_name)
    all_names = set(getattr(mod, "__all__", []))
    submodules_in_all = sorted(
        name for name in all_names if isinstance(getattr(mod, name, None), types.ModuleType)
    )
    assert not submodules_in_all, (
        f"{pkg_name}.__all__ lists submodule names: {submodules_in_all}. "
        "Subpackages may be imported for attribute access convenience but must not "
        "be listed in __all__."
    )


@pytest.mark.parametrize("pkg_name", _PUBLIC_PACKAGES)
def test_all_matches_public_namespace(pkg_name: str) -> None:
    """Every name imported into the package must be in __all__, and every name in __all__ must exist
    on the package.

    Does not check whether symbols defined in child submodules are re-exported here;
    that is covered by test_module_public_symbols_are_reexported.
    """
    mod = importlib.import_module(pkg_name)
    all_names = set(getattr(mod, "__all__", []))

    dunder_names = {name for name in all_names if name.startswith("__") and name.endswith("__")}
    missing_dunders = sorted(name for name in dunder_names if not hasattr(mod, name))
    assert not missing_dunders, (
        f"{pkg_name}.__all__ lists dunder names that don't exist on the module: {missing_dunders}"
    )

    # Private names (single leading _) must never appear in __all__. Dunder names like
    # __version__ are excluded since they are legitimate public exports.
    private_in_all = sorted(
        name for name in all_names if name.startswith("_") and name not in dunder_names
    )
    assert not private_in_all, (
        f"{pkg_name}.__all__ contains private names (leading _): {private_in_all}"
    )

    # Symbols originating from a private module path (any segment starting with '_') must not
    # be re-exported via a public __all__, regardless of their own name.
    private_origin_in_all = sorted(
        name
        for name in all_names - dunder_names
        if not isinstance(getattr(mod, name, None), types.ModuleType)
        and _has_private_origin(getattr(mod, name, None))
    )
    assert not private_origin_in_all, (
        f"{pkg_name}.__all__ re-exports symbols from private module paths: "
        f"{private_origin_in_all}. "
        "Move the symbol to a public module path or prefix it with '_'."
    )

    accessible = accessible_public_names(mod)
    declared = {
        name
        for name in all_names - dunder_names
        if not isinstance(getattr(mod, name, None), types.ModuleType)
    }

    missing_from_all = sorted(accessible - declared)
    stale_in_all = sorted(declared - accessible)

    errors: list[str] = []
    if missing_from_all:
        errors.append(
            f"Names accessible as {pkg_name}.xxx but missing from __all__: {missing_from_all}",
        )
    if stale_in_all:
        errors.append(
            f"Names in {pkg_name}.__all__ that don't exist on the module: {stale_in_all}",
        )

    assert not errors, "\n".join(errors)


# --------------------------------------------------------------------------------------
# Module-level visibility tests
# --------------------------------------------------------------------------------------

# Build a set of object ids declared in any package or module __all__, so we can check whether
# a symbol is already authoritatively declared public.
_DECLARED_OBJ_IDS = collect_declared_obj_ids(_PUBLIC_PACKAGES + _PUBLIC_MODULES)


@pytest.mark.parametrize("mod_name", _PUBLIC_MODULES)
def test_module_public_symbols_are_reexported(mod_name: str) -> None:
    """Every public symbol defined in a module must be authoritatively declared.

    Symbols on a public import path (no '_' in any path segment) are importable by users. Each
    such symbol should be resolved by one of:
    - Listing it in the module's own __all__ (if the module itself forms a self-contained public
      API), or
    - Re-exporting it via __all__ in a package __init__.py, or
    - Prefixing it with '_' to make it private.
    """
    mod = importlib.import_module(mod_name)
    defined = originating_public_names(mod)

    not_declared = sorted(
        name for name in defined if id(getattr(mod, name)) not in _DECLARED_OBJ_IDS
    )

    assert not not_declared, (
        f"{mod_name} has public symbols not declared in any __all__.\n"
        "Resolve each by adding to the module's own __all__, re-exporting via a "
        "package __init__.py's __all__, or prefixing with '_'.\n"
        "See DEVELOPER.md for guidance on public API visibility.\n"
        "Undeclared symbols:\n" + "\n".join(f"  - {mod_name}.{name}" for name in not_declared)
    )
