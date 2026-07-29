#!/usr/bin/env python3

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause


"""Enforce `_`-prefix aliasing of project-internal imports per code style guide.

A module is "private" when any segment of its path starts with a single
underscore (e.g., ``coreai_opt/_utils/helpers.py``,
``coreai_opt/quantization/_eager/quantizer.py``). When a public module imports
a symbol from a private module, the bound name must be aliased with a ``_``
prefix so it cannot be re-imported through the public namespace::

    # public module
    from coreai_opt._utils import helper as _helper  # OK

When a private module imports from another private module, the alias is
unnecessary and adds visual noise::

    # private module (path contains `_`)
    from coreai_opt._utils import helper  # OK, no alias needed

This script flags both kinds of mismatch and, with ``--fix``, rewrites the
import statement and renames every reference to the bound name within the
file. See ``docs/contributing/code_style_guide.md`` section 3.3.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterator

# Keep in sync with the formatter line-length in configs/darker.toml and
# pyproject.toml ([tool.ruff]); this hook wraps its own rewrites because it
# runs after the formatter hooks in .pre-commit-config.yaml.
LINE_LIMIT = 100


def _is_private_part(part: str) -> bool:
    """Return True for single-underscore segments, False for dunder/public."""
    return part.startswith("_") and not part.startswith("__")


def _is_private_module(module: str) -> bool:
    """Return True if any dotted segment of ``module`` is private."""
    return any(_is_private_part(p) for p in module.split("."))


def _is_private_file(file_path: Path, package_root: Path) -> bool:
    """Return True if any path segment below ``package_root`` is private."""
    try:
        rel = file_path.resolve().relative_to(package_root.resolve())
    except ValueError:
        return False
    parts = list(rel.parts)
    parts[-1] = Path(parts[-1]).stem
    return any(_is_private_part(p) for p in parts)


def _line_starts(source: str) -> list[int]:
    """Cumulative UTF-8 byte offset at the start of each (1-indexed) line.

    ``ast`` reports ``col_offset``/``end_col_offset`` as byte offsets into the source,
    so offsets must be computed and applied in bytes to stay correct on lines that
    contain multibyte characters.
    """
    starts = [0]
    for line in source.splitlines(keepends=True):
        starts.append(starts[-1] + len(line.encode("utf-8")))
    return starts


class _Positioned(Protocol):
    """An AST node carrying source position attributes (any ``ast.expr`` or ``ast.stmt``)."""

    lineno: int
    col_offset: int
    end_lineno: int | None
    end_col_offset: int | None


def _node_span(node: _Positioned, line_starts: list[int]) -> tuple[int, int]:
    """Convert an AST node's (lineno, col_offset, end_lineno, end_col_offset) to byte offsets."""
    start = line_starts[node.lineno - 1] + node.col_offset
    end_line = node.end_lineno or node.lineno
    end_col = node.end_col_offset if node.end_col_offset is not None else node.col_offset
    end = line_starts[end_line - 1] + end_col
    return start, end


@dataclass(frozen=True)
class _Edit:
    """A byte-offset span replacement in the source text."""

    start: int
    end: int
    replacement: str


@dataclass
class Violation:
    """A single import-aliasing rule violation."""

    line: int
    message: str
    edits: list[_Edit]
    fixable: bool
    skip_reason: str = ""  # why the auto-fix was withheld (empty when fixable)


def _is_named_target(target: ast.expr, name: str) -> bool:
    """Return True if assignment target ``target`` binds ``name`` (handles unpacking)."""
    if isinstance(target, ast.Name):
        return target.id == name
    if isinstance(target, (ast.Tuple, ast.List)):
        return any(_is_named_target(t, name) for t in target.elts)
    if isinstance(target, ast.Starred):
        return _is_named_target(target.value, name)
    return False


def _node_binds_name(node: ast.AST, name: str) -> bool:
    """Return True if ``node`` binds ``name`` (assignment/loop/with/walrus/comprehension).

    The caller's traversal decides which node kinds reach here: the module-scope pass visits
    statements only (never walrus/comprehension); the function-scope pass visits every node.
    """
    if isinstance(node, ast.Assign):
        # `name = ...`, chained `a = name = ...`, or unpacking `a, name = ...`
        return any(_is_named_target(t, name) for t in node.targets)
    if isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr, ast.For, ast.AsyncFor)):
        # `name: int = ...`, `name += ...`, walrus `(name := ...)`, or `for name in ...`
        return _is_named_target(node.target, name)
    if isinstance(node, (ast.With, ast.AsyncWith)):
        # `with ... as name:` (optional_vars is None for a bare `with ...:`)
        return any(
            item.optional_vars is not None and _is_named_target(item.optional_vars, name)
            for item in node.items
        )
    if isinstance(node, ast.comprehension):
        # the `for name in ...` clause of a list/set/dict/generator comprehension
        return _is_named_target(node.target, name)
    return False


def _module_binding_stmt(node: ast.AST, name: str, import_line: int) -> bool:
    """Return True if statement ``node`` binds ``name`` directly in its enclosing scope."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name == name
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return node.lineno != import_line and any(
            (alias.asname or alias.name) == name for alias in node.names
        )
    if isinstance(node, ast.Try):
        return any(handler.name == name for handler in node.handlers)
    return _node_binds_name(node, name)


def _module_scope_blocks(node: ast.AST) -> list[list[ast.stmt]]:
    """Return ``node``'s sub-statement blocks that stay in the enclosing scope.

    Control-flow constructs are transparent; ``def``/``class`` bodies are never
    returned, so bindings inside nested scopes are not treated as module-level.
    """
    if isinstance(node, ast.If):
        return [node.body, node.orelse]
    if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
        return [node.body, node.orelse]
    if isinstance(node, (ast.With, ast.AsyncWith)):
        return [node.body]
    if isinstance(node, ast.Try):
        return [node.body, node.orelse, node.finalbody, *(h.body for h in node.handlers)]
    return []


def _has_module_level_binding(tree: ast.Module, name: str, import_line: int) -> bool:
    """Return True if ``name`` is bound at module scope outside the target import.

    Recurses through control-flow constructs (``if``/``for``/``try``/``with``)
    but never into ``def``/``class``/``lambda`` bodies — those introduce inner
    scopes whose bindings don't shadow the module-level import for references in
    unrelated scopes.
    """

    def _visit(stmts: list[ast.stmt]) -> bool:
        for node in stmts:
            if _module_binding_stmt(node, name, import_line):
                return True
            if any(_visit(block) for block in _module_scope_blocks(node)):
                return True
        return False

    return _visit(list(tree.body))


def _iter_body_nodes(
    func: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
) -> Iterator[ast.AST]:
    """Yield every descendant node in ``func``'s body (a ``Lambda`` body is a single expr)."""
    if isinstance(func, ast.Lambda):
        yield from ast.walk(func.body)
    else:
        for stmt in func.body:
            yield from ast.walk(stmt)


def _function_locally_binds(
    func: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda,
    name: str,
) -> bool:
    """Return True if ``func`` binds ``name`` as a parameter or local assignment."""
    args = func.args
    params = (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg)
    if any(param is not None and param.arg == name for param in params):
        return True
    return any(_node_binds_name(node, name) for node in _iter_body_nodes(func))


def _has_shadowing_function(tree: ast.Module, name: str) -> bool:
    """Return True if any function locally binds ``name`` *and* references it.

    A function that takes ``helper`` as a parameter, then uses ``helper`` in
    its body, refers to the parameter — not the module-level import. Renaming
    those references would be wrong, so we refuse the auto-fix.
    """
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if not _function_locally_binds(func, name):
            continue
        if any(isinstance(node, ast.Name) and node.id == name for node in _iter_body_nodes(func)):
            return True
    return False


def _is_safe_to_fix(tree: ast.Module, name: str, import_line: int) -> bool:
    """Return True if it's safe to rename references to ``name`` across the file."""
    return not _has_module_level_binding(tree, name, import_line) and not _has_shadowing_function(
        tree,
        name,
    )


def _name_reference_edits(
    tree: ast.Module,
    old_name: str,
    new_name: str,
    import_line: int,
    line_starts: list[int],
) -> list[_Edit]:
    """Return edits that rename every ``Name(id=old_name)`` to ``new_name``.

    Skips the import statement itself (positions inside it are not ``Name`` nodes
    anyway, but the guard makes intent explicit).
    """
    edits: list[_Edit] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name):
            continue
        if node.id != old_name:
            continue
        if node.lineno == import_line:
            continue
        start, end = _node_span(node, line_starts)
        edits.append(_Edit(start=start, end=end, replacement=new_name))
    return edits


def _alias_text(name: str, asname: str | None) -> str:
    """Format an alias as it would appear in source (``name`` or ``name as asname``)."""
    if asname is None or asname == name:
        return name
    return f"{name} as {asname}"


def _format_importfrom(
    module: str,
    aliases: list[tuple[str, str | None]],
    indent_col: int,
) -> str:
    """Format ``from module import ...`` choosing single-line or paren-wrapped form.

    The first line is assumed to start at column ``indent_col`` (the position of
    the ``from`` keyword in the original source). Continuation lines include
    their own leading whitespace so the rewrite preserves the original indent.
    """
    parts = [_alias_text(name, asname) for name, asname in aliases]
    single_line = f"from {module} import {', '.join(parts)}"
    if indent_col + len(single_line) <= LINE_LIMIT:
        return single_line
    body_indent = " " * (indent_col + 4)
    close_indent = " " * indent_col
    lines = [f"from {module} import ("]
    lines.extend(f"{body_indent}{part}," for part in parts)
    lines.append(f"{close_indent})")
    return "\n".join(lines)


def _format_import(aliases: list[tuple[str, str | None]]) -> str:
    """Format an ``import a, b as c`` statement from (name, asname) pairs.

    Unlike ``from`` imports, plain ``import`` statements cannot be parenthesized, so this
    always emits a single line (aliased private-module imports are short in practice).
    """
    return "import " + ", ".join(_alias_text(name, asname) for name, asname in aliases)


class _AliasChange(NamedTuple):
    """The rewrite for one imported name whose ``_`` aliasing must change."""

    rename: tuple[str, str]  # (old_bound_name, new_bound_name)
    message: str


def _resolve_alias(
    alias: ast.alias,
    module: str,
    *,
    file_is_private: bool,
) -> tuple[tuple[str, str | None], _AliasChange | None]:
    """Decide the target aliasing for one imported name.

    Returns ``(new_alias, change)`` where ``new_alias`` is the ``(name, asname)``
    to emit and ``change`` is ``None`` when the import already complies.

    A public module must ``_``-prefix the *bound* name (a custom ``as`` alias included),
    since any un-prefixed name re-exports the private symbol. A private module only strips
    the redundant canonical ``_name`` alias; a custom alias there is left as the author chose.
    """
    original = alias.name
    current_alias = alias.asname
    if original.startswith("_"):
        # The imported symbol is itself private-named; leave it untouched.
        return (original, current_alias), None

    bound_name = current_alias or original
    if file_is_private:
        # Only the canonical ``name as _name`` alias is redundant noise worth stripping;
        # renaming a custom alias would change a name the author deliberately picked.
        if current_alias != "_" + original:
            return (original, current_alias), None
        new_asname: str | None = None
        target_bound = original
        message = (
            f"Unnecessary `_` alias for `{original}` imported from "
            f"private module `{module}` into a private module."
        )
    else:
        # Public module: the bound name must be `_`-prefixed so it cannot be re-imported.
        if bound_name.startswith("_"):
            return (original, current_alias), None
        target_bound = "_" + bound_name
        new_asname = target_bound
        message = (
            f"Missing `_` alias for `{original}` imported from "
            f"private module `{module}` into a public module. "
            f"Use `{original} as {target_bound}`."
        )
    return (
        (original, new_asname),
        _AliasChange(rename=(bound_name, target_bound), message=message),
    )


def _resolve_plain_alias(
    alias: ast.alias,
    project_package: str,
    *,
    file_is_private: bool,
) -> _AliasChange | None:
    """Decide the ``_``-alias change for one ``import module as name`` clause.

    Only an aliased import of an in-project private module into a public module is flagged
    (``import pkg._priv.mod as name`` -> ``as _name``): a bare ``import pkg._priv.mod`` binds
    the public top-level package rather than the private module, so it is left alone.
    """
    module = alias.name
    if alias.asname is None or alias.asname.startswith("_") or file_is_private:
        return None
    if module != project_package and not module.startswith(project_package + "."):
        return None
    if not _is_private_module(module):
        return None
    target = "_" + alias.asname
    message = (
        f"Missing `_` alias for private module `{module}` bound as `{alias.asname}` "
        f"in a public module. Use `import {module} as {target}`."
    )
    return _AliasChange(rename=(alias.asname, target), message=message)


def _statement_edits(
    node: ast.ImportFrom,
    module: str,
    new_aliases: list[tuple[str, str | None]],
    renames: list[tuple[str, str]],
    tree: ast.Module,
    line_starts: list[int],
) -> list[_Edit]:
    """Build the import rewrite plus every bound-name rename edit for one statement."""
    import_start, import_end = _node_span(node, line_starts)
    replacement = _format_importfrom(module, new_aliases, node.col_offset)
    edits = [_Edit(start=import_start, end=import_end, replacement=replacement)]
    for old, new in renames:
        edits.extend(_name_reference_edits(tree, old, new, node.lineno, line_starts))
    return edits


def _span_has_comment(source_bytes: bytes, start: int, end: int) -> bool:
    """Return True if the byte span holds a ``#`` — a comment the rewrite would drop.

    Import statements contain no string literals, so any ``#`` within the replaced span
    is a comment (typically inside a paren-wrapped, multi-line import). A trailing comment
    on a single-line import falls *after* the span and is preserved regardless.
    """
    return b"#" in source_bytes[start:end]


def _alias_span_edits(
    node: ast.ImportFrom,
    new_aliases: list[tuple[str, str | None]],
    renames: list[tuple[str, str]],
    tree: ast.Module,
    line_starts: list[int],
) -> list[_Edit]:
    """Rewrite only the changed ``alias`` nodes in place, preserving surrounding comments.

    Used when the statement span holds a comment (always a paren-wrapped, multi-line
    import): replacing the whole statement would drop the comment, so each imported name
    is edited where it sits, leaving the commas, comments, and layout untouched.
    """
    edits: list[_Edit] = []
    for alias, (name, new_asname) in zip(node.names, new_aliases, strict=True):
        if (name, new_asname) == (alias.name, alias.asname):
            continue  # this import name already complies
        start, end = _node_span(alias, line_starts)
        edits.append(_Edit(start=start, end=end, replacement=_alias_text(name, new_asname)))
    for old, new in renames:
        edits.extend(_name_reference_edits(tree, old, new, node.lineno, line_starts))
    return edits


def _build_violation(
    node: ast.stmt,
    messages: list[str],
    edits: list[_Edit],
    *,
    all_safe: bool,
) -> Violation:
    """Assemble the Violation for one import statement (``edits`` already gated on fixability)."""
    return Violation(
        line=node.lineno,
        message="; ".join(messages),
        edits=edits,
        fixable=all_safe,
        skip_reason="" if all_safe else "bound name is shadowed",
    )


def _check_import_from(
    node: ast.ImportFrom,
    project_package: str,
    tree: ast.Module,
    line_starts: list[int],
    source_bytes: bytes,
    *,
    file_is_private: bool,
    fix: bool,
) -> Violation | None:
    """Flag a ``from module import ...`` (absolute or relative) that mis-aliases private imports."""
    # Preserve the written form: relative imports keep their leading dots. They are
    # always in-package, so they bypass the absolute ``project_package`` prefix check.
    module = "." * node.level + (node.module or "")
    is_absolute = node.level == 0
    if is_absolute and module != project_package and not module.startswith(project_package + "."):
        return None
    if not _is_private_module(module):
        return None

    new_aliases: list[tuple[str, str | None]] = []
    messages: list[str] = []
    renames: list[tuple[str, str]] = []  # (old_bound, new_bound) pairs
    all_safe = True
    for alias in node.names:
        new_alias, change = _resolve_alias(alias, module, file_is_private=file_is_private)
        new_aliases.append(new_alias)
        if change is None:
            continue
        renames.append(change.rename)
        messages.append(change.message)
        if not _is_safe_to_fix(tree, change.rename[0], node.lineno):
            all_safe = False
    if not renames:
        return None

    # Only build edits under --fix (report-only mode discards them). The whole-statement
    # rewrite can re-wrap a long line but drops any comment inside the span; when the
    # statement carries a comment, edit each name in place instead.
    edits: list[_Edit] = []
    if fix and all_safe:
        import_start, import_end = _node_span(node, line_starts)
        if _span_has_comment(source_bytes, import_start, import_end):
            edits = _alias_span_edits(node, new_aliases, renames, tree, line_starts)
        else:
            edits = _statement_edits(node, module, new_aliases, renames, tree, line_starts)
    return _build_violation(node, messages, edits, all_safe=all_safe)


def _check_plain_import(
    node: ast.Import,
    project_package: str,
    tree: ast.Module,
    line_starts: list[int],
    *,
    file_is_private: bool,
    fix: bool,
) -> Violation | None:
    """Flag ``import pkg._priv.mod as name`` that binds a private module to a public name."""
    new_aliases: list[tuple[str, str | None]] = []
    messages: list[str] = []
    renames: list[tuple[str, str]] = []  # (old_bound, new_bound) pairs
    all_safe = True
    for alias in node.names:
        change = _resolve_plain_alias(alias, project_package, file_is_private=file_is_private)
        if change is None:
            new_aliases.append((alias.name, alias.asname))
            continue
        new_aliases.append((alias.name, change.rename[1]))
        renames.append(change.rename)
        messages.append(change.message)
        if not _is_safe_to_fix(tree, change.rename[0], node.lineno):
            all_safe = False
    if not renames:
        return None

    # Only build edits under --fix. A plain ``import`` cannot be parenthesized, so it never
    # holds an inner comment; a trailing comment sits past the span and survives the rewrite.
    edits: list[_Edit] = []
    if fix and all_safe:
        import_start, import_end = _node_span(node, line_starts)
        edits.append(
            _Edit(start=import_start, end=import_end, replacement=_format_import(new_aliases)),
        )
        for old, new in renames:
            edits.extend(_name_reference_edits(tree, old, new, node.lineno, line_starts))
    return _build_violation(node, messages, edits, all_safe=all_safe)


def _check_module(
    tree: ast.Module,
    project_package: str,
    source: str,
    *,
    file_is_private: bool,
    fix: bool,
) -> list[Violation]:
    """Build one ``Violation`` per import statement that needs fixing.

    Covers ``from`` imports (absolute and relative) and aliased plain ``import``
    statements; grouping at the statement level lets us emit one clean replacement each.
    Rewrite edits are built only when ``fix`` is set — report-only runs skip that work.
    """
    line_starts = _line_starts(source)
    source_bytes = source.encode("utf-8")
    violations: list[Violation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            violation = _check_import_from(
                node,
                project_package,
                tree,
                line_starts,
                source_bytes,
                file_is_private=file_is_private,
                fix=fix,
            )
        elif isinstance(node, ast.Import):
            violation = _check_plain_import(
                node,
                project_package,
                tree,
                line_starts,
                file_is_private=file_is_private,
                fix=fix,
            )
        else:
            continue
        if violation is not None:
            violations.append(violation)
    return violations


def _apply_edits(source: str, edits: list[_Edit]) -> str:
    """Apply edits in reverse byte-offset order so earlier positions stay valid.

    Edits carry UTF-8 byte spans (from ``ast`` positions), so splice on the encoded
    bytes rather than the ``str`` to stay aligned with multibyte characters.
    """
    data = source.encode("utf-8")
    for edit in sorted(edits, key=lambda e: e.start, reverse=True):
        data = data[: edit.start] + edit.replacement.encode("utf-8") + data[edit.end :]
    return data.decode("utf-8")


def check_file(
    file_path: Path,
    package_root: Path,
    *,
    fix: bool,
) -> tuple[list[Violation], list[Violation]]:
    """Return (applied, remaining) violations for one file."""
    try:
        source = file_path.read_text()
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, OSError):
        return [], []

    file_is_private = _is_private_file(file_path, package_root)
    violations = _check_module(
        tree,
        package_root.name,
        source,
        file_is_private=file_is_private,
        fix=fix,
    )
    if not violations or not fix:
        return [], violations

    fixable = [v for v in violations if v.fixable]
    unfixable = [v for v in violations if not v.fixable]
    if not fixable:
        return [], unfixable

    all_edits = [e for v in fixable for e in v.edits]
    new_source = _apply_edits(source, all_edits)
    if new_source != source:
        file_path.write_text(new_source)
    return fixable, unfixable


def _iter_target_files(paths: list[Path], package_root: Path) -> list[Path]:
    """Yield Python files inside ``package_root``.

    If ``paths`` is empty, walk ``package_root``. Otherwise, only keep paths
    that resolve under it (so external files passed by pre-commit are ignored).
    """
    if not paths:
        return sorted(package_root.rglob("*.py"))
    inside: list[Path] = []
    for p in paths:
        if p.suffix != ".py":
            continue
        try:
            p.resolve().relative_to(package_root.resolve())
        except ValueError:
            continue
        inside.append(p)
    return sorted(set(inside))


def main(argv: list[str]) -> int:
    """Report internal-import aliasing issues; with ``--fix`` apply the suggested fixes.

    Detection (the default, run by the pre-commit hook) never edits files and fails the commit
    when it finds a violation, so the offender is forced to resolve it — by hand or by running
    ``--fix`` manually. Keeping the fix opt-in stops its rewrites from getting entangled with
    the formatting/linting hooks' changes.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply the suggested fixes in place (default: report violations and fail).",
    )
    parser.add_argument(
        "--root",
        default="src/coreai_opt",
        help="Package root used to decide file privacy and bound import paths.",
    )
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args(argv)

    package_root = Path(args.root)
    if not package_root.exists():
        sys.stderr.write(f"Package root not found: {package_root}\n")
        return 1

    reported = 0
    for file_path in _iter_target_files(args.paths, package_root):
        applied, remaining = check_file(file_path, package_root, fix=args.fix)
        for v in applied:
            sys.stdout.write(f"{file_path}:{v.line}: FIXED — {v.message}\n")
        for v in remaining:
            hint = "" if v.fixable else f" (needs a manual fix: {v.skip_reason})"
            sys.stdout.write(f"{file_path}:{v.line}: {v.message}{hint}\n")
            reported += 1

    if reported:
        if args.fix:
            sys.stdout.write(
                f"\n{reported} issue(s) could not be auto-fixed; resolve them by hand.\n"
            )
        else:
            sys.stdout.write(
                f"\n{reported} issue(s) found. Fix them before committing — run "
                f"`python {sys.argv[0]} --fix` to apply the fixable ones automatically.\n"
            )
    # Fail the commit whenever anything remains unresolved so it can't slip through.
    return 1 if reported else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
