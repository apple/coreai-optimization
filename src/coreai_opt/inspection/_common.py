# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause


"""Shared utilities for building module trees from discovered ops."""

from .types import (
    ModuleInfo,
    OpInfo,
)

FORWARD_FUNCTION_NAME = "forward"


def _get_or_create_child(parent: ModuleInfo, module_name: str, module_type: str) -> ModuleInfo:
    """Get an existing child module or create a new one."""
    child = parent.child_modules.get(module_name)
    if child is None:
        child = ModuleInfo(
            module_name=module_name,
            module_type=module_type,
            child_modules={},
            ops=[],
            input_ops={},
            output_ops={},
        )
        parent.child_modules[module_name] = child
    return child


def build_module_tree(
    root_module_type: str,
    all_ops: list[OpInfo],
) -> ModuleInfo:
    """Build a ModuleInfo tree from ops with populated module stacks.

    Each op is attached to the deepest module named in its ``module_stack``.
    Ops whose ``module_stack`` consists entirely of root entries (or whose
    stack is empty after skipping root entries) are attached to the root.

    Args:
        root_module_type (str): Fully-qualified type name of the root module.
        all_ops (list[_OpInfo]): All discovered ops. Ops with an empty
            ``module_stack`` are excluded from the tree.

    Returns:
        _ModuleInfo: The root of the module tree.
    """
    root = ModuleInfo(
        module_name="",
        module_type=root_module_type,
        child_modules={},
        ops=[],
        input_ops={},
        output_ops={},
    )

    for op in all_ops:
        if op.is_state:
            continue
        if not op.module_stack:
            continue
        current = root
        for ctx in op.module_stack:
            if ctx.module_name == "":
                continue
            current = _get_or_create_child(current, ctx.module_name, ctx.module_type)
        current.ops.append(op)

    return root


def filter_module_tree(module: ModuleInfo, keep_op_names: set[str]) -> ModuleInfo:
    """Recursively filter a ModuleInfo tree, keeping only matching ops.

    Boundary info (input_ops/output_ops) is preserved from the unfiltered tree.

    Args:
        module (_ModuleInfo): The module tree to filter.
        keep_op_names (set[str]): Op names to retain in the tree.

    Returns:
        _ModuleInfo: A filtered copy with only matching ops.
    """
    filtered_children = {
        fqn: filter_module_tree(child, keep_op_names) for fqn, child in module.child_modules.items()
    }
    filtered_ops = [op for op in module.ops if op.op_name in keep_op_names]

    return ModuleInfo(
        module_name=module.module_name,
        module_type=module.module_type,
        child_modules=filtered_children,
        ops=filtered_ops,
        input_ops=module.input_ops,
        output_ops=module.output_ops,
    )
