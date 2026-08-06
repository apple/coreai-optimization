# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tree-based formatting for model operation summaries using rich."""

from __future__ import annotations

import os

from rich.console import Console
from rich.text import Text
from rich.tree import Tree

from .types import InputEdge, ModelSummary, ModuleInfo, OpInfo

_FRAMEWORK_PATH_MARKERS = ("torch/nn/modules/", "torch/nn/functional", "torch/_")

_LEGEND = (
    "Legend:\n"
    "  ■ module_name (module_type)  ◆ op_name [op_type]\n"
    "\n"
    "  op inputs:  {I: producer[N]}  —  I = op_input_spec index;"
    " N = output slot of the producing op\n"
    "  op states:     param_name    —  model parameter or buffer\n"
    "  op outputs: {N: [consumers]} —  N = output slot index;"
    " consumers = ops receiving that output\n"
    "  untracked_N                  —  input tensor whose producer was not intercepted"
    " (e.g. raw attribute or global tensor); still quantizable via op_input_spec\n"
    "  module inputs:  {I: [op[N], ...]}  —  I = module_input_spec index;"
    " op[N] = op and its input slot receiving data from outside; absent keys = non-quantizable\n"
    "  module outputs: {I: op[N]}         —  I = module_output_spec index;"
    " op[N] = op and its output slot leaving the module; absent keys = non-quantizable"
)


def _source_for_op(op: OpInfo) -> tuple[str, str]:
    """Return the source file path and code context as separate strings."""
    if not op.source_frames:
        return "", ""
    for f in reversed(op.source_frames):
        if not any(marker in f.filename for marker in _FRAMEWORK_PATH_MARKERS):
            frame = f
            break
    else:
        return "", ""
    try:
        rel_path = os.path.relpath(frame.filename)
    except ValueError:
        rel_path = frame.filename
    return f"{rel_path}:{frame.lineno}", frame.code_context


def _producer_output_label(inp: InputEdge) -> str:
    """Return the display label for one input edge.

    When ``output_idx`` is ``None`` (registered states, ephemeral/untracked tensors)
    only the name is shown. Otherwise the output slot index is appended: ``name[N]``.
    """
    if inp.output_idx is None:
        return inp.op_name
    return f"{inp.op_name}[{inp.output_idx}]"


def _styled_op_label(op: OpInfo) -> Text:
    """Build the styled multi-line label for an op leaf node."""
    label = Text()

    # Line 1: ◆ op_name [op_type]
    label.append("◆ ")
    label.append(op.op_name, style="bold cyan")
    op_type_str = op.op_type if op.op_type else "?"
    label.append(" [")
    label.append(op_type_str, style="yellow")
    label.append("]")

    # Line 2: op inputs as {arg_idx: producer_label}, excluding states.
    # arg_idx is the full positional index (matching op_input_spec), not the filtered position.
    non_state_input_items = [(i, inp) for i, inp in enumerate(op.inputs) if not inp.is_state]
    if non_state_input_items:
        parts = ", ".join(f"{i}: {_producer_output_label(inp)}" for i, inp in non_state_input_items)
        label.append(f"\n  op inputs:  {{{parts}}}")

    # Line 3: op states
    op_state_names = [inp._display_name for inp in op.inputs if inp.is_state]
    if op_state_names:
        state_names = ", ".join(name for name in op_state_names)
        label.append(f"\n  op states:  {state_names}")

    # Line 4: op outputs
    if op.outputs:
        parts = ", ".join(
            f"{idx}: [{', '.join(out.op_name for out in consumers)}]"
            for idx, consumers in sorted(op.outputs.items())
        )
        label.append(f"\n  op outputs: {{{parts}}}")

    # Lines 5-6: source
    source_path, source_code = _source_for_op(op)
    if source_path:
        label.append(f"\n  filepath:  {source_path}", style="dim")
        if source_code:
            label.append(f"\n  code:      {source_code}", style="dim")

    return label


def _format_input_ops(input_ops: dict) -> str:
    """Format module input_ops dict as '{I: [op[N], ...], ...}'."""
    parts = []
    for k, edges in sorted(input_ops.items()):
        edge_strs = [f"{e.op.op_name}[{e.index}]" for e in edges]
        parts.append(f"{k}: [{', '.join(edge_strs)}]")
    return "{" + ", ".join(parts) + "}"


def _format_output_ops(output_ops: dict) -> str:
    """Format module output_ops dict as '{I: op[N], ...}'."""
    parts = [f"{k}: {e.op.op_name}[{e.index}]" for k, e in sorted(output_ops.items())]
    return "{" + ", ".join(parts) + "}"


def _styled_module_label(module: ModuleInfo) -> Text:
    """Build the styled label for a module node."""
    label = Text()
    label.append("■ ")
    label.append(module.module_name, style="green")
    label.append(" (")
    label.append(module.module_type, style="magenta")
    label.append(")")
    if module.input_ops:
        label.append(f"\n    module inputs:  {_format_input_ops(module.input_ops)}", style="dim")
    if module.output_ops:
        label.append(f"\n    module outputs: {_format_output_ops(module.output_ops)}", style="dim")
    return label


def _render_tree(module: ModuleInfo, tree: Tree) -> None:
    """Recursively add module children and ops to a rich Tree."""
    for child in module.child_modules.values():
        child_label = _styled_module_label(child)
        child_branch = tree.add(child_label)
        _render_tree(child, child_branch)

    for op in module.ops:
        label = _styled_op_label(op)
        tree.add(label)


def format_model_summary(summary: ModelSummary, colorize: bool | None = None) -> str:
    """Format a :class:`ModelSummary` as a module-hierarchy tree string.

    Args:
        summary (ModelSummary): The operation summary to format.
        colorize (bool | None): Whether to include ANSI color codes in the
            output. ``None`` (default) auto-detects based on terminal
            capabilities and environment variables (``NO_COLOR``,
            ``FORCE_COLOR``). ``True`` forces color on, ``False`` forces
            color off.

    Returns:
        str: The formatted tree as a string.
    """
    if not summary.model.ops and not summary.model.child_modules:
        return "(no compressible operations found)"

    # Root label
    root_label = Text("(")
    root_label.append(summary.model.module_type, style="magenta")
    root_label.append(")")
    if summary.model.input_ops:
        root_label.append(
            f"\n    module inputs:  {_format_input_ops(summary.model.input_ops)}", style="dim"
        )
    if summary.model.output_ops:
        root_label.append(
            f"\n    module outputs: {_format_output_ops(summary.model.output_ops)}", style="dim"
        )

    tree = Tree(root_label)
    _render_tree(summary.model, tree)

    console_kwargs: dict[str, bool] = {"highlight": False}
    if colorize is True:
        console_kwargs["force_terminal"] = True
    elif colorize is False:
        console_kwargs["no_color"] = True

    console = Console(**console_kwargs)
    with console.capture() as capture:
        console.print(tree)

    output = capture.get().rstrip("\n")
    return f"{_LEGEND}\n\n{output}"
