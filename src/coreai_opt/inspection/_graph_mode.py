# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Graph module op discovery implementation.

Walks an exported ``torch.fx.GraphModule`` graph and extracts operation
metadata from FX node attributes and metadata dictionaries.
"""

from __future__ import annotations

import re
from collections import defaultdict

import torch
from torch.fx import Node

from coreai_opt._utils.fx_utils import (
    get_module_boundary_nodes,
    get_node_type,
    normalize_module_fqn,
)
from coreai_opt.base_model_compressor import _BaseModelCompressor
from coreai_opt.quantization import Quantizer
from coreai_opt.quantization._graph.quantizer import GraphQuantizer
from coreai_opt.quantization.config.quantization_config import ExecutionMode

from ._common import (
    FORWARD_FUNCTION_NAME,
    build_module_tree,
    filter_module_tree,
)
from .types import (
    BoundaryEdge,
    InputEdge,
    ModelSummary,
    ModuleContext,
    ModuleInfo,
    OpInfo,
    SourceFrame,
)


def _extract_module_stack(node: Node) -> tuple[ModuleContext, ...]:
    """Build the module nesting hierarchy from ``nn_module_stack`` metadata."""
    stack = node.meta.get("nn_module_stack", {})
    return tuple(
        ModuleContext(module_name=normalize_module_fqn(module_fqn), module_type=module_type)
        for module_fqn, module_type in stack.values()
    )


def _parse_stack_trace(stack_trace: str | None) -> tuple[SourceFrame, ...]:
    """Parse the ``stack_trace`` metadata string into filtered source frames.

    The ``stack_trace`` stored in ``node.meta["stack_trace"]`` is a multi-line
    string formatted like a Python traceback::

        File "path/to/file.py", line 42, in forward
          x = self.conv(x)

    Only frames from ``forward()`` methods are kept, filtering out framework
    dispatch machinery, C++ internals, and other non-informative frames.
    """
    if not stack_trace:
        return ()

    frames: list[SourceFrame] = []
    lines = stack_trace.strip().splitlines()
    # Lines come in pairs: the first is a location header of the form
    #   File "path/to/file.py", line 42, in forward
    # and the second is the source line at that location:
    #   x = self.conv(x)
    # We parse each header, peek ahead for the source line, and keep
    # only frames originating from ``forward()`` methods.
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Match: File "...", line N, in func_name
        match = re.match(r'^File "(.+)", line (\d+), in (.+)$', line)
        if match:
            filename = match.group(1)
            lineno = int(match.group(2))
            function_name = match.group(3)
            # Next line (if present) is the source code
            code_context = ""
            if i + 1 < len(lines) and not lines[i + 1].strip().startswith("File "):
                code_context = lines[i + 1].strip()
                i += 1
            if function_name == FORWARD_FUNCTION_NAME:
                frames.append(
                    SourceFrame(
                        filename=filename,
                        lineno=lineno,
                        function_name=function_name,
                        code_context=code_context,
                    )
                )
        i += 1
    return tuple(frames)


def _populate_boundary_ops_graph(
    root: ModuleInfo,
    model: torch.fx.GraphModule,
    node_name_to_op_info: dict[str, OpInfo],
) -> None:
    """Populate input_ops and output_ops for all modules in topological order.

    Reuses the method for which graph mode Quantizer uses to determine module
    boundary inputs and outputs.
    A single pass over model.graph.nodes buckets each node under every module
    in its nn_module_stack. Per module, a node is an input_op if any of its
    non-get_attr inputs falls outside the module's subtree, and an output_op if
    any of its users falls outside the subtree.
    """
    module_to_nodes: defaultdict[str, list[Node]] = defaultdict(list)
    for node in model.graph.nodes:
        if node.op == "get_attr":
            continue
        for ctx in _extract_module_stack(node):
            module_to_nodes[ctx.module_name].append(node)

    def _recurse(module: ModuleInfo) -> None:
        for child in module.child_modules.values():
            _recurse(child)

        subtree_nodes = module_to_nodes.get(module.module_name, [])
        input_consumer_tuples, output_nodes = get_module_boundary_nodes(subtree_nodes)
        # input_ops: keyed by spec index (enumerate position in input_consumer_tuples,
        # which already excludes state nodes via is_coreai_compressed_state_node).
        module.input_ops = {
            idx: [
                BoundaryEdge(
                    op=node_name_to_op_info[consumer.name],
                    index=consumer.all_input_nodes.index(external),
                )
            ]
            for idx, (external, consumer) in enumerate(input_consumer_tuples)
        }
        # output_ops: keyed by spec index (enumerate position in output_nodes).
        module.output_ops = {
            idx: BoundaryEdge(
                op=node_name_to_op_info[node.name],
                # outputs is always {0: consumers} in graph mode (see phase 2), so this yields 0.
                index=next(iter(node_name_to_op_info[node.name].outputs)),
            )
            for idx, node in enumerate(output_nodes)
        }

    _recurse(root)


def parse_ops_for_graph(
    model: torch.fx.GraphModule,
    compressor: type[_BaseModelCompressor] | None = None,
) -> ModelSummary:
    """Discover all operations in a graph exported model.

    Args:
        model (torch.fx.GraphModule): An exported ``torch.fx.GraphModule``
            (from ``torch.export``).
        compressor (type[_BaseModelCompressor] | None): A compressor class to
            filter ops to only those supported by that compression algorithm.
            When ``None``, all ops are included.

    Returns:
        ModelSummary: Operations nested in a :class:`ModuleInfo` tree
        mirroring the ``nn.Module`` hierarchy.

    Raises:
        ValueError: If *compressor* is not supported in graph mode.
    """
    # Phase 1: Build OpInfo stubs (empty inputs/outputs) for every node.
    node_name_to_op_info_dict: dict[str, OpInfo] = {}
    node_op_list: list[tuple[torch.fx.Node, OpInfo]] = []
    all_ops: list[OpInfo] = []
    seen_op_names: set[str] = set()
    root_module_type = ""

    for node in model.graph.nodes:
        op_type = get_node_type(node, warn_on_failure=False)
        module_stack = _extract_module_stack(node)
        source_frames = _parse_stack_trace(node.meta.get("stack_trace"))

        # One time processing to fill in root_module_type
        if not root_module_type:
            for ctx in module_stack:
                if ctx.module_name == "":
                    root_module_type = ctx.module_type
                    break

        is_state = node.op == "get_attr"
        op_info = OpInfo(
            op_name=(node.target if is_state else node.name),
            op_type=op_type,
            module_stack=module_stack,
            source_frames=source_frames,
            inputs=(),
            outputs={},
            is_state=is_state,
        )
        node_name_to_op_info_dict[node.name] = op_info
        node_op_list.append((node, op_info))
        assert op_info.op_name not in seen_op_names, f"duplicate op_name {op_info.op_name}"
        seen_op_names.add(op_info.op_name)
        all_ops.append(op_info)

    # Phase 2: Fill in op inputs/outputs.
    for node, op_info in node_op_list:
        # Graph mode: all outputs are at slot 0, so output_idx is always 0.
        op_info.inputs = tuple(
            InputEdge(op=node_name_to_op_info_dict[inp.name], output_idx=0)
            for inp in node.all_input_nodes
            if inp.name in node_name_to_op_info_dict
        )
        # For graph mode, graph annotation has no concept of multiple outputs. Graph quantizer
        # lumps them all as a single output index. Hardcode all outputs to index 0.
        op_info.outputs = {
            0: tuple(
                node_name_to_op_info_dict[user.name]
                for user in node.users
                if user.name in node_name_to_op_info_dict
            )
        }

    # Phase 3: Build the module tree.
    root = build_module_tree(root_module_type, all_ops)
    _populate_boundary_ops_graph(root, model, node_name_to_op_info_dict)

    if compressor is not None:
        if issubclass(compressor, Quantizer):
            compressible_names = GraphQuantizer.get_compressible_op_names(model)
        else:
            msg = f"No graph mode op filtering for compressor {compressor.__name__}."
            raise ValueError(msg)
        root = filter_module_tree(root, compressible_names)

    return ModelSummary(model=root, mode=ExecutionMode.GRAPH)
