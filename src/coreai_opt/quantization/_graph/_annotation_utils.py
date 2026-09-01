# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import Node
from torch.fx.passes.utils.matcher_utils import InternalMatch
from torch.fx.passes.utils.matcher_with_name_node_map_utils import (
    SubgraphMatcherWithNameNodeMap,
)
from torch.fx.passes.utils.source_matcher_utils import SourcePartition
from torchao.quantization.pt2e import WrapperModule, find_sequential_partitions
from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec as TorchAOQuantizationSpec,
    get_module_name_filter,
)
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY

from coreai_opt._utils.config_utils import (
    ALL_TENSORS,
    ConfigLevel,
    get_last_matching_spec,
)
from coreai_opt._utils.fx_utils import (
    get_local_state_name,
    get_module_boundary_nodes,
    is_coreai_compressed_state_node,
)
from coreai_opt._utils.python_utils import get_fn_arg_names
from coreai_opt._utils.version_utils import version_ge
from coreai_opt.config.compression_config import ModuleConfigDict
from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.quantization._graph._utils import get_source_module_name
from coreai_opt.quantization.config import ModuleQuantizerConfig
from coreai_opt.quantization.config.quantization_config import (
    _ACTIVATION_SPEC_DICT,
    _STATE_SPEC_DICT,
)
from coreai_opt.quantization.spec import (
    QuantizationScheme,
    QuantizationSpec,
)

from ._annotation_config import AnnotationConfig, AnnotationContext

logger = logging.getLogger(__name__)


INPUT_NODE_PREFIX = "input::"
PARAM_NODE_PREFIX = "param::"


def _get_aten_graph_module_for_pattern(
    pattern: Callable,
    example_inputs: tuple[Any, ...],
    is_cuda: bool = False,
    **kwargs: Any,
) -> torch.fx.GraphModule:
    """Capture a small pattern callable as an aten-decomposed FX GraphModule.

    Drop-in replacement for ``torchao.quantization.pt2e.utils._get_aten_graph_module_for_pattern``
    that uses ``torch.export.export(strict=False)`` instead of ``strict=True``.

    The strict path routes through dynamo's compile pipeline, which is slower and leaks
    ``__compiled_fn_*`` entries permanently. Each pattern capture in this file's annotators leaks
    one entry per call; multiplied across ~190 weighted-mod patterns x 2 variants
    (with/without BN) per ``Quantizer.prepare`` call, that drives the
    dominant memory growth in the slow-test suite.

    The ``strict=False`` path skips the dynamo compile pipeline entirely and
    produces equivalent FX graphs for the patterns this file uses
    (Conv/Linear, optionally followed by BN/ReLU).
    """
    if is_cuda:
        example_inputs = tuple(
            x.cuda() if isinstance(x, torch.Tensor) else x for x in example_inputs
        )

    exported_program = torch.export.export(pattern, example_inputs, kwargs, strict=False)
    if version_ge(torch, "2.9"):
        aten_pattern = exported_program.module(check_guards=False)
    else:
        aten_pattern = exported_program.module()

    aten_pattern.graph.eliminate_dead_code()
    aten_pattern.recompile()

    # ep.module() adds copy_ nodes for the mutated inputs. For patterns,
    # they don't matter and would interfere with subgraph matching.
    for node in list(aten_pattern.graph.nodes):
        if (
            node.op == "call_function"
            and node.target == torch.ops.aten.copy_.default
            and len(node.users) == 0
        ):
            aten_pattern.graph.erase_node(node)

    aten_pattern.graph.eliminate_dead_code()
    aten_pattern.recompile()
    return aten_pattern


@dataclass
class OpsListPattern:
    """
    Represents a list of torch functions and operations making up a sequential pattern
    """

    pattern: list[Callable]


# All activations recognized for conv-act/conv-bn-act patterns
_supported_activations = (
    F.relu,
    F.relu6,
    F.leaky_relu,
    F.silu,
    F.elu,
    F.celu,
    F.selu,
    F.mish,
    F.hardtanh,
    F.hardswish,
    F.hardsigmoid,
)

# Dictionary mapping ops with known output bounds to (qscheme, float_range).
# float_range elements may be None to leave that side data-driven.
_fixed_q_params_ops = {
    # tanh: bounded to [-1, 1]
    torch.ops.aten.tanh.default: (QuantizationScheme.SYMMETRIC, (-1.0, 1.0)),
    torch.ops.aten.tanh_.default: (QuantizationScheme.SYMMETRIC, (-1.0, 1.0)),
    # sigmoid: bounded to [0, 1]
    torch.ops.aten.sigmoid.default: (QuantizationScheme.ASYMMETRIC, (0.0, 1.0)),
    torch.ops.aten.sigmoid_.default: (QuantizationScheme.ASYMMETRIC, (0.0, 1.0)),
    # hardsigmoid: bounded to [0, 1]
    torch.ops.aten.hardsigmoid.default: (QuantizationScheme.ASYMMETRIC, (0.0, 1.0)),
    torch.ops.aten.hardsigmoid_.default: (QuantizationScheme.ASYMMETRIC, (0.0, 1.0)),
    # relu: always >= 0, upper bound is data-driven
    torch.ops.aten.relu.default: (QuantizationScheme.ASYMMETRIC, (0.0, None)),
    torch.ops.aten.relu_.default: (QuantizationScheme.ASYMMETRIC, (0.0, None)),
    # relu6: clipped to [0, 6]
    torch.ops.aten.relu6.default: (QuantizationScheme.ASYMMETRIC, (0.0, 6.0)),
    torch.ops.aten.relu6_.default: (QuantizationScheme.ASYMMETRIC, (0.0, 6.0)),
}

# hardtanh bounds are configurable via node arguments; handled separately.
_hardtanh_ops = (
    torch.ops.aten.hardtanh.default,
    torch.ops.aten.hardtanh_.default,
)


# These activation functions don't have an inplace argument
_supported_activations_no_inplace = (F.gelu, F.sigmoid, F.logsigmoid, F.tanh)


# Map of dimension to convolution and convtranspose functions
_conv_fn_map = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}
_conv_transpose_fn_map = {1: F.conv_transpose1d, 2: F.conv_transpose2d, 3: F.conv_transpose3d}


def is_node_annotated(node: Node) -> bool:
    """
    Returns True if the node is annotated, otherwise returns False
    """
    return node and Q_ANNOTATION_KEY in node.meta and node.meta[Q_ANNOTATION_KEY]._annotated


def _get_weighted_mod_pattern(
    mod_fn: Callable,
    example_inputs: tuple[torch.Tensor, ...],
    act_fn: Callable | None = None,
    act_in_place: bool = False,
) -> torch.nn.Module:
    """
    Returns an aten graph corresponding to a sequence of these ops:
    input -> weighted_mod -> activation -> output

    A weighted mod is a module which has a weight and bias, such as a
    convolution module or a linear module. Only weight is quantized.

    No activation is used if ``act_fn`` is ``None``.
    ``act_fn`` is an activation function from _supported_activations or
    _supported_activations_no_inplace
    """

    def _weighted_mod(input, weight, bias):
        mod_out = mod_fn(input, weight, bias)
        output = mod_out
        node_dict = {
            "input": input,
            "mod": mod_out,
            "weight": weight,
            "bias": bias,
        }
        if act_fn is not None:
            # Only add output if activation function is applied to model output
            output = act_fn(output, inplace=True) if act_in_place else act_fn(output)
            node_dict["output"] = output
        return output, node_dict

    return _get_aten_graph_module_for_pattern(WrapperModule(_weighted_mod), example_inputs)


def _get_weighted_mod_bn_pattern(
    mod_fn: Callable,
    example_inputs: tuple[torch.Tensor, ...],
    act_fn: Callable | None = None,
    act_in_place: bool = False,
) -> torch.nn.Module:
    """
    Returns an aten graph corresponding to a sequence of these ops:
    input -> weighted_mod -> batch_norm -> activation -> output

    A weighted mod is a module which has a weight and bias, such as a
    convolution module or a linear module.

    No activation is used if ``act_fn`` is ``None``.
    ``act_fn`` is an activation function from _supported_activations or
    _supported_activations_no_inplace
    """

    def _weight_mod_bn(input, weight, bias, bn_weight, bn_bias, bn_run_mean, bn_run_var):
        mod_out = mod_fn(input, weight, bias)
        bn_out = F.batch_norm(mod_out, bn_run_mean, bn_run_var, bn_weight, bn_bias, training=True)
        output = bn_out
        if act_fn is not None:
            # Only add output if activation function is applied to model output
            output = act_fn(bn_out, inplace=True) if act_in_place else act_fn(bn_out)

        node_dict = {
            "input": input,
            "mod": mod_out,
            "bn": bn_out,
            "weight": weight,
            "bias": bias,
            "output": output,
        }
        return output, node_dict

    return _get_aten_graph_module_for_pattern(WrapperModule(_weight_mod_bn), example_inputs)


def _create_module_graph_pattern(
    module: nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    example_kwargs: dict | None = None,
    inputs_to_quantize: list[str] | None = None,
    params_to_quantize: list[str] | None = None,
) -> torch.fx.GraphModule:
    """
    Create a pattern graph module for an nn.Module.

    This function wraps the custom module to return both the output and a name node map
    that can be used by the annotation system.

    Args:
        module: The custom nn.Module to create a pattern for
        example_inputs: Example inputs to trace the module
        example_kwargs: Optional dictionary of keyword arguments to pass to the module
            during tracing
        inputs_to_quantize: Input names that should be quantized
        params_to_quantize: Parameter names that should be quantized

    Returns:
        A torch.fx.GraphModule representing the pattern
    """

    class CustomModuleWrapper(nn.Module):
        def __init__(
            self,
            wrapped_module: nn.Module,
            inputs_to_quantize: list[str] | None,
            params_to_quantize: list[str] | None,
        ):
            super().__init__()
            self.wrapped_module = wrapped_module
            self.inputs_to_quantize = inputs_to_quantize
            self.params_to_quantize = params_to_quantize

        def forward(self, *args, **kwargs):
            # Call the wrapped module
            output = self.wrapped_module(*args, **kwargs)

            # Create name node map for annotation
            node_dict = {}

            # Add input nodes
            if self.inputs_to_quantize is not None:
                arg_names = get_fn_arg_names(self.wrapped_module.forward)

                # Add args
                for idx, arg in enumerate(args):
                    if idx < len(arg_names) and arg_names[idx] in self.inputs_to_quantize:
                        input_node_name = INPUT_NODE_PREFIX + str(idx)
                        node_dict[input_node_name] = arg

                # Add kwargs
                idx = len(args)
                for name, arg in kwargs.items():
                    if name in self.inputs_to_quantize:
                        input_node_name = INPUT_NODE_PREFIX + str(idx)
                        node_dict[input_node_name] = arg
                        idx += 1

            # Add module node
            node_dict["mod"] = output

            # Add parameter nodes
            if self.params_to_quantize is not None:
                params = {}
                for name, p in self.wrapped_module.named_parameters():
                    if name in self.params_to_quantize and p is not None:
                        params[name] = p

                for param_name, param in params.items():
                    param_node_name = PARAM_NODE_PREFIX + param_name
                    node_dict[param_node_name] = param

            return output, node_dict

    # Create the wrapper module
    wrapper_module = CustomModuleWrapper(module, inputs_to_quantize, params_to_quantize)

    # Convert example_kwargs to empty dict if None
    if example_kwargs is None:
        example_kwargs = {}

    # Convert to ATEN graph using the same utility
    return _get_aten_graph_module_for_pattern(
        wrapper_module, example_inputs, is_cuda=False, **example_kwargs
    )


def get_conv_pattern(
    conv_dim: int,
    is_transpose: bool = False,
    act_fn: Callable | None = None,
    act_in_place: bool = False,
) -> torch.nn.Module:
    """
    Returns an aten graph corresponding to a sequence of these ops:
    input -> conv/conv_transpose -> activation -> output

    No activation is used if ``act_fn`` is ``None``.
    ``act_fn`` is an activation function from _supported_activations or
    _supported_activations_no_inplace
    """
    if is_transpose:
        assert conv_dim in _conv_transpose_fn_map, (
            f"Dimension {conv_dim} is not supported for ConvTranspose layers."
        )
    else:
        assert conv_dim in _conv_fn_map, (
            f"Dimension {conv_dim} is not supported for Convolution layers."
        )

    conv_func = _conv_transpose_fn_map[conv_dim] if is_transpose else _conv_fn_map[conv_dim]

    example_inputs = (
        torch.randn(1, 1, *[3] * conv_dim),  # input
        torch.randn(1, 1, *[1] * conv_dim),  # conv weight
        torch.randn(1),  # conv bias
    )
    return _get_weighted_mod_pattern(conv_func, example_inputs, act_fn, act_in_place)


def get_conv_bn_pattern(
    conv_dim: int, is_transpose: bool, act_fn: Callable | None = None, act_in_place: bool = False
) -> torch.nn.Module:
    """
    Returns an aten graph corresponding to a sequence of these ops:
    input -> conv -> batch_norm -> activation -> output

    No activation is used if ``act_fn`` is ``None``.
    ``act_fn`` is an activation function from _supported_activations or
    _supported_activations_no_inplace
    """
    if is_transpose:
        assert conv_dim in _conv_transpose_fn_map, (
            f"Dimension {conv_dim} is not supported for ConvTranspose layers."
        )
    else:
        assert conv_dim in _conv_fn_map, (
            f"Dimension {conv_dim} is not supported for Convolution layers."
        )

    conv_func = _conv_transpose_fn_map[conv_dim] if is_transpose else _conv_fn_map[conv_dim]

    example_inputs = (
        torch.randn(1, 1, *[3] * conv_dim),  # input
        torch.randn(1, 1, *[1] * conv_dim),  # conv weight
        torch.randn(1),  # conv bias
        torch.randn(1),  # bn_weight
        torch.randn(1),  # bn_bias
        torch.randn(1),  # bn_run_mean
        torch.randn(1),  # bn_run_var
    )
    return _get_weighted_mod_bn_pattern(conv_func, example_inputs, act_fn, act_in_place)


def get_linear_pattern(
    act_fn: Callable | None = None, act_in_place: bool = False
) -> torch.nn.Module:
    """
    Returns an aten graph corresponding to a sequence of these ops:
    input -> linear -> activation -> output

    No activation is used if ``act_fn`` is ``None``.
    ``act_fn`` is an activation function from _supported_activations or
    _supported_activations_no_inplace
    """
    example_inputs = (
        torch.randn(1, 1),  # input
        torch.randn(3, 1),  # linear weight
        torch.randn(3),  # linear bias
    )
    return _get_weighted_mod_pattern(F.linear, example_inputs, act_fn, act_in_place)


def get_linear_bn_pattern(
    act_fn: Callable | None = None, act_in_place: bool = False
) -> torch.nn.Module:
    """
    Returns an aten graph corresponding to a sequence of these ops:
    input -> linear -> batch_norm -> activation -> output

    No activation is used if ``act_fn`` is ``None``.
    ``act_fn`` is an activation function from _supported_activations or
    _supported_activations_no_inplace
    """
    example_inputs = (
        torch.randn(2, 1),  # input
        torch.randn(3, 1),  # linear weight
        torch.randn(3),  # linear bias
        torch.randn(3),  # bn_weight
        torch.randn(3),  # bn_bias
        torch.randn(3),  # bn_run_mean
        torch.randn(3),  # bn_run_var
    )
    return _get_weighted_mod_bn_pattern(F.linear, example_inputs, act_fn, act_in_place)


def get_embedding_pattern() -> torch.nn.Module:
    """
    Returns an aten graph corresponding to a sequence of these ops:
    input -> embedding -> output
    """
    example_inputs = (
        torch.randint(low=0, high=100, size=(1, 3), dtype=torch.long),  # input
        torch.randn(10, 3),  # embedding weight
    )

    def _embedding(input, weight):
        mod_out = F.embedding(input, weight)
        output = mod_out
        node_dict = {
            "input": input,
            "mod": mod_out,
            "weight": weight,
        }
        return output, node_dict

    return _get_aten_graph_module_for_pattern(WrapperModule(_embedding), example_inputs)


def _is_fx_node_floating_point(node: torch.fx.Node) -> bool:
    """
    Check if a fx node has floating point data.

    Args:
        node: The FX node to check

    Returns:
        bool: True if node has floating point data, False otherwise
    """
    # If 'val' is not present, assume tensor is not quantizable
    if "val" in node.meta:
        fake_tensor = node.meta["val"]
        if isinstance(fake_tensor, torch.Tensor):
            # Only quantize floating-point activations, skip integers, SymInt etc
            return fake_tensor.dtype.is_floating_point
    return False


def _get_state_aliases(
    state_node: torch.fx.Node,
    module_name_to_state_names_map: Mapping[str, Mapping[str, list[str]]],
) -> set[str]:
    """Return all local names any module uses for the state tensor at ``state_node.target``.

    A single state tensor may be aliased under different attribute names by different
    modules. This collects every such name across all modules so that spec lookups and
    warning checks are not limited to a single module's perspective.
    """
    return {
        name
        for module_states in module_name_to_state_names_map.values()
        if state_node.target in module_states
        for name in module_states[state_node.target]
    }


def _warn_non_quantizable_tensor_setting(
    node: torch.fx.Node,
    spec_type: Literal["input", "output", "state"],
    identifier: int | str,
    spec_dict: dict[str | int, TorchAOQuantizationSpec | None],
) -> None:
    """
    Warn that the user has configured quantization for a non-floating-point tensor.

    Args:
        node: The non-quantizable node
        spec_type: Type of spec being configured ("input", "state", or "output")
        identifier: The index or state name used in the config
        spec_dict: The full spec dictionary for context in the warning
    """
    spec_dict_name = f"op_{spec_type}_spec"

    if isinstance(identifier, int):
        identifier_str = f"idx {identifier}"
        setting_type = "index"
    else:
        identifier_str = identifier
        setting_type = "state"

    warning_msg = (
        f"Config is attempting to set {spec_dict_name} {identifier_str} for node "
        f"{node.name}, but the tensor is not a quantizable floating point "
        "tensor. No quantization will be performed on the tensor. Remove the "
        f"{setting_type} setting from the config to disable this warning.\n"
        f"{spec_dict_name}: {spec_dict}"
    )
    logger.warning(warning_msg)


def _validate_state_referenced_as_input(
    node: torch.fx.Node,
    input_idx: int,
    op_input_spec: dict[int | str, TorchAOQuantizationSpec | None],
) -> None:
    """
    Raise error if the user attempts to set a state tensor using input idx in
    op_input_spec.
    """
    if is_coreai_compressed_state_node(node) and input_idx in op_input_spec:
        raise RuntimeError(
            f"Config is attempting to set op_input_spec idx {input_idx}, but the input "
            f"is a state tensor (node: {node.name}). Use op_state_spec to configure "
            "state inputs instead.\n"
            f"op_input_spec: {op_input_spec}"
        )


def _get_input_qspec_map(
    input_and_state_nodes: list[torch.fx.Node],
    quantization_config: AnnotationConfig,
    context: AnnotationContext,
) -> dict[torch.fx.Node, TorchAOQuantizationSpec | None]:
    """
    Get input_qspec_map for a node according to the settings in quantization_config.
    """
    input_qspec_map: dict[torch.fx.Node, TorchAOQuantizationSpec | None] = {}
    op_input_spec = quantization_config.op_input_spec
    op_state_spec = quantization_config.op_state_spec

    for idx, node in enumerate(input_and_state_nodes):
        if not _is_fx_node_floating_point(node):
            # If input/state was specifically configured to set some qspec, log a
            # warning (settings using "*" will not be flagged)
            if idx in op_input_spec:
                _warn_non_quantizable_tensor_setting(node, "input", idx, op_input_spec)
            if is_coreai_compressed_state_node(node):
                state_names = _get_state_aliases(node, context.module_name_to_state_names_map)
                matching_keys = [key for key in op_state_spec if key in state_names]
                if matching_keys:
                    _warn_non_quantizable_tensor_setting(
                        node, "state", matching_keys[-1], op_state_spec
                    )

            input_qspec_map[node] = None
            continue

        _validate_state_referenced_as_input(node, idx, op_input_spec)

        if is_coreai_compressed_state_node(node):
            _fill_input_qspec_map_for_state(input_qspec_map, node, op_state_spec, context)
        else:
            _fill_input_qspec_map_for_input(input_qspec_map, node, idx, op_input_spec)
    return input_qspec_map


def _fill_input_qspec_map_for_state(
    input_qspec_map: dict[torch.fx.Node, TorchAOQuantizationSpec | None],
    state_node: torch.fx.Node,
    op_state_spec: dict[str, TorchAOQuantizationSpec | None],
    context: AnnotationContext,
) -> None:
    """
    Fill input_qspec_map with state_node as the key.

    For already compressed state nodes (e.g., lut_to_dense outputs from palettization),
    no quantization is applied since they don't have a traditional state name
    and represent already-compressed data.
    """
    found, spec = _get_state_node_shared_spec(state_node)
    if not found:
        state_name = get_local_state_name(state_node)
        if state_name is None:
            # Already compressed state (e.g., lut_to_dense from palettization) - don't quantize
            spec = None
        else:
            state_names = _get_state_aliases(state_node, context.module_name_to_state_names_map)
            spec, _ = get_last_matching_spec(state_names, op_state_spec)
    input_qspec_map[state_node] = spec


def _get_state_node_shared_spec(
    state_node: torch.fx.Node,
) -> tuple[bool, TorchAOQuantizationSpec | None]:
    """
    Check whether a state node already has an annotated user. If so, return the qspec
    associated with the user's input corresponding to the state_node (can be None).

    This ensures that for any states which are shared, a single common qspec (or None)
    is used for that state for all consumers of the state. Annotations are performed in
    priority order so any user which has already been annotated will have a higher
    priority state setting than whatever is present in the current user's op_state_spec.

    Returns a tuple of 2 values:
        - True if an annotated user was found, False otherwise
        - qspec for the annotated user's state input (can be None). Always returns None
          if no annotated user was found.
    """
    for user in state_node.users:
        if is_node_annotated(user):
            return True, user.meta[Q_ANNOTATION_KEY].input_qspec_map.get(state_node)
    return False, None


def _fill_input_qspec_map_for_input(
    input_qspec_map: dict[torch.fx.Node, TorchAOQuantizationSpec | None],
    input_node: torch.fx.Node,
    idx: int,
    op_input_spec: dict[int | str, TorchAOQuantizationSpec | None],
) -> None:
    """
    Fill input_qspec_map for the given node and input idx. Function arguments may change
    once tensor identification using string naming is supported.
    """
    # Check if any qspec is already set from a parent node output. If so, simply
    # use that spec.
    if not is_node_annotated(input_node) or input_node.meta[Q_ANNOTATION_KEY].output_qspec is None:
        spec, _ = get_last_matching_spec([idx], op_input_spec)
        input_qspec_map[input_node] = spec
    else:
        input_qspec_map[input_node] = input_node.meta[Q_ANNOTATION_KEY].output_qspec


def _get_call_function_node_from_partition(partition: SourcePartition) -> torch.fx.Node:
    """
    Given a partition, return the call function node associated with the partition.

    We expect there to be only one call function node in the partition.
    """
    call_function_nodes = [node for node in partition.nodes if node.op == "call_function"]
    if len(call_function_nodes) != 1:
        # torch.export's insert_deferred_runtime_asserts synthesizes one SymInt mul per
        # shape-runtime assertion, all sharing one torch_fn tag, so several can collapse
        # into a single partition. They carry no tensor value to annotate, so picking any
        # one of them is safe here; downstream floating-point filtering no-ops on SymInt.
        if call_function_nodes and all(
            isinstance(node.meta.get("val"), torch.SymInt) for node in call_function_nodes
        ):
            return call_function_nodes[0]

        module_names = {
            name
            for node in call_function_nodes
            if (name := get_source_module_name(node)) is not None
        }
        module_hint = ""
        if module_names:
            module_hint = (
                f"\nSource module(s): {', '.join(sorted(module_names))}. "
                f"Consider excluding this module from quantization via "
                f"module_name_configs."
            )
        error_msg = (
            f"Expected exactly 1 call function node in source partition but got "
            f"{call_function_nodes}.{module_hint}"
        )
        raise RuntimeError(error_msg)
    return call_function_nodes[0]


def match_pattern_with_sequential_partitions(
    model: torch.fx.GraphModule, pattern: OpsListPattern
) -> dict[torch.fx.Node, tuple[SourcePartition]]:
    """
    Helper function for matching model nodes with pattern using
    find_sequential_partitions.

    Returns a dictionary mapping nodes to matches.
    """
    node_to_match_dict: dict[torch.fx.Node, tuple[SourcePartition]] = {}

    partitions = find_sequential_partitions(model, pattern.pattern)
    # Each sequential partition is a tuple of source partitions, one per pattern element.
    for sequential_partition in partitions:
        if _is_branching_partition(sequential_partition):
            continue
        first_op_node = _get_call_function_node_from_partition(sequential_partition[0])
        assert first_op_node not in node_to_match_dict

        # Use the first node in the sequences as the key
        node_to_match_dict[first_op_node] = sequential_partition
    return node_to_match_dict


def _is_branching_partition(sequential_partition: tuple[SourcePartition]) -> bool:
    """
    Return True if the partition is a branching partition, False otherwise.

    A branching partition is defined as any intermediate node in the partition having more than
    one child node.
    """
    # Ignore the final node in the pattern since it is ok for it to have multiple outputs
    for source_partition in sequential_partition[:-1]:
        node = _get_call_function_node_from_partition(source_partition)
        if len(node.users) > 1:
            return True
    return False


def match_pattern_with_subgraph_matcher(
    model: torch.fx.GraphModule, pattern_gm: torch.fx.GraphModule
) -> dict[torch.fx.Node, InternalMatch]:
    """
    Helper function for matching model nodes with pattern using
    SubgraphMatcherWithNameNodeMap.

    Returns a dictionary mapping nodes to matches.
    """
    node_to_match_dict: dict[torch.fx.Node, InternalMatch] = {}
    matcher = SubgraphMatcherWithNameNodeMap(pattern_gm, ignore_literals=True)
    matches = matcher.match(model.graph)

    for subgraph_match in matches:
        name_node_map = subgraph_match.name_node_map
        mod_node = name_node_map["mod"]
        assert mod_node not in node_to_match_dict

        # Use the node marked with "mod" as the key
        node_to_match_dict[mod_node] = subgraph_match
    return node_to_match_dict


def annotate_module_level_specs(
    module_configs: ModuleConfigDict,
    module_name_to_state_names_map: Mapping[str, Mapping[str, list[str]]],
    model: torch.fx.GraphModule,
) -> None:
    """
    Annotate nodes for module_input_spec, module_output_spec, and module_state_spec.

    Args:
        module_configs: Dictionary mapping module names to their quantization
            configurations, separating modules by config level.
        module_name_to_state_names_map: A two level dictionary mapping module names
            to another dictionary. The inner dictionary maps full state names to all
            local names in that module which points to the state object referenced
            by the full state name.
        model: Model to annotate.
    """
    for config_level in [ConfigLevel.MODULE_TYPE, ConfigLevel.MODULE_NAME]:
        for module_name, module_config in module_configs[config_level].items():
            if _module_config_has_module_level_input_output_spec(module_config):
                _annotate_nodes_for_module_level_input_output_spec(
                    module_config,
                    _get_nodes_in_module(module_name, model),
                )

    # Module state specs are handled separately since nodes outside of the module with
    # the state spec config may be affected.
    _annotate_nodes_for_module_level_state_spec(
        module_configs, module_name_to_state_names_map, model
    )


def _annotate_nodes_for_module_level_state_spec(
    module_configs: ModuleConfigDict,
    module_name_to_state_names_map: Mapping[str, Mapping[str, list[str]]],
    model: torch.fx.GraphModule,
) -> None:
    """
    Annotate state nodes in the model for module state spec.

    In contrast with module input and output settings which affect nodes at the module
    bounary, module state settings correspond to state nodes which may be used by other
    modules outside of the module which defined the state (see the below example).

    As a result of this, we annotate nodes for module level state settings in a separate
    pass compared to input/output settings, iterating through all nodes in the graph
    without considering what module they lie in.

    Consider the following model definition:

    class InnerModule(torch.nn.Module):
        def __init__(self):
            ...
            self.inner_param = torch.nn.Parameter(...)

        def forward(self, inp):
            ...

    class OuterModule(torch.nn.Module):
        def __init__(self):
            ...
            self.outer_param = torch.nn.Parameter(...)
            self.inner_module = InnerModule()
            self.some_op = SomeOp()

        def forward(self, inp):
            ...
            x = self.some_op(self.inner_module.inner_param)
            self.inner_module(self.outer_param)
            ...

    2 things to note:
        - inner_param is defined in InnerModule, so a specification for inner_param can
          be defined in a module type or name config for InnerModule.
          However, since OuterModule makes use of InnerModule's inner_param, this leads
          to OuterModule.some_op's input needing to use inner_param's spec.
        - OuterModule's outer_param is passed as input to self.inner_module. If a
          specification was set for OuterModule's outer_param, whatever operation within
          self.inner_module which makes use of the passed in state would need to be
          configured with outer_param's spec.
    """
    for node in model.graph.nodes:
        if is_coreai_compressed_state_node(node):
            _match_and_annotate_state_node(node, module_configs, module_name_to_state_names_map)


def _match_and_annotate_state_node(
    node: torch.fx.Node,
    module_configs: ModuleConfigDict,
    module_name_to_state_names_map: Mapping[str, Mapping[str, list[str]]],
) -> None:
    """
    Given a state node, check if any of the module_configs have applicable
    module_state_specs and apply if so.
    """
    for level in ConfigLevel.priority_order():
        # Reversed is needed because two different modules may have shared params where
        # both module configs are setting module_state_spec for the param using their
        # respective local names for the same parameter.
        # Follow the standard config behavior of using the highest priority last
        # matching config.
        for module_name, config in module_configs[level].items():
            if node.target in module_name_to_state_names_map[module_name] and config is not None:
                local_state_names = module_name_to_state_names_map[module_name][node.target]
                found, spec = _get_spec_from_spec_dict(config.module_state_spec, local_state_names)
                if found:
                    _annotate_state_node_consumers(node, spec)
                    # Skip processing other module configs for the current node if a
                    # match is found.
                    return


def _annotate_state_node_consumers(
    state_node: torch.fx.Node, spec: QuantizationSpec | None
) -> None:
    """
    Annotate state node given spec.
    """
    converted_spec = AnnotationConfig._convert_to_pt2e_spec(spec, CompressionTargetTensor.WEIGHT)
    for consumer_node in state_node.users:
        _annotate_node_input_qspec(consumer_node, state_node, converted_spec)


def _get_nodes_in_module(module_name: str, model: torch.fx.GraphModule) -> list[torch.fx.Node]:
    """
    Given a module name, return a list of nodes which correspond to that name.
    """
    module_name_filter = get_module_name_filter(module_name)
    return [node for node in model.graph.nodes if module_name_filter(node)]


def _module_config_has_module_level_input_output_spec(module_config: ModuleQuantizerConfig) -> bool:
    """
    Return True if module_config has a non-empty module level input or output spec,
    False otherwise.
    """
    return bool(module_config.module_input_spec or module_config.module_output_spec)


def _get_spec_from_spec_dict(
    spec_dict: _ACTIVATION_SPEC_DICT | _STATE_SPEC_DICT,
    identifier: int | str | list[int] | list[str],
) -> tuple[bool, QuantizationSpec | None]:
    """
    Get spec from module config dict, with fallback to _ALL_TENSORS.

    The first element of the tuple is a boolean representing whether a spec in spec dict
    was found. The second element is the actual spec itself. Because a found spec can
    be None, a distinction is made between a found spec vs. one that is None due to not
    having any match.

    Args:
        spec_dict: The spec dictionary to look up (module_input_spec, module_state_spec,
            or module_output_spec)
        index: The index of the input/output to look up

    Returns:
        A tuple of (found, spec) where:
        - found: True if key was found (or _ALL_TENSORS fallback exists), False
        otherwise
        - spec: The quantization spec (may be None if explicitly set to None)
    """
    if not isinstance(identifier, list):
        identifier = [identifier]

    for i in identifier:
        if i in spec_dict:
            return (True, spec_dict[i])
    if ALL_TENSORS in spec_dict:
        return (True, spec_dict[ALL_TENSORS])
    return (False, None)


def _annotate_node_input_qspec(
    node: torch.fx.Node, input_node: torch.fx.Node, qspec: TorchAOQuantizationSpec | None
) -> None:
    """
    Annotate a node's input quantization spec. If a quantization spec exists, it will
    be overwritten.

    Args:
        node: The consumer node to annotate
        input_node: The input node being consumed
        qspec: The quantization spec to apply
    """
    if Q_ANNOTATION_KEY not in node.meta:
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation()
    node.meta[Q_ANNOTATION_KEY].input_qspec_map[input_node] = qspec


def _annotate_node_output_qspec(node: torch.fx.Node, qspec: TorchAOQuantizationSpec | None) -> None:
    """
    Annotate a node's output quantization spec. If a quantization spec exists, it will
    be overwritten.

    Args:
        node: The node to annotate
        qspec: The quantization spec to apply
    """
    if Q_ANNOTATION_KEY not in node.meta:
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation()
    node.meta[Q_ANNOTATION_KEY].output_qspec = qspec


def _annotate_nodes_for_module_level_input_output_spec(
    module_config: ModuleQuantizerConfig,
    nodes_in_module: list[torch.fx.Node],
) -> None:
    """
    Annotate nodes with module-level input and output quantization specs.

    Args:
        module_config: The module quantizer config containing module-level specs
        nodes_in_module: List of nodes present in the module being annotated
    """
    (input_consumer_tuples, outputs) = get_module_boundary_nodes(nodes_in_module)

    # Annotate module inputs
    if module_config.module_input_spec:
        for idx, (input_node, consumer_node) in enumerate(input_consumer_tuples):
            _find_and_apply_module_level_spec(
                consumer_node, idx, input_node, module_config.module_input_spec, True
            )

    # Annotate module outputs
    if module_config.module_output_spec:
        for idx, output_node in enumerate(outputs):
            _find_and_apply_module_level_spec(
                output_node, idx, None, module_config.module_output_spec, False
            )


def _find_and_apply_module_level_spec(
    node_to_annotate: torch.fx.Node,
    identifier: int | str,
    input_node: torch.fx.Node | None,
    spec_dict: _ACTIVATION_SPEC_DICT,
    is_input: bool,
) -> None:
    """
    Possibly annotate a node if a matching spec for identifier is found.

    Args:
        node_to_annotate: The node to annotate
        identifier: Index or string name to match for an entry in spec_dict
        input_node: Only used for input annotation. Needed for filling the
            corresponding entry in input_qspec_map with the spec that was found
        spec_dict: Dictionary mapping tensor identifiers (e.g. input/output index, state
            names, "*") to specs
        is_input: True if the input of the node is to be annotated, False for output
    """
    found, spec = _get_spec_from_spec_dict(spec_dict, identifier)
    # If spec is not found, do not do any more processing (as opposed to disabling
    # quantization for the node as is done in op-level processing)
    if found:
        converted_spec = AnnotationConfig._convert_to_pt2e_spec(
            spec, CompressionTargetTensor.ACTIVATION
        )
        if is_input:
            _annotate_node_input_qspec(node_to_annotate, input_node, converted_spec)
        else:
            _annotate_node_output_qspec(node_to_annotate, converted_spec)
