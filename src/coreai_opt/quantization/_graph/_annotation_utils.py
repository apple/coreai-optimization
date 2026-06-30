# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import logging
from collections.abc import Callable, Iterable, Mapping
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
    FixedQParamsQuantizationSpec,
    QuantizationAnnotation,
    QuantizationSpec as TorchAOQuantizationSpec,
    SharedQuantizationSpec as _SharedQuantizationSpec,
    get_module_name_filter,
)
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY

from coreai_opt._utils.config_utils import (
    ALL_TENSORS as _ALL_TENSORS,
    ConfigLevel as _ConfigLevel,
    get_last_matching_spec,
)
from coreai_opt._utils.fx_utils import (
    get_local_state_name,
    get_module_boundary_nodes,
    is_coreai_compressed_state_node,
)
from coreai_opt._utils.python_utils import get_fn_arg_names
from coreai_opt._utils.version_utils import version_ge as _version_ge
from coreai_opt.config.compression_config import ModuleConfigDict
from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.quantization.config import ModuleQuantizerConfig
from coreai_opt.quantization.config.quantization_config import (
    _ACTIVATION_SPEC_DICT,
    _STATE_SPEC_DICT,
)
from coreai_opt.quantization.spec import QuantizationSpec

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
    if _version_ge(torch, "2.9"):
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

_tanh_qspec = FixedQParamsQuantizationSpec(
    dtype=torch.uint8,
    scale=2.0 / 256.0,
    zero_point=128,
    quant_min=0,
    quant_max=255,
    qscheme=torch.per_tensor_symmetric,
)

_sigmoid_qspec = FixedQParamsQuantizationSpec(
    dtype=torch.uint8,
    scale=1.0 / 256.0,
    zero_point=0,
    quant_min=0,
    quant_max=255,
    qscheme=torch.per_tensor_affine,
)

_fixed_q_params_ops = {
    torch.ops.aten.tanh.default: _tanh_qspec,
    torch.ops.aten.tanh_.default: _tanh_qspec,
    torch.ops.aten.sigmoid.default: _sigmoid_qspec,
    torch.ops.aten.sigmoid_.default: _sigmoid_qspec,
    torch.ops.aten.hardsigmoid.default: _sigmoid_qspec,
    torch.ops.aten.hardsigmoid_.default: _sigmoid_qspec,
}

_always_affine_ops = (
    torch.ops.aten.relu.default,
    torch.ops.aten.relu_.default,
    torch.ops.aten.relu6.default,
    torch.ops.aten.relu6_.default,
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


def is_any_annotated(nodes: list[Node]) -> bool:
    """
    Given a list of nodes (that represents an operator pattern),
    check if any of the node is annotated, return True if any of the node
    is annotated, otherwise return False.
    """
    return any(is_node_annotated(node) for node in nodes)


def is_all_annotated(nodes: list[Node]) -> bool:
    """
    Given a list of nodes (that represents an operator pattern),
    return True if all of the node is annotated, otherwise return False.
    """
    return all(is_node_annotated(node) for node in nodes)


def mark_nodes_as_annotated(nodes: Iterable[Node]) -> None:
    for node in nodes:
        if node is not None:
            if Q_ANNOTATION_KEY not in node.meta:
                node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation()
            node.meta[Q_ANNOTATION_KEY]._annotated = True


def _propagate_qscheme_to_child_nodes(
    root_node: torch.fx.Node,
    qscheme: torch.qscheme,
    shared_observer_nodes: set[torch.fx.Node] | None = None,
) -> None:
    """
    Given a qscheme, propagate the qscheme to all applicable children. Any input qspecs
    which are not shared qspecs will have qschemes updated. The propagation logic
    continues downwards through the graph until we encounter a non-shared observer op.
    """
    nodes_to_propagate = [(root_node, user) for user in root_node.users.keys()]
    while nodes_to_propagate:
        parent, curr_node = nodes_to_propagate.pop(0)
        if not is_node_annotated(curr_node):
            continue
        curr_input_qspec = curr_node.meta[Q_ANNOTATION_KEY].input_qspec_map.get(parent)
        if curr_input_qspec is None:
            continue
        if (
            isinstance(curr_input_qspec, _SharedQuantizationSpec)
            and root_node not in curr_input_qspec.edge_or_node
        ):
            # This is a case in which we encounter a multi input op which already has a
            # shared input qspec referencing a different edge. Here, we will not update
            # the qscheme since it would unnecessarily constrain the range of the other
            # edge. Here, there is the possiblity of having back to back quantize
            # dequantize ops inserted.
            continue
        if not isinstance(curr_input_qspec, _SharedQuantizationSpec):
            adjusted_qspec = TorchAOQuantizationSpec(
                observer_or_fake_quant_ctr=curr_input_qspec.observer_or_fake_quant_ctr,
                dtype=curr_input_qspec.dtype,
                qscheme=qscheme,
                quant_min=curr_input_qspec.quant_min,
                quant_max=curr_input_qspec.quant_max,
            )
            curr_node.meta[Q_ANNOTATION_KEY].input_qspec_map[parent] = adjusted_qspec

        if curr_node in shared_observer_nodes:
            nodes_to_propagate.extend([(curr_node, user) for user in curr_node.users.keys()])


def adjust_output_qspec_for_qscheme_and_propagate(
    node: torch.fx.Node, shared_observer_nodes: set[torch.fx.Node]
) -> None:
    """
    Adjust output quantization spec for ops which can use fixed qparams
    or ops for which we can use affine quantization mode during
    symmetric quantization because their output is always positive.
    Propagate the updated qscheme to input qspecs of child ops.

    Args:
        node: The node whose output qspec should be updated if necessary
        shared_observer_nodes: Set of shared observer nodes for which adjusted qschemes
            should propagate through
    """
    if not is_node_annotated(node):
        return

    qspec = node.meta[Q_ANNOTATION_KEY].output_qspec
    if qspec is None:
        return

    # ReLU6 activation maps to torch.ops.aten.hardtanh.default with
    # min_val = 0 and max_val = 6
    is_always_affine_op = node.target in _always_affine_ops or (
        node.target in [torch.ops.aten.hardtanh.default, torch.ops.aten.hardtanh_.default]
        and node.args[1] == 0  # min_val, corresponding to ReLU6
        and node.args[2] == 6  # max_val, corresponding to ReLU6
    )

    adjusted_qspec = None
    if node.target in _fixed_q_params_ops:
        adjusted_qspec = TorchAOQuantizationSpec(
            observer_or_fake_quant_ctr=qspec.observer_or_fake_quant_ctr,
            dtype=qspec.dtype,
            qscheme=_fixed_q_params_ops[node.target].qscheme,
            quant_min=qspec.quant_min,
            quant_max=qspec.quant_max,
        )
        # FIXME: Because of a bug in PyTorch in function _create_obs_or_fq_from_qspec
        #        in module torch/ao/quantization/fx/prepare.py  which creates a
        #        FixedQParamsFakeQuantize partial, instead of an instance, we cannot
        #        actually create FixedQParamsQuantizationSpec
    elif is_always_affine_op:
        adjusted_qspec = TorchAOQuantizationSpec(
            observer_or_fake_quant_ctr=qspec.observer_or_fake_quant_ctr,
            dtype=qspec.dtype,
            qscheme=torch.per_tensor_affine,
            quant_min=qspec.quant_min,
            quant_max=qspec.quant_max,
        )

    if adjusted_qspec is not None:
        node.meta[Q_ANNOTATION_KEY].output_qspec = adjusted_qspec
        _propagate_qscheme_to_child_nodes(node, adjusted_qspec.qscheme, shared_observer_nodes)


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


def find_consumers(
    pattern_nodes: Iterable[torch.fx.Node], target_nodes: list[torch.fx.Node]
) -> set[torch.fx.Node]:
    """Find all nodes that take as input any of the target nodes"""
    consumers = []
    for node in pattern_nodes:
        if node is None:
            continue
        for arg in node.all_input_nodes:
            if arg in target_nodes:
                consumers.append(node)
    return set(consumers)


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


def _get_output_qspec(
    node: torch.fx.Node,
    quantization_config: AnnotationConfig,
    shared_observer_nodes: set[torch.fx.Node] | None = None,
) -> TorchAOQuantizationSpec | None:
    """
    Get the output qspec which should be associated with the node. Check child nodes
    first to see if any input qspec is already set, and if so, reuse the qspec.
    If there are multiple child nodes, the qspec will be the first valid one found when
    checking all child nodes.
    When encountering shared observer nodes as child nodes, continue checking their
    children if no valid input qspec is found.
    """
    op_output_spec = quantization_config.op_output_spec

    # Early exit if input node is not floating point dtype
    if not _is_fx_node_floating_point(node):
        # Hardcoding for index 0 for now while we only support single output setting
        if 0 in op_output_spec:
            _warn_non_quantizable_tensor_setting(node, "output", 0, op_output_spec)
        return None

    # First read qspec from config without applying it yet.
    qspec_from_config, _ = get_last_matching_spec([0], op_output_spec)

    # Don't set output qspec if it is specified to be None. If the op has multiple child
    # ops where a subset of child ops don't have input quantization, we should not
    # insert a quantizer for the op's output.
    if qspec_from_config is None:
        return None

    # Check child ops to see if qspec settings should be taken from them.
    # Current logic simply uses the first child qspec found as the qspec to use.
    # This logic might need to be updated to be smarter in how the qspec to use is
    # determined.
    if not shared_observer_nodes:
        shared_observer_nodes = set()
    nodes_to_check = [user for user in node.users]
    while nodes_to_check:
        curr_node = nodes_to_check.pop(0)
        if (
            is_node_annotated(curr_node)
            and curr_node.meta[Q_ANNOTATION_KEY].input_qspec_map.get(node) is not None
        ):
            return curr_node.meta[Q_ANNOTATION_KEY].input_qspec_map.get(node)
        # Check children of shared observer nodes as well in a DFS fashion
        if curr_node in shared_observer_nodes:
            nodes_to_check[:0] = curr_node.users

    # If no valid qspec was found, refer to quantization config
    return qspec_from_config


def _propagate_output_qspec(
    node: torch.fx.Node,
    output_qspec: TorchAOQuantizationSpec,
    shared_observer_nodes: set[torch.fx.Node] | None = None,
):
    """
    Propagate output qspec to child ops which are shared observer nodes. These node
    types should simply echo the qspec which is coming from its input.
    Propagated specs will be SharedQuantizationSpecs except for the first input qspec
    that is set, since all subsequent SharedQuantizationSpecs will be referring to the
    first input qspec set.
    This function should be called for each annotation function which is defined.
    """
    if output_qspec is None or not shared_observer_nodes or not node.users:
        return
    if not shared_observer_nodes:
        shared_observer_nodes = set()
    spec_to_propagate = output_qspec
    nodes_to_propagate = [user for user in node.users if user in shared_observer_nodes]

    while nodes_to_propagate:
        user = nodes_to_propagate.pop(0)
        # Skip propagation for a node which is already annotated
        if is_any_annotated([user]):
            continue
        input_qspec_map = {}
        if isinstance(spec_to_propagate, _SharedQuantizationSpec):
            # If spec to propagate is already a shared qspec, we can simply update all
            # shared observer inputs and outputs with the shared qspec
            for input_node in user.all_input_nodes:
                input_qspec_map[input_node] = spec_to_propagate
        else:
            # If spec is not a shared qspec, this is the very first qspec we are
            # setting. Set the first input to be this qspec, then update
            # spec_to_propagate to be a shared qspec to be used for all other updates
            first_user_input = user.all_input_nodes[0]
            input_qspec_map[first_user_input] = spec_to_propagate
            spec_to_propagate = _SharedQuantizationSpec((first_user_input, user))
            for input_node in user.all_input_nodes[1:]:
                input_qspec_map[input_node] = spec_to_propagate

        user.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map, output_qspec=spec_to_propagate
        )
        mark_nodes_as_annotated([user])

        nodes_to_propagate.extend([child for child in user.users if child in shared_observer_nodes])


def _get_call_function_node_from_partition(partition: SourcePartition) -> torch.fx.Node:
    """Return the first call_function node in the partition."""
    return [node for node in partition.nodes if node.op == "call_function"][0]


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


def annotate_weighted_mod_match(
    annotator_match: InternalMatch,
    quantization_config: AnnotationConfig,
    context: AnnotationContext,
) -> None:
    """
    Try to annotate specific nodes in the model designated by ``annotator_match`` using
    ``quantization_config``.
    """
    # Entries in name_node_map will be determined by what is populated in
    # node_dict when the pattern was created.
    name_node_map = annotator_match.name_node_map
    mod_node = name_node_map["mod"]
    bn_node = name_node_map.get("bn")
    output_node = None
    if "output" in name_node_map:
        # In this case, an activation is applied to the weighted module output
        output_node = name_node_map["output"]
        # If the output is same as bn_node or mod_node, it means we have an inplace
        # activation, so we need to correct the node.
        if bn_node is not None and bn_node == output_node:
            bn_node = bn_node.args[0]
        elif mod_node == output_node:
            mod_node = mod_node.args[0]

    # TODO: skip partition if any intermediate node output is used by an op outside the pattern.

    # Skip partition if already annotated
    partition = [mod_node]
    partition.extend(filter(None, [bn_node, output_node]))

    if is_any_annotated(partition):
        return

    shared_observer_nodes = context.shared_observer_nodes
    input_qspec_map = _get_input_qspec_map(
        mod_node.all_input_nodes,
        quantization_config,
        context,
    )
    output_qspec = _get_output_qspec(
        output_node or mod_node, quantization_config, shared_observer_nodes
    )
    # set mod_node
    mod_node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=None if output_node else output_qspec,
    )
    # set output_node if exists
    if output_node:
        output_node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(output_qspec=output_qspec)

    # apply output spec to either final output_node or mod_node
    _propagate_output_qspec(output_node or mod_node, output_qspec, shared_observer_nodes)

    # Mark all nodes in pattern as annotated
    mark_nodes_as_annotated(partition)


def annotate_n_ary_act_match(
    annotator_match: tuple[SourcePartition],
    quantization_config: AnnotationConfig,
    context: AnnotationContext,
) -> None:
    """
    Try to annotate specific nodes in the model designated by ``annotator_match`` using
    ``quantization_config``.
    """
    if len(annotator_match) > 2:
        error_msg = (
            "Sequential list of ops is longer than 2. Only lists of up to "
            "length 2 (op + [act]_ are supported."
        )
        raise RuntimeError(error_msg)
    nodes_to_annotate = [
        _get_call_function_node_from_partition(partition) for partition in annotator_match
    ]
    first_op_node = nodes_to_annotate[0]
    last_op_node = nodes_to_annotate[-1]

    if is_any_annotated(nodes_to_annotate):
        return

    # TODO: skip partition if any intermediate node output is used by an op outside the pattern.

    shared_observer_nodes = context.shared_observer_nodes
    input_qspec_map = _get_input_qspec_map(
        first_op_node.all_input_nodes,
        quantization_config,
        context,
    )
    output_qspec = _get_output_qspec(last_op_node, quantization_config, shared_observer_nodes)
    if len(nodes_to_annotate) == 1:
        first_op_node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=output_qspec,
            _annotated=True,
        )
    else:
        first_op_node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
        )
        last_op_node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            output_qspec=output_qspec,
            _annotated=True,
        )
    _propagate_output_qspec(last_op_node, output_qspec, shared_observer_nodes)


def _adjust_input_qspec_map_for_shared_observers(
    op_node: Node, input_qspec_map: dict[Node, TorchAOQuantizationSpec]
) -> _SharedQuantizationSpec | None:
    """
    For shared observer ops, check if any inputs have qspecs. If so,
    1. set the first input to the first valid qspec
    2. set all subsequent inputs to use SharedObserverQspec tied to the first input

    This logic is mainly needed for multi-input shared observer ops (concat).
    A SharedObserverQspec is returned for use as output_qspec.
    """
    shared_qspec = None
    for input_node in op_node.all_input_nodes:
        if input_qspec_map[input_node] is not None:
            input_qspec_map[op_node.all_input_nodes[0]] = input_qspec_map[input_node]
    if input_qspec_map[op_node.all_input_nodes[0]] is not None:
        shared_qspec = _SharedQuantizationSpec((op_node.all_input_nodes[0], op_node))
        for input_node in op_node.all_input_nodes[1:]:
            input_qspec_map[input_node] = shared_qspec
    return shared_qspec


def annotate_shared_observer_match(
    annotator_match: tuple[SourcePartition],
    quantization_config: AnnotationConfig,
    context: AnnotationContext,
) -> None:
    """
    Try to annotate specific nodes in the model designated by ``annotator_match`` using
    ``quantization_config``.
    """
    if len(annotator_match) > 1:
        error_msg = (
            "Shared observer pattern is expected to be length 1, but got "
            f"{len(annotator_match)}. Annotator match: {annotator_match}."
        )
        raise RuntimeError(error_msg)

    op_node = _get_call_function_node_from_partition(annotator_match[0])
    if is_node_annotated(op_node):
        return

    shared_observer_nodes = context.shared_observer_nodes
    input_qspec_map = _get_input_qspec_map(
        op_node.all_input_nodes,
        quantization_config,
        context,
    )
    output_qspec = _adjust_input_qspec_map_for_shared_observers(op_node, input_qspec_map)

    if output_qspec is None:
        # Only use a different output qspec if it isn't sharing with its input
        output_qspec = _get_output_qspec(op_node, quantization_config, shared_observer_nodes)

    # input and output of op will share quantization parameter with input of op
    op_node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
        input_qspec_map=input_qspec_map,
        output_qspec=output_qspec,
        _annotated=True,
    )
    _propagate_output_qspec(op_node, output_qspec, shared_observer_nodes)


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
    for config_level in [_ConfigLevel.MODULE_TYPE, _ConfigLevel.MODULE_NAME]:
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
    for level in _ConfigLevel.priority_order():
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
    if _ALL_TENSORS in spec_dict:
        return (True, spec_dict[_ALL_TENSORS])
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
