# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Export preparation utilities for quantized models.

This module provides functions to prepare PT2E-quantized models for different
export backends (CoreML, CoreAI) by converting fake quantization operations to
backend-specific representations.
"""

import logging
import operator
from collections.abc import Mapping

import torch
from torch.fx import Node

from coreai_opt._utils.export_utils import validate_coreml_compatibility
from coreai_opt._utils.fx_utils import get_node_type as _get_node_type
from coreai_opt._utils.import_utils import lazy_import_coreai_torch
from coreai_opt._utils.metadata_utils import CompressionType, MILCompressionMetadata
from coreai_opt._utils.torch_utils import (
    is_float4_dtype as _is_float4_dtype,
    sanitize_module_name as _sanitize_module_name,
)
from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.quantization._export_utils import (
    canonicalize_qparam_shape as _canonicalize_qparam_shape,
    convert_dtype_for_torch_quantize as _convert_dtype_for_torch_quantize,
    create_mil_act_quant_seq as _create_mil_act_quant_seq,
    extract_quantization_params as _extract_quantization_params,
    pack_fp4_to_float4tensor as _pack_fp4_to_float4tensor,
    select_export_qparams_by_formulation as _select_export_qparams_by_formulation,
    validate_fp4_export as _validate_fp4_export,
    validate_qformulation_for_mil_export as _validate_qformulation_for_mil_export,
)
from coreai_opt.quantization._graph._utils import (
    remove_fake_quant_module,
    resolve_attr,
)
from coreai_opt.quantization.spec.fake_quantize import FakeQuantizeImplBase

logger = logging.getLogger(__name__)

# Module name prefix for MIL activation quantization modules
MIL_ACT_QUANT_MODULE_PREFIX = "_mil_act_quant_"


def _get_module_from_node(
    node: Node, named_modules: dict[str, torch.nn.Module]
) -> torch.nn.Module | None:
    """
    If `node` refers to a call_module node, return the module, else None.
    """
    if node.op == "call_module":
        return named_modules.get(str(node.target))
    else:
        return None


def _collect_fake_quantize_nodes(
    model: torch.fx.GraphModule,
    modules: Mapping[str, torch.nn.Module] | None = None,
) -> list[tuple[Node, FakeQuantizeImplBase]]:
    """Collect all fake quantization nodes and their corresponding modules.

    Args:
        model (torch.fx.GraphModule): The graph module to search for fake
            quantization nodes.
        modules (Mapping[str, torch.nn.Module] | None): Optional pre-computed
            mapping of module names to modules. If None, will be computed from
            model.

    Returns:
        list[tuple[Node, FakeQuantizeImplBase]]: A list of
            (node, fake_quantize_module) pairs.

    """
    if modules is None:
        modules = dict(model.named_modules(remove_duplicate=False))

    fake_quant_nodes_and_modules = []
    for node in list(model.graph.nodes):
        if node.op == "call_module":
            mod = _get_module_from_node(node, modules)
            if isinstance(mod, FakeQuantizeImplBase):
                fake_quant_nodes_and_modules.append((node, mod))

    return fake_quant_nodes_and_modules


def _is_weight_fake_quant(node: Node, module: FakeQuantizeImplBase) -> bool:
    """Determine if a fake quantization node is for weight quantization.

    Weight quantization nodes have get_attr as their input (accessing model parameters),
    while activation quantization nodes have other operations as input.

    Args:
        node: The fake quantization node to check
        module: The fake quantization module to check

    Returns:
        True if this is a weight quantization node, False otherwise

    """
    input_node = node.args[0]
    return (
        module.quantization_target == CompressionTargetTensor.WEIGHT and input_node.op == "get_attr"
    )


def _get_fake_quant_input(fake_quant_node: Node) -> Node:
    """Get the input node from a fake quantization node.

    For any fake quantization node (weight or activation), the input is
    always the first argument.

    Args:
        fake_quant_node: The fake quantization node

    Returns:
        The input node

    """
    return fake_quant_node.args[0]


def _get_weight_input_names(
    fake_quant_node: Node,
    module: FakeQuantizeImplBase,
) -> tuple[str, str]:
    """Extract module name and parameter name from a weight quantization node.

    Args:
        fake_quant_node: The fake quantization node (must be weight quantization)
        module: The fake quantization module

    Returns:
        Tuple of (module_name, param_name)
        - module_name: e.g., "conv", "layer1.0"
        - param_name: e.g., "weight", "bias"

    Raises:
        ValueError: If node is not a weight quantization node
        ValueError: If weight target path is invalid

    """
    if not _is_weight_fake_quant(fake_quant_node, module):
        msg = f"Node {fake_quant_node.name} is not a weight quantization node"
        raise ValueError(msg)

    input_node = _get_fake_quant_input(fake_quant_node)

    # Extract module and parameter name from target path
    # e.g., "conv.weight" -> ("conv", "weight")
    # e.g., "layer1.0.weight" -> ("layer1.0", "weight")
    target_path = str(input_node.target)
    last_dot_idx = target_path.rfind(".")
    if last_dot_idx == -1:
        msg = f"Invalid weight target path: {target_path}"
        raise ValueError(msg)

    module_name = target_path[:last_dot_idx]
    param_name = target_path[last_dot_idx + 1 :]

    return module_name, param_name


def _get_weight_module(
    modules: Mapping[str, torch.nn.Module],
    module_name: str,
) -> torch.nn.Module:
    """Get a module by name from the modules mapping.

    Args:
        modules: Mapping of module names to modules
        module_name: Dot-separated module name (e.g., "conv", "layer1.0")

    Returns:
        The module with the given name

    Raises:
        ValueError: If module not found

    """
    weight_module = modules.get(module_name)
    if weight_module is None:
        msg = f"Module {module_name} not found in model"
        raise ValueError(msg)

    return weight_module


def _register_quantization_buffers(
    model: torch.fx.GraphModule,
    base_name: str,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None = None,
    quantized_data: torch.Tensor | None = None,
    minval: torch.Tensor | None = None,
) -> dict[str, str | None]:
    """
    Register quantization parameter buffers in the model and return buffer names.

    Args:
        model: The graph module to register buffers in
        base_name: Base name for the buffers (e.g., parameter name)
        scale: Scale tensor to register
        zero_point: Optional zero_point tensor
        quantized_data: Optional quantized data tensor
        minval: Optional minval tensor (used by the MINVAL formulation)

    Returns:
        Dictionary with the created buffer names
    """
    buffer_names = {
        "scale": f"{base_name}_scale",
        "quantized_data": f"{base_name}_quantized",
        "zero_point": f"{base_name}_zero_point",
        "minval": f"{base_name}_minval",
    }

    model.register_buffer(buffer_names["scale"], scale)

    if zero_point is not None:
        model.register_buffer(buffer_names["zero_point"], zero_point)

    if quantized_data is not None:
        model.register_buffer(buffer_names["quantized_data"], quantized_data)

    if minval is not None:
        model.register_buffer(buffer_names["minval"], minval)

    return buffer_names


def _process_mlir_weight_quantization(
    model: torch.fx.GraphModule,
    node: Node,
    fake_quant_mod: FakeQuantizeImplBase,
) -> None:
    """
    Process weight quantization by replacing fake quantization with MLIR operations.

    Args:
        model: The graph module being modified
        node: The fake quantization node to replace
        fake_quant_mod: The fake quantization module
    """

    def _import_coreai_custom_ops():
        import coreai_torch._compression.custom_layers  # noqa: PLC0415, F401
        from torch.ops import coreai  # noqa: PLC0415

        return coreai

    coreai = lazy_import_coreai_torch(_import_coreai_custom_ops)

    if not node.args:
        raise ValueError(f"Node {node} has no input arguments")

    input_node = node.args[0]
    if input_node.op != "get_attr":
        raise ValueError(f"Expected get_attr node for weight quantization, got {input_node.op}")

    # Extract and prepare quantization parameters
    scale, zero_point, minval = _extract_quantization_params(fake_quant_mod)
    # Cast scale and minval to appropriate dtype for MLIR backend inference
    _compute_dtype_for_export = fake_quant_mod.qparams_calculator._compute_dtype_for_export
    scale = scale.to(dtype=_compute_dtype_for_export)
    if minval is not None:
        minval = minval.to(dtype=_compute_dtype_for_export)

    # Construct quantized weights
    dense_weight = resolve_attr(model, input_node.target).data
    quantized_data = fake_quant_mod.quantize(dense_weight, scale, zero_point, minval)

    # Drop one of the offsets so that the export
    # module / runtime selects the right dequant path.
    zero_point, minval = _select_export_qparams_by_formulation(fake_quant_mod, zero_point, minval)

    # Convert to export dtypes for MLIR ops
    if _is_float4_dtype(fake_quant_mod.dtype):
        _validate_fp4_export(fake_quant_mod, quantized_data)
        quantized_data = _pack_fp4_to_float4tensor(quantized_data)
    if fake_quant_mod.qparams_calculator.scale_dtype == torch.float8_e8m0fnu:
        scale = scale.to(torch.float8_e8m0fnu)

    # Register buffers and get buffer names
    param_name = str(input_node.target).replace(".", "_")
    buffer_names = _register_quantization_buffers(
        model, param_name, scale, zero_point, quantized_data, minval
    )

    # Create graph nodes and replace fake quantization
    with model.graph.inserting_before(node):
        quantized_data_node = model.graph.get_attr(buffer_names["quantized_data"])
        scale_node = model.graph.get_attr(buffer_names["scale"])

        if zero_point is not None:
            zp_node = model.graph.get_attr(buffer_names["zero_point"])
        else:
            zp_node = None

        if minval is not None:
            minval_node = model.graph.get_attr(buffer_names["minval"])
        else:
            minval_node = None

        # Pass input_dtype for integer quantization
        # needed for determining n_bits for subbyte (eg. int4) quantization
        input_dtype = fake_quant_mod.dtype if not fake_quant_mod.dtype.is_floating_point else None
        args = (quantized_data_node, scale_node, zp_node, minval_node, input_dtype)

        kwargs = {}
        if fake_quant_mod.qparams_calculator.scale_dtype == torch.float8_e8m0fnu:
            kwargs["output_dtype"] = _compute_dtype_for_export

        new_node = model.graph.call_function(coreai.constexpr_blockwise_shift_scale, args, kwargs)
        node.replace_all_uses_with(new_node)

    model.graph.erase_node(node)
    remove_fake_quant_module(model, node)


def _process_mlir_activation_quantization(
    model: torch.fx.GraphModule,
    node: Node,
    fake_quant_mod: FakeQuantizeImplBase,
) -> None:
    """
    Process activation quantization by replacing fake quantization with MLIR operations.

    Args:
        model: The graph module being modified
        node: The fake quantization node to replace
        fake_quant_mod: The fake quantization module
    """
    if _is_float4_dtype(fake_quant_mod.dtype):
        raise ValueError("FP4 activation quantization is not supported for MLIR export.")

    def _import_coreai_custom_ops():
        import coreai_torch._compression.custom_layers  # noqa: PLC0415, F401
        from torch.ops import coreai  # noqa: PLC0415

        return coreai

    coreai = lazy_import_coreai_torch(_import_coreai_custom_ops)

    if not node.args:
        raise ValueError(f"Node {node} has no input arguments")

    # Extract and prepare quantization parameters
    scale, zero_point, minval = _extract_quantization_params(fake_quant_mod)

    # Drop the offset the active formulation doesn't consume so the runtime op
    # selects the right dequant path.
    zero_point, minval = _select_export_qparams_by_formulation(fake_quant_mod, zero_point, minval)

    # Cast scale and minval to appropriate dtype for MLIR backend inference
    _compute_dtype_for_export = fake_quant_mod.qparams_calculator._compute_dtype_for_export
    scale = scale.to(dtype=_compute_dtype_for_export)
    if minval is not None:
        minval = minval.to(dtype=_compute_dtype_for_export)

    if fake_quant_mod.qparams_calculator.scale_dtype == torch.float8_e8m0fnu:
        scale = scale.to(torch.float8_e8m0fnu)

    # Canonicalize scale/zero_point/minval to 0-D (per-tensor) or 1-D (per-channel)
    granularity = fake_quant_mod.granularity
    scale = _canonicalize_qparam_shape(scale, granularity)
    if zero_point is not None:
        zero_point = _canonicalize_qparam_shape(zero_point, granularity)
    if minval is not None:
        minval = _canonicalize_qparam_shape(minval, granularity)

    # Register buffers and get buffer names
    base_name = node.name.replace(".", "_")
    buffer_names = _register_quantization_buffers(
        model, base_name, scale, zero_point, minval=minval
    )

    # Use non-negative axis for export (None for per-tensor)
    axis = fake_quant_mod.qparams_calculator._resolved_axis

    # Determine output_dtype for dequantize (needed for FP8 when scale is float8_e8m0fnu)
    if fake_quant_mod.qparams_calculator.scale_dtype == torch.float8_e8m0fnu:
        dequant_output_dtype = _compute_dtype_for_export
    else:
        dequant_output_dtype = None

    # Create graph nodes and replace fake quantization
    with model.graph.inserting_before(node):
        scale_node = model.graph.get_attr(buffer_names["scale"])

        zp_node = None
        if zero_point is not None:
            zp_node = model.graph.get_attr(buffer_names["zero_point"])

        minval_node = None
        if minval is not None:
            minval_node = model.graph.get_attr(buffer_names["minval"])

        # coreai.quantize(input, scale, output_dtype, zero_point=, minval=, axis=)
        quant_args = (node.args[0], scale_node, fake_quant_mod.dtype)
        quant_kwargs = {}
        if zp_node is not None:
            quant_kwargs["zero_point"] = zp_node
        if minval_node is not None:
            quant_kwargs["minval"] = minval_node
        if axis is not None:
            quant_kwargs["axis"] = axis
        quantize_node = model.graph.call_function(coreai.quantize, quant_args, quant_kwargs)

        # coreai.dequantize(input, scale, zero_point=, minval=, axis=, input_dtype=, output_dtype=)
        dequant_args = (quantize_node, scale_node)
        dequant_kwargs = {"output_dtype": dequant_output_dtype}
        if zp_node is not None:
            # output = scale * (input - zero_point)
            dequant_kwargs["zero_point"] = zp_node
        if minval_node is not None:
            # output = scale * (input - q_min) + minval
            dequant_kwargs["minval"] = minval_node

        # input_dtype is needed for determining n_bits for subbyte (e.g. int4) quantization
        # and for deriving q_min in the MINVAL formulation.
        if not fake_quant_mod.dtype.is_floating_point:
            dequant_kwargs["input_dtype"] = fake_quant_mod.dtype
        if axis is not None:
            dequant_kwargs["axis"] = axis
        dequantize_node = model.graph.call_function(coreai.dequantize, dequant_args, dequant_kwargs)

        node.replace_all_uses_with(dequantize_node)

    model.graph.erase_node(node)
    remove_fake_quant_module(model, node)


def _process_mil_weight_quantization(
    model: torch.fx.GraphModule,
    fake_quant_node: Node,
    fake_quant_mod: FakeQuantizeImplBase,
    modules: Mapping[str, torch.nn.Module],
) -> None:
    """Process a single weight quantization node for MIL export.

    Extracts quantization metadata, registers it as buffers, and removes the fake
    quantization node from the graph.

    Args:
        model: The graph module being modified
        fake_quant_node: The fake quantization node to process
        fake_quant_mod: The fake quantization module
        modules: Mapping of module names to modules

    """
    _validate_qformulation_for_mil_export(fake_quant_mod)
    # Extract quantization parameters
    scale, zero_point, _ = _extract_quantization_params(fake_quant_mod)
    module_name, param_name = _get_weight_input_names(fake_quant_node, fake_quant_mod)

    # Get the module that owns the weight parameter
    weight_module: torch.nn.Module = _get_weight_module(modules, module_name)

    # Create and register metadata
    metadata = MILCompressionMetadata(
        param_name=param_name,
        compression_type=CompressionType.QUANTIZATION,
        quantization_n_bits=fake_quant_mod.n_bits,
        quantization_scale=scale,
        zero_point=zero_point,
    )
    metadata.register(weight_module)

    # Replace weight data with fake-quantized version so the model produces
    # quantization-aware outputs when run as a PyTorch model.
    input_node = fake_quant_node.args[0]
    weight_param = resolve_attr(model, input_node.target)
    with torch.no_grad():
        weight_param.data = fake_quant_mod(weight_param.data)

    # Bypass FakeQuantize node — the get_attr now yields fake-quantized weights
    fake_quant_node.replace_all_uses_with(input_node)
    model.graph.erase_node(fake_quant_node)

    # Remove the FakeQuantize module from the model
    remove_fake_quant_module(model, fake_quant_node)


def _process_mil_activation_quantization(
    model: torch.fx.GraphModule,
    fake_quant_node: Node,
    fake_quant_mod: FakeQuantizeImplBase,
) -> None:
    """Process activation quantization node for MIL export.

    Converts activation FakeQuantize modules to Sequential modules containing
    quantize and dequantize operations that CoreMLTools can understand.
    Buffers are stored inside the modules. By the time this runs, CoreML
    export compatibility has already been validated, so only per-tensor
    activation granularity ever reaches here.

    Args:
        model: The graph module being modified
        fake_quant_node: The fake quantization node to process
        fake_quant_mod: The fake quantization module

    """
    _validate_qformulation_for_mil_export(fake_quant_mod)

    scale, zero_point, _ = _extract_quantization_params(fake_quant_mod)
    converted_dtype, converted_zero_point = _convert_dtype_for_torch_quantize(
        fake_quant_mod.dtype,
        zero_point,
    )

    # Use non-negative axis for export (None for per-tensor)
    axis = fake_quant_mod.qparams_calculator._resolved_axis

    sequential_module = _create_mil_act_quant_seq(
        scale=scale,
        zero_point=converted_zero_point,
        dtype=converted_dtype,
        axis=axis,
    )

    # Add Sequential module to the model
    module_name = f"{MIL_ACT_QUANT_MODULE_PREFIX}{_sanitize_module_name(fake_quant_node.name)}"
    model.add_module(module_name, sequential_module)

    # Replace fake quant node with call_module to the Sequential
    input_node = _get_fake_quant_input(fake_quant_node)
    with model.graph.inserting_before(fake_quant_node):
        replacement_node = model.graph.call_module(module_name, args=(input_node,))

    fake_quant_node.replace_all_uses_with(replacement_node)
    model.graph.erase_node(fake_quant_node)
    remove_fake_quant_module(model, fake_quant_node)


def prepare_for_mil_export(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Register compression metadata as buffers for CoreML export.

    This function processes all fake quantization nodes in the model:
    - Weight quantization: Registers metadata as _COREML_/* buffers and removes nodes
    - Activation quantization: Replaces with Sequential(quantize, dequantize) modules

    Weight quantization metadata is registered in the module._COREML_/* format that
    CoreMLTools converter expects. The fake quantization nodes are removed since
    CoreMLTools reads metadata and performs quantization during conversion.

    Activation quantization is handled by creating Sequential modules containing
    separate quantize and dequantize operations. Quantization parameters (scale,
    zero_point) are stored as buffers inside the quantize module.

    IMPORTANT: This function modifies the model in place. The input model's graph
    and state_dict will be modified by adding metadata buffers, adding activation
    modules, and removing fake quantization nodes.

    Args:
        model: The quantized GraphModule containing fake quantization nodes.
            This model will be modified in place.

    Returns:
        The same GraphModule instance with compression metadata registered and
        fake quantization nodes replaced.

    Raises:
        TypeError: If model is not a torch.fx.GraphModule or metadata value
            cannot be converted to tensor
        ValueError: If model contains no fake quantization nodes or
            node processing fails

    """
    if not isinstance(model, torch.fx.GraphModule):
        msg = "Model must be a torch.fx.GraphModule"
        raise TypeError(msg)

    # Create modules dict once to avoid redundant computation
    modules: dict[str, torch.nn.Module] = dict(
        model.named_modules(remove_duplicate=False),
    )

    fake_quant_nodes = _collect_fake_quantize_nodes(model, modules)
    if not fake_quant_nodes:
        msg = "Model contains no fake quantization nodes"
        raise ValueError(msg)

    # Fail fast if model is not coreml-exportable
    for fake_quant_node, fake_quant_mod in fake_quant_nodes:
        node_id = str(fake_quant_node.target)
        if _is_weight_fake_quant(fake_quant_node, fake_quant_mod):
            validate_coreml_compatibility(
                CompressionTargetTensor.WEIGHT,
                fake_quant_mod.dtype,
                f"weight quantizer '{node_id}'",
            )
        else:
            validate_coreml_compatibility(
                CompressionTargetTensor.ACTIVATION,
                fake_quant_mod.dtype,
                f"activation quantizer '{node_id}'",
                fake_quant_mod.granularity,
            )

    # Process all fake quantization nodes
    for fake_quant_node, fake_quant_mod in fake_quant_nodes:
        if _is_weight_fake_quant(fake_quant_node, fake_quant_mod):
            # Process weight quantization - register metadata
            _process_mil_weight_quantization(
                model,
                fake_quant_node,
                fake_quant_mod,
                modules,
            )
        else:
            # Process activation quantization - convert to supported ops
            _process_mil_activation_quantization(
                model,
                fake_quant_node,
                fake_quant_mod,
            )

    # Register metadata version
    MILCompressionMetadata.register_version(model)

    model.graph.eliminate_dead_code()
    model.recompile()

    return model


def prepare_for_mlir_export(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """
    Prepare a quantized PyTorch model for Core AI export by replacing fake quantization
    with quantization custom ops.

    This function processes all fake quantization nodes in the model and replaces them
    with appropriate quantization operations:
    - Weight quantization: Uses constexpr_blockwise_shift_scale op with
                           pre-quantized weights
    - Activation quantization: Uses quantize/dequantize operation pairs

    Args:
        model: The quantized GraphModule containing fake quantization nodes

    Returns:
        The modified GraphModule with quantization operations

    Raises:
        ImportError: If coreai-torch package is not installed (required for MLIR export)
    """

    # Lazy import: coreai_torch is required for MLIR export (registers torch.ops.coreai)
    def _import_coreai_torch():
        import coreai_torch._compression.custom_layers  # noqa: PLC0415, F401

    lazy_import_coreai_torch(_import_coreai_torch)

    if not isinstance(model, torch.fx.GraphModule):
        raise TypeError("Model must be a torch.fx.GraphModule")

    fake_quant_nodes = _collect_fake_quantize_nodes(model)
    if not fake_quant_nodes:
        raise ValueError("Model contains no fake quantization nodes to convert")

    for node, fake_quant_mod in fake_quant_nodes:
        try:
            # Process based on quantization type
            if _is_weight_fake_quant(node, fake_quant_mod):
                _process_mlir_weight_quantization(model, node, fake_quant_mod)
            else:
                _process_mlir_activation_quantization(model, node, fake_quant_mod)
        except Exception as e:
            raise RuntimeError(f"Failed to process fake quantization node {node.name}: {e}") from e

    model.graph.eliminate_dead_code()
    model.recompile()

    return model


def _move_cache_dequant_to_output(
    model: torch.fx.GraphModule,
    op_type: str,
    quant_input_idx: int,
) -> torch.fx.GraphModule:
    """Relocate the dequantize on a cache-update op's input edge to its output edge.

    Run after :func:`prepare_for_mlir_export`. The fake-quant graph

        update -> coreai.quantize -> coreai.dequantize -> cache_op(x, dq, ...) -> consumer

    is rewritten in-place to

        update -> coreai.quantize -> cache_op(x, q, ...) -> coreai.dequantize -> consumer

    The cache state's dtype (placeholder ``meta['val']`` and op ``meta['val']``) is
    also retyped to the quantized dtype.

    Precondition: the cache op must commute with quantize/dequantize — i.e.
    a pure data-movement op (slicing, narrowing, copy). Arithmetic on cached
    values would silently produce a numerically wrong model.

    Args:
        model: GraphModule already processed by :func:`prepare_for_mlir_export`.
        op_type: Short op-type name as returned by ``get_node_type``
        quant_input_idx: Index of the op input that the prepare-side spec annotated.

    Returns:
        The mutated GraphModule.

    Raises:
        NotImplementedError: If the op has ``getitem`` consumers (multi-output ops
            are not supported), or the cache state cannot be located as a single
            placeholder among the op's args.
        RuntimeError: If the expected ``coreai.quantize`` -> ``coreai.dequantize``
            chain is not present on the op's input edge.
    """

    def _import_coreai_custom_ops():
        import coreai_torch._compression.custom_layers  # noqa: PLC0415, F401
        from torch.ops import coreai  # noqa: PLC0415

        return coreai

    coreai = lazy_import_coreai_torch(_import_coreai_custom_ops)

    rewrites = 0
    for op_node in model.graph.nodes:
        if op_node.op != "call_function":
            continue
        if _get_node_type(op_node, warn_on_failure=False) != op_type:
            continue

        if len(op_node.all_input_nodes) <= quant_input_idx:
            raise RuntimeError(
                f"Op {op_node.name} has {len(op_node.all_input_nodes)} input nodes; "
                f"quant_input_idx={quant_input_idx} is out of range."
            )
        dq_node = op_node.all_input_nodes[quant_input_idx]
        if not (isinstance(dq_node, Node) and dq_node.target is coreai.dequantize):
            raise RuntimeError(
                f"Expected coreai.dequantize on input {quant_input_idx} of "
                f"{op_node.name}, found {dq_node!r} "
                f"(target={getattr(dq_node, 'target', None)!r})."
            )
        q_node = dq_node.args[0]
        if not (isinstance(q_node, Node) and q_node.target is coreai.quantize):
            raise RuntimeError(
                f"Expected coreai.quantize feeding {dq_node.name}, found {q_node!r}."
            )

        getitem_users = [
            u for u in op_node.users if u.op == "call_function" and u.target is operator.getitem
        ]
        if getitem_users:
            raise NotImplementedError(
                f"Op {op_node.name} has getitem consumers "
                f"({[u.name for u in getitem_users]}), indicating a multi-output op. "
                "Relocation for multi-output cache ops is not implemented yet."
            )

        original_consumers = list(op_node.users)
        quantized_dtype = q_node.args[2]
        if not isinstance(quantized_dtype, torch.dtype):
            raise RuntimeError(
                f"Expected coreai.quantize's dtype arg (args[2]) to be a torch.dtype; "
                f"got {type(quantized_dtype).__name__}: {quantized_dtype!r}."
            )

        # Find the cache placeholder among the op's args.
        cache_placeholders = [
            a for a in op_node.args if isinstance(a, Node) and a.op == "placeholder"
        ]
        if len(cache_placeholders) != 1:
            raise NotImplementedError(
                f"Expected exactly one placeholder among {op_node.name}'s args; "
                f"found {len(cache_placeholders)}. Cannot determine which input "
                "is the cache state."
            )
        cache_placeholder = cache_placeholders[0]

        op_node.replace_input_with(dq_node, q_node)
        with model.graph.inserting_after(op_node):
            new_dq = model.graph.call_function(
                coreai.dequantize, (op_node, *dq_node.args[1:]), dict(dq_node.kwargs)
            )
        # new_dq outputs the original compute dtype — which is what
        # op_node.meta["val"] still holds at this point. Clone so any later
        # in-place mutation of one meta["val"] doesn't bleed into the other.
        if "val" in op_node.meta:
            new_dq.meta["val"] = op_node.meta["val"].clone()
            op_node.meta["val"] = op_node.meta["val"].to(quantized_dtype)
        for consumer in original_consumers:
            consumer.replace_input_with(op_node, new_dq)

        if "val" in cache_placeholder.meta:
            cache_placeholder.meta["val"] = cache_placeholder.meta["val"].to(quantized_dtype)

        rewrites += 1

    if rewrites == 0:
        raise RuntimeError(
            f"_move_cache_dequant_to_output found no nodes matching {op_type!r}; "
            "the dequantize was not relocated to the cache-op output and the "
            "cache buffer was not retyped — the deployed model will not have "
            "stored cache quantization. The op was present at prepare time, so "
            "this likely indicates a graph mutation between prepare and finalize."
        )

    model.graph.eliminate_dead_code()
    model.recompile()
    return model
