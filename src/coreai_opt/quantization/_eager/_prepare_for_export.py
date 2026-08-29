# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import logging
from collections import OrderedDict
from os import PathLike
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P

from coreai_opt._utils.export_utils import clear_parametrization_original, prepare_mmap_dir
from coreai_opt._utils.import_utils import lazy_import_coreai_torch
from coreai_opt._utils.torch_utils import (
    get_parent_module_and_attr_name,
    is_float4_dtype,
    mmap_module_state_dict,
)
from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.quantization._export_utils import (
    dequant_output_dtype,
    extract_export_qparams,
    extract_quantization_params,
    get_activation_export_handler,
    pack_fp4_to_float4tensor,
    select_export_qparams_by_formulation,
    validate_activation_export_supported,
    validate_fp4_export,
)
from coreai_opt.quantization.config.quantization_config import ExecutionMode
from coreai_opt.quantization.spec.fake_quantize import FakeQuantizeImplBase

logger = logging.getLogger(__name__)


def _add_weight_dequantization_parametrization(
    module: nn.Module,
    module_name: str,
    param_name: str,
    fake_quant_idx: int,
    fake_quant_mod: FakeQuantizeImplBase,
    weight_dequant_cls: type[nn.Module],
    mmap_dir: str | PathLike[str] | None,
) -> nn.Module:
    """Replace one weight's FakeQuantize parametrization with a dequantize
    parametrization; optionally persist to ``mmap_dir`` and reload via mmap.
    """

    # Extract and prepare quantization parameters
    scale, zero_point, minval = extract_quantization_params(fake_quant_mod)

    # Cast scale and minval to appropriate dtype for MLIR backend inference
    _compute_dtype_for_export = fake_quant_mod.qparams_calculator._compute_dtype_for_export
    scale = scale.to(dtype=_compute_dtype_for_export)
    if minval is not None:
        minval = minval.to(dtype=_compute_dtype_for_export)

    dense_weight = module.parametrizations[param_name].original.detach()
    quantized_data = fake_quant_mod.quantize(dense_weight, scale, zero_point, minval)

    # Drop one of the offsets so that the export
    # module / runtime selects the right dequant path.
    zero_point, minval = select_export_qparams_by_formulation(fake_quant_mod, zero_point, minval)

    if is_float4_dtype(fake_quant_mod.dtype):
        validate_fp4_export(fake_quant_mod, quantized_data)
        quantized_data = pack_fp4_to_float4tensor(quantized_data)
    if fake_quant_mod.qparams_calculator.scale_dtype == torch.float8_e8m0fnu:
        output_dtype = _compute_dtype_for_export
        scale = scale.to(torch.float8_e8m0fnu)
    else:
        output_dtype = None

    # Pass input_dtype for integer quantization
    # needed for determining n_bits for subbyte (eg. int4) quantization
    input_dtype = fake_quant_mod.dtype if not fake_quant_mod.dtype.is_floating_point else None

    weight_dequant_mod = weight_dequant_cls(
        quantized_data=quantized_data,
        scale=scale,
        zero_point=zero_point,
        minval=minval,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
    )

    if mmap_dir is not None:
        stem = f"{module_name}.{param_name}" if module_name else param_name
        path = Path(mmap_dir) / f"{stem}.safetensors"
        mmap_module_state_dict(weight_dequant_mod, path)

    module.parametrizations[param_name][fake_quant_idx] = weight_dequant_mod
    clear_parametrization_original(module, param_name)
    return weight_dequant_mod


def _process_weight_quantization(model: nn.Module, mmap_dir: str | PathLike[str] | None = None):
    """
    Replace FakeQuantizeImplBase parametrizations on weights with
    WeightDequantizedParametrization.

    This function iterates through all parametrized modules in the model, finds those
    with FakeQuantizeImplBase parametrizations, extracts their quantization parameters,
    and replaces them with WeightDequantizedParametrization modules
    that contain the quantized weights.

    Args:
        model (nn.Module): The prepared quantized model to finalize in-place.
        mmap_dir (str | None): If provided, serialize each finalized
            WeightDequantizedParametrization to a safetensors file under this
            directory and reload it via mmap, so large-model finalization does
            not hold full quantized weights in RAM.
    """
    prepare_mmap_dir(mmap_dir)

    # Lazy import: coreai_torch is required for MLIR export
    def _import_coreai_torch_modules():
        from coreai_torch._compression.custom_layers import (  # noqa: PLC0415
            WeightDequantizeModule,
        )
        from coreai_torch._compression.utils import wrap_for_parametrization  # noqa: PLC0415

        return WeightDequantizeModule, wrap_for_parametrization

    WeightDequantizeModule, wrap_for_parametrization = lazy_import_coreai_torch(
        _import_coreai_torch_modules
    )

    WeightDequantizedParametrization = wrap_for_parametrization(WeightDequantizeModule)

    # For each weight fake quant module, create corresponding MLIR module
    # and replace the weight fake quant module with it
    fq_id_to_dequant_mod: dict[int, nn.Module] = {}

    for module_name, module in model.named_modules():
        if not P.is_parametrized(module):
            continue
        for param_name, parametrizations in list(module.parametrizations.items()):
            # Find the weight FakeQuantize parametrization, if any.
            fake_quant_idx, fake_quant_mod = next(
                (
                    (idx, p)
                    for idx, p in enumerate(parametrizations)
                    if isinstance(p, FakeQuantizeImplBase)
                    and p.quantization_target == CompressionTargetTensor.WEIGHT
                ),
                (None, None),
            )
            if fake_quant_idx is None:
                continue

            cached = fq_id_to_dequant_mod.get(id(fake_quant_mod))
            if cached is not None:
                # Shared FakeQuantize across modules (e.g. weight tying): reuse
                # the same dequant module so post-finalize sharing is preserved.
                module.parametrizations[param_name][fake_quant_idx] = cached
                clear_parametrization_original(module, param_name)
                continue

            weight_dequant_mod = _add_weight_dequantization_parametrization(
                module=module,
                module_name=module_name,
                param_name=param_name,
                fake_quant_idx=fake_quant_idx,
                fake_quant_mod=fake_quant_mod,
                weight_dequant_cls=WeightDequantizedParametrization,
                mmap_dir=mmap_dir,
            )
            fq_id_to_dequant_mod[id(fake_quant_mod)] = weight_dequant_mod


def _build_activation_replacement_module(
    fake_quant_module: FakeQuantizeImplBase,
) -> nn.Module:
    """
    Build the module that replaces one activation fake quantization module.

    Mirrors the graph path's ``_process_mlir_activation_quantization``: a registered
    handler for the granularity takes precedence, and the built-in axis-based path is
    the fallback. Only the shape differs -- graph mode rewrites the node in place,
    while this returns a module for the caller to swap in.

    Args:
        fake_quant_module (FakeQuantizeImplBase): The activation fake quantization
            module being replaced.

    Returns:
        nn.Module: The replacement, a ``Sequential`` of a quantize and a dequantize
            module for the built-in path, or whatever a registered handler returns.
    """
    # A registered handler owns this granularity end to end, including its own
    # dtype and shape validation, and returns the replacement module; the
    # caller's swap is shared.
    handler = get_activation_export_handler(fake_quant_module.granularity, ExecutionMode.EAGER)
    if handler is not None:
        return handler(fake_quant_module)  # type: ignore[call-arg,return-value]

    validate_activation_export_supported(fake_quant_module)

    # Lazy import: coreai_torch is required for MLIR export
    def _import_coreai_torch_modules():
        from coreai_torch._compression.custom_layers import (  # noqa: PLC0415
            ActivationDequantizeModule,
            ActivationQuantizeModule,
        )

        return ActivationDequantizeModule, ActivationQuantizeModule

    ActivationDequantizeModule, ActivationQuantizeModule = lazy_import_coreai_torch(
        _import_coreai_torch_modules
    )

    scale, zero_point, minval = extract_export_qparams(fake_quant_module)

    axis = fake_quant_module.qparams_calculator._resolved_axis
    axis = axis if axis is not None else 0

    # Create the replacement module sequence
    # First: ActivationQuantizeParametrization
    quant_module = ActivationQuantizeModule(
        scale=scale,
        output_dtype=fake_quant_module.dtype,
        zero_point=zero_point.clone() if zero_point is not None else None,
        minval=minval.clone() if minval is not None else None,
        axis=axis,
    )

    # Pass input_dtype for integer quantization
    # needed for determining n_bits for subbyte (eg. int4) quantization
    input_dtype = fake_quant_module.dtype if not fake_quant_module.dtype.is_floating_point else None

    # Second: ActivationDequantizeParametrization
    dequant_module = ActivationDequantizeModule(
        scale=scale.clone(),  # make one more copy for dequantize module buffers
        zero_point=zero_point.clone() if zero_point is not None else None,
        minval=minval.clone() if minval is not None else None,
        axis=axis,
        input_dtype=input_dtype,
        output_dtype=dequant_output_dtype(fake_quant_module),
    )

    # Create a sequential module to combine both operations
    return nn.Sequential(
        OrderedDict(
            [
                ("quantize", quant_module),
                ("dequantize", dequant_module),
            ]
        )
    )


def _process_activation_quantization(model: nn.Module):
    """
    Replace FakeQuantizeImplBase modules with ActivationQuantizeParametrization
    followed by ActivationDequantizeParametrization.
    """
    modules_to_replace = []

    # Collect modules that need to be replaced
    # If there are duplicated fake quant modules they should ideally have duplicated
    # parent modules as well, unless manually inserted. Since we don't yet support
    # manual insertion, we aren't handling duplicates separately (remove_duplicate=True)
    #
    # Collect before mutating: named_modules() walks the live tree, so replacing a
    # module mid-iteration would invalidate the traversal.
    for name, module in list(model.named_modules(remove_duplicate=True)):
        if isinstance(module, FakeQuantizeImplBase) and module.quantization_target in (
            CompressionTargetTensor.ACTIVATION,
        ):
            modules_to_replace.append((name, module))

    # Replace each FakeQuantizeImplBase module
    for name, fake_quant_module in modules_to_replace:
        replacement_module = _build_activation_replacement_module(fake_quant_module)
        parent_module, attr_name = get_parent_module_and_attr_name(model, name)
        setattr(parent_module, attr_name, replacement_module)


def prepare_for_mlir_export(
    model: nn.Module, mmap_dir: str | PathLike[str] | None = None
) -> nn.Module:
    """
    Prepare a quantized PyTorch model for Core AI export by replacing fake quantization
    parametrizations and modules with corresponding quantization custom ops.

    Args:
        model (nn.Module): The PyTorch model to be prepared for export
        mmap_dir (str | None): If provided, serialize finalized quantized weights
            to safetensors files under this directory and re-load them via mmap,
            so large-model finalization does not hold full quantized weights in RAM.
            The files in ``mmap_dir`` must remain in place for the lifetime of
            the returned model; removing them invalidates the mmap-backed weights.

    Returns:
        nn.Module: The input model that has now been modified for Core AI export

    Raises:
        ImportError: If coreai-torch package is not installed (required for MLIR export)

    Note:
        This function frees the dense pre-quantization weights in place: on each
        parametrized weight, ``parametrizations[...].original`` is replaced with
        a zero-size placeholder so its storage can be released.
    """
    # Replace FakeQuantize weight parametrizations with MLIR weight dequantize module
    _process_weight_quantization(model, mmap_dir=mmap_dir)

    # Replace FakeQuantize modules with MLIR quantize dequantize modules
    _process_activation_quantization(model)

    return model
