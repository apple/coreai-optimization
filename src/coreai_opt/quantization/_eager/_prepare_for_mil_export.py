# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""MIL export preparation for eager mode quantized models.

This module provides functions to prepare eager mode quantized models for MIL
(Model Intermediate Language) export by extracting quantization metadata and
converting fake quantization operations to CoreMLTools-compatible representations.
"""

from __future__ import annotations

from typing import Any

import torch.nn.utils.parametrize as P
from torch import nn

from coreai_opt._utils.export_utils import validate_coreml_compatibility
from coreai_opt._utils.metadata_utils import CompressionType, MILCompressionMetadata
from coreai_opt._utils.torch_utils import get_parent_module_and_attr_name
from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.quantization._export_utils import (
    convert_dtype_for_torch_quantize,
    create_mil_act_quant_seq,
    extract_quantization_params,
    is_module_fake_quant_target,
    validate_qformulation_for_mil_export,
)
from coreai_opt.quantization.spec.fake_quantize import FakeQuantizeImplBase


def _process_weight_quantization(
    module: nn.Module,
    param_name: str,
    fake_quant_param: FakeQuantizeImplBase,
) -> None:
    """Process weight quantization by extracting metadata and removing parametrization.

    Args:
        module: The module containing the parametrized weight
        param_name: The name of the parametrized parameter (e.g., "weight", "bias")
        fake_quant_param: The fake quantization parametrization to process

    """
    validate_qformulation_for_mil_export(fake_quant_param)
    scale, zero_point, _ = extract_quantization_params(fake_quant_param)

    # Remove parametrization but keep fake-quantized weights
    P.remove_parametrizations(
        module,
        param_name,
        leave_parametrized=True,
    )

    metadata = MILCompressionMetadata(
        param_name=param_name,
        compression_type=CompressionType.QUANTIZATION,
        quantization_n_bits=fake_quant_param.n_bits,
        quantization_scale=scale,
        zero_point=zero_point,
    )
    metadata.register(module)


def _mark_if_not_already_processed(obj: Any, processed_ids: set[int]) -> bool:
    """Mark object as processed and return whether it was already processed.

    Args:
        obj: Object to check and mark (typically a fake quantizer)
        processed_ids: Set of IDs of already processed objects

    Returns:
        True if already processed before marking, False if newly added

    """
    obj_id = id(obj)
    if obj_id in processed_ids:
        return True
    processed_ids.add(obj_id)
    return False


def _process_activation_quantization(
    parent_module: nn.Module,
    attr_name: str,
    fake_quant_mod: FakeQuantizeImplBase,
) -> None:
    """Process activation quantization by replacing with Sequential quantize/dequantize.

    By the time this runs, CoreML export compatibility has already been
    validated, so only per-tensor activation granularity ever reaches here.

    Args:
        parent_module: The parent module containing the fake quantizer
        attr_name: The attribute name of the fake quantizer in the parent
        fake_quant_mod: The fake quantization module to replace

    """
    validate_qformulation_for_mil_export(fake_quant_mod)

    scale, zero_point, _ = extract_quantization_params(fake_quant_mod)
    converted_dtype, converted_zero_point = convert_dtype_for_torch_quantize(
        fake_quant_mod.dtype,
        zero_point,
    )

    # Use non-negative axis for export (None for per-tensor)
    axis = fake_quant_mod.qparams_calculator._resolved_axis

    replacement_module = create_mil_act_quant_seq(
        scale=scale,
        zero_point=converted_zero_point,
        dtype=converted_dtype,
        axis=axis,
    )

    setattr(parent_module, attr_name, replacement_module)


def prepare_for_mil_export(model: nn.Module) -> nn.Module:
    """Register compression metadata as buffers for CoreML export.

    This function processes eager mode quantized models by:
    1. Processing weight quantization: extracting metadata, removing parametrizations.
    2. Processing activation quantization: replacing fake quantizers with
       Sequential(quantize, dequantize).
    3. Registering metadata version buffer.

    Args:
        model: The quantized model containing fake quantization components

    Returns:
        The same model instance with compression metadata registered and
        fake quantization components replaced with coremltool supported ops

    """
    processed_fq_ids: set[int] = set()

    # Fail fast if model is not coreml-exportable
    for module_name, module in model.named_modules():
        if P.is_parametrized(module):
            for param_name, parametrizations in module.parametrizations.items():
                for p in parametrizations:
                    if is_module_fake_quant_target(p, CompressionTargetTensor.WEIGHT):
                        validate_coreml_compatibility(
                            CompressionTargetTensor.WEIGHT,
                            p.dtype,
                            f"weight '{param_name}' of module '{module_name}'",
                        )
        if is_module_fake_quant_target(module, CompressionTargetTensor.ACTIVATION):
            validate_coreml_compatibility(
                CompressionTargetTensor.ACTIVATION,
                module.dtype,
                f"activation quantizer of module '{module_name}'",
                module.granularity,
            )

    for name, module in list(model.named_modules()):
        # Handle weight quantization parametrizations
        if P.is_parametrized(module):
            for param_name, parametrizations in list(module.parametrizations.items()):
                for p in parametrizations:
                    if not is_module_fake_quant_target(p, CompressionTargetTensor.WEIGHT):
                        continue

                    if _mark_if_not_already_processed(p, processed_fq_ids):
                        continue

                    _process_weight_quantization(module, param_name, p)

        # Handle activation quantization modules
        if is_module_fake_quant_target(module, CompressionTargetTensor.ACTIVATION):
            if _mark_if_not_already_processed(module, processed_fq_ids):
                continue

            parent_module, attr_name = get_parent_module_and_attr_name(model, name)
            _process_activation_quantization(parent_module, attr_name, module)

    # Register metadata version
    MILCompressionMetadata.register_version(model)

    return model
