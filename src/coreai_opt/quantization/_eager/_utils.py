# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from collections.abc import Mapping

import torch.nn as nn

from coreai_opt._utils.insertion.torch_function.registered_optimizers_tracker import (
    RegisteredOptimizersTracker,
)
from coreai_opt._utils.torch_utils import (
    get_parent_module_and_attr_name,
    remove_compression_parametrizations,
)
from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.quantization.spec.fake_quantize import FakeQuantizeImplBase


def remove_act_fq_from_reference_tracker(
    disabled_fq: Mapping[str, FakeQuantizeImplBase],
    tracker: RegisteredOptimizersTracker,
) -> None:
    """Remove activation FQ optimizer names from a reference tracker.

    For each disabled activation FakeQuantize module, determines its parent
    module name (the tracker key) and attribute name (the optimizer name),
    then scrubs that name from only the corresponding module's registrations
    in the tracker.

    Args:
        disabled_fq (Mapping[str, FakeQuantizeImplBase]): Mapping of fully
            qualified module name to disabled FakeQuantize module instance.
        tracker (RegisteredOptimizersTracker): The reference tracker to
            update.

    """
    # Group attr_names by their parent module name (the tracker key)
    per_module: dict[str, set[str]] = {}
    for name, module in disabled_fq.items():
        if module.quantization_target != CompressionTargetTensor.ACTIVATION:
            continue
        if "." in name:
            parent_name, attr_name = name.rsplit(".", 1)
        else:
            parent_name = ""
            attr_name = name
        per_module.setdefault(parent_name, set()).add(attr_name)

    for module_name, attr_names in per_module.items():
        tracker.remove_optimizer_names(module_name, attr_names)


def remove_fake_quant_modules(
    model: nn.Module,
    modules_to_remove: set[FakeQuantizeImplBase],
) -> int:
    """Remove specified FakeQuantize modules from a model.

    Handles both weight parametrizations and standalone activation modules.
    For weight FQ: removes the parametrization from the module.
    For activation FQ: deletes the module from its parent.

    Args:
        model (nn.Module): The model to modify.
        modules_to_remove (set[FakeQuantizeImplBase]): Set of FakeQuantize
            module instances to remove.

    Returns:
        int: Number of modules removed.

    """
    num_removed = remove_compression_parametrizations(model, modules_to_remove)

    # Remove activation FQ standalone modules
    ids_to_remove = {id(m) for m in modules_to_remove}
    for name, module in list(model.named_modules(remove_duplicate=True)):
        if (
            isinstance(module, FakeQuantizeImplBase)
            and module.quantization_target == CompressionTargetTensor.ACTIVATION
            and id(module) in ids_to_remove
        ):
            parent_module, attr_name = get_parent_module_and_attr_name(model, name)
            delattr(parent_module, attr_name)
            num_removed += 1

    return num_removed
