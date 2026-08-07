# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Record which model tensor each weight fake-quantize module quantizes.

``FakeQuantizeImplBase`` warns and disables itself from inside ``forward`` when a
tensor is incompatible with the configured block size, but a module cannot
discover its own position in the model at that point. The passes here run during
``prepare()``, before the first forward pass, and stamp each weight
fake-quantize module with the FQN of the parameter it quantizes so the warning
can name the offending weight.

Weights only: activation fake-quantize modules have no backing parameter, so
they keep the unnamed warning.

Like :mod:`coreai_opt.quantization._axis_defaults`, this module deliberately
implements its own graph and eager walks instead of reusing the helpers in
``_graph``/``_eager``. Both mode-specific quantizers import it, so importing
either subpackage from here would create a circular import.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.utils.parametrize as P
from torch.fx import GraphModule

from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.quantization.spec.fake_quantize import FakeQuantizeImplBase


def record_weight_source_names_graph(model: GraphModule) -> None:
    """Stamp weight fake-quantize modules in a graph-mode ``GraphModule``.

    A weight fake-quantize node takes its value from the ``get_attr`` node for
    the parameter, whose target is the dotted parameter FQN (e.g.
    ``"layer1.0.weight"``), so the name needs no inference. This mirrors
    ``_graph._prepare_for_export._get_weight_input_names``.

    Args:
        model (GraphModule): The prepared graph-mode ``GraphModule``.
    """
    modules = dict(model.named_modules(remove_duplicate=False))

    for node in model.graph.nodes:
        if node.op != "call_module":
            continue
        fake_quant = modules.get(str(node.target))
        if not isinstance(fake_quant, FakeQuantizeImplBase):
            continue
        if fake_quant.quantization_target != CompressionTargetTensor.WEIGHT:
            continue

        # Activation fake-quantize nodes read from an op rather than a parameter.
        input_node = node.args[0]
        if input_node.op != "get_attr":
            continue

        # "layer1.0.weight" -> ("layer1.0", "weight"); a root-module parameter
        # such as "weight" has no module part.
        target_path = str(input_node.target)
        module_name, _, param_name = target_path.rpartition(".")
        fake_quant.set_source_name(module_name, param_name)


def record_weight_source_names_eager(model: nn.Module) -> None:
    """Stamp weight fake-quantize modules in an eager-mode model.

    Weight fake-quantize modules live in the ``ParametrizationList`` registered
    for the parameter they quantize, so the owning module name and the parameter
    name both come straight from ``named_modules()``.

    Args:
        model (nn.Module): The prepared eager-mode model.
    """
    for module_name, module in model.named_modules(remove_duplicate=True):
        if not P.is_parametrized(module):
            continue
        for param_name, parametrizations in module.parametrizations.items():
            for fake_quant in parametrizations:
                if not isinstance(fake_quant, FakeQuantizeImplBase):
                    continue
                if fake_quant.quantization_target != CompressionTargetTensor.WEIGHT:
                    continue
                fake_quant.set_source_name(module_name, param_name)
