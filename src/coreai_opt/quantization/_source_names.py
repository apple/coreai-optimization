# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Record which parameter each weight fake-quantize module quantizes.

``FakeQuantizeImplBase`` warns and disables itself from inside ``forward``, where a
module cannot know its own position in the model. These passes run during
``prepare()``, before the first forward, so the warning can name the weight.

Like :mod:`coreai_opt.quantization._axis_defaults`, this module implements its own
graph and eager walks rather than reusing the ``_graph``/``_eager`` helpers: both
mode-specific quantizers import it, so importing either subpackage here would be a
circular import.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.utils.parametrize as P
from torch.fx import GraphModule

from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.quantization.spec.fake_quantize import FakeQuantizeImplBase


def record_weight_source_names_graph(model: GraphModule) -> None:
    """Stamp weight fake-quantize modules in a graph-mode ``GraphModule``.

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

        # An already-compressed weight reaches the fake quantize through a
        # decompression op (e.g. coreai.lut_to_dense) instead of a get_attr, and
        # then carries no parameter name. See is_coreai_compressed_state_node.
        input_node = node.args[0]
        if input_node.op != "get_attr":
            continue

        # A get_attr target is the dotted parameter FQN: "layer1.0.weight" ->
        # ("layer1.0", "weight"). A root-module parameter has no module part.
        module_name, _, param_name = str(input_node.target).rpartition(".")
        fake_quant.set_source_name(module_name, param_name)


def record_weight_source_names_eager(model: nn.Module) -> None:
    """Stamp weight fake-quantize modules in an eager-mode model.

    Args:
        model (nn.Module): The prepared eager-mode model.
    """
    for module_name, module in model.named_modules(remove_duplicate=True):
        if not P.is_parametrized(module):
            continue
        # A weight fake quantize lives in the ParametrizationList of the
        # parameter it quantizes.
        for param_name, parametrizations in module.parametrizations.items():
            for fake_quant in parametrizations:
                if not isinstance(fake_quant, FakeQuantizeImplBase):
                    continue
                if fake_quant.quantization_target != CompressionTargetTensor.WEIGHT:
                    continue
                fake_quant.set_source_name(module_name, param_name)
