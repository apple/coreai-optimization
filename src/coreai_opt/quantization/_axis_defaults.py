# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Axis defaults and validation for quantization granularities.

Provides default weight axes for per-channel (output channel) and per-block
(input channel) granularity, with mode-specific entry points for graph and
eager workflows. Also validates that activation fake-quantize modules have
explicit axes when required.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
from torch.fx import GraphModule

from coreai_opt._utils.torch_utils import ATEN_OP_TO_MODULE_TYPE
from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.quantization.spec.fake_quantize import FakeQuantizeImplBase
from coreai_opt.quantization.spec.granularity import (
    PerBlockGranularity,
    PerChannelGranularity,
    QuantizationGranularity,
)

logger = logging.getLogger(__name__)

# Maps each weight FQ to the ops/modules that consume it.
# Used to look up default axes.
_ConsumerInfo = tuple[type[nn.Module] | torch._ops.OpOverload | None, str]
_WeightFQMap = defaultdict[FakeQuantizeImplBase, list[_ConsumerInfo]]


@dataclass(frozen=True)
class _WeightAxisSpec:
    """Default axis indices for a single ``nn.Module`` type.

    Attributes:
        per_channel_axis (int): Output-channel axis for per-channel granularity.
        per_block_axis (int): Input-channel (reduction) axis for per-block granularity.
    """

    per_channel_axis: int
    per_block_axis: int

    def default_axis_for(self, granularity: QuantizationGranularity) -> int | None:
        """Return the default axis for the given granularity, or ``None`` if not applicable."""
        if isinstance(granularity, PerChannelGranularity):
            return self.per_channel_axis
        if isinstance(granularity, PerBlockGranularity):
            return self.per_block_axis
        return None


# Conv weights are [out_ch, in_ch, ...]; ConvTranspose weights are [in_ch, out_ch, ...].
# Embedding weights are [num_embeddings, embedding_dim].
_WEIGHT_AXIS_SPECS: dict[type[nn.Module], _WeightAxisSpec] = {
    nn.Conv1d: _WeightAxisSpec(0, 1),
    nn.Conv2d: _WeightAxisSpec(0, 1),
    nn.Conv3d: _WeightAxisSpec(0, 1),
    nn.ConvTranspose1d: _WeightAxisSpec(1, 0),
    nn.ConvTranspose2d: _WeightAxisSpec(1, 0),
    nn.ConvTranspose3d: _WeightAxisSpec(1, 0),
    nn.Linear: _WeightAxisSpec(0, 1),
    nn.Embedding: _WeightAxisSpec(0, 1),
}


def _granularity_needs_axis_default(granularity: QuantizationGranularity) -> bool:
    """Return True if granularity has an unresolved ``axis=None`` that needs a default."""
    if isinstance(granularity, PerChannelGranularity):
        return granularity.axis is None
    if isinstance(granularity, PerBlockGranularity):
        # Tuple block_size (multi-axis mode) expects axis=None. Only scalar
        # block_size requires an explicit axis that we can default
        return isinstance(granularity.block_size, int) and granularity.axis is None
    return False


def _resolve_axis_on_fake_quantize(
    fake_quant: FakeQuantizeImplBase,
    default_axis: int,
    name: str,
) -> None:
    """Set the default weight axis on a fake-quantize module's granularity.

    The caller must ensure the granularity needs an axis default before
    calling this function.

    Args:
        fake_quant (FakeQuantizeImplBase): The fake-quantize module to update.
        default_axis (int): The default axis value to set.
        name (str): Module name used for debug logging.
    """
    fake_quant.granularity = fake_quant.granularity.model_copy(
        update={"axis": default_axis},
    )
    logger.debug(f"Default weight axis applied for {name}")


def _raise_axis_default_errors(unresolved: list[str], conflicting: list[str]) -> None:
    """Raise ``ValueError`` if any weight FQ axes could not be resolved or conflicted.

    Args:
        unresolved (list[str]): Names of weight fake-quantize modules whose
            ``axis=None`` could not be resolved to a default.
        conflicting (list[str]): Names of weight fake-quantize modules whose
            consumers resolve to different default axes.

    Raises:
        ValueError: If either list is non-empty.
    """
    if not unresolved and not conflicting:
        return
    parts: list[str] = []
    if unresolved:
        names = ", ".join(sorted(unresolved))
        parts.append(
            f"Weight fake-quantize modules with unresolved axis=None remain "
            f"after applying defaults: {names}. The ops consuming these "
            f"modules do not have defaults. Provide an explicit axis value in "
            f"the granularity configuration (e.g., PerChannelGranularity(axis=0))."
        )
    if conflicting:
        names = ", ".join(sorted(conflicting))
        parts.append(
            f"Conflicting default axes for shared weight fake-quantize modules: "
            f"{names}. All consumers of a shared weight must resolve to the same "
            f"default axis. Provide an explicit axis."
        )
    raise ValueError("\n".join(parts))


def _collect_weight_fq_entries_graph(model: GraphModule) -> _WeightFQMap:
    """Collect weight fake-quantize entries from a graph-mode ``GraphModule``.

    Iterates ``call_module`` nodes to find weight ``FakeQuantizeImplBase``
    instances. For each, walks ``node.users`` to find the consuming aten op.

    Args:
        model (GraphModule): The prepared graph-mode ``GraphModule``.

    Returns:
        _WeightFQMap: Map from fake-quantize instance to its consumers.
            Each consumer is a ``(consuming_op, name)`` tuple where
            ``consuming_op`` is the aten ``OpOverload``, or ``None``
            when no consuming op is found.
    """
    modules = dict(model.named_modules(remove_duplicate=False))
    fq_map: _WeightFQMap = defaultdict(list)

    for node in model.graph.nodes:
        if node.op != "call_module":
            continue
        mod = modules.get(str(node.target))
        if not isinstance(mod, FakeQuantizeImplBase):
            continue
        if mod.quantization_target != CompressionTargetTensor.WEIGHT:
            continue

        consuming_ops = [
            user.target
            for user in node.users
            if user.op == "call_function" and isinstance(user.target, torch._ops.OpOverload)
        ]
        # preserve a None entry for raising unresolved axis
        # in case of no consuming op
        for op in consuming_ops or [
            None,
        ]:
            fq_map[mod].append((op, str(node.target)))

    return fq_map


def _collect_weight_fq_entries_eager(model: nn.Module) -> _WeightFQMap:
    """Collect weight fake-quantize entries from an eager-mode model.

    Iterates ``named_modules()`` to find modules with parametrized weights.
    For each, finds ``FakeQuantizeImplBase`` instances on
    ``module.parametrizations["weight"]`` and records the owner module's
    base type.

    Args:
        model (nn.Module): The prepared eager-mode model.

    Returns:
        _WeightFQMap: Map from fake-quantize instance to its consumers.
            Each consumer is a ``(module_type, name)`` tuple where
            ``module_type`` is the base type of the parametrized owner
            module.
    """
    fq_map: _WeightFQMap = defaultdict(list)

    for name, module in model.named_modules():
        if not P.is_parametrized(module, "weight"):
            continue

        owner_type = type(module).__bases__[0]

        for fq in module.parametrizations["weight"]:
            if not isinstance(fq, FakeQuantizeImplBase):
                continue
            if fq.quantization_target != CompressionTargetTensor.WEIGHT:
                continue
            fq_map[fq].append((owner_type, name))

    return fq_map


def _apply_defaults(fq_map: _WeightFQMap) -> None:
    """Apply default weight axes and raise on unresolved or conflicting entries.

    For each FQ in the map, resolves each consumer's identifier to a module
    type (aten ``OpOverload`` entries from graph-mode are mapped via
    ``_ATEN_OP_TO_MODULE_TYPE``), then looks up the default axis from
    ``_WEIGHT_AXIS_SPECS``. When a single FQ has multiple consumers (shared
    weight), all consumers must agree on the same default axis.

    Args:
        fq_map (_WeightFQMap): Map from fake-quantize instance to its
            consumers, produced by a mode-specific collection function.

    Raises:
        ValueError: If any FQ entries need axis defaults but have no
            default for their module type, or if consumers of a shared
            weight resolve to different default axes.
    """
    unresolved: list[str] = []
    conflicting: list[str] = []

    for fq, consumers in fq_map.items():
        if not _granularity_needs_axis_default(fq.granularity):
            continue

        # Resolve each consumer to a module type and look up its default axis
        resolved_axes: set[int] = set()
        all_consumer_names: list[str] = []
        for module_type_or_op, name in consumers:
            # graph-mode consumers are aten ops, so map them to module types first
            if isinstance(module_type_or_op, torch._ops.OpOverload):
                module_type = ATEN_OP_TO_MODULE_TYPE.get(module_type_or_op)
            else:
                module_type = module_type_or_op

            spec = _WEIGHT_AXIS_SPECS.get(module_type) if module_type is not None else None
            default_axis = spec.default_axis_for(fq.granularity) if spec is not None else None
            if default_axis is not None:
                resolved_axes.add(default_axis)

            all_consumer_names.append(name)

        # All consumers must agree on a single default axis
        if not resolved_axes:
            unresolved.extend(all_consumer_names)
        elif len(resolved_axes) == 1:
            _resolve_axis_on_fake_quantize(
                fq, next(iter(resolved_axes)), ", ".join(all_consumer_names)
            )
        else:
            # multiple possible unique resolved axes for single FQ
            conflicting.extend(all_consumer_names)

    _raise_axis_default_errors(unresolved, conflicting)


def apply_weight_axis_defaults_graph(model: GraphModule) -> None:
    """Resolve ``axis=None`` on weight fake-quantize modules in a graph-mode ``GraphModule``.

    Raises:
        ValueError: If any weight fake-quantize modules still have
            ``axis=None`` after the defaults pass (i.e., the consuming op
            is not in the defaults tables).
    """
    _apply_defaults(_collect_weight_fq_entries_graph(model))


def apply_weight_axis_defaults_eager(model: nn.Module) -> None:
    """Resolve ``axis=None`` on weight fake-quantize parametrizations in eager mode.

    Raises:
        ValueError: If any weight fake-quantize modules still have
            ``axis=None`` after the defaults pass.
    """
    _apply_defaults(_collect_weight_fq_entries_eager(model))


def validate_activation_axes(model: nn.Module) -> None:
    """Raise if any activation fake-quantize has unresolved ``axis=None``.

    Validation-only pass for activations. Unlike weights, there are no default
    axis tables for activations, so per-channel or single-axis per-block
    granularity must specify an explicit axis.

    Works for both eager and graph-mode models (both support ``named_modules()``).

    Args:
        model (nn.Module): The prepared model.

    Raises:
        ValueError: If any activation fake-quantize modules have granularity
            that requires an axis but ``axis`` is ``None``.
    """
    unresolved: list[tuple[str, str]] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, FakeQuantizeImplBase):
            continue
        if mod.quantization_target != CompressionTargetTensor.ACTIVATION:
            continue
        if _granularity_needs_axis_default(mod.granularity):
            unresolved.append((name, type(mod.granularity).__name__))

    if unresolved:
        details = ", ".join(f"{name} ({gran_type})" for name, gran_type in sorted(unresolved))
        error_msg = (
            f"Activation fake-quantize modules with unresolved axis=None: "
            f"{details}. Activation quantization does not support axis=None. "
            f"Provide an explicit axis value in the granularity configuration "
            f"(e.g., PerChannelGranularity(axis=0))."
        )
        raise ValueError(error_msg)
