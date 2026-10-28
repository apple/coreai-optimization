# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Shared helpers for the joint sparsity + palettization tests."""

import torch
import torch.nn as nn

from coreai_opt.palettization import (
    KMeansPalettizer,
    KMeansPalettizerConfig,
    ModuleKMeansPalettizerConfig,
    PalettizationSpec,
)
from coreai_opt.palettization.config import VectorKMeansConfig
from coreai_opt.palettization.spec import PerGroupedChannelGranularity, PerTensorGranularity
from coreai_opt.quantization.spec import QuantizationScheme, QuantizationSpec

SPARSITY = 0.5
GROUP_SIZE = 2
SYMMETRIC_LUT_QSPEC = QuantizationSpec(dtype=torch.int8, qscheme=QuantizationScheme.SYMMETRIC)


def presparsify(
    model: nn.Module,
    sparsity: float,
    group_size: int | None = None,
    layers: list[str] | None = None,
) -> nn.Module:
    """Zero the smallest-magnitude weights of each layer in place.

    Args:
        model (nn.Module): Model whose Conv2d, ConvTranspose2d and Linear weights are pruned.
        sparsity (float): Fraction of each group (or tensor) to zero.
        group_size (int | None): Axis-0 group size, so every group keeps the same count. If
            ``None``, prune per tensor.
        layers (list[str] | None): Names of the layers to prune; all of them if ``None``.

    Returns:
        nn.Module: The pruned model in eval mode.
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, nn.Conv2d | nn.ConvTranspose2d | nn.Linear):
                continue
            if layers is not None and name not in layers:
                continue
            weight = module.weight
            for group in weight.split(group_size or weight.shape[0], dim=0):
                num_pruned = int(group.numel() * sparsity)
                group.view(-1)[group.abs().flatten().argsort()[:num_pruned]] = 0
    return model.eval()


def spec_kwargs(n_bits: int, group_size: int | None, axis: int = 0) -> dict:
    """Return per-tensor spec kwargs, or axis-``axis`` grouped ones if ``group_size`` is set."""
    granularity = (
        PerTensorGranularity()
        if group_size is None
        else PerGroupedChannelGranularity(axis=axis, group_size=group_size)
    )
    return {"n_bits": n_bits, "granularity": granularity}


def build_palettizer(model: nn.Module, **kwargs) -> KMeansPalettizer:
    """Build a palettizer for every ``weight`` with a ``PalettizationSpec(**kwargs)``.

    Sparsity detection defaults to a threshold of 0.4 and the granularity to per-tensor.
    """
    kwargs.setdefault("granularity", PerTensorGranularity())
    kwargs.setdefault("sparsity_detection_threshold", 0.4)
    cluster_dim = kwargs.get("cluster_dim", 1)
    config = KMeansPalettizerConfig(
        global_config=ModuleKMeansPalettizerConfig(
            op_state_spec={"weight": PalettizationSpec(**kwargs)},
            # cluster_dim > 1 requires fast mode off; scalar palettization keeps it on,
            # since non-fast 1-D k-means allocates O(2**n_bits * weight.numel()) memory.
            enable_fast_kmeans_mode=cluster_dim == 1,
            vector_kmeans_config=(
                VectorKMeansConfig(n_init=1, max_iter=1) if cluster_dim > 1 else None
            ),
        )
    )
    return KMeansPalettizer(model, config)
