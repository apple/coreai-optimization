# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests verifying prepare() example_inputs type annotations and multi-input support."""

import typing
from typing import Any

import pytest
import torch
import torch.nn as nn

from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig
from coreai_opt.pruning import MagnitudePruner, MagnitudePrunerConfig
from coreai_opt.quantization import ExecutionMode, Quantizer, QuantizerConfig
from coreai_opt.quantization._eager.quantizer import EagerQuantizer
from coreai_opt.quantization._graph.quantizer import GraphQuantizer


class TwoInputModel(nn.Module):
    """Simple model accepting two tensor inputs for multi-input testing."""

    def __init__(self, in_features: int = 4, out_features: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.fc1(x) + self.fc2(y)


@pytest.mark.parametrize(
    "cls_or_fn",
    [
        Quantizer.prepare,
        EagerQuantizer.prepare,
        GraphQuantizer.prepare,
        KMeansPalettizer.prepare,
        MagnitudePruner.prepare,
    ],
    ids=[
        "Quantizer.prepare",
        "EagerQuantizer.prepare",
        "GraphQuantizer.prepare",
        "KMeansPalettizer.prepare",
        "MagnitudePruner.prepare",
    ],
)
def test_prepare_example_inputs_type_annotation(cls_or_fn):
    """Verify example_inputs is annotated as tuple[Any, ...] across all compressors."""
    type_hints = typing.get_type_hints(cls_or_fn)
    assert "example_inputs" in type_hints, f"example_inputs missing from type hints of {cls_or_fn}"
    expected_type = tuple[Any, ...]
    assert type_hints["example_inputs"] == expected_type, (
        f"{cls_or_fn.__qualname__} has unexpected type hint {type_hints['example_inputs']!r}, "
        f"expected {expected_type!r}"
    )


def test_eager_quantizer_multi_input_prepare():
    """Verify EagerQuantizer prepares and runs a multi-input model with a 2-tuple."""
    model = TwoInputModel()
    example_inputs = (torch.randn(1, 4), torch.randn(1, 4))
    config = QuantizerConfig(execution_mode=ExecutionMode.EAGER)
    quantizer = Quantizer(model, config)
    prepared_model = quantizer.prepare(example_inputs)

    output = prepared_model(*example_inputs)
    assert output.shape == (1, 4)


def test_graph_quantizer_multi_input_prepare():
    """Verify _GraphQuantizer prepares and runs a multi-input model with a 2-tuple."""
    model = TwoInputModel()
    example_inputs = (torch.randn(1, 4), torch.randn(1, 4))
    config = QuantizerConfig(execution_mode=ExecutionMode.GRAPH)
    quantizer = Quantizer(model, config)
    prepared_model = quantizer.prepare(example_inputs)

    output = prepared_model(*example_inputs)
    assert output.shape == (1, 4)


def test_kmeans_palettizer_multi_input_prepare():
    """Verify KMeansPalettizer prepares and runs a multi-input model with a 2-tuple."""
    model = TwoInputModel()
    example_inputs = (torch.randn(1, 4), torch.randn(1, 4))
    config = KMeansPalettizerConfig()
    palettizer = KMeansPalettizer(model, config)
    prepared_model = palettizer.prepare(example_inputs)

    output = prepared_model(*example_inputs)
    assert output.shape == (1, 4)


def test_magnitude_pruner_multi_input_prepare():
    """Verify MagnitudePruner prepares and runs a multi-input model with a 2-tuple."""
    model = TwoInputModel()
    example_inputs = (torch.randn(1, 4), torch.randn(1, 4))
    config = MagnitudePrunerConfig()
    pruner = MagnitudePruner(model, config)
    prepared_model = pruner.prepare(example_inputs)

    output = prepared_model(*example_inputs)
    assert output.shape == (1, 4)
