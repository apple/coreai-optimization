# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import copy
from collections.abc import Callable

import pytest
import torch
import torch.nn as nn

from coreai_opt.base_model_compressor import (
    _COREAI_OPT_PREPARED_ATTR,
    _BaseModelCompressor,
)
from coreai_opt.palettization import KMeansPalettizer
from coreai_opt.pruning import MagnitudePruner
from coreai_opt.quantization import ModuleQuantizerConfig, Quantizer, QuantizerConfig
from coreai_opt.quantization.spec import (
    default_activation_quantization_spec,
    default_weight_quantization_spec,
)


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


class TestIsModelPrepared:
    """Tests for _is_model_prepared static method."""

    def test_fresh_model_not_prepared(self):
        """Test that a newly created model is not marked as prepared."""
        model = SimpleModel()
        assert not _BaseModelCompressor._is_model_prepared(model)


class TestMarkModelAsPrepared:
    """Tests for _mark_model_as_prepared static method."""

    def test_mark_simple_model_as_prepared(self):
        """Test marking a simple model as prepared."""
        model = SimpleModel()
        assert not _BaseModelCompressor._is_model_prepared(model)

        _BaseModelCompressor._mark_model_as_prepared(model)

        assert _BaseModelCompressor._is_model_prepared(model)
        # Marker is registered as a non-persistent buffer.
        assert _COREAI_OPT_PREPARED_ATTR in dict(model.named_buffers())
        assert _COREAI_OPT_PREPARED_ATTR not in model.state_dict()

    def test_mark_already_prepared_model(self):
        """Test that marking an already prepared model is idempotent."""
        model = SimpleModel()
        _BaseModelCompressor._mark_model_as_prepared(model)
        assert _BaseModelCompressor._is_model_prepared(model)

        # Mark again
        _BaseModelCompressor._mark_model_as_prepared(model)
        assert _BaseModelCompressor._is_model_prepared(model)
        assert _COREAI_OPT_PREPARED_ATTR in dict(model.named_buffers())


class TestPreparedSurvivesDeepcopy:
    """The marker must survive deepcopy across module types used by coreai-opt.

    ``torch.fx.GraphModule.__deepcopy__`` only copies parameters, buffers, and
    submodules; arbitrary attributes set via ``setattr`` are dropped. Storing the
    marker as a buffer ensures PT2E-prepared models stay marked after deepcopy.
    """

    def test_nn_module_deepcopy(self):
        model = SimpleModel()
        _BaseModelCompressor._mark_model_as_prepared(model)
        assert _BaseModelCompressor._is_model_prepared(copy.deepcopy(model))

    def test_exported_graph_module_deepcopy(self):
        exported = torch.export.export(SimpleModel(), (torch.randn(2, 10),))
        graph_module = exported.module()
        _BaseModelCompressor._mark_model_as_prepared(graph_module)
        assert _BaseModelCompressor._is_model_prepared(copy.deepcopy(graph_module))


def _quantizer(execution_mode: str) -> Callable[[nn.Module], Quantizer]:
    config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={"weight": default_weight_quantization_spec()},
            op_input_spec={"*": default_activation_quantization_spec()},
            op_output_spec=None,
        ),
        execution_mode=execution_mode,
    )
    return lambda model: Quantizer(model, config)


_COMPRESSORS = [
    pytest.param(KMeansPalettizer, id="palettizer"),
    pytest.param(_quantizer("eager"), id="quantizer-eager"),
    pytest.param(_quantizer("graph"), id="quantizer-graph"),
    pytest.param(MagnitudePruner, id="pruner"),
]


@pytest.mark.parametrize("make_compressor", _COMPRESSORS)
class TestPrepareFromStateDict:
    """``prepare(state_dict=...)`` reloads a prepared model's compression state."""

    @staticmethod
    def _prepare_source(make_compressor, model, example_input):
        """Prepare ``model`` and, for quantizers, calibrate it on scaled inputs."""
        compressor = make_compressor(model)
        prepared = compressor.prepare((example_input,))
        if isinstance(compressor, Quantizer):
            with compressor.calibration_mode():
                prepared(example_input * 3.0)
        return prepared

    def test_reproduces_prepared_model(self, make_compressor, tmp_path):
        """A model reloaded from a saved state dict reproduces the source model's output."""
        torch.manual_seed(0)
        model, example_input = SimpleModel(), torch.randn(4, 10)
        source = self._prepare_source(make_compressor, copy.deepcopy(model), example_input)
        expected = source(example_input)
        path = tmp_path / "prepared.pt"
        torch.save(source.state_dict(), path)

        reloaded = make_compressor(copy.deepcopy(model)).prepare(
            (example_input,), state_dict=torch.load(path)
        )

        torch.testing.assert_close(reloaded(example_input), expected, rtol=0, atol=0)

    def test_flags_match_regular_prepare(self, make_compressor):
        """Enable flags are left as a regular ``prepare`` leaves them, whatever was saved."""
        torch.manual_seed(0)
        model, example_input = SimpleModel(), torch.randn(4, 10)
        regular = make_compressor(copy.deepcopy(model)).prepare((example_input,))
        flipped = {
            name: 1 - tensor if name.endswith("_enabled") else tensor
            for name, tensor in regular.state_dict().items()
        }

        reloaded = make_compressor(copy.deepcopy(model)).prepare(
            (example_input,), state_dict=flipped
        )

        for name, tensor in regular.state_dict().items():
            if name.endswith("_enabled"):
                assert torch.equal(reloaded.state_dict()[name], tensor), name

    def test_mismatched_state_dict_raises(self, make_compressor):
        """A state dict that does not match the prepared model raises a clear error."""
        torch.manual_seed(0)
        model, example_input = SimpleModel(), torch.randn(4, 10)
        state_dict = make_compressor(copy.deepcopy(model)).prepare((example_input,)).state_dict()
        state_dict.pop(next(iter(state_dict)))

        with pytest.raises(RuntimeError, match="State dict does not match"):
            make_compressor(copy.deepcopy(model)).prepare((example_input,), state_dict=state_dict)
