# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""coreai-opt without the optional coreai packages.

The ``no_coreai`` fixture makes coreai and coreai-torch unimportable, so these tests run, and
have to pass, whether or not those packages are installed.
"""

import sys

import pytest

from coreai_opt import ExportBackend
from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig
from coreai_opt.pruning import MagnitudePruner
from coreai_opt.quantization import ExecutionMode, Quantizer, QuantizerConfig

_COMPRESSORS = {
    "graph_quantizer": lambda model: Quantizer(model, QuantizerConfig.presets.w8()),
    "eager_quantizer": lambda model: Quantizer(
        model, QuantizerConfig.presets.w8(execution_mode=ExecutionMode.EAGER)
    ),
    "palettizer": lambda model: KMeansPalettizer(model, KMeansPalettizerConfig.presets.w4()),
    "pruner": MagnitudePruner,
}


@pytest.fixture(autouse=True)
def no_coreai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make coreai and coreai-torch unimportable, including submodules imported earlier."""
    for name in [*sys.modules, "coreai", "coreai_torch"]:
        if name.split(".")[0] in ("coreai", "coreai_torch"):
            monkeypatch.setitem(sys.modules, name, None)


@pytest.fixture(params=list(_COMPRESSORS))
def prepared(request, simple_linear_model, simple_linear_model_input):
    """Return a compressor of each kind, prepared on ``simple_linear_model`` without coreai."""
    compressor = _COMPRESSORS[request.param](simple_linear_model.eval())
    compressor.prepare((simple_linear_model_input,))
    return compressor, simple_linear_model_input


def test_torch_finalize_works_without_coreai(prepared):
    compressor, example_input = prepared
    compressor.finalize(backend=ExportBackend.Torch)(example_input)


def test_coreai_finalize_without_coreai_points_to_the_torch_backend(prepared):
    compressor, _ = prepared
    with pytest.raises(ImportError, match=r"`coreai` extra.*ExportBackend\.Torch"):
        compressor.finalize()
