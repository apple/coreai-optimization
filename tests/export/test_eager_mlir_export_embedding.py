# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Eager Core AI export tests for FP4-quantized nn.Embedding variants.

These mirror the embedding patterns seen in common LLMs:
  - plain  : standard nn.Embedding
  - scaled : embedding output scaled by sqrt(hidden_size)
  - tied   : embedding weight shared with an lm_head Linear

All variants use FP4 (float4_e2m1fn) weight-only quantization with symmetric
per-block granularity (block_size=32), which is the only granularity supported
for FP4 MLIR export.
"""

import pytest
import torch
from torch import nn

from coreai_opt import ExportBackend
from coreai_opt.quantization import (
    ModuleQuantizerConfig,
    QuantizationSpec,
    Quantizer,
    QuantizerConfig,
)
from coreai_opt.quantization.spec import PerBlockGranularity, QuantizationScheme
from tests.models.simple import (
    PlainEmbeddingModel,
    ScaledEmbeddingModel,
    TiedEmbeddingModel,
)

from . import export_utils

_VOCAB_SIZE = 256
_EMBED_DIM = 64
_SEQ_LEN = 8


def _fp4_config() -> QuantizerConfig:
    """FP4 weight-only config: symmetric, per-block (block_size=32) along axis 1."""
    return QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={
                "weight": QuantizationSpec(
                    dtype="float4_e2m1fn",
                    qscheme=QuantizationScheme.SYMMETRIC,
                    granularity=PerBlockGranularity(axis=1, block_size=32),
                ),
            },
            op_input_spec=None,
            op_output_spec=None,
        ),
        execution_mode="eager",
    )


def _run_fp4_embedding_export(model: nn.Module, expected_shift_scale_ops: int) -> None:
    """Quantize the embedding model with FP4, finalize, and export/run on Core AI."""
    model = model.eval().to(dtype=torch.float16)
    input_ids = torch.randint(0, _VOCAB_SIZE, (1, _SEQ_LEN), dtype=torch.int32)

    quantizer = Quantizer(model, _fp4_config())
    prepared_model = quantizer.prepare((input_ids,))

    with torch.no_grad():
        prepared_model_output = prepared_model(input_ids)

    finalized_model = quantizer.finalize(backend=ExportBackend.CoreAI)

    export_utils.convert_and_verify(
        finalized_model=finalized_model,
        input_data=input_ids,
        expected_ops={"constexpr_blockwise_shift_scale": expected_shift_scale_ops},
        export_backend=ExportBackend.CoreAI,
        prepared_model_output=prepared_model_output,
    )


@pytest.mark.parametrize(
    "model_factory, expected_shift_scale_ops",
    [
        pytest.param(PlainEmbeddingModel, 1, id="plain"),
        pytest.param(ScaledEmbeddingModel, 1, id="scaled"),
        pytest.param(TiedEmbeddingModel, 2, id="tied"),
    ],
)
def test_fp4_embedding_export(
    model_factory: type[nn.Module], expected_shift_scale_ops: int
) -> None:
    """Eager Core AI export with FP4-quantized nn.Embedding variants."""
    _run_fp4_embedding_export(
        model_factory(vocab_size=_VOCAB_SIZE, embed_dim=_EMBED_DIM),
        expected_shift_scale_ops,
    )
