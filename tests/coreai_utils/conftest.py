# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import pytest

pytest.importorskip("coreai")

import torch
from torch import nn

from tests.export.export_utils import MLIRConverter


class _LinearModel(torch.nn.Module):
    """Toy large linear model for compression testing."""

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.linear = nn.Linear(2048, 32).to(dtype)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.linear(input_tensor)


@pytest.fixture(params=["fp32", "fp16"])
def _exported_program(request) -> tuple[torch.export.ExportedProgram, torch.Tensor, str]:
    uncompressed_dtype = request.param
    torch_dtype = torch.float16 if uncompressed_dtype == "fp16" else torch.float32
    input_tensor = torch.randn(2, 2048, dtype=torch_dtype)
    with torch.no_grad():
        exported_program = torch.export.export(
            _LinearModel(torch_dtype).eval(),
            args=(),
            kwargs={"input_tensor": input_tensor},
        )
        exported_program = exported_program.run_decompositions()
    return exported_program, input_tensor, uncompressed_dtype


@pytest.fixture
def _coreai_program(_exported_program):
    exported_program, input_tensor, uncompressed_dtype = _exported_program
    coreai_program = MLIRConverter._lower_to_coreai(exported_program)
    return coreai_program, input_tensor, uncompressed_dtype
