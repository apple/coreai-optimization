# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Test models with composite ops, for externalization test coverage."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tests.utils import test_artifact_path

# Externalize specs shared by the externalization test modules.


def rmsnorm_externalize_spec():
    """ExternalizeSpec targeting the RMSNormImpl composite."""
    from coreai_torch import ExternalizeSpec  # noqa: PLC0415
    from coreai_torch.composite_ops import RMSNormImpl  # noqa: PLC0415

    return ExternalizeSpec(
        target_class=RMSNormImpl,
        composite_op_name="rms_norm",
        composite_attrs=["axes", "eps"],
    )


def sdpa_externalize_spec():
    """ExternalizeSpec targeting the SDPA composite."""
    from coreai_torch import ExternalizeSpec  # noqa: PLC0415
    from coreai_torch.composite_ops import SDPA  # noqa: PLC0415

    return ExternalizeSpec(
        target_class=SDPA,
        composite_op_name="scaled_dot_product_attention",
        composite_attrs=["scale", "is_causal", "window_size"],
    )


class MNISTCompositeRMSNormModel(nn.Module):
    """Tiny MNIST classifier with an embedded RMSNormImpl composite op.

    Architecture: Flatten -> Linear(28*28, 128) -> RMSNormImpl ->
    ReLU -> Linear(128, 10) -> LogSoftmax.
    """

    def __init__(
        self,
        hidden: int = 128,
        num_classes: int = 10,
        eps: float = 1e-5,
    ) -> None:
        from coreai_torch.composite_ops import RMSNormImpl  # noqa: PLC0415

        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.norm = RMSNormImpl(eps=eps)
        self.scale = nn.Parameter(torch.ones(hidden))
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, num_classes)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.norm(x, self.scale)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)


# Function scoped so each test gets an independently mutable model.
@pytest.fixture(scope="function")
def mnist_composite_rmsnorm_pretrained_model() -> MNISTCompositeRMSNormModel:
    """Load the committed 1-epoch MNISTCompositeRMSNormModel checkpoint.

    Trained with seed 42 for one epoch over the MNIST train split: Adam,
    lr=1e-3, batch_size=128, shuffle=False, nll_loss.
    """
    model = MNISTCompositeRMSNormModel()
    model.load_state_dict(
        torch.load(
            test_artifact_path("mnist/mnist_composite_rmsnorm_pretrained_1epoch_08132026.pt")
        )
    )
    return model


@pytest.fixture
def mnist_composite_rmsnorm_example_input() -> torch.Tensor:
    """A canonical example-input tensor matching MNIST shape, for prepare()."""
    return torch.ones(1, 1, 28, 28, dtype=torch.float32)


class CompositeRMSNormOnlyModel(nn.Module):
    """A single RMSNormImpl composite and nothing else."""

    def __init__(self, dim: int = 32, eps: float = 1e-5) -> None:
        from coreai_torch.composite_ops import RMSNormImpl  # noqa: PLC0415

        super().__init__()
        self.norm = RMSNormImpl(eps=eps)
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x, self.scale)


class CompositeSDPAModel(nn.Module):
    """proj -> SDPA(composite) -> output_proj, fp16, single-head fake-dim."""

    def __init__(self, dim: int = 32) -> None:
        from coreai_torch.composite_ops import SDPA  # noqa: PLC0415

        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.composite = SDPA(scale=None, is_causal=True)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        # Insert a single fake head dim so SDPA sees rank-4 inputs.
        q = q.unsqueeze(1)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)
        attn = self.composite(q, k, v).squeeze(1)
        return self.out(attn)
