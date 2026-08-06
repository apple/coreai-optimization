# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Test models with composite ops, for externalization test coverage."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

_DIM = 32
_HIDDEN = 64
_SEQ = 4
_BATCH = 2


class CompositeRMSNormModel(nn.Module):
    """Linear -> RMSNormImpl (composite) -> Linear over rank-3 activations."""

    def __init__(
        self,
        dim: int = _DIM,
        hidden: int = _HIDDEN,
        eps: float = 1e-5,
    ) -> None:
        from coreai_torch.composite_ops import RMSNormImpl  # noqa: PLC0415

        super().__init__()
        self.up = nn.Linear(dim, hidden, bias=False)
        self.norm = RMSNormImpl(eps=eps)
        self.scale = nn.Parameter(torch.ones(hidden))
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up(x)
        h = self.norm(h, self.scale)
        return self.down(h)


@pytest.fixture
def composite_rmsnorm_model() -> CompositeRMSNormModel:
    """Eval-half model with a single RMSNormImpl composite op."""
    return CompositeRMSNormModel().eval().half()


@pytest.fixture
def composite_rmsnorm_input() -> torch.Tensor:
    """Rank-3 fp16 sample input matching CompositeRMSNormModel's shapes."""
    return torch.randn(_BATCH, _SEQ, _DIM, dtype=torch.float16)


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


@pytest.fixture(scope="function")
def mnist_composite_rmsnorm_pretrained_state(mnist_dataset) -> dict:
    """One-epoch-pretrained state_dict for MNISTCompositeRMSNormModel.

    Training costs ~1.5s. This fixture is function scoped so the
    repo-wide seeding policy applies: determinism comes from the
    consuming test's ``@pytest.mark.seed`` marker, which the autouse
    ``seed_every_test`` fixture in ``tests/conftest.py`` honors before
    this fixture runs.
    """
    model = MNISTCompositeRMSNormModel()

    train_ds, _ = mnist_dataset
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _data, _target in train_loader:
        optimizer.zero_grad()
        output = model(_data)
        loss = F.nll_loss(output, _target)
        loss.backward()
        optimizer.step()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


@pytest.fixture(scope="function")
def mnist_composite_rmsnorm_pretrained_model(
    mnist_composite_rmsnorm_pretrained_state: dict,
) -> MNISTCompositeRMSNormModel:
    """Fresh MNISTCompositeRMSNormModel loaded from the pretrained state fixture."""
    model = MNISTCompositeRMSNormModel()
    model.load_state_dict(mnist_composite_rmsnorm_pretrained_state)
    return model


@pytest.fixture
def mnist_composite_rmsnorm_example_input() -> torch.Tensor:
    """A canonical example-input tensor matching MNIST shape, for prepare()."""
    return torch.ones(1, 1, 28, 28, dtype=torch.float32)


class CompositeRMSNormOnlyModel(nn.Module):
    """A single RMSNormImpl composite and nothing else."""

    def __init__(self, dim: int = _DIM, eps: float = 1e-5) -> None:
        from coreai_torch.composite_ops import RMSNormImpl  # noqa: PLC0415

        super().__init__()
        self.norm = RMSNormImpl(eps=eps)
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x, self.scale)


class CompositeSDPAModel(nn.Module):
    """proj -> SDPA(composite) -> output_proj, fp16, single-head fake-dim."""

    def __init__(self, dim: int = _DIM) -> None:
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
