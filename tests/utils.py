# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.nn.utils.parametrize import is_parametrized

from coreai_opt.palettization.kmeans.kmeans_fake_palettize import _KMeansFakePalettize


def spy_cluster_weights(cluster_dim: int) -> AbstractContextManager[Any]:
    """Patch the k-means method for ``cluster_dim`` with a call-recording mock that runs it."""
    name = "_cluster_weights_1d" if cluster_dim == 1 else "_cluster_weights_2d"
    return patch.object(
        _KMeansFakePalettize, name, autospec=True, side_effect=getattr(_KMeansFakePalettize, name)
    )


def clustered_block_count(spy: Any, cluster_dim: int) -> int:
    """Blocks a ``spy_cluster_weights`` mock clustered.

    ``_cluster_weights_1d`` takes one block per call, while ``_cluster_weights_2d`` takes all
    remaining blocks of a tensor in one batched call, so call counts are not comparable.
    """
    if cluster_dim == 1:
        return spy.call_count
    # autospec records ``self`` first, so the list of blocks is the second argument.
    return sum(len(call.args[1]) for call in spy.call_args_list)


def count_weight_parametrizations(model: nn.Module, parametrization_cls: type) -> int:
    """Count modules in ``model`` whose ``weight`` is parametrized with ``parametrization_cls``."""
    return sum(
        1
        for module in model.modules()
        if is_parametrized(module, "weight")
        and any(isinstance(p, parametrization_cls) for p in module.parametrizations["weight"])
    )


def test_data_path():
    return Path(__file__).parent.absolute() / "_test_data"


def test_artifact_path(rel: str | Path) -> Path:
    """Return the absolute path of a committed test artifact under ``_test_artifacts/``."""
    return Path(__file__).parent.absolute() / "_test_artifacts" / rel


def setup_data_loaders(dataset, batch_size, shuffle=False):
    train, test = dataset
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    test_loader = torch.utils.data.DataLoader(
        test,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return train_loader, test_loader


def train_step(model, optimizer, train_loader, data, target, batch_idx, epoch):
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % 100 == 0:
        print(
            f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
            f"({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}"
        )
    return loss


def eval_model(model, test_loader):
    # move model to eval
    model.eval()

    if len(test_loader.dataset) == 0:
        raise ValueError("Found empty test set")

    test_loss = 0
    correct = 0
    accuracy = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100.0 * correct / len(test_loader.dataset)

        print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.0f}%\n")
    return accuracy


def create_yaml_file(directory: Path, filename: str, content: dict) -> Path:
    """Helper function to create YAML files"""
    file_path = directory / filename
    with open(file_path, "w") as f:
        yaml.dump(content, f, default_flow_style=False)
    return file_path
