# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Pytest configuration file for coreai_opt tests.

Generic pytest configurations live in ``tests/config/plugin.py``, which any conftest can
import and register.
"""

# ruff: noqa: E402

from tests.config.env import apply_thread_defaults

# Has to happen before torch is imported; see tests/config/env.py.
apply_thread_defaults()

import pytest
import torch

from tests.utils import test_artifact_path

pytest_plugins = [
    "tests.config.plugin",
    "tests.fixtures.quantization",
    "tests.fixtures.palettization",
    "tests.fixtures.fp8",
    "tests.fixtures.fp4",
    "tests.fixtures.compression",
    "tests.fixtures.pruning",
    "tests.models.mnist",
    "tests.models.resnet",
    "tests.models.simple",
    "tests.models.composite",
]


@pytest.fixture(scope="function")
def mnist_pretrained_model(custom_test_mnist_model):
    """Load the committed 1-epoch MNIST checkpoint into a fresh model."""
    model = custom_test_mnist_model
    model.load_state_dict(
        torch.load(test_artifact_path("mnist/mnist_pretrained_1epoch_09032025.pt"))
    )
    return model
