# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Pytest configuration file for coreai_opt tests."""

# ruff: noqa: E402

import os
import sys

if sys.platform == "linux":
    os.environ.setdefault("OMP_NUM_THREADS", "1")

import random
import tempfile

import numpy as np
import pytest
import torch

from tests.utils import test_artifact_path

pytest_plugins = [
    "tests.fixtures.quantization",
    "tests.fixtures.palettization",
    "tests.fixtures.fp8",
    "tests.fixtures.fp4",
    "tests.fixtures.compression",
    "tests.fixtures.pruning",
    "tests.models.mnist",
    "tests.models.resnet",
    "tests.models.simple",
]

_DEFAULT_SEED: int = 42


@pytest.fixture(autouse=True)
def seed_every_test(request: pytest.FixtureRequest) -> None:
    """Seeding policy for test reproducibility.

    By default, tests run with nondeterministic seeding.

    Use markers to enable deterministic seeding when reproducibility is needed:
    - No marker: doesn't do anything special
    - @pytest.mark.seed: Use default seed (42) for deterministic behavior
    - @pytest.mark.seed(N): Use specific seed N for deterministic behavior
    - @pytest.mark.seed(None): Explicitly use nondeterministic seeding
    """
    marker = request.node.get_closest_marker("seed")

    if marker is None:
        # No marker: don't do anything special
        return

    # @pytest.mark.seed (no argument): use default seed
    # @pytest.mark.seed(N): use specified seed, `N` can be `None`
    seed = _DEFAULT_SEED if not marker.args else marker.args[0]

    # Validate seed type
    if seed is not None and not isinstance(seed, int):
        pytest.fail(
            f"@pytest.mark.seed expects int or None, got {type(seed).__name__}: {seed!r}",
        )

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    if seed is None:
        torch.seed()
    else:
        torch.manual_seed(seed)


@pytest.fixture(autouse=True)
def reset_dynamo() -> None:
    """Reset torch._dynamo state before each test.

    This ensures tests don't interfere with each other through cached
    dynamo compilation state.
    """
    torch._dynamo.reset()


@pytest.fixture(scope="session")
def temp_dir():
    """Fixture to provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture(scope="function")
def mnist_pretrained_model(custom_test_mnist_model):
    """Load the committed 1-epoch MNIST checkpoint into a fresh model."""
    model = custom_test_mnist_model
    model.load_state_dict(
        torch.load(test_artifact_path("mnist/mnist_pretrained_1epoch_09032025.pt"))
    )
    return model
