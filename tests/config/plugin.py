# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Reusable pytest plugin: compute unit selection, seeding, and generic fixtures."""

import random
import tempfile
from collections.abc import Iterator

import numpy as np
import pytest
import torch

from .env import (
    COMPUTE_UNIT_KINDS,
    DEFAULT_COMPUTE_UNIT_KIND,
    select_compute_unit_kind,
)

_DEFAULT_SEED: int = 42

_SEED_MARKER = (
    "seed(value): enable deterministic seeding; "
    f"no value uses default seed ({_DEFAULT_SEED}), "
    "value=None means nondeterministic seeding"
)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register CLI options."""
    parser.addoption(
        "--compute-unit-kind",
        choices=list(COMPUTE_UNIT_KINDS),
        default=DEFAULT_COMPUTE_UNIT_KIND,
        help=(
            "Compute unit used by CoreAI runtime inference in tests:\n"
            "  interpreter (default) - bundled runtime (USE_LOCAL_COREAI=1)\n"
            "  cpu                   - SpecializationOptions.cpu_only() (BNNS)\n"
            "  gpu                   - preferred ComputeUnitKind.gpu() (MPSGraph)\n"
            "  neural_engine         - preferred ComputeUnitKind.neural_engine()\n"
            "Anything other than 'interpreter' unsets USE_LOCAL_COREAI so the OS\n"
            "runtime is used."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register this plugin's marker and publish the selected compute unit."""
    config.addinivalue_line("markers", _SEED_MARKER)
    select_compute_unit_kind(config.getoption("--compute-unit-kind"))


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
def temp_dir() -> Iterator[str]:
    """Fixture to provide a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def accelerator_device() -> str:
    """The available accelerator device type ("cuda" or "mps"); skip if neither."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    pytest.skip("requires a CUDA or MPS accelerator")
