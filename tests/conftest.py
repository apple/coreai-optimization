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

_COMPUTE_UNIT_KIND_CHOICES = ("interpreter", "cpu", "gpu", "neural_engine")
_COMPUTE_UNIT_KIND_DEFAULT = "interpreter"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register CLI options."""
    parser.addoption(
        "--compute-unit-kind",
        choices=list(_COMPUTE_UNIT_KIND_CHOICES),
        default=_COMPUTE_UNIT_KIND_DEFAULT,
        help=(
            "Compute unit used by MLIRConverter inference:\n"
            "  interpreter (default) - bundled runtime (USE_LOCAL_COREAI=1)\n"
            "  cpu                   - SpecializationOptions.cpu_only() (BNNS)\n"
            "  gpu                   - preferred ComputeUnitKind.gpu() (MPSGraph)\n"
            "  neural_engine         - preferred ComputeUnitKind.neural_engine()\n"
            "Anything other than 'interpreter' unsets USE_LOCAL_COREAI so the OS\n"
            "runtime is used."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    """Publish the selected compute unit to the export test utils.

    For ``--compute-unit-kind=interpreter`` we pin ``USE_LOCAL_COREAI=1`` so the
    bundled runtime is used. For any real compute unit (cpu/gpu/neural_engine)
    we drop the env var so the OS runtime — which actually exposes those compute
    units — gets picked up.
    """
    compute_unit_kind = config.getoption("--compute-unit-kind")
    if compute_unit_kind == "interpreter":
        os.environ.setdefault("USE_LOCAL_COREAI", "1")
    else:
        os.environ.pop("USE_LOCAL_COREAI", None)

    # Imported here — after the env var is adjusted — because export_utils imports
    # coreai_torch at module load, and coreai_torch reads USE_LOCAL_COREAI at
    # dlopen time. Importing earlier would lock in the wrong runtime.
    from tests.export.export_utils import set_test_compute_unit_kind  # noqa: PLC0415

    set_test_compute_unit_kind(compute_unit_kind)


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
