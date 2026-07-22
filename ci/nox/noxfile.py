# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Nox sessions for testing against multiple Python versions.

This module defines nox sessions to test the coreai-opt package against:
1. Supported Python versions (blocking for CI)
"""

import os
import sys
from pathlib import Path

from nox import Session, options
from nox_uv import session

from coreai_opt._utils.repo_utils import find_repo_root

# Find repository root (where pyproject.toml is located)
REPO_ROOT = find_repo_root(__file__)

# Add repository root to sys.path so we can import ci package
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("UV_PROJECT", str(REPO_ROOT))

from ci.nox.utils import (  # noqa: E402
    change_dir_to_project_root,
    get_pytest_executable,
    get_supported_python_versions,
)

options.default_venv_backend = "uv"
options.error_on_missing_interpreters = True

TORCH_GROUP = os.environ.get("TORCH_GROUP")

# Optional path to a pre-built distribution (wheel or sdist) to smoke test
# instead of building one from source. Set by the release workflow so the exact
# artifact that will be uploaded to PyPI is what gets tested. Relative paths are
# resolved against the working directory (the project root, set just below).
SMOKE_TEST_DIST = os.environ.get("SMOKE_TEST_DIST")


@session(
    python=get_supported_python_versions(),
    uv_extras=["coreai"],
    uv_groups=["test", TORCH_GROUP],
    # When testing a pre-built distribution, install only the project's
    # dependencies (not the project from source) so the distribution under test
    # is the sole coreai_opt on the path.
    uv_no_install_project=bool(SMOKE_TEST_DIST),
)
def smoke_tests(session: Session) -> None:
    """Smoke test the package and coreai_opt imports and basic functionality.

    By default, builds the package using the nox session's Python version,
    installs it in a clean environment, and runs smoke tests to verify
    functionality. When the ``SMOKE_TEST_DIST`` environment variable points to a
    pre-built wheel or sdist, that distribution is installed and tested instead
    of building one — used by the release workflow to smoke test the exact
    artifact that will be published to PyPI.
    """
    change_dir_to_project_root(session)

    if SMOKE_TEST_DIST:
        dist_path = Path(SMOKE_TEST_DIST).absolute()
        if not dist_path.is_file():
            session.error(f"SMOKE_TEST_DIST does not point to a file: {dist_path}")
        session.log(f"Installing pre-built distribution: {dist_path}")
        session.install(str(dist_path))
    else:
        session.log(f"Building package with Python {session.python}")
        session.install("build")
        session.run("make", "build-dev", external=True)
        session.log("Installing built package")

        # Find the built wheel
        wheels = list(Path("dist").glob("*.whl"))
        if not wheels:
            session.error(f"Build unsuccessful for Python {session.python}")
            session.error("No wheel found in dist/")
        latest_wheel = max(wheels, key=lambda p: p.stat().st_mtime)
        session.install(str(latest_wheel))
        session.log("Build Succeeded!")

    # setuptools is needed by torch.utils.cpp_extension (used by PT2E quantization);
    # required on Python 3.12+ where distutils was removed from stdlib.
    session.install("setuptools")

    session.log("Running smoke tests")

    # Use run_tests.sh to properly handle --junit and other custom flags
    # The script handles --junit by converting it to --junitxml
    # Pass the session's pytest executable to ensure we use the nox venv's pytest
    # Process posargs to handle --junit flag for unique filenames per Python version
    if session.posargs and "--junit" in session.posargs:
        test_args = [arg for arg in session.posargs if arg != "--junit"]
        test_args.extend(
            [
                f"--junitxml=test-results/pytest-results-{session.python}.xml",
                "--cov-append",
            ]
        )
    else:
        test_args = list(session.posargs) if session.posargs else []
    session.run(
        str(REPO_ROOT / "scripts" / "make" / "run_tests.sh"),
        "--pytest",
        get_pytest_executable(session),
        "--path",
        str(REPO_ROOT / "tests" / "test_smoke.py"),
        # Disable pytest-xdist for smoke tests because it makes test suite much slower
        # This can be overriden by user by setting workers in test_args
        "--workers",
        "0",
        "--noconftest",
        *test_args,
        external=True,
    )

    session.log("Smoke test passed!")
