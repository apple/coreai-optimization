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


@session(python=get_supported_python_versions(), uv_extras=["coreai"], uv_groups=["test"])
def smoke_tests(session: Session) -> None:
    """Smoke test the package build and coreai_opt imports and basic functionality.

    Builds the package using the nox session's Python version, installs it
    in a clean environment, and runs smoke tests to verify functionality.
    """
    change_dir_to_project_root(session)
    session.log(f"Building package with Python {session.python}")
    session.install("build")
    session.run("make", "build", external=True)
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
