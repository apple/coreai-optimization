# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Environment variables that native runtimes read once, when they are loaded.

``USE_LOCAL_COREAI`` and ``OMP_NUM_THREADS`` are read by native libraries as they load.
"""

import os
import sys

#: Selects the local coreai runtime
USE_LOCAL_COREAI_ENV_VAR = "USE_LOCAL_COREAI"

#: Carries the ``--compute-unit-kind`` choice to the helpers that act on it.
COMPUTE_UNIT_KIND_ENV_VAR = "COREAI_OPT_TEST_COMPUTE_UNIT_KIND"

COMPUTE_UNIT_KINDS: tuple[str, ...] = ("interpreter", "cpu", "gpu", "neural_engine")
DEFAULT_COMPUTE_UNIT_KIND = "interpreter"


def apply_thread_defaults() -> None:
    """Cap OpenMP at one thread on Linux unless the caller already chose a value."""
    if sys.platform == "linux":
        os.environ.setdefault("OMP_NUM_THREADS", "1")


def select_compute_unit_kind(kind: str) -> None:
    """Record the compute unit and point ``coreai_torch`` at the matching runtime.

    ``interpreter`` needs the bundled runtime, so ``USE_LOCAL_COREAI`` is
    defaulted to ``1`` — ``setdefault`` rather than assignment, so an explicit
    ``USE_LOCAL_COREAI=0`` (the override documented in the Makefile) still wins.
    The real compute units are only exposed by the OS runtime, so the variable is
    dropped for those.

    Args:
        kind (str): One of :data:`COMPUTE_UNIT_KINDS`.

    Raises:
        ValueError: If ``kind`` is not a known compute unit kind.

    """
    _validate_compute_unit_kind(kind)
    os.environ[COMPUTE_UNIT_KIND_ENV_VAR] = kind
    if kind == "interpreter":
        os.environ.setdefault(USE_LOCAL_COREAI_ENV_VAR, "1")
    else:
        os.environ.pop(USE_LOCAL_COREAI_ENV_VAR, None)


def get_compute_unit_kind() -> str:
    """Return the compute unit chosen by ``--compute-unit-kind``.

    Falls back to :data:`DEFAULT_COMPUTE_UNIT_KIND` when
    :func:`select_compute_unit_kind` has not run, so importing a test helper
    outside a pytest session still works.

    Returns:
        str: One of :data:`COMPUTE_UNIT_KINDS`.

    Raises:
        ValueError: If the environment holds an unknown compute unit kind.

    """
    kind = os.environ.get(COMPUTE_UNIT_KIND_ENV_VAR, DEFAULT_COMPUTE_UNIT_KIND)
    _validate_compute_unit_kind(kind)
    return kind


def _validate_compute_unit_kind(kind: str) -> None:
    """Raise ``ValueError`` unless ``kind`` is one of :data:`COMPUTE_UNIT_KINDS`."""
    if kind not in COMPUTE_UNIT_KINDS:
        msg = f"Unknown compute unit kind: {kind!r}; expected one of {COMPUTE_UNIT_KINDS}"
        raise ValueError(msg)
