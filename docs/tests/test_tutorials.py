# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause


"""Test that tutorial notebooks execute without errors."""

from __future__ import annotations

from pathlib import Path

import papermill as pm
import pytest

from coreai_opt._utils.repo_utils import find_repo_root

NOTEBOOK_CELL_TIMEOUT_SECONDS = 300

# Notebooks whose filename contains this token must export deployment models.
MNIST_TOKEN = "mnist"
EXPECTED_MNIST_EXPORTS = [
    "exported_model.aimodel",
]

_repo_root = find_repo_root(__file__)
_tutorials_dir = _repo_root / "docs" / "src" / "tutorials"
_notebooks = sorted(_tutorials_dir.glob("*.ipynb"))


def _notebook_id(path: Path) -> str:
    return path.stem


def test_tutorials_dir_is_non_empty() -> None:
    """Guard against an empty parametrize set silently producing zero tests."""
    assert _notebooks, f"No tutorial notebooks found under {_tutorials_dir}"


@pytest.mark.parametrize("notebook", _notebooks, ids=_notebook_id)
def test_tutorial_notebook_executes(notebook: Path, tmp_path: Path) -> None:
    """Execute a tutorial notebook end-to-end with papermill and verify outputs.

    ``SAVE_DIRECTORY`` is injected as a pytest ``tmp_path`` so the notebook
    writes its dataset and exported models into a temporary directory rather
    than the source tree. Any notebook whose filename contains "mnist" must
    export both ``exported_model.aimodel`` and ``exported_model.mlpackage``.
    """
    pm.execute_notebook(
        str(notebook),
        str(tmp_path / notebook.name),
        parameters={"SAVE_DIRECTORY": str(tmp_path)},
        kernel_name="python3",
        execution_timeout=NOTEBOOK_CELL_TIMEOUT_SECONDS,
    )

    if MNIST_TOKEN not in notebook.stem:
        return

    for name in EXPECTED_MNIST_EXPORTS:
        export_path = tmp_path / name
        assert export_path.exists(), (
            f"MNIST notebook {notebook.name} did not produce expected export: "
            f"{name} (looked in {tmp_path})"
        )
