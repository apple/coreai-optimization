# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause


"""Test that the tutorial notebooks execute without errors."""

from pathlib import Path

import pytest
from _tutorial_execution import collect_notebooks, run_tutorial_notebook

from coreai_opt._utils.repo_utils import find_repo_root

_repo_root = find_repo_root(__file__)
_notebooks = collect_notebooks([_repo_root / "docs" / "src" / "tutorials"])


def _notebook_id(path: Path) -> str:
    return path.stem


def test_tutorials_dir_is_non_empty() -> None:
    """Guard against an empty parametrize set silently producing zero tests."""
    assert _notebooks, "No tutorial notebooks found"


@pytest.mark.parametrize("notebook", _notebooks, ids=_notebook_id)
def test_tutorial_notebook_executes(notebook: Path, tmp_path: Path) -> None:
    """Execute a tutorial notebook end-to-end and verify its expected exports."""
    run_tutorial_notebook(notebook, tmp_path, parameters={"SAVE_DIRECTORY": str(tmp_path)})

    # MNIST tutorials must export a deployable model.
    if "mnist" in notebook.stem:
        export_path = tmp_path / "exported_model.aimodel"
        assert export_path.exists(), (
            f"{notebook.name} did not produce expected export {export_path.name} "
            f"(looked in {tmp_path})"
        )
