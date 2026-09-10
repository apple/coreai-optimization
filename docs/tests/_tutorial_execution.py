# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause


"""Collection and execution helpers for tutorial notebook tests."""

from pathlib import Path

import papermill as pm
from nbformat import NotebookNode

NOTEBOOK_CELL_TIMEOUT_SECONDS = 300


def collect_notebooks(dirs: list[Path]) -> list[Path]:
    """Return the ``*.ipynb`` files under each directory (missing dirs yield none).

    Args:
        dirs (list[Path]): List of directories containing ipynb files.

    Returns:
        list[Path]: List of all ipynb files found.
    """
    notebooks: list[Path] = []
    for directory in dirs:
        notebooks.extend(sorted(directory.glob("*.ipynb")))
    return notebooks


def run_tutorial_notebook(
    notebook: Path, save_dir: Path, parameters: dict | None = None
) -> NotebookNode:
    """Execute a tutorial notebook end-to-end with papermill and return it.

    Args:
        notebook (Path): The path to the notebook to execute.
        save_dir (Path): Where papermill writes the executed copy of the notebook.
        parameters (dict | None): Optional parameters injected into the notebook's
            ``parameters``-tagged cell.

    Returns:
        NotebookNode: The executed notebook.
    """
    return pm.execute_notebook(
        str(notebook),
        str(save_dir / notebook.name),
        parameters=parameters or {},
        kernel_name="python3",
        execution_timeout=NOTEBOOK_CELL_TIMEOUT_SECONDS,
    )
