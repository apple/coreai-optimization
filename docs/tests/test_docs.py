# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Test that documentation builds correctly.
"""

import subprocess
from pathlib import Path

import pytest

# Lines of build output to quote in the failure message. `make docs` emits a few
# hundred lines, almost all of it Sphinx progress; the cause of a failure is
# always in the last handful.
_FAILURE_CONTEXT_LINES = 40


def _build_failure_message(returncode: int, output: str) -> str:
    """Return an assertion message that carries the build's own diagnostics.

    pytest renders a failing fixture's ``CompletedProcess`` with the middle
    elided, so the actual error is the part that gets cut. Quoting the tail of the
    output inside the message puts it in every reported ERROR block, where it
    survives both that truncation and a trimmed CI log.
    """
    lines = output.splitlines()
    excerpt = lines[-_FAILURE_CONTEXT_LINES:]
    omitted = len(lines) - len(excerpt)
    header = f"`make docs` failed with exit code {returncode}."
    if omitted > 0:
        header += f" Last {len(excerpt)} of {len(lines)} output lines ({omitted} omitted):"
    else:
        header += " Full output:"
    # Blank line before the block so pytest's `E ` prefixes stay readable.
    return "\n".join([header, ""] + excerpt)


@pytest.fixture(scope="session")
def built_docs(repo_root: Path) -> Path:
    """Build the docs once with ``make docs`` and return the HTML output directory.

    Session-scoped so both docs tests share a single build and each remains runnable
    in isolation, without an implicit ordering dependency on another test building first.
    """
    # Combine stdout and stderr so errors appear in context.
    result = subprocess.run(
        ["make", "docs"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Always print output for visibility.
    print(f"\n=== make docs output ===\n{result.stdout}")

    assert result.returncode == 0, _build_failure_message(result.returncode, result.stdout)
    return repo_root / "docs" / "build" / "html"


def test_docs_built_index_html(built_docs: Path) -> None:
    """Verify the docs build emitted the HTML entry point (index.html)."""
    assert (built_docs / "index.html").exists(), (
        f"Expected documentation output at {built_docs / 'index.html'} does not exist"
    )


def test_docs_built_llms_txt(built_docs: Path) -> None:
    """Verify the docs build emitted the llms.txt / llms-full.txt summaries."""
    assert (built_docs / "llms.txt").exists(), (
        f"Expected llms.txt at {built_docs / 'llms.txt'} does not exist"
    )
    assert (built_docs / "llms-full.txt").exists(), (
        f"Expected llms-full.txt at {built_docs / 'llms-full.txt'} does not exist"
    )


def test_mermaid_svgs_generated(built_docs: Path) -> None:
    """Verify ``make docs`` rendered mermaid diagrams to real, non-empty SVGs."""
    images_dir = built_docs / "_images"
    svgs = list(images_dir.glob("mermaid-*.svg"))
    assert svgs, f"No mermaid SVGs were generated in {images_dir}"

    # A real mermaid diagram contains an ``<svg>`` root; an empty/degenerate placeholder
    # (the silent-failure mode) does not.
    malformed = [p.name for p in svgs if b"<svg" not in p.read_bytes()]
    assert not malformed, (
        f"Mermaid SVGs were generated but are empty/malformed (no <svg> root): {malformed}"
    )
