# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import tomllib

import pytest
from packaging import version

from coreai_opt._utils.repo_utils import find_repo_root
from coreai_opt._utils.version_utils import (
    _MAX_TESTED_TORCH,
    torchao_torch_incompatibility,
    untested_torch_version,
)

INCOMPATIBLE = [
    ("0.18.0", "2.8.0"),
    ("0.18.0", "2.9.1"),
    ("0.18.0", "2.10.0"),
    ("0.18.0", "2.10.0+cu128"),
    ("0.19.0", "2.10.0"),
    # A source-built torchao reports a local version, which sorts above the base.
    ("0.18.0+gitabc1234", "2.10.0"),
]

COMPATIBLE = [
    # torch is new enough.
    ("0.18.0", "2.11.0"),
    ("0.18.0", "2.11.0+cu128"),
    ("0.18.0", "2.12.0.dev20260805+cu128"),
    # A 2.11 pre-release counts as 2.11.
    ("0.18.0", "2.11.0rc1"),
    # torchao still supports older torch.
    ("0.17.0", "2.8.0"),
    ("0.16.0", "2.10.0"),
    ("0.15.0", "2.8.0"),
]


@pytest.mark.parametrize(("torchao_version", "torch_version"), INCOMPATIBLE)
def test_returns_message_for_incompatible_pair(torchao_version, torch_version):
    message = torchao_torch_incompatibility(torchao_version, torch_version)

    assert message is not None
    assert torchao_version in message
    assert torch_version in message
    assert "https://github.com/pytorch/ao/releases/tag/v0.18.0" in message


@pytest.mark.parametrize(("torchao_version", "torch_version"), COMPATIBLE)
def test_returns_none_for_compatible_pair(torchao_version, torch_version):
    assert torchao_torch_incompatibility(torchao_version, torch_version) is None


UNTESTED_TORCH = [
    "2.15.0",
    "2.15.0+cu128",
    "2.16.0.dev20260805+cu128",
    "2.15.0rc1",
    "3.0.0",
]

TESTED_TORCH = [
    "2.8.0",
    "2.10.0",
    "2.14.0",
    "2.14.5",
    "2.14.0+cu128",
    "2.14.0rc1",
    "2.14.0+gitabc1234",
]


@pytest.mark.parametrize("torch_version", UNTESTED_TORCH)
def test_returns_message_for_untested_torch(torch_version):
    message = untested_torch_version(torch_version)

    assert message is not None
    assert torch_version in message
    assert "2.14" in message


@pytest.mark.parametrize("torch_version", TESTED_TORCH)
def test_returns_none_for_tested_torch(torch_version):
    assert untested_torch_version(torch_version) is None


def test_max_tested_torch_matches_pyproject():
    """_MAX_TESTED_TORCH must track the highest torch_2_* dependency group.

    Repo-only: pyproject.toml is not shipped in the wheel, so this must never
    move into tests/test_smoke.py (which runs against an installed wheel).
    """
    pyproject_path = find_repo_root(__file__) / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        pyproject = tomllib.load(f)

    groups = pyproject["dependency-groups"]
    torch_pins = [
        version.parse(spec.split("==")[1])
        for name, specs in groups.items()
        if name.startswith("torch_2_")
        for spec in specs
        if isinstance(spec, str) and spec.startswith("torch==")
    ]

    assert torch_pins, "no torch_2_* groups pinning torch== found in pyproject.toml"
    highest = max(torch_pins)
    expected = f"{highest.major}.{highest.minor}"

    assert _MAX_TESTED_TORCH == expected, (
        f"_MAX_TESTED_TORCH is {_MAX_TESTED_TORCH} but the highest torch_2_* "
        f"group pins torch {highest}. Update _MAX_TESTED_TORCH in "
        f"src/coreai_opt/_utils/version_utils.py."
    )
