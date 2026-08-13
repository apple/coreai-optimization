# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import pytest

from coreai_opt._utils.version_utils import torchao_torch_incompatibility

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
