# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from collections import OrderedDict

import pytest
import torch

from coreai_opt.palettization.kmeans.dkm import _dispatch, _forward


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch):
    monkeypatch.setattr(_dispatch.DKMBackend, "REGISTRY", OrderedDict())
    _dispatch.get_forward.cache_clear()
    yield
    _dispatch.get_forward.cache_clear()


def test_no_backends_returns_pytorch_forward_and_logs(caplog):
    with caplog.at_level("INFO", logger=_dispatch.logger.name):
        assert _dispatch.get_forward(torch.device("cpu")) is _forward.dkm_forward
    assert "DKM: using the PyTorch implementation on cpu" in caplog.text


def test_first_applicable_backend_wins(caplog):
    def forward(*args, **kwargs):
        raise NotImplementedError

    _dispatch.DKMBackend.register("unusable")(lambda device: None)
    _dispatch.DKMBackend.register("good")(lambda device: forward)
    _dispatch.DKMBackend.register("later")(lambda device: pytest.fail("loaded after a match"))

    with caplog.at_level("INFO", logger=_dispatch.logger.name):
        assert _dispatch.get_forward(torch.device("cpu")) is forward
    assert "DKM: using the good backend on cpu" in caplog.text
