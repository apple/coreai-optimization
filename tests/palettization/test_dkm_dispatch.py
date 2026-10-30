# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from unittest import mock

import pytest
import torch

from coreai_opt.palettization.kmeans.dkm import _dispatch


@pytest.fixture(autouse=True)
def _clear_caches():
    _dispatch._providers.cache_clear()
    _dispatch.get_forward.cache_clear()
    yield
    _dispatch._providers.cache_clear()
    _dispatch.get_forward.cache_clear()


def _plugin_entry_point(plugin):
    entry_point = mock.Mock()
    entry_point.load.return_value = plugin
    return entry_point


def test_no_providers_returns_none_and_logs(caplog):
    with (
        mock.patch.object(_dispatch, "entry_points", return_value=[]),
        caplog.at_level("INFO", logger=_dispatch.logger.name),
    ):
        assert _dispatch.get_forward(torch.device("cpu")) is None
    assert "DKM: using the PyTorch implementation on cpu" in caplog.text


def test_providers_are_chained_and_first_applicable_wins(caplog):
    def forward(*args, **kwargs):
        raise NotImplementedError

    def plugin(providers):
        return [
            *providers,
            _dispatch.Provider("unusable", lambda device: None),
            _dispatch.Provider("good", lambda device: forward),
        ]

    with (
        mock.patch.object(_dispatch, "entry_points", return_value=[_plugin_entry_point(plugin)]),
        caplog.at_level("INFO", logger=_dispatch.logger.name),
    ):
        assert _dispatch.get_forward(torch.device("cpu")) is forward
    assert "DKM: using the good provider on cpu" in caplog.text


def test_failing_plugin_is_skipped_with_warning(caplog):
    def plugin(providers):
        raise ImportError("boom")

    with (
        mock.patch.object(_dispatch, "entry_points", return_value=[_plugin_entry_point(plugin)]),
        caplog.at_level("WARNING", logger=_dispatch.logger.name),
    ):
        assert _dispatch.get_forward(torch.device("cpu")) is None
    assert "failed to load; skipping" in caplog.text
