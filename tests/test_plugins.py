# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for entry-point plugin discovery."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from coreai_opt import _plugins

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class _FakeEntryPoint:
    """Stands in for ``importlib.metadata.EntryPoint``, whose ``load`` reads real metadata."""

    name: str
    value: str
    loaded: Callable[[], Any] | Exception

    def load(self) -> Callable[[], Any]:
        if isinstance(self.loaded, Exception):
            raise self.loaded
        return self.loaded


def _patch_entry_points(
    monkeypatch: pytest.MonkeyPatch,
    *entry_points: _FakeEntryPoint,
) -> list[str]:
    """Make ``load_plugins`` see exactly ``entry_points``, and record the group asked for."""
    groups: list[str] = []

    def fake_entry_points(*, group: str) -> tuple[_FakeEntryPoint, ...]:
        groups.append(group)
        return entry_points

    monkeypatch.setattr(_plugins, "entry_points", fake_entry_points)
    return groups


def test_an_advertised_callable_is_called(monkeypatch):
    """Discovery is the activation step: finding a plugin means calling it."""
    called = []
    groups = _patch_entry_points(
        monkeypatch,
        _FakeEntryPoint("ext", "ext:register", lambda: called.append("ext")),
    )

    _plugins.load_plugins()

    assert called == ["ext"]
    assert groups == [_plugins.PLUGIN_GROUP]


def test_no_plugins_installed_is_a_no_op(monkeypatch):
    """The OSS case. Nothing advertised means nothing changes and nothing warns."""
    _patch_entry_points(monkeypatch)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _plugins.load_plugins()


def test_a_plugin_that_fails_to_import_warns_and_is_skipped(monkeypatch):
    """A broken plugin must not take `import coreai_opt` down with it."""
    _patch_entry_points(
        monkeypatch,
        _FakeEntryPoint("broken", "broken:register", ModuleNotFoundError("no module")),
    )

    with pytest.warns(RuntimeWarning, match="'broken'"):
        _plugins.load_plugins()


def test_a_plugin_that_raises_when_called_warns_and_is_skipped(monkeypatch):
    """Failing during registration is as survivable as failing to import."""

    def explode() -> None:
        msg = "registration failed"
        raise RuntimeError(msg)

    _patch_entry_points(monkeypatch, _FakeEntryPoint("broken", "broken:register", explode))

    with pytest.warns(RuntimeWarning, match="registration failed"):
        _plugins.load_plugins()


def test_one_broken_plugin_does_not_stop_the_next(monkeypatch):
    """Plugins are independent, so the loop continues past a failure."""
    called = []
    _patch_entry_points(
        monkeypatch,
        _FakeEntryPoint("broken", "broken:register", ValueError("boom")),
        _FakeEntryPoint("healthy", "healthy:register", lambda: called.append("healthy")),
    )

    with pytest.warns(RuntimeWarning):
        _plugins.load_plugins()

    assert called == ["healthy"]
