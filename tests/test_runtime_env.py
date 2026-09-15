# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for tests/config/env.py and the hooks in tests/config/plugin.py."""

import os
import random

import pytest

from tests.config.env import (
    COMPUTE_UNIT_KIND_ENV_VAR,
    COMPUTE_UNIT_KINDS,
    DEFAULT_COMPUTE_UNIT_KIND,
    USE_LOCAL_COREAI_ENV_VAR,
    apply_thread_defaults,
    get_compute_unit_kind,
    select_compute_unit_kind,
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop both variables, letting monkeypatch restore the session's values on teardown."""
    monkeypatch.delenv(COMPUTE_UNIT_KIND_ENV_VAR, raising=False)
    monkeypatch.delenv(USE_LOCAL_COREAI_ENV_VAR, raising=False)


class TestSelectComputeUnitKind:
    """Tests for select_compute_unit_kind."""

    def test_interpreter_pins_the_bundled_runtime(self, clean_env) -> None:
        select_compute_unit_kind("interpreter")
        assert os.environ[USE_LOCAL_COREAI_ENV_VAR] == "1"

    def test_interpreter_defers_to_an_explicit_setting(self, clean_env, monkeypatch) -> None:
        """An explicit USE_LOCAL_COREAI=0 must survive, so the override still works."""
        monkeypatch.setenv(USE_LOCAL_COREAI_ENV_VAR, "0")
        select_compute_unit_kind("interpreter")
        assert os.environ[USE_LOCAL_COREAI_ENV_VAR] == "0"

    @pytest.mark.parametrize("kind", ["cpu", "gpu", "neural_engine"])
    def test_real_compute_units_drop_the_bundled_runtime(
        self, clean_env, monkeypatch, kind: str
    ) -> None:
        monkeypatch.setenv(USE_LOCAL_COREAI_ENV_VAR, "1")
        select_compute_unit_kind(kind)
        assert USE_LOCAL_COREAI_ENV_VAR not in os.environ

    def test_rejects_an_unknown_kind(self, clean_env) -> None:
        with pytest.raises(ValueError, match="Unknown compute unit kind"):
            select_compute_unit_kind("quantum")


class TestGetComputeUnitKind:
    """Tests for get_compute_unit_kind."""

    @pytest.mark.parametrize("kind", COMPUTE_UNIT_KINDS)
    def test_round_trips_every_kind(self, clean_env, kind: str) -> None:
        select_compute_unit_kind(kind)
        assert get_compute_unit_kind() == kind

    def test_falls_back_when_nothing_selected(self, clean_env) -> None:
        """Importing a test helper outside a pytest session still has to work."""
        assert get_compute_unit_kind() == DEFAULT_COMPUTE_UNIT_KIND

    def test_rejects_an_unknown_kind_from_the_environment(self, clean_env, monkeypatch) -> None:
        monkeypatch.setenv(COMPUTE_UNIT_KIND_ENV_VAR, "quantum")
        with pytest.raises(ValueError, match="Unknown compute unit kind"):
            get_compute_unit_kind()


class TestApplyThreadDefaults:
    """Tests for apply_thread_defaults."""

    def test_caps_openmp_on_linux(self, monkeypatch) -> None:
        monkeypatch.setattr("sys.platform", "linux")
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        apply_thread_defaults()
        assert os.environ["OMP_NUM_THREADS"] == "1"

    def test_defers_to_an_explicit_thread_count(self, monkeypatch) -> None:
        monkeypatch.setattr("sys.platform", "linux")
        monkeypatch.setenv("OMP_NUM_THREADS", "8")
        apply_thread_defaults()
        assert os.environ["OMP_NUM_THREADS"] == "8"

    def test_is_a_no_op_off_linux(self, monkeypatch) -> None:
        monkeypatch.setattr("sys.platform", "darwin")
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        apply_thread_defaults()
        assert "OMP_NUM_THREADS" not in os.environ


class TestPytestPlugin:
    """Tests for the plugin hooks in tests/config/plugin.py."""

    def test_configure_published_the_selected_compute_unit(
        self,
        pytestconfig: pytest.Config,
    ) -> None:
        """The running session's --compute-unit-kind must be readable via the getter."""
        assert get_compute_unit_kind() == pytestconfig.getoption("--compute-unit-kind")

    @pytest.mark.seed(123)
    def test_seed_marker_is_registered_and_applied(self) -> None:
        """A registered marker plus a deterministic draw proves the fixture ran."""
        assert random.random() == random.Random(123).random()
