# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for device and toolchain availability probes."""

import builtins
import sys
import types

import torch

from coreai_opt._utils import device_utils


def test_cuda_available_matches_torch():
    assert device_utils.cuda_available() == torch.cuda.is_available()


def test_nvcc_available_on_path(monkeypatch):
    monkeypatch.setattr(device_utils.shutil, "which", lambda _name: "/usr/bin/nvcc")
    assert device_utils.nvcc_available() is True


def test_nvcc_available_via_cuda_home(monkeypatch, tmp_path):
    monkeypatch.setattr(device_utils.shutil, "which", lambda _name: None)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "nvcc").write_text("")
    monkeypatch.setattr(device_utils, "CUDA_HOME", str(tmp_path))
    assert device_utils.nvcc_available() is True


def test_nvcc_unavailable(monkeypatch):
    monkeypatch.setattr(device_utils.shutil, "which", lambda _name: None)
    monkeypatch.setattr(device_utils, "CUDA_HOME", None)
    assert device_utils.nvcc_available() is False


def test_triton_available_true(monkeypatch):
    # A stub module in sys.modules makes ``import triton`` succeed regardless of
    # whether triton is actually installed.
    monkeypatch.setitem(sys.modules, "triton", types.ModuleType("triton"))
    assert device_utils.triton_available() is True


def test_triton_unavailable_on_import_error(monkeypatch):
    monkeypatch.delitem(sys.modules, "triton", raising=False)
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "triton":
            raise ImportError("no triton")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert device_utils.triton_available() is False
