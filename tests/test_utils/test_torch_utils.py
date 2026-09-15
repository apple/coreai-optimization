# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for torch_utils."""

import pytest
import torch
from coreai_torch._compression._floatx import Float4Tensor
from safetensors.torch import load_file
from torch import nn
from torchao.quantization.pt2e import allow_exported_model_train_eval

from coreai_opt._utils.fx_utils import normalize_module_fqn
from coreai_opt._utils.torch_utils import (
    mmap_module_state_dict,
    mmap_named_tensors,
    move_model_to_eval,
    move_model_to_train,
    normalize_axis,
)


class TestMoveModelContextManagers:
    """Test move_model_to_train / move_model_to_eval context managers."""

    @staticmethod
    def test_raises_when_exported_training_state_unknown():
        """Context managers raise when _exported_training is not set."""
        model = torch.nn.Linear(4, 4)
        exported = torch.export.export(model, (torch.randn(1, 4),)).module()
        allow_exported_model_train_eval(exported)

        with pytest.raises(RuntimeError, match=r"Call \.train\(\) or \.eval\(\)"):
            with move_model_to_train(exported):
                pass

        with pytest.raises(RuntimeError, match=r"Call \.train\(\) or \.eval\(\)"):
            with move_model_to_eval(exported):
                pass

    @staticmethod
    def test_works_when_exported_training_state_known():
        """Context managers work when _exported_training is set."""
        model = torch.nn.Linear(4, 4)
        exported = torch.export.export(model, (torch.randn(1, 4),)).module()
        allow_exported_model_train_eval(exported)

        exported.eval()

        with move_model_to_train(exported):
            pass

        with move_model_to_eval(exported):
            pass

    @staticmethod
    def test_works_with_explicit_original_state():
        """Context managers work when original_state is passed explicitly."""
        model = torch.nn.Linear(4, 4)
        exported = torch.export.export(model, (torch.randn(1, 4),)).module()
        allow_exported_model_train_eval(exported)

        # Should not raise when original_state is provided
        with move_model_to_train(exported, original_state=True):
            pass

        with move_model_to_eval(exported, original_state=False):
            pass


class TestNormalizeModuleFqn:
    """Test normalize_module_fqn path normalization."""

    @staticmethod
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("model.layers.0.norm", "model.layers.0.norm"),
            ("L['self'].model", "model"),
            ("L['fn'].model", "model"),
            ("L['args'][0].model.layers[0]", "model.layers.0"),
            (
                "_modules['model']._modules['layers']._modules['0']",
                "model.layers.0",
            ),
            ("L['self'].encoder.conv1", "encoder.conv1"),
            ("L['self']._modules['layer1']._modules['0'].conv1", "layer1.0.conv1"),
            ("layers[2].block[0].norm", "layers.2.block.0.norm"),
            ("", ""),
        ],
    )
    def test_normalize_module_fqn(raw: str, expected: str) -> None:
        """Verify various path formats are normalized correctly."""
        assert normalize_module_fqn(raw) == expected


class TestNormalizeAxis:
    """Test normalize_axis resolution of negative axes."""

    @staticmethod
    @pytest.mark.parametrize(
        ("axis", "ndim", "expected"),
        [
            (0, 2, 0),
            (1, 2, 1),
            (-1, 2, 1),
            (-2, 2, 0),
            (-4, 4, 0),
            (-1, 1, 0),
            # Out of range passes through unchanged; the caller validates.
            (-5, 2, -3),
            (3, 2, 3),
        ],
    )
    def test_resolves_to_non_negative(axis: int, ndim: int, expected: int) -> None:
        """A negative axis resolves to its non-negative equivalent."""
        assert normalize_axis(axis, ndim) == expected


class TestMmapModuleStateDict:
    """Test mmap_module_state_dict serialization and reload."""

    @staticmethod
    def test_standard_tensors_roundtrip(tmp_path):
        """Standard (non-subbyte) tensors are saved and reloaded correctly."""
        model = nn.Linear(8, 4, bias=True)
        original_weight = model.weight.data.clone()
        original_bias = model.bias.data.clone()

        mmap_module_state_dict(model, tmp_path / "model.safetensors")

        assert torch.equal(model.weight.data, original_weight)
        assert torch.equal(model.bias.data, original_bias)

    @staticmethod
    def test_mixed_standard_and_float4(tmp_path):
        """Module with both standard and Float4Tensor parameters roundtrips."""

        module = nn.Module()
        module.register_buffer("normal", torch.randn(4, 4))
        uint8_data = torch.randint(0, 255, (2, 8), dtype=torch.uint8)
        module.register_buffer("compressed", Float4Tensor(uint8_data))

        original_normal = module.normal.clone()

        mmap_module_state_dict(module, tmp_path / "model.safetensors")

        assert torch.equal(module.normal, original_normal)
        assert isinstance(module.compressed, Float4Tensor)
        assert torch.equal(module.compressed.elem, uint8_data)

    @staticmethod
    def test_raises_on_non_cpu_tensor(tmp_path, accelerator_device):
        """Raises ValueError when a tensor is not on CPU."""
        model = nn.Linear(4, 4).to(accelerator_device)

        with pytest.raises(ValueError, match="requires CPU tensors"):
            mmap_module_state_dict(model, tmp_path / "model.safetensors")


class TestMmapNamedTensors:
    """Test mmap_named_tensors partial remapping of a module's tensors."""

    @staticmethod
    def _module_with_buffers():
        module = nn.Module()
        module.register_buffer("kept", torch.randn(4, 4))
        module.register_buffer("moved", torch.randn(8, 8))
        return module

    @staticmethod
    def test_remaps_only_the_named_tensors(tmp_path):
        """The named buffer is remapped and the others are left alone."""
        module = TestMmapNamedTensors._module_with_buffers()
        original_moved = module.moved.clone()
        original_kept_ptr = module.kept.data_ptr()

        path = tmp_path / "partial.safetensors"
        mmap_named_tensors(module, path, ["moved"])

        assert torch.equal(module.moved, original_moved)
        assert module.kept.data_ptr() == original_kept_ptr, "an unnamed buffer was remapped"
        assert set(load_file(path)) == {"moved"}, "the file must hold only the named tensor"
        assert not module.moved.untyped_storage().resizable(), (
            "the remapped buffer is still an ordinary allocation, so it was not mmap-backed"
        )
        assert "moved" in module.state_dict(), "the remapped tensor not in buffer registry"

    @staticmethod
    def test_float4_tensor_is_rewrapped(tmp_path):
        """A ``Float4Tensor`` buffer comes back as a ``Float4Tensor``, not a raw elem."""
        module = nn.Module()
        uint8_data = torch.randint(0, 255, (2, 8), dtype=torch.uint8)
        module.register_buffer("compressed", Float4Tensor(uint8_data))

        mmap_named_tensors(module, tmp_path / "fp4.safetensors", ["compressed"])

        assert isinstance(module.compressed, Float4Tensor)
        assert torch.equal(module.compressed.elem, uint8_data)

    @staticmethod
    def test_raises_on_non_tensor_name(tmp_path):
        """A name that does not resolve to a tensor is rejected."""
        module = nn.Module()
        module.not_a_tensor = "hello"

        with pytest.raises(TypeError, match="expected a tensor"):
            mmap_named_tensors(module, tmp_path / "bad.safetensors", ["not_a_tensor"])

    @staticmethod
    def test_raises_on_non_cpu_tensor(tmp_path, accelerator_device):
        """Raises ValueError when a named tensor is not on CPU."""
        module = nn.Module()
        module.register_buffer("moved", torch.randn(4, 4, device=accelerator_device))

        with pytest.raises(ValueError, match="requires CPU tensors"):
            mmap_named_tensors(module, tmp_path / "module.safetensors", ["moved"])
