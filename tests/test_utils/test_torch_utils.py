# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for torch_utils."""

import pytest
import torch
from torchao.quantization.pt2e import allow_exported_model_train_eval

from coreai_opt._utils.fx_utils import normalize_module_fqn
from coreai_opt._utils.torch_utils import (
    move_model_to_eval,
    move_model_to_train,
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
