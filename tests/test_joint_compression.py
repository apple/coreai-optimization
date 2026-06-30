# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""End-to-end accuracy tests for P4-A8 compression (palettization + quantization)."""

from copy import deepcopy

import pytest
import torch

import tests.utils as utils
from coreai_opt import ExportBackend
from coreai_opt.palettization.kmeans import KMeansPalettizer
from coreai_opt.palettization.spec.fake_palettize import _FakePalettizeImplBase
from coreai_opt.quantization import Quantizer
from tests.fixtures.compression import ParametrizedP4A8CompressionConfigs
from tests.test_utils.general import COREAI_AVAILABLE

batch_size = 128
num_calibration_batches = 17


@pytest.mark.skipif(not COREAI_AVAILABLE, reason="Requires coreai")
@pytest.mark.slow
def test_p4a8_compression_mnist_accuracy(
    mnist_pretrained_model: torch.nn.Module,
    mnist_dataset: tuple,
    parametrized_p4a8_compression_config: ParametrizedP4A8CompressionConfigs,
) -> None:
    """Verify P4-A8 compression preserves acceptable accuracy on MNIST.

    Runs the full P4-A8 compression pipeline: palettize weights (4-bit) then
    quantize activations (int8 symmetric), with calibration on training data.
    Asserts accuracy stays above a threshold after both compressions.
    """
    model = deepcopy(mnist_pretrained_model)
    config = parametrized_p4a8_compression_config
    example_input = torch.rand(1, 1, 28, 28)

    train_loader, test_loader = utils.setup_data_loaders(mnist_dataset, batch_size)

    # Baseline accuracy
    baseline_acc = utils.eval_model(model, test_loader)
    assert baseline_acc > 97.0, f"Pre-trained MNIST model accuracy too low: {baseline_acc:.2f}%"

    # Palettize weights
    palettizer = KMeansPalettizer(model, config.palett_config)
    prepared_palettized = palettizer.prepare((example_input,))

    # MNIST model has 6 weight-bearing layers (conv1, conv2, conv_transpose1,
    # conv_transpose2, dense1, dense2). With 4-bit per-tensor palettization, all
    # 6 layers are palettized.
    palettized_count = utils.count_weight_parametrizations(
        prepared_palettized, _FakePalettizeImplBase
    )
    assert palettized_count == 6, f"Expected 6 palettized layers, got {palettized_count}"

    palettized = palettizer.finalize(backend=ExportBackend.CoreAI)

    # Quantize activations on the palettized model
    quantizer = Quantizer(palettized, config.quant_config)
    prepared = quantizer.prepare((example_input,))

    # Calibrate activation ranges with training data
    with quantizer.calibration_mode():
        prepared.eval()
        for i, (data, _target) in enumerate(train_loader):
            prepared(data)
            if i >= num_calibration_batches - 1:
                break

    # Evaluate post-calibration
    post_calib_acc = utils.eval_model(prepared, test_loader)
    assert post_calib_acc > 90.0, (
        f"Joint compression accuracy too low after calibration: {post_calib_acc:.2f}% "
        f"(baseline: {baseline_acc:.2f}%)"
    )

    # Finalize and verify accuracy is preserved
    finalized = quantizer.finalize(backend=ExportBackend._TORCH)
    finalized_acc = utils.eval_model(finalized, test_loader)
    assert post_calib_acc == finalized_acc, (
        f"Post-calibration accuracy ({post_calib_acc:.2f}%) != "
        f"post-finalize accuracy ({finalized_acc:.2f}%)"
    )
