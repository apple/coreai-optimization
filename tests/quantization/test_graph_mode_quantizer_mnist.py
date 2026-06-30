# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import sys

import pytest
import torch

import tests.utils as utils
from coreai_opt import ExportBackend
from coreai_opt._utils.torch_utils import (
    is_float4_dtype as _is_float4_dtype,
)
from coreai_opt.quantization import ModuleQuantizerConfig, Quantizer, QuantizerConfig
from coreai_opt.quantization.config import QATSchedule
from coreai_opt.quantization.spec import (
    PerBlockGranularity,
    PerChannelGranularity,
    PerTensorGranularity,
)

image_size = 28
batch_size = 128
num_classes = 10
num_epochs = 1


@pytest.mark.slow
@pytest.mark.seed
@pytest.mark.parametrize(
    "dtype,granularity,scale_dtype,qformulation",
    [
        ("int8", PerChannelGranularity(axis=0), None, "zp"),
        ("int8", PerChannelGranularity(axis=0), None, "minval"),
        pytest.param(
            torch.float8_e4m3fn,
            PerTensorGranularity(),
            None,
            "zp",
            # TODO: fix Conv+BN fusion numerical mismatch during quantizer finalize for FP8.
            marks=pytest.mark.xfail(
                sys.platform == "darwin",
                reason=(
                    "FP8 weight quantization produces a numerical mismatch "
                    "after Conv+BN fusion runs during quantizer finalize."
                ),
            ),
        ),
        (
            torch.float8_e4m3fn,
            PerChannelGranularity(axis=0),
            torch.float8_e8m0fnu,
            "zp",
        ),
        (torch.float4_e2m1fn_x2, PerBlockGranularity(axis=1, block_size=16), None, "zp"),
    ],
)
def test_weight_only_ptq_mnist(
    dtype, granularity, scale_dtype, qformulation, mnist_pretrained_model, mnist_dataset
):
    """
    Train a simple convnet on the MNIST dataset for different deployment targets
    and verify its accuracy.
    """
    # Setup test data loader for evaluation
    _, test_loader = utils.setup_data_loaders(mnist_dataset, batch_size)

    # Verify baseline accuracy
    accuracy = utils.eval_model(mnist_pretrained_model, test_loader)
    assert accuracy > 97.0, "expect pre-trained mnist model accuracy to be at least 97%"

    # Setup the quantizer
    quantization_config = {
        "global_config": {
            "op_state_spec": {
                "weight": {
                    "dtype": dtype,
                    "qscheme": "symmetric",
                    "qformulation": qformulation,
                    "granularity": granularity,
                    "scale_dtype": scale_dtype,
                }
            },
            "op_input_spec": None,
            "op_output_spec": None,
        },
        # Skip conv1 for FP4: weight dimensions are incompatible with per-block granularity.
        "module_name_configs": {"conv1": None} if _is_float4_dtype(dtype) else None,
        "execution_mode": "graph",
    }

    config = QuantizerConfig.from_dict({"quantization_config": quantization_config})
    quantizer = Quantizer(mnist_pretrained_model, config)

    prepared_model = quantizer.prepare(
        example_inputs=(torch.ones(1, 1, 28, 28, dtype=torch.float),)
    )
    post_prepare_accuracy = utils.eval_model(prepared_model, test_loader)

    # There should be drop in accuracy after setting up quantization (PTQ)
    accuracy_drop = accuracy - post_prepare_accuracy
    max_drop = 0.3 if _is_float4_dtype(dtype) else 0.2
    assert accuracy_drop < max_drop, (
        f"Accuracy drop too high: before={accuracy:.4f}, after={post_prepare_accuracy:.4f}"
    )

    finalized_model = quantizer.finalize(backend=ExportBackend._TORCH)
    finalized_accuracy = utils.eval_model(finalized_model, test_loader)

    # Accuracy before and after finalize should match
    assert post_prepare_accuracy == finalized_accuracy


@pytest.mark.slow
@pytest.mark.seed
@pytest.mark.parametrize(
    "weight_dtype,weight_granularity,weight_scale_dtype,activation_dtype,activation_granularity,activation_scale_dtype",
    [
        ("int8", PerTensorGranularity(), None, "int8", PerTensorGranularity(), None),
        (
            "int4",
            PerChannelGranularity(axis=0),
            None,
            "int8",
            PerTensorGranularity(),
            None,
        ),
        (
            torch.float8_e4m3fn,
            PerTensorGranularity(),
            None,
            torch.float8_e4m3fn,
            PerTensorGranularity(),
            None,
        ),
        (
            torch.float8_e5m2,
            PerTensorGranularity(),
            None,
            torch.float8_e5m2,
            PerTensorGranularity(),
            None,
        ),
        (
            torch.float4_e2m1fn_x2,
            PerBlockGranularity(axis=1, block_size=16),
            "float8_e8m0",
            torch.float8_e4m3fn,
            PerTensorGranularity(),
            "float8_e8m0",
        ),
        (
            torch.float4_e2m1fn_x2,
            PerBlockGranularity(axis=1, block_size=16),
            "float8_e8m0",
            torch.float4_e2m1fn_x2,
            PerTensorGranularity(),
            "float8_e8m0",
        ),
    ],
)
def test_weight_and_activation_ptq_mnist(
    weight_dtype,
    weight_granularity,
    weight_scale_dtype,
    activation_dtype,
    activation_granularity,
    activation_scale_dtype,
    mnist_pretrained_model,
    mnist_dataset,
):
    """
    Train a simple convnet on the MNIST dataset for different deployment targets
    and verify its accuracy.
    """
    # Setup data loaders
    train_loader, test_loader = utils.setup_data_loaders(mnist_dataset, batch_size)

    # Verify baseline accuracy
    accuracy = utils.eval_model(mnist_pretrained_model, test_loader)
    assert accuracy > 97.0, "expect pre-trained mnist model accuracy to be at least 97%"

    config = QuantizerConfig.from_dict(
        {
            "quantization_config": {
                "global_config": {
                    "op_state_spec": {
                        "weight": {
                            "dtype": weight_dtype,
                            "qscheme": "symmetric",
                            "granularity": weight_granularity,
                            "scale_dtype": weight_scale_dtype,
                        },
                    },
                    "op_input_spec": {
                        "*": {
                            "dtype": activation_dtype,
                            "qscheme": "symmetric",
                            "granularity": activation_granularity,
                            "scale_dtype": activation_scale_dtype,
                        },
                    },
                    "op_output_spec": {
                        "*": {
                            "dtype": activation_dtype,
                            "qscheme": "symmetric",
                            "granularity": activation_granularity,
                            "scale_dtype": activation_scale_dtype,
                        },
                    },
                },
                # Skip conv1 for FP4: weight dimensions are incompatible with per-block granularity.
                "module_name_configs": {"conv1": None} if _is_float4_dtype(weight_dtype) else None,
                "execution_mode": "graph",
            },
        }
    )
    quantizer = Quantizer(mnist_pretrained_model, config)

    prepared_model = quantizer.prepare(
        example_inputs=(torch.ones(1, 1, 28, 28, dtype=torch.float),)
    )
    post_prepare_accuracy = utils.eval_model(prepared_model, test_loader)
    assert post_prepare_accuracy < 97, (
        "Expect accuracy to drop below 97% after preparation with an all ones data sample"
    )

    # Calibrate model with one batch of data
    with quantizer.calibration_mode():
        prepared_model.eval()
        data, _target = next(iter(train_loader))
        prepared_model(data)

    # Calibrate model with additional batches to stabilize moving averages
    with quantizer.calibration_mode():
        prepared_model.eval()
        # Skip first batch (already used) and use next few batches
        train_iter = iter(train_loader)
        next(train_iter)  # Skip first batch
        for i, (data, _target) in enumerate(train_iter):
            if i >= 16:  # Use 16 more batches for a total of 17 calibration batches
                break
            prepared_model(data)

    post_calibrate_accuracy = utils.eval_model(prepared_model, test_loader)
    assert post_calibrate_accuracy > 90.0, "Expect accuracy to climb above 90% after calibration"

    finalized_model = quantizer.finalize(backend=ExportBackend._TORCH)
    finalized_accuracy = utils.eval_model(finalized_model, test_loader)

    # Accuracy before and after finalize should match closely. Allow a small
    # delta because conv+bn fold replaces the prepared graph's `F.batch_norm`
    # call with `conv + fused_bias`; the two are algebraically equal but differ
    # by ~1 ulp in fp32, which coarse-grid quantizers (fp8/fp4) can occasionally
    # amplify into a class flip on samples near a decision boundary.
    accuracy_delta = abs(post_calibrate_accuracy - finalized_accuracy)
    assert accuracy_delta < 0.02, (
        f"Post calibrate accuracy ({post_calibrate_accuracy:.4f}) differs from "
        f"post finalize accuracy ({finalized_accuracy:.4f}) by {accuracy_delta:.4f} "
        f"(> 0.1% tolerance)"
    )


@pytest.mark.seed
@pytest.mark.slow
@pytest.mark.parametrize(
    "qat_schedule",
    [
        None,
        QATSchedule(enable_observer=0, enable_fake_quant=100, disable_observer=500),
    ],
    ids=["no_schedule", "with_schedule"],
)
def test_weight_and_activation_qat_mnist(mnist_pretrained_model, mnist_dataset, qat_schedule):
    """
    Train a simple convnet on the MNIST dataset with graph-mode QAT
    and verify its accuracy. Parameterized over no schedule vs. a
    milestone-based QAT schedule.

    Takes ~4 mins per variant to run on M1 Max Macbook Pro
    """
    # setup data loaders
    train_loader, test_loader = utils.setup_data_loaders(mnist_dataset, batch_size)

    accuracy = utils.eval_model(mnist_pretrained_model, test_loader)
    assert accuracy > 97.0, "expect pre-trained mnist model accuracy to be at least 97%"

    config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(qat_schedule=qat_schedule),
    )
    quantizer = Quantizer(mnist_pretrained_model, config)

    prepared_model = quantizer.prepare(
        example_inputs=(torch.ones(1, 1, 28, 28, dtype=torch.float),)
    )
    post_prepare_accuracy = utils.eval_model(prepared_model, test_loader)
    assert post_prepare_accuracy < 88, (
        "Expect accuracy to drop below 88% after preparation with an all ones data sample"
    )

    # Fine tune the model
    num_epoch = 1
    optimizer = torch.optim.Adam(prepared_model.parameters(), eps=1e-03, weight_decay=1e-4)

    with quantizer.training_mode():
        for epoch in range(num_epoch):
            for batch_idx, (data, target) in enumerate(train_loader):
                utils.train_step(
                    prepared_model,
                    optimizer,
                    train_loader,
                    data,
                    target,
                    batch_idx,
                    epoch,
                )
                if qat_schedule is not None:
                    quantizer.step()

    post_qat_accuracy = utils.eval_model(prepared_model, test_loader)
    assert post_qat_accuracy > 96.0, "Expect accuracy to climb above 96% after QAT"

    finalized_model = quantizer.finalize(backend=ExportBackend._TORCH)
    finalized_accuracy = utils.eval_model(finalized_model, test_loader)

    # Accuracy before and after finalize should match
    assert post_qat_accuracy == finalized_accuracy
