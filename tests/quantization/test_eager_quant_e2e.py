# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from collections import OrderedDict

import pytest
import torch
import torch.nn as nn
import torch.nn.init as init

import tests.utils as utils
from coreai_opt import ExportBackend
from coreai_opt._utils.torch_utils import (
    is_float4_dtype as _is_float4_dtype,
)
from coreai_opt.quantization import (
    ModuleQuantizerConfig,
    Quantizer,
    QuantizerConfig,
)
from coreai_opt.quantization.config import ExecutionMode, QATSchedule
from coreai_opt.quantization.spec import (
    PerBlockGranularity,
    PerChannelGranularity,
    PerTensorGranularity,
    QuantizationScheme,
    QuantizationSpec,
    default_weight_quantization_spec,
)


def test_eager_quantizer():
    model = nn.Sequential(
        OrderedDict(
            [
                ("linear", nn.Linear(10, 100)),
                ("relu", nn.ReLU()),
            ]
        )
    )
    config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={"weight": default_weight_quantization_spec()},
            op_input_spec=None,
            op_name_config=None,
        ),
        execution_mode="eager",
    )
    quantizer = Quantizer(model, config)

    prepared_model = quantizer.prepare(example_inputs=(torch.rand(10, 10),))

    assert prepared_model is not None


def test_eager_quantizer_export_non_strict():
    model = nn.Sequential(
        OrderedDict(
            [
                ("linear", nn.Linear(10, 100)),
                ("relu", nn.ReLU()),
            ]
        )
    )
    config = QuantizerConfig(
        execution_mode="eager",
    )
    quantizer = Quantizer(model, config)

    prepared_model = quantizer.prepare(example_inputs=(torch.rand(10, 10),))
    finalized_model = quantizer.finalize(prepared_model, backend=ExportBackend.CoreAI)
    exported_model = torch.export.export(finalized_model, (torch.randn(10, 10),), strict=False)
    assert exported_model is not None


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
        (torch.float8_e4m3fn, PerTensorGranularity(), None, "zp"),
        (
            torch.float8_e4m3fn,
            PerChannelGranularity(axis=0),
            torch.float8_e8m0fnu,
            "zp",
        ),
        (torch.float4_e2m1fn_x2, PerBlockGranularity(axis=1, block_size=16), None, "zp"),
    ],
)
def test_weight_ptq_mnist(
    dtype, granularity, scale_dtype, qformulation, mnist_pretrained_model, mnist_dataset
):
    """
    Train a simple convnet on the MNIST dataset for different deployment targets
    and verify its accuracy.
    """
    # setup data loaders
    _, test_loader = utils.setup_data_loaders(mnist_dataset, batch_size)

    # Verify model accuracy
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
        "execution_mode": "eager",
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

    finalized_model = quantizer.finalize(backend=ExportBackend.Torch)
    finalized_accuracy = utils.eval_model(finalized_model, test_loader)

    # Accuracy before and after finalize should match
    assert post_prepare_accuracy == finalized_accuracy


@pytest.mark.slow
@pytest.mark.seed
@pytest.mark.parametrize(
    "weight_dtype,weight_granularity,weight_scale_dtype,activation_dtype,activation_granularity,activation_scale_dtype",
    [
        pytest.param(
            "int4",
            PerChannelGranularity(axis=0),
            None,
            "int4",
            PerTensorGranularity(),
            None,
            id="int4_w_int4_a",
        ),
        pytest.param(
            torch.float8_e4m3fn,
            PerChannelGranularity(axis=0),
            None,
            torch.float8_e4m3fn,
            PerTensorGranularity(),
            None,
            id="fp8_e4m3fn_w_fp8_e4m3fn_a",
        ),
        pytest.param(
            torch.float8_e5m2,
            PerChannelGranularity(axis=0),
            None,
            torch.float8_e5m2,
            PerTensorGranularity(),
            None,
            id="fp8_e5m2_w_fp8_e5m2_a",
        ),
        pytest.param(
            torch.float4_e2m1fn_x2,
            PerBlockGranularity(axis=1, block_size=16),
            "float8_e8m0",
            torch.float8_e4m3fn,
            PerTensorGranularity(),
            "float8_e8m0",
            id="fp4_w_fp8_e4m3fn_a",
        ),
        pytest.param(
            torch.float4_e2m1fn_x2,
            PerBlockGranularity(axis=1, block_size=16),
            "float8_e8m0",
            torch.float4_e2m1fn_x2,
            PerTensorGranularity(),
            "float8_e8m0",
            id="fp4_w_fp4_a",
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

    Roughly takes ~30-40 seconds to run, if the pre-trained model can be loaded from
    S3 bucket. If not, then takes about ~2-3 mins.

    """
    # setup data loaders
    train_loader, test_loader = utils.setup_data_loaders(mnist_dataset, batch_size)

    # Verify model accuracy
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
                "execution_mode": "eager",
            },
        }
    )
    quantizer = Quantizer(mnist_pretrained_model, config)

    prepared_model = quantizer.prepare(
        example_inputs=(torch.ones(1, 1, 28, 28, dtype=torch.float),)
    )
    post_prepare_accuracy = utils.eval_model(prepared_model, test_loader)
    assert post_prepare_accuracy < 90.0, (
        "Expect accuracy to drop below 90% after preparation with an all ones data sample"
    )

    # Calibrate model with one batch of data
    with quantizer.calibration_mode():
        mnist_pretrained_model.eval()
        data, _ = next(iter(train_loader))
        prepared_model(data)

    # Calibrate model with additional batches to stabilize moving averages
    with quantizer.calibration_mode():
        mnist_pretrained_model.eval()
        # Skip first batch (already used) and use next few batches
        train_iter = iter(train_loader)
        next(train_iter)  # Skip first batch
        for i, (data, _target) in enumerate(train_iter):
            if i >= 16:  # Use 16 more batches for a total of 17 calibration batches
                break
            prepared_model(data)

    post_calibrate_accuracy = utils.eval_model(prepared_model, test_loader)
    assert post_calibrate_accuracy > 90.0, "Expect accuracy to climb above 90% after calibration"

    finalized_model = quantizer.finalize(backend=ExportBackend.Torch)
    finalized_accuracy = utils.eval_model(finalized_model, test_loader)

    # Accuracy before and after finalize should match
    assert post_calibrate_accuracy == finalized_accuracy, (
        f"Post calibrate accuracy ({post_calibrate_accuracy:.4f}) is not the same as "
        f"post finalize accuracy ({finalized_accuracy:.4f})"
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
    Train a simple convnet on the MNIST dataset with eager mode QAT
    and verify its accuracy. Parameterized over no schedule vs. a
    milestone-based QAT schedule.

    Takes ~3min per variant to run on an M2 Max Macbook Pro
    """
    # setup data loaders
    train_loader, test_loader = utils.setup_data_loaders(mnist_dataset, batch_size)

    accuracy = utils.eval_model(mnist_pretrained_model, test_loader)
    assert accuracy > 97.0, "expect pre-trained mnist model accuracy to be at least 97%"

    config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(qat_schedule=qat_schedule),
        execution_mode="eager",
    )
    quantizer = Quantizer(mnist_pretrained_model, config)

    prepared_model = quantizer.prepare(
        example_inputs=(torch.ones(1, 1, 28, 28, dtype=torch.float),)
    )
    post_prepare_accuracy = utils.eval_model(prepared_model, test_loader)
    assert post_prepare_accuracy < 80, (
        "Expect accuracy to drop below 80% after preparation with an all ones data sample"
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
    print("\n accuracy of post qat model: ", post_qat_accuracy)
    assert post_qat_accuracy > 96.0, "Expect accuracy to climb above 96% after QAT"

    finalized_model = quantizer.finalize(backend=ExportBackend.Torch)
    finalized_accuracy = utils.eval_model(finalized_model, test_loader)

    # Accuracy before and after finalize should match
    assert post_qat_accuracy == finalized_accuracy


@pytest.mark.parametrize("param_dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_weight_scale_consistent_with_export_dtype(param_dtype):
    """
    Test that after prepare+finalize, the scale stored in the finalized model
    matches a manually computed scale using the export dtype (weight.dtype).

    This verifies that scale is cast to export dtype before quantization,
    so that quantization and dequantization are self-consistent.
    """

    model = nn.Conv2d(64, 64, kernel_size=3).to(param_dtype)
    init.kaiming_normal_(model.weight)

    example_input = torch.randn(1, 64, 3, 3).to(param_dtype)

    weight_dtype = torch.int4
    weight_qscheme = QuantizationScheme.SYMMETRIC
    quant_config = QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={
                "weight": QuantizationSpec(
                    dtype=weight_dtype,
                    qscheme=weight_qscheme,
                    granularity=PerTensorGranularity(),
                )
            },
            op_input_spec=None,
            op_output_spec=None,
        ),
        execution_mode=ExecutionMode.EAGER,
    )
    original_weight = model.weight.data.clone()

    quantizer = Quantizer(model, quant_config)
    quantizer.prepare((example_input,))
    finalized_model = quantizer.finalize(backend=ExportBackend.CoreAI)

    weight_scale_buffers = {
        key: value
        for key, value in finalized_model.named_buffers()
        if "scale" in key and "activation" not in key
    }
    assert len(weight_scale_buffers) == 1, (
        f"Expected 1 weight scale buffer, got {len(weight_scale_buffers)}: "
        f"{list(weight_scale_buffers.keys())}"
    )
    stored_scale = next(iter(weight_scale_buffers.values()))

    quant_min, quant_max = QuantizationSpec.get_quant_range(weight_dtype, weight_qscheme)
    expected_scale = (2 * original_weight.abs().max() / (quant_max - quant_min)).to(param_dtype)

    assert stored_scale.dtype == param_dtype, (
        f"Expected scale dtype {param_dtype}, got {stored_scale.dtype}"
    )
    assert stored_scale == expected_scale, f"Scale mismatch for param_dtype={param_dtype}"

    weight_quantized_data_buffers = {
        key: value for key, value in finalized_model.named_buffers() if "quantized_data" in key
    }
    assert len(weight_quantized_data_buffers) == 1, (
        f"Expected 1 quantized_data buffer, got {len(weight_quantized_data_buffers)}: "
        f"{list(weight_quantized_data_buffers.keys())}"
    )
    stored_quantized_data = next(iter(weight_quantized_data_buffers.values()))

    expected_quantized_data = torch.clamp(
        torch.round(original_weight.to(torch.float32) / expected_scale),
        quant_min,
        quant_max,
    ).to(stored_quantized_data.dtype)

    assert torch.equal(stored_quantized_data, expected_quantized_data), (
        f"Quantized weight mismatch for param_dtype={param_dtype}: "
        f"num mismatches={((stored_quantized_data != expected_quantized_data).sum().item())}"
    )
