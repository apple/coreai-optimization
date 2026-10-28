# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""End-to-end export of already-sparse models with joint sparsity + palettization.

Pre-sparsified MNIST and ResNet-50 models are palettized, finalized and converted to CoreAI and
CoreML. The converted model's ops are counted (MNIST also checks their order), and its output
is compared with the prepared model's. The logic of ``prepare()`` and ``finalize()`` is tested
in ``tests/palettization/test_joint_sparsity_palettization.py``.

Configs tested on MNIST (``test_mnist`` and ``test_mnist_vector_palettization``). A "-" means
the backend does not support the config, so it is not tested there. "Uneven groups" means the
weights were pruned per tensor, so channel groups hold different numbers of nonzeros.
(*) means the weights were pruned to 49.99% instead of 50%, so most layers' nonzero counts are
not a multiple of cluster_dim and the smallest nonzeros are dropped.

| Config                                                               | CoreAI | CoreML |
| -------------------------------------------------------------------- | ------ | ------ |
| Per-tensor, 8-bit                                                    | yes    | yes    |
| Axis-0 groups (size 2), 4-bit, uneven groups                         | yes    | yes    |
| Axis-1 groups (size 1), 4-bit, uneven groups                         | -      | yes    |
| Per-tensor, 4-bit, per-channel scale                                 | yes    | yes    |
| Per-tensor, 4-bit, per-channel scale + int8 LUT                      | yes    | -      |
| Per-tensor, 4-bit, vector (cluster_dim 2)                            | yes    | -      |
| Axis-0 groups (size 2), 4-bit, vector (cluster_dim 2), uneven groups | yes    | -      |
| Per-tensor, 4-bit, vector (cluster_dim 2) (*)                        | yes    | -      |
| Per-tensor, 4-bit, vector (cluster_dim 4), per-channel scale (*)     | yes    | -      |

Configs tested on ResNet-50 (``test_resnet50``, slow). "Even groups" means the weights were
pruned per group, so every channel group holds the same number of nonzeros.

| Config                                     | CoreAI | CoreML |
| ------------------------------------------ | ------ | ------ |
| Per-tensor, 8-bit                          | yes    | yes    |
| Axis-0 groups (size 2), 4-bit, even groups | yes    | yes    |
"""

import copy

import pytest
import torch

from coreai_opt import ExportBackend
from tests.test_utils.joint_sparsity import (
    GROUP_SIZE,
    SPARSITY,
    SYMMETRIC_LUT_QSPEC,
    build_palettizer,
    presparsify,
    spec_kwargs,
)

from . import export_utils

_COREAI, _COREML = ExportBackend.CoreAI, ExportBackend.CoreML
_MNIST_LAYERS = ["conv1", "conv2", "conv_transpose1", "conv_transpose2", "dense1", "dense2"]
_RESNET_LAYER_COUNT = 54

_JOINT_OPS = {
    _COREAI: ["lut_to_dense", "sparse_to_dense"],
    _COREML: ["constexpr_lut_to_sparse", "constexpr_sparse_to_dense"],
}


def _target_name(node) -> str | None:
    """Return the name of the custom op a graph node calls, e.g. ``coreai::lut_to_dense``."""
    target = getattr(node, "target", None)
    return target.name() if hasattr(target, "name") else None


def _assert_coreai_op_order(exported_program, num_layers: int, scaled: bool):
    """Assert each weight is a LUT lookup on the nonzeros, then sparse-to-dense, then a scale.

    The order is checked on the exported torch program, so it does not depend on how a Core AI
    version lowers or prints these ops.
    """

    def nodes(op_name: str) -> list:
        return [n for n in exported_program.graph.nodes if _target_name(n) == f"coreai::{op_name}"]

    to_dense = nodes("sparse_to_dense")
    assert len(to_dense) == num_layers
    assert all(_target_name(n.args[0]) == "coreai::lut_to_dense" for n in to_dense)
    scale_ops = nodes("constexpr_blockwise_shift_scale")
    assert len(scale_ops) == (num_layers if scaled else 0)
    assert all(_target_name(n.args[0]) == "coreai::sparse_to_dense" for n in scale_ops)


def _assert_coreml_op_order(converted_model, num_layers: int, scaled: bool):
    """Assert each weight is a sparse LUT lookup, then sparse-to-dense, then a scale."""
    main = converted_model._mil_program.functions["main"]
    to_dense = main.find_ops(op_type="constexpr_sparse_to_dense")
    assert len(to_dense) == num_layers
    for op in to_dense:
        assert op.nonzero_data.op.op_type == "constexpr_lut_to_sparse"
        assert op.mask.op.op_type == "constexpr_lut_to_sparse"
    scale_ops = main.find_ops(op_type="constexpr_blockwise_shift_scale")
    assert len(scale_ops) == (num_layers if scaled else 0)
    assert all(op.data.op.op_type == "constexpr_sparse_to_dense" for op in scale_ops)


def _export(
    backend, model, input_data, kwargs, num_layers, scaled=False, check_order=False, **verify_kwargs
):
    """Palettize ``model``, finalize and convert it, and return the converted model.

    With ``check_order``, also assert the order of the ops that rebuild each weight.
    """
    palettizer = build_palettizer(model, **kwargs)
    prepared = palettizer.prepare((input_data,))
    with torch.no_grad():
        prepared_output = prepared(input_data)
    ops = _JOINT_OPS[backend] + (["constexpr_blockwise_shift_scale"] if scaled else [])
    expected_ops = dict.fromkeys(ops, num_layers)
    finalized_model = palettizer.finalize(backend=backend)
    if check_order and backend == _COREAI:
        # Traced on a copy, since converting may modify the model.
        exported_program = export_utils.create_converter(backend).trace(
            copy.deepcopy(finalized_model), input_data, expected_ops
        )
        _assert_coreai_op_order(exported_program, num_layers, scaled)
    converted_model = export_utils.convert_and_verify(
        finalized_model=finalized_model,
        input_data=input_data,
        expected_ops=expected_ops,
        export_backend=backend,
        prepared_model_output=prepared_output,
        **verify_kwargs,
    )
    if check_order and backend == _COREML:
        _assert_coreml_op_order(converted_model, num_layers, scaled)
    return converted_model


_SCALE = {"enable_per_channel_scale": True}
_LUT_QSPEC = {"lut_qspec": SYMMETRIC_LUT_QSPEC}
_VECTOR = {"cluster_dim": 2}
# id, backends, n_bits, group_size, group size the model was pruned with (None: per tensor,
# so the groups' nonzero counts differ), group axis, extra spec kwargs (a scale, LUT quant
# and/or vector palettization).
_MNIST_CONFIGS = [
    ("per_tensor_8bit", [_COREAI, _COREML], 8, None, None, 0, {}),
    ("grouped_uneven_4bit", [_COREAI, _COREML], 4, GROUP_SIZE, None, 0, {}),
    # CoreAI maps contiguous blocks of nonzeros to groups, which needs axis 0. A group size of
    # 1, since conv1 has a single input channel.
    ("grouped_axis1_4bit", [_COREML], 4, 1, None, 1, {}),
    ("per_channel_scale", [_COREAI, _COREML], 4, None, None, 0, _SCALE),
    ("per_channel_scale+lut_qspec", [_COREAI], 4, None, None, 0, {**_SCALE, **_LUT_QSPEC}),
    ("vector", [_COREAI], 4, None, None, 0, _VECTOR),
    ("vector_grouped_uneven", [_COREAI], 4, GROUP_SIZE, None, 0, _VECTOR),
]


@pytest.mark.parametrize(
    "backend,n_bits,group_size,prune_group_size,axis,extra",
    [
        pytest.param(backend, *config[2:], id=f"{backend.name.lower()}-{config[0]}")
        for config in _MNIST_CONFIGS
        for backend in config[1]
    ],
)
def test_mnist(
    backend,
    n_bits,
    group_size,
    prune_group_size,
    axis,
    extra,
    custom_test_mnist_model,
    mnist_example_input,
):
    model = presparsify(custom_test_mnist_model, SPARSITY, prune_group_size)
    scaled = "enable_per_channel_scale" in extra or "lut_qspec" in extra
    _export(
        backend,
        model,
        mnist_example_input,
        {**spec_kwargs(n_bits, group_size, axis), **extra},
        len(_MNIST_LAYERS),
        scaled=scaled,
        check_order=True,
        snr_thresh=40.0,
        psnr_thresh=40.0,
    )


@pytest.mark.parametrize(
    "cluster_dim,extra",
    [(2, {}), (4, _SCALE)],
    ids=["dim2", "dim4_per_channel_scale"],
)
def test_mnist_vector_palettization(
    cluster_dim, extra, custom_test_mnist_model, mnist_example_input
):
    # Pruning a hair under half leaves most layers with a nonzero count that is not a multiple
    # of cluster_dim, so the smallest nonzeros are dropped from the mask.
    model = presparsify(custom_test_mnist_model, 0.4999)
    _export(
        _COREAI,
        model,
        mnist_example_input,
        {**spec_kwargs(4, None), "cluster_dim": cluster_dim, **extra},
        len(_MNIST_LAYERS),
        scaled=bool(extra),
        check_order=True,
        snr_thresh=40.0,
        psnr_thresh=40.0,
    )


@pytest.mark.slow
@pytest.mark.parametrize("backend", [_COREAI, _COREML], ids=["coreai", "coreml"])
@pytest.mark.parametrize(
    "n_bits,group_size",
    [(8, None), (4, GROUP_SIZE)],
    ids=["per_tensor_8bit", "grouped_4bit"],
)
def test_resnet50(backend, n_bits, group_size, resnet50_model, resnet_example_input):
    model = presparsify(resnet50_model, SPARSITY, group_size)
    _export(
        backend, model, resnet_example_input, spec_kwargs(n_bits, group_size), _RESNET_LAYER_COUNT
    )
