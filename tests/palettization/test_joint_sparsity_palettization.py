# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Logic tests of joint sparsity + palettization of an already-sparse model.

Layers with at least ``sparsity_detection_threshold`` exact zeros are palettized jointly with
their sparsity. These tests check ``prepare()`` and ``finalize()`` (detection, centroids, the
modules inserted for CoreAI, rejections) without converting to a backend; the converted models
are tested in ``tests/export/test_joint_sparsity_palettization_e2e.py``.
"""

import copy
import logging
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from coreai_opt import ExportBackend
from coreai_opt.palettization import ModuleKMeansPalettizerConfig, PalettizationSpec
from coreai_opt.palettization.kmeans._prepare_for_export import (
    _pad_mask_to_equal_groups,
    _SparsePalettizeModule,
)
from coreai_opt.palettization.kmeans.kmeans_fake_palettize import _KMeansFakePalettize
from coreai_opt.quantization.spec import QuantizationScheme, QuantizationSpec
from tests.test_utils.joint_sparsity import (
    GROUP_SIZE,
    SPARSITY,
    SYMMETRIC_LUT_QSPEC,
    build_palettizer,
    presparsify,
    spec_kwargs,
)

_LAYERS = ["conv1", "conv2", "fc"]
_SPARSE_LAYERS = ["conv1", "fc"]
_COREAI, _COREML = ExportBackend.CoreAI, ExportBackend.CoreML


class _SmallModel(nn.Module):
    """Two convs and a linear layer, with weights large enough for k-means in groups of 2."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.fc = nn.Linear(16 * 4 * 4, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        return self.fc(x.flatten(1))


@pytest.fixture
def small_model() -> nn.Module:
    torch.manual_seed(0)
    return _SmallModel().eval()


@pytest.fixture
def small_input() -> torch.Tensor:
    return torch.randn(1, 3, 8, 8)


def _fake(model: nn.Module, name: str):
    """Return the fake palettization module of layer ``name`` of a prepared model."""
    return getattr(model, name).parametrizations.weight[0]


def _sparse_module(layer: nn.Module) -> _SparsePalettizeModule | None:
    """Return the sparse palettize module of a finalized (CoreAI) layer, if it has one."""
    return next(
        (p for p in layer.parametrizations.weight if isinstance(p, _SparsePalettizeModule)), None
    )


def _weights(model: nn.Module, names: list[str]) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        return {name: getattr(model, name).weight.clone() for name in names}


def test_sparsity_detection_threshold(small_model, small_input):
    assert PalettizationSpec(n_bits=4).sparsity_detection_threshold is None
    for invalid in (-0.1, 1.1):
        with pytest.raises(ValueError):
            PalettizationSpec(n_bits=4, sparsity_detection_threshold=invalid)
    spec = PalettizationSpec(n_bits=4, sparsity_detection_threshold=1.0)
    assert spec.sparsity_detection_threshold == 1.0
    assert spec.model_copy().sparsity_detection_threshold == 1.0
    assert PalettizationSpec(**spec.model_dump()).sparsity_detection_threshold == 1.0
    # A public option, so it can be set from a dict config.
    config = ModuleKMeansPalettizerConfig(
        op_state_spec={"weight": {"n_bits": 4, "sparsity_detection_threshold": 0.4}}
    )
    assert config.op_state_spec["weight"].sparsity_detection_threshold == 0.4

    # Only two layers are pre-sparsified; conv2 has no zeros at all (0% sparsity), so even a
    # threshold of 0 leaves it as plain palettization.
    model = presparsify(small_model, SPARSITY, layers=_SPARSE_LAYERS)
    for threshold, detected in [
        (None, []),
        (0.0, _SPARSE_LAYERS),
        (0.4, _SPARSE_LAYERS),
        (0.6, []),
    ]:
        prepared = build_palettizer(
            copy.deepcopy(model), n_bits=4, sparsity_detection_threshold=threshold
        ).prepare((small_input,))
        for name in _LAYERS:
            is_detected = _fake(prepared, name)._sparsity_mask is not None
            assert is_detected == (name in detected), (threshold, name)


@pytest.mark.parametrize(
    "group_size,presparse_group_size",
    [(None, None), (GROUP_SIZE, GROUP_SIZE), (GROUP_SIZE, None)],
    ids=["per_tensor", "grouped_balanced", "grouped_uneven"],
)
def test_centroid_count(group_size, presparse_group_size, small_model, small_input):
    """Check which LUTs reserve entry 0 for exactly 0.0.

    Pruning each group of ``presparse_group_size`` channels separately gives every LUT group the
    same number of nonzeros; pruning the whole tensor (``None``) leaves them uneven. Uneven
    groups with fewer nonzeros than the fullest group are padded with stored zeros at export, so
    their LUT reserves entry 0 as 0.0 and fits the other 2**n_bits - 1 centroids. The fullest
    group, and every group of a balanced or per-tensor layer, uses all 2**n_bits centroids.
    """
    n_bits = 4
    model = presparsify(small_model, SPARSITY, presparse_group_size)
    prepared = build_palettizer(model, **spec_kwargs(n_bits, group_size)).prepare((small_input,))
    uneven = group_size is not None and presparse_group_size is None
    for name in _LAYERS:
        fake = _fake(prepared, name)
        num_groups = 1 if group_size is None else getattr(model, name).weight.shape[0] // group_size
        lut = fake.lut.reshape(num_groups, 2**n_bits)
        kept = fake._sparsity_mask.reshape(num_groups, -1)
        indices = fake.indices.reshape(num_groups, -1)
        num_nonzeros = kept.sum(dim=1)
        padded = num_nonzeros < num_nonzeros.max()
        assert padded.any() == uneven, name
        for group in range(num_groups):
            kept_indices = indices[group][kept[group]]
            if padded[group]:
                # (A block with at most 2**n_bits distinct values reuses its own palette, with
                # 0.0 at another position; these blocks are larger.)
                assert lut[group, 0] == 0, (name, group)
                assert (kept_indices != 0).all(), (name, group)
            else:
                assert lut[group, 0] != 0, (name, group)
                assert (kept_indices == 0).any(), (name, group)


@pytest.mark.parametrize("all_zero", ["group", "layer"])
def test_all_zero_group_or_layer(all_zero, small_model, small_input, caplog):
    n_bits = 4
    if all_zero == "group":
        model = presparsify(small_model, SPARSITY, GROUP_SIZE)
        with torch.no_grad():
            for name in _LAYERS:
                getattr(model, name).weight[:GROUP_SIZE] = 0
        kwargs = spec_kwargs(n_bits, GROUP_SIZE)
    else:
        model = presparsify(small_model, SPARSITY)
        with torch.no_grad():
            for name in ["conv1", "fc"]:
                getattr(model, name).weight.zero_()
        # Only the fully zero layers reach a threshold of 1.0.
        kwargs = {"n_bits": n_bits, "sparsity_detection_threshold": 1.0}
    zeros_before = {name: getattr(model, name).weight == 0 for name in _LAYERS}
    palettizer = build_palettizer(model, **kwargs)
    prepared = palettizer.prepare((small_input,))
    for name in _LAYERS:
        assert _fake(prepared, name).lut.shape[-2] == 2**n_bits, name
    weights = _weights(prepared, _LAYERS)
    # The zero groups or layers, and every pruned weight, are still exactly zero.
    for name, weight in weights.items():
        assert torch.equal(weight == 0, zeros_before[name]), name

    with caplog.at_level(logging.WARNING):
        finalized = palettizer.finalize(backend=_COREAI)

    # An all-zero layer is exported densely (its palette is all zeros, so this is exact); an
    # all-zero group only makes the groups uneven, so the layer is padded like any other.
    sparse_layers = [n for n in _LAYERS if _sparse_module(getattr(finalized, n)) is not None]
    assert sparse_layers == (_LAYERS if all_zero == "group" else [])
    assert ("all weights are zero; exporting without joint sparsity" in caplog.text) == (
        all_zero == "layer"
    )
    for name, weight in weights.items():
        torch.testing.assert_close(getattr(finalized, name).weight, weight)
        assert torch.equal(getattr(finalized, name).weight == 0, zeros_before[name]), name


@pytest.mark.parametrize(
    "group_size,presparse_group_size,uneven,n_bits",
    [
        (None, None, False, 4),
        (GROUP_SIZE, GROUP_SIZE, False, 4),
        # Pruned per tensor, so the groups' nonzero counts differ.
        (GROUP_SIZE, None, True, 4),
        (GROUP_SIZE, None, True, 2),
        (GROUP_SIZE, None, True, 1),
    ],
    ids=["per_tensor", "grouped", "grouped_uneven", "grouped_uneven_2bit", "grouped_uneven_1bit"],
)
def test_round_trip(group_size, presparse_group_size, uneven, n_bits, small_model, small_input):
    model = presparsify(small_model, SPARSITY, presparse_group_size)
    zeros_before = {name: getattr(model, name).weight == 0 for name in _LAYERS}
    palettizer = build_palettizer(model, **spec_kwargs(n_bits, group_size))
    prepared = palettizer.prepare((small_input,))
    # The default strategy is post-training, so only the layers around the palettized weights
    # get gradients; the weights themselves are left unchanged.
    with palettizer.training_mode():
        prepared(small_input).sum().backward()
    assert all(getattr(prepared, name).bias.grad is not None for name in _LAYERS)
    prepared_weights = _weights(prepared, _LAYERS)

    finalized = palettizer.finalize(backend=_COREAI)
    for name in _LAYERS:
        layer = getattr(finalized, name)
        sparse_module = _sparse_module(layer)
        lut, mask = sparse_module.lut, sparse_module.mask
        indices = sparse_module.nonzero_indices.long()
        assert lut.shape[1] == 2**n_bits, name
        if uneven:
            # Groups with fewer nonzeros also store zeros so that every group holds as many
            # elements as the fullest one, and reserve entry 0 as 0.0 for them (these blocks are
            # large enough to run k-means). The fullest group uses all its entries for nonzeros.
            nonzeros_per_group = (~zeros_before[name]).reshape(lut.shape[0], -1).sum(dim=1)
            padded = nonzeros_per_group < nonzeros_per_group.max()
            assert padded.any(), name
            assert (lut[padded, 0, 0] == 0).all(), name
            assert (lut[~padded, 0, 0] != 0).all(), name
            assert not (~zeros_before[name] & ~mask).any(), name
            stored = torch.stack([g.sum() for g in mask.split(GROUP_SIZE, dim=0)])
            assert stored.unique().numel() == 1, name
            assert mask.sum() > (~zeros_before[name]).sum(), name
        else:
            # Every entry, including entry 0, is a real centroid of the nonzero weights.
            assert (lut != 0).all(), name
            assert torch.equal(mask, ~zeros_before[name]), name

        # De-palettize by hand: block g of the nonzero indices looks up LUT g, and the values
        # scatter back into the mask's positions.
        group = torch.arange(indices.numel()) // (indices.numel() // lut.shape[0])
        depalettized = torch.zeros(mask.shape, dtype=lut.dtype)
        depalettized[mask] = lut[group, indices, 0]
        assert torch.equal(depalettized == 0, zeros_before[name]), name
        assert torch.equal(depalettized, prepared_weights[name]), name
        assert torch.equal(layer.weight, prepared_weights[name]), name


def _palettize_linear_weight(weight, **spec):
    """Palettize ``weight`` as a bias-free linear layer and return (palettizer, prepared)."""
    model = nn.Sequential(nn.Linear(weight.shape[1], weight.shape[0]))
    with torch.no_grad():
        model[0].weight.copy_(weight)
    palettizer = build_palettizer(model, **spec)
    return palettizer, palettizer.prepare((torch.randn(1, weight.shape[1]),))


def test_vector_palettization_of_nonzeros():
    # In row-major order the nonzeros are 1 2 1 2 0.1 3 4 3 4 5 6. Their count is odd, so with
    # cluster_dim=2 the smallest one (0.1) is dropped, leaving the vectors (1,2) (1,2) (3,4)
    # (3,4) (5,6): three distinct vectors, which fit a 4-entry LUT exactly.
    weight = torch.tensor(
        [
            [1, 2, 0, 1, 2],
            [0, 0.1, 3, 0, 4],
            [3, 4, 0, 0, 5],
            [0, 6, 0, 0, 0],
        ]
    )
    expected = weight.clone()
    expected[1, 1] = 0
    palettizer, prepared = _palettize_linear_weight(weight, n_bits=2, cluster_dim=2)

    fake = prepared[0].parametrizations.weight[0]
    assert torch.equal(fake._sparsity_mask, expected != 0)
    # One LUT of four 2-vectors and one index per vector; equal vectors share an index.
    assert fake.lut.shape == (1, 1, 4, 2)
    indices = fake.indices.tolist()
    assert len(indices) == 5
    assert indices[0] == indices[1] and indices[2] == indices[3] and len(set(indices)) == 3
    assert torch.equal(prepared[0].weight, expected)

    finalized = palettizer.finalize(backend=_COREAI)
    sparse_module = _sparse_module(finalized[0])
    assert torch.equal(sparse_module.mask, expected != 0)
    assert sparse_module.lut.shape == (1, 4, 2)
    assert sparse_module.nonzero_indices.shape == (5,)
    assert torch.equal(finalized[0].weight, expected)


def test_vector_palettization_of_nonzeros_grouped():
    # Two groups of two rows, one LUT each. Group 0 has 7 nonzeros, so its smallest (0.1) is
    # dropped to leave three whole vectors. Group 1 has 4, so it also stores its first two
    # zeros (row 2, columns 0 and 2) to hold as many vectors as group 0.
    weight = torch.tensor(
        [
            [1, 2, 0, 3, 4],
            [0, 5, 6, 0, 0.1],
            [0, 7, 0, 8, 0],
            [9, 0, 0, 0, 10],
        ]
    )
    expected = weight.clone()
    expected[1, 4] = 0
    palettizer, prepared = _palettize_linear_weight(weight, cluster_dim=2, **spec_kwargs(2, 2))

    fake = prepared[0].parametrizations.weight[0]
    mask = fake._sparsity_mask.clone()
    assert mask[:2].sum() == mask[2:].sum() == 6
    assert torch.equal(mask & (expected != 0), expected != 0)
    assert (mask & (expected == 0)).nonzero().tolist() == [[2, 0], [2, 2]]
    assert fake.lut.shape == (2, 1, 4, 2)
    assert fake.indices.shape == (6,)
    # Each group has three distinct vectors, so its LUT is exact and the padding costs nothing.
    assert torch.equal(prepared[0].weight, expected)

    finalized = palettizer.finalize(backend=_COREAI)
    sparse_module = _sparse_module(finalized[0])
    assert torch.equal(sparse_module.mask, mask)
    assert sparse_module.lut.shape == (2, 4, 2)
    assert sparse_module.nonzero_indices.shape == (6,)
    # Group g's three indices look up LUT g, and the values scatter into the mask row by row.
    indices = sparse_module.nonzero_indices.long().reshape(2, 3)
    values = sparse_module.lut[torch.arange(2).unsqueeze(1), indices].flatten()
    assert torch.equal(torch.zeros(weight.shape).masked_scatter(mask, values), expected)
    assert torch.equal(finalized[0].weight, expected)


def test_vector_palettization_of_nonzeros_edge_cases():
    # Only axis-0 groups are contiguous in the nonzero list, so a sparse layer with LUTs grouped
    # along another axis fails at detection.
    model = presparsify(nn.Sequential(nn.Linear(8, 4)), SPARSITY)
    grouped = build_palettizer(model, cluster_dim=2, **spec_kwargs(2, GROUP_SIZE, axis=1))
    with pytest.raises(RuntimeError, match="Centroid calculation failed") as exc_info:
        grouped.prepare((torch.randn(1, 8),))
    assert "requires per-tensor or axis-0 grouped" in str(exc_info.value.__cause__)

    # A single nonzero cannot fill a vector, so the layer is palettized densely, which is exact
    # here since it holds only two distinct vectors.
    model = nn.Sequential(nn.Linear(8, 4))
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].weight[1, 2] = 1.0
    original = model[0].weight.detach().clone()
    palettizer = build_palettizer(model, n_bits=2, cluster_dim=2)
    prepared = palettizer.prepare((torch.randn(1, 8),))
    assert prepared[0].parametrizations.weight[0]._sparsity_mask is None
    finalized = palettizer.finalize(backend=_COREAI)
    assert _sparse_module(finalized[0]) is None
    assert torch.equal(finalized[0].weight, original)


_SCALE_FEATURES = {
    "per_channel_scale": {"enable_per_channel_scale": True},
    "lut_qspec": {"lut_qspec": SYMMETRIC_LUT_QSPEC},
    "per_channel_scale+lut_qspec": {
        "enable_per_channel_scale": True,
        "lut_qspec": SYMMETRIC_LUT_QSPEC,
    },
}


@pytest.mark.parametrize("feature", list(_SCALE_FEATURES))
@pytest.mark.parametrize(
    "group_size,presparse_group_size",
    [(None, None), (4, 4), (4, None)],
    ids=["per_tensor", "grouped", "grouped_uneven"],
)
def test_scale_variants_rebuild_prepared_weight(feature, group_size, presparse_group_size):
    # The scale (per-channel and/or LUT dequantization) is applied after sparse_to_dense, so
    # the zeros stay exactly zero.
    torch.manual_seed(0)
    model = presparsify(nn.Sequential(nn.Conv2d(8, 16, 3)), SPARSITY, presparse_group_size)
    palettizer = build_palettizer(model, **spec_kwargs(4, group_size), **_SCALE_FEATURES[feature])
    prepared = palettizer.prepare((torch.randn(1, 8, 8, 8),))
    (prepared_weight,) = _weights(prepared, ["0"]).values()

    finalized = palettizer.finalize(backend=_COREAI)
    sparse_module = _sparse_module(finalized[0])
    assert sparse_module.scale is not None
    assert (sparse_module.lut.dtype == torch.int8) == ("lut_qspec" in feature)
    torch.testing.assert_close(finalized[0].weight, prepared_weight, rtol=1e-6, atol=1e-8)
    assert torch.equal(finalized[0].weight == 0, prepared_weight == 0)


@pytest.mark.parametrize("group_size", [None, GROUP_SIZE], ids=["per_tensor", "grouped_uneven"])
def test_already_palettized_sparse_weights_skip_kmeans(group_size, small_model, small_input):
    model = presparsify(small_model, SPARSITY)
    # Palettize at 3 bits, then reuse the result as an already-palettized sparse model.
    palettized = copy.deepcopy(model)
    prepared = build_palettizer(model, **spec_kwargs(3, group_size)).prepare((small_input,))
    with torch.no_grad():
        for name, weight in _weights(prepared, _LAYERS).items():
            getattr(palettized, name).weight.copy_(weight)

    # Raising proves that k-means is skipped for every layer.
    with patch.object(
        _KMeansFakePalettize, "_cluster_weights_1d_masked", side_effect=AssertionError
    ):
        palettizer = build_palettizer(palettized.eval(), **spec_kwargs(4, group_size))
        reprepared = palettizer.prepare((small_input,))
    weights = _weights(reprepared, _LAYERS)

    finalized = palettizer.finalize(backend=_COREAI)
    for name, weight in weights.items():
        assert _sparse_module(getattr(finalized, name)) is not None, name
        assert torch.equal(getattr(finalized, name).weight, weight), name


def test_torch_backend_keeps_zeros(small_model, small_input):
    model = presparsify(small_model, SPARSITY, GROUP_SIZE)
    zeros_before = {name: getattr(model, name).weight == 0 for name in _LAYERS}
    palettizer = build_palettizer(model, **spec_kwargs(4, GROUP_SIZE))
    prepared = palettizer.prepare((small_input,))
    with torch.no_grad():
        prepared_output = prepared(small_input)

    finalized = palettizer.finalize(backend=ExportBackend.Torch)
    with torch.no_grad():
        torch.testing.assert_close(finalized(small_input), prepared_output)
    for name in _LAYERS:
        assert torch.equal(getattr(finalized, name).weight == 0, zeros_before[name]), name


def test_pad_mask_to_equal_groups():
    context = "'layer.weight': "
    # Group 0 (rows 0-1) holds 4 nonzeros and group 1 (rows 2-3) holds 2.
    mask = torch.tensor([[1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 0, 1]], dtype=torch.bool)
    indices = torch.tensor([[1, 2, 0], [3, 1, 2], [0, 3, 0], [0, 0, 1]], dtype=torch.uint8)
    lut = torch.tensor([[[0.0], [1.0], [2.0], [3.0]], [[0.0], [4.0], [5.0], [6.0]]])

    padded = _pad_mask_to_equal_groups(mask, indices, lut, 2, context)
    assert [int(g.sum()) for g in padded.split(2, dim=0)] == [4, 4]
    assert (padded | mask).equal(padded)
    # Only zeros of the original mask were added, and only in the sparser group.
    assert not (padded & ~mask)[:2].any()
    assert (padded & ~mask)[2:].sum() == 2

    # A LUT whose zero entry changed after prepare() cannot back the padded zeros.
    drifted = lut.clone()
    drifted[1, 0, 0] = 0.5
    with pytest.raises(ValueError, match=r"'layer\.weight': .*exactly 0\.0"):
        _pad_mask_to_equal_groups(mask, indices, drifted, 2, context)

    # The trailing group of one row is too small to hold the fullest group's nonzeros.
    ragged_mask = torch.ones(5, 3, dtype=torch.bool)
    ragged_mask[4] = False
    with pytest.raises(ValueError, match=r"'layer\.weight': group 2 .*cannot hold 6"):
        _pad_mask_to_equal_groups(
            ragged_mask, torch.zeros(5, 3, dtype=torch.uint8), lut, 2, context
        )


_VECTOR = {"n_bits": 4, "cluster_dim": 2}
_REJECTS = {
    # CoreAI supports per-tensor or axis-0 grouped LUTs, scalar or vector, and zero-point-free LUT
    # quantization of scalar LUTs.
    _COREAI: {
        "vector_lut_qspec": {**_VECTOR, "lut_qspec": SYMMETRIC_LUT_QSPEC},
        "grouped_axis1": spec_kwargs(4, 4, axis=1),
        "lut_qspec_int8_zero_point": {
            "n_bits": 4,
            "lut_qspec": QuantizationSpec(dtype=torch.int8, qscheme=QuantizationScheme.ASYMMETRIC),
        },
        "lut_qspec_uint8_zero_point": {
            "n_bits": 4,
            "lut_qspec": QuantizationSpec(dtype=torch.uint8, qscheme=QuantizationScheme.ASYMMETRIC),
        },
    },
    # CoreML takes a vector mask on the grid of vectors, so it cannot express vectors of nonzero
    # values, and it cannot convert the scale after its sparse LUT op.
    _COREML: {
        "vector_per_tensor": _VECTOR,
        "vector_grouped": {**_VECTOR, **spec_kwargs(4, GROUP_SIZE)},
        "lut_qspec": {"n_bits": 4, "lut_qspec": SYMMETRIC_LUT_QSPEC},
    },
}


@pytest.mark.parametrize(
    "backend,kwargs",
    [
        pytest.param(backend, kwargs, id=f"{backend.name.lower()}-{name}")
        for backend, cases in _REJECTS.items()
        for name, kwargs in cases.items()
    ],
)
def test_finalize_rejects_unsupported(backend, kwargs):
    # A fixed seed, so that the asymmetric LUT quantization gets a nonzero zero point.
    torch.manual_seed(0)
    model = presparsify(nn.Sequential(nn.Conv2d(8, 16, 3)), SPARSITY)
    palettizer = build_palettizer(model, **kwargs)
    palettizer.prepare((torch.randn(1, 8, 8, 8),))
    with pytest.raises(ValueError, match="not supported|zero point of 0"):
        palettizer.finalize(backend=backend)
