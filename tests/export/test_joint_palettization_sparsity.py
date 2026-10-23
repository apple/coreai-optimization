# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import copy
import logging
from collections.abc import Callable
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from coreai_opt import ExportBackend
from coreai_opt.palettization import (
    KMeansPalettizer,
    KMeansPalettizerConfig,
    ModuleKMeansPalettizerConfig,
    PalettizationSpec,
)
from coreai_opt.palettization.kmeans._prepare_for_export import _pad_mask_to_equal_groups
from coreai_opt.palettization.kmeans.kmeans_fake_palettize import _KMeansFakePalettize
from coreai_opt.palettization.spec import PerGroupedChannelGranularity, PerTensorGranularity
from coreai_opt.quantization.spec import QuantizationScheme, QuantizationSpec

from . import export_utils

_SPARSITY = 0.5
_THRESHOLD = 0.4
_GROUP_SIZE = 2
_MNIST_LAYERS = ["conv1", "conv2", "conv_transpose1", "conv_transpose2", "dense1", "dense2"]
_RESNET_LAYER_COUNT = 54
_BACKENDS = [ExportBackend.CoreAI, ExportBackend.CoreML]

_JOINT_OPS = {
    ExportBackend.CoreAI: lambda n: {"lut_to_dense": n, "sparse_to_dense": n},
    ExportBackend.CoreML: lambda n: {"constexpr_lut_to_sparse": n, "constexpr_sparse_to_dense": n},
}


def _presparsify(
    model: nn.Module,
    sparsity: float,
    group_size: int | None = None,
    layers: list[str] | None = None,
) -> nn.Module:
    """Zero the smallest-magnitude weights of each layer in place.

    Args:
        model (nn.Module): Model whose Conv2d, ConvTranspose2d and Linear weights are pruned.
        sparsity (float): Fraction of each group (or tensor) to zero.
        group_size (int | None): Axis-0 group size, so every group keeps the same count. If
            ``None``, prune per tensor.
        layers (list[str] | None): Names of the layers to prune; all of them if ``None``.

    Returns:
        nn.Module: The pruned model in eval mode.
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, nn.Conv2d | nn.ConvTranspose2d | nn.Linear):
                continue
            if layers is not None and name not in layers:
                continue
            weight = module.weight
            for group in weight.split(group_size or weight.shape[0], dim=0):
                num_pruned = int(group.numel() * sparsity)
                group.view(-1)[group.abs().flatten().argsort()[:num_pruned]] = 0
    return model.eval()


class TestJointPalettizationSparsity:
    """Test palettization of an already-sparse model.

    Layers with at least ``_sparsity_detection_threshold`` exact zeros are palettized jointly
    with their sparsity.

    Masking flattens indices to rank 1 before the LUT lookup. CoreAI supports scalar
    (cluster_dim=1) palettization with per-tensor or axis-0 grouped LUTs; a grouped LUT
    maps the ``g``-th contiguous block of nonzero indices to group ``g``, so every group
    must hold the same number of nonzeros, so groups with fewer nonzeros also store some of
    their zeros, which index a LUT entry pinned to 0.0.

    CoreML keeps one index per weight element, so coremltools' sparse LUT op recovers
    per-position group context for any mask. Vector palettization (``cluster_dim>1``) and
    ``enable_per_channel_scale`` are rejected for CoreML: both are coremltools limitations
    (an index-count mismatch and a scale/data rank mismatch, respectively).
    """

    # Scalar (cluster_dim=1) palettization gets the joint sparse op chain on both backends.
    VALID_CONFIGS: list[tuple[str, dict]] = [
        ("4bit", {"n_bits": 4}),
        ("6bit", {"n_bits": 6}),
        ("8bit", {"n_bits": 8}),
        (
            "grouped_channel",
            {
                "n_bits": 4,
                "granularity": PerGroupedChannelGranularity(axis=0, group_size=_GROUP_SIZE),
            },
        ),
    ]
    COREAI_REJECTS_CONFIGS: list[tuple[str, dict]] = [
        ("vector_ndim", {"n_bits": 4, "cluster_dim": 2}),
        (
            "lut_qspec",
            {
                "n_bits": 4,
                "lut_qspec": QuantizationSpec(
                    dtype=torch.int8, qscheme=QuantizationScheme.SYMMETRIC
                ),
            },
        ),
        ("per_channel_scale", {"n_bits": 4, "enable_per_channel_scale": True}),
        (
            "axis1_grouped",
            {"n_bits": 4, "granularity": PerGroupedChannelGranularity(axis=1, group_size=1)},
        ),
    ]
    COREML_REJECTS_CONFIGS: list[tuple[str, dict]] = [
        ("vector_ndim", {"n_bits": 4, "cluster_dim": 2}),
        ("per_channel_scale", {"n_bits": 4, "enable_per_channel_scale": True}),
    ]

    @staticmethod
    def _build_palettizer(model: nn.Module, **spec_kwargs) -> KMeansPalettizer:
        spec_kwargs.setdefault("granularity", PerTensorGranularity())
        spec_kwargs.setdefault("_sparsity_detection_threshold", _THRESHOLD)
        config = KMeansPalettizerConfig(
            global_config=ModuleKMeansPalettizerConfig(
                op_state_spec={"weight": PalettizationSpec(**spec_kwargs)},
                # cluster_dim > 1 requires fast mode off; scalar palettization keeps it on,
                # since non-fast 1-D k-means allocates O(2**n_bits * weight.numel()) memory.
                enable_fast_kmeans_mode=spec_kwargs.get("cluster_dim", 1) == 1,
            )
        )
        return KMeansPalettizer(model, config)

    @classmethod
    def _run(
        cls,
        backend: ExportBackend,
        model: nn.Module,
        input_data: torch.Tensor,
        spec_kwargs: dict,
        expected_ops: dict[str, int],
        prepared_check: Callable[[nn.Module], None] | None = None,
    ) -> None:
        palettizer = cls._build_palettizer(model, **spec_kwargs)
        prepared_model = palettizer.prepare((input_data,))
        if prepared_check is not None:
            prepared_check(prepared_model)
        with torch.no_grad():
            prepared_model_output = prepared_model(input_data)

        export_utils.convert_and_verify(
            finalized_model=palettizer.finalize(backend=backend),
            input_data=input_data,
            expected_ops=expected_ops,
            export_backend=backend,
            prepared_model_output=prepared_model_output,
        )

    @classmethod
    def _run_rejects(
        cls,
        backend: ExportBackend,
        model: nn.Module,
        input_data: torch.Tensor,
        spec_kwargs: dict,
    ) -> None:
        palettizer = cls._build_palettizer(model, **spec_kwargs)
        prepared_model = palettizer.prepare((input_data,))
        with torch.no_grad():
            prepared_model(input_data)

        with pytest.raises((RuntimeError, ValueError)):
            palettizer.finalize(backend=backend)

    @staticmethod
    def _group_size(spec_kwargs: dict) -> int | None:
        return getattr(spec_kwargs.get("granularity"), "group_size", None)

    @pytest.mark.parametrize("backend", _BACKENDS, ids=["coreai", "coreml"])
    @pytest.mark.parametrize(
        "spec_kwargs", [c[1] for c in VALID_CONFIGS], ids=[c[0] for c in VALID_CONFIGS]
    )
    def test_accepts_scalar_mnist(
        self, backend, spec_kwargs, custom_test_mnist_model, mnist_example_input
    ):
        model = _presparsify(custom_test_mnist_model, _SPARSITY, self._group_size(spec_kwargs))
        self._run(
            backend,
            model,
            mnist_example_input,
            spec_kwargs,
            _JOINT_OPS[backend](len(_MNIST_LAYERS)),
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("backend", _BACKENDS, ids=["coreai", "coreml"])
    @pytest.mark.parametrize(
        "spec_kwargs", [c[1] for c in VALID_CONFIGS], ids=[c[0] for c in VALID_CONFIGS]
    )
    def test_accepts_scalar_resnet(
        self, backend, spec_kwargs, resnet50_model, resnet_example_input
    ):
        model = _presparsify(resnet50_model, _SPARSITY, self._group_size(spec_kwargs))
        self._run(
            backend,
            model,
            resnet_example_input,
            spec_kwargs,
            _JOINT_OPS[backend](_RESNET_LAYER_COUNT),
        )

    @pytest.mark.parametrize(
        "threshold,num_joint",
        [(None, 0), (_THRESHOLD, 3), (0.6, 0)],
        ids=["detection_off", "below_threshold_skipped", "threshold_above_sparsity"],
    )
    def test_detects_only_layers_at_or_above_threshold(
        self, threshold, num_joint, custom_test_mnist_model, mnist_example_input
    ):
        # Only three layers are pre-sparsified; the rest stay dense.
        model = _presparsify(
            custom_test_mnist_model, _SPARSITY, layers=["conv1", "conv_transpose1", "dense1"]
        )
        self._run(
            ExportBackend.CoreAI,
            model,
            mnist_example_input,
            {"n_bits": 4, "_sparsity_detection_threshold": threshold},
            {"lut_to_dense": len(_MNIST_LAYERS), "sparse_to_dense": num_joint},
        )

    @pytest.mark.parametrize("backend", _BACKENDS, ids=["coreai", "coreml"])
    def test_grouped_with_uneven_groups(
        self, backend, custom_test_mnist_model, mnist_example_input
    ):
        # Pruning per tensor leaves the channel groups with different nonzero counts.
        model = _presparsify(custom_test_mnist_model, _SPARSITY)
        spec_kwargs = {
            "n_bits": 4,
            "granularity": PerGroupedChannelGranularity(axis=0, group_size=_GROUP_SIZE),
        }
        self._run(
            backend,
            model,
            mnist_example_input,
            spec_kwargs,
            _JOINT_OPS[backend](len(_MNIST_LAYERS)),
        )

    def test_sparsity_detection_threshold_spec(self, custom_test_mnist_model, mnist_example_input):
        assert PalettizationSpec(n_bits=4)._sparsity_detection_threshold is None
        for invalid in (-0.1, 1.1):
            with pytest.raises(ValueError):
                PalettizationSpec(n_bits=4, _sparsity_detection_threshold=invalid)
        spec = PalettizationSpec(n_bits=4, _sparsity_detection_threshold=1.0)
        assert spec._sparsity_detection_threshold == 1.0
        assert spec.model_copy()._sparsity_detection_threshold == 1.0

        # A threshold of 0 still leaves a fully dense model undetected.
        self._run(
            ExportBackend.CoreAI,
            custom_test_mnist_model.eval(),
            mnist_example_input,
            {"n_bits": 4, "_sparsity_detection_threshold": 0.0},
            {"lut_to_dense": len(_MNIST_LAYERS), "sparse_to_dense": 0},
        )

    @pytest.mark.parametrize("backend", _BACKENDS, ids=["coreai", "coreml"])
    @pytest.mark.parametrize("all_zero", ["group", "layer"])
    def test_all_zero_group_or_layer(
        self, backend, all_zero, custom_test_mnist_model, mnist_example_input, caplog
    ):
        n_bits = 4
        zero_layers = ["conv1", "dense1"]
        if all_zero == "group":
            model = _presparsify(custom_test_mnist_model, _SPARSITY, _GROUP_SIZE)
            with torch.no_grad():
                for name in _MNIST_LAYERS:
                    getattr(model, name).weight[:_GROUP_SIZE] = 0
            spec_kwargs = {
                "n_bits": n_bits,
                "granularity": PerGroupedChannelGranularity(axis=0, group_size=_GROUP_SIZE),
            }
            num_joint = len(_MNIST_LAYERS)
        else:
            model = _presparsify(custom_test_mnist_model, _SPARSITY)
            with torch.no_grad():
                for name in zero_layers:
                    getattr(model, name).weight.zero_()
            # Only the fully zero layers reach a threshold of 1.0.
            spec_kwargs = {"n_bits": n_bits, "_sparsity_detection_threshold": 1.0}
            num_joint = len(zero_layers)

        def check_full_size_luts(prepared_model: nn.Module) -> None:
            for name in _MNIST_LAYERS:
                fake_palett_mod = getattr(prepared_model, name).parametrizations.weight[0]
                assert fake_palett_mod.lut.shape[-2] == 2**n_bits, name

        # CoreAI exports all-zero layers densely, which is exact since their palette is zero.
        exported_densely = backend == ExportBackend.CoreAI and all_zero == "layer"
        expected_ops = (
            {"lut_to_dense": len(_MNIST_LAYERS), "sparse_to_dense": 0}
            if exported_densely
            else _JOINT_OPS[backend](num_joint)
        )
        with caplog.at_level(logging.WARNING):
            self._run(
                backend,
                model,
                mnist_example_input,
                spec_kwargs,
                expected_ops,
                prepared_check=check_full_size_luts,
            )
        if exported_densely:
            assert "all weights are zero; exporting without joint sparsity" in caplog.text

    @pytest.mark.parametrize(
        "granularity",
        [PerTensorGranularity(), PerGroupedChannelGranularity(axis=0, group_size=_GROUP_SIZE)],
        ids=["per_tensor", "grouped_channel"],
    )
    def test_pruned_weights_stay_zero(
        self, granularity, custom_test_mnist_model, mnist_example_input
    ):
        # At 2 bits and 25% sparsity, a zero centroid fit on pruned and unpruned weights
        # alike drifts off 0.
        model = _presparsify(
            custom_test_mnist_model, 0.25, getattr(granularity, "group_size", None)
        )
        zeros_before = {name: getattr(model, name).weight == 0 for name in _MNIST_LAYERS}
        palettizer = self._build_palettizer(
            model, n_bits=2, granularity=granularity, _sparsity_detection_threshold=0.2
        )
        prepared_model = palettizer.prepare((mnist_example_input,))
        for name in _MNIST_LAYERS:
            with torch.no_grad():
                weight = getattr(prepared_model, name).weight
            assert torch.equal(weight == 0, zeros_before[name]), name

    @pytest.mark.parametrize(
        "granularity,presparse_group_size,uneven,n_bits",
        [
            (PerTensorGranularity(), None, False, 4),
            (PerGroupedChannelGranularity(axis=0, group_size=_GROUP_SIZE), _GROUP_SIZE, False, 4),
            # Pruned per tensor, so the groups' nonzero counts differ.
            (PerGroupedChannelGranularity(axis=0, group_size=_GROUP_SIZE), None, True, 4),
            (PerGroupedChannelGranularity(axis=0, group_size=_GROUP_SIZE), None, True, 2),
            (PerGroupedChannelGranularity(axis=0, group_size=_GROUP_SIZE), None, True, 1),
        ],
        ids=[
            "per_tensor",
            "grouped_channel",
            "grouped_channel_uneven",
            "grouped_channel_uneven_2bit",
            "grouped_channel_uneven_1bit",
        ],
    )
    def test_round_trip(
        self,
        granularity,
        presparse_group_size,
        uneven,
        n_bits,
        custom_test_mnist_model,
        mnist_example_input,
    ):
        model = _presparsify(custom_test_mnist_model, _SPARSITY, presparse_group_size)
        zeros_before = {name: getattr(model, name).weight == 0 for name in _MNIST_LAYERS}
        palettizer = self._build_palettizer(model, n_bits=n_bits, granularity=granularity)
        prepared_model = palettizer.prepare((mnist_example_input,))
        with torch.no_grad():
            prepared_weights = {
                name: getattr(prepared_model, name).weight.clone() for name in _MNIST_LAYERS
            }

        finalized_model = palettizer.finalize(backend=ExportBackend.CoreAI)
        for name in _MNIST_LAYERS:
            module = getattr(finalized_model, name)
            reconstruction = next(
                p for p in module.parametrizations.weight if hasattr(p, "nonzero_indices")
            )
            lut, mask = reconstruction.lut, reconstruction.mask
            indices = reconstruction.nonzero_indices.long()
            assert lut.shape[1] == 2**n_bits, name
            if uneven:
                # Entry 0 is pinned to 0.0, and groups with fewer nonzeros also store zeros so
                # that every group holds as many elements as the fullest one.
                assert (lut[:, 0, 0] == 0).all(), name
                assert not (~zeros_before[name] & ~mask).any(), name
                stored = torch.stack([g.sum() for g in mask.split(_GROUP_SIZE, dim=0)])
                assert stored.unique().numel() == 1, name
                assert mask.sum() > (~zeros_before[name]).sum(), name
            else:
                # Every entry, including entry 0, is a real centroid of the nonzero weights.
                assert (lut != 0).all(), name
                assert torch.equal(mask, ~zeros_before[name]), name

            # De-palettize by hand: block g of the nonzero indices looks up LUT g, and the
            # values scatter back into the mask's positions.
            group = torch.arange(indices.numel()) // (indices.numel() // lut.shape[0])
            depalettized = torch.zeros(mask.shape, dtype=lut.dtype)
            depalettized[mask] = lut[group, indices, 0]
            assert torch.equal(depalettized == 0, zeros_before[name]), name
            # Weights that were zero before palettization stay exactly zero.
            assert (prepared_weights[name][zeros_before[name]] == 0).all(), name
            assert torch.equal(depalettized, prepared_weights[name]), name
            with torch.no_grad():
                assert torch.equal(module.weight, prepared_weights[name]), name

    @pytest.mark.parametrize("uneven", [False, True], ids=["per_tensor", "grouped_channel_uneven"])
    def test_already_palettized_sparse_weights_skip_kmeans(
        self, uneven, custom_test_mnist_model, mnist_example_input
    ):
        granularity = (
            PerGroupedChannelGranularity(axis=0, group_size=_GROUP_SIZE)
            if uneven
            else PerTensorGranularity()
        )
        model = _presparsify(custom_test_mnist_model, _SPARSITY)
        # Palettize at 3 bits, then reuse the result as an already-palettized sparse model.
        palettized = copy.deepcopy(model)
        prepared = self._build_palettizer(model, n_bits=3, granularity=granularity).prepare(
            (mnist_example_input,)
        )
        with torch.no_grad():
            for name in _MNIST_LAYERS:
                getattr(palettized, name).weight.copy_(getattr(prepared, name).weight)

        # Raising proves that k-means is skipped for every layer.
        with patch.object(
            _KMeansFakePalettize, "_cluster_weights_1d_masked", side_effect=AssertionError
        ):
            self._run(
                ExportBackend.CoreAI,
                palettized.eval(),
                mnist_example_input,
                {"n_bits": 4, "granularity": granularity},
                _JOINT_OPS[ExportBackend.CoreAI](len(_MNIST_LAYERS)),
            )

    def test_train_mode_forwards_uneven_groups(self, custom_test_mnist_model, mnist_example_input):
        model = _presparsify(custom_test_mnist_model, _SPARSITY)
        palettizer = self._build_palettizer(
            model,
            n_bits=4,
            granularity=PerGroupedChannelGranularity(axis=0, group_size=_GROUP_SIZE),
        )
        prepared_model = palettizer.prepare((mnist_example_input,))
        with palettizer.training_mode():
            for _ in range(3):
                prepared_model(mnist_example_input).sum().backward()
        with torch.no_grad():
            prepared_model_output = prepared_model(mnist_example_input)

        export_utils.convert_and_verify(
            finalized_model=palettizer.finalize(backend=ExportBackend.CoreAI),
            input_data=mnist_example_input,
            expected_ops=_JOINT_OPS[ExportBackend.CoreAI](len(_MNIST_LAYERS)),
            export_backend=ExportBackend.CoreAI,
            prepared_model_output=prepared_model_output,
        )

    def test_pad_mask_to_equal_groups(self):
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

        # A drifted LUT entry 0 cannot back the padded zeros.
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

    @pytest.mark.parametrize(
        "spec_kwargs",
        [c[1] for c in COREAI_REJECTS_CONFIGS],
        ids=[c[0] for c in COREAI_REJECTS_CONFIGS],
    )
    def test_coreai_rejects_vector_mnist(
        self, spec_kwargs, custom_test_mnist_model, mnist_example_input
    ):
        model = _presparsify(custom_test_mnist_model, _SPARSITY)
        self._run_rejects(ExportBackend.CoreAI, model, mnist_example_input, spec_kwargs)

    @pytest.mark.parametrize(
        "spec_kwargs",
        [c[1] for c in COREML_REJECTS_CONFIGS],
        ids=[c[0] for c in COREML_REJECTS_CONFIGS],
    )
    def test_coreml_rejects_vector_or_per_channel_scale_mnist(
        self, spec_kwargs, custom_test_mnist_model, mnist_example_input
    ):
        model = _presparsify(custom_test_mnist_model, _SPARSITY)
        self._run_rejects(ExportBackend.CoreML, model, mnist_example_input, spec_kwargs)
