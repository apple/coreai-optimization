# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

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
from coreai_opt.palettization.spec import PerGroupedChannelGranularity, PerTensorGranularity

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
    """Zero the smallest-magnitude weights of each layer in place, per axis-0 group of
    ``group_size`` channels if given (so every group keeps the same count), else per tensor.
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
    """PTP of an already-sparse model: layers with at least ``_sparsity_detection_threshold``
    exact zeros are palettized jointly with their sparsity.

    Masking flattens indices to rank 1 before the LUT lookup. CoreAI supports scalar
    (cluster_dim=1) palettization with per-tensor or axis-0 grouped LUTs; a grouped LUT
    maps the ``g``-th contiguous block of nonzero indices to group ``g``, so every group
    must hold the same number of nonzeros, and uneven layers are exported densely instead.
    It rejects vector palettization outright.

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
    ) -> None:
        palettizer = cls._build_palettizer(model, **spec_kwargs)
        prepared_model = palettizer.prepare((input_data,))
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

    @pytest.mark.parametrize(
        "backend,expected_ops",
        [
            (ExportBackend.CoreAI, {"lut_to_dense": len(_MNIST_LAYERS), "sparse_to_dense": 0}),
            (ExportBackend.CoreML, _JOINT_OPS[ExportBackend.CoreML](len(_MNIST_LAYERS))),
        ],
        ids=["coreai_exports_dense", "coreml_stays_joint"],
    )
    def test_grouped_with_uneven_groups(
        self, backend, expected_ops, custom_test_mnist_model, mnist_example_input
    ):
        # Pruning per tensor leaves the channel groups with different nonzero counts.
        model = _presparsify(custom_test_mnist_model, _SPARSITY)
        granularity = PerGroupedChannelGranularity(axis=0, group_size=_GROUP_SIZE)
        self._run(
            backend,
            model,
            mnist_example_input,
            {"n_bits": 4, "granularity": granularity},
            expected_ops,
        )

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
