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
from coreai_opt.palettization.spec import (
    PerGroupedChannelGranularity,
    PerTensorGranularity as PalettPerTensorGranularity,
)
from coreai_opt.quantization import ModuleQuantizerConfig, Quantizer, QuantizerConfig
from coreai_opt.quantization.config import ExecutionMode
from coreai_opt.quantization.spec import (
    PerChannelGranularity,
    PerTensorGranularity,
    QuantizationGranularity,
    QuantizationScheme,
    QuantizationSpec,
)

from . import export_utils

_SPARSITY = 0.5
_MNIST_LAYER_COUNT = 6
_RESNET_LAYER_COUNT = 54
_BACKENDS = [ExportBackend.CoreAI, ExportBackend.CoreML]

_QUANT_EXPECTED_OPS = {
    ExportBackend.CoreAI: lambda n: {
        "sparse_to_dense": n,
        "constexpr_blockwise_shift_scale": n,
    },
    ExportBackend.CoreML: lambda n: {
        "constexpr_sparse_to_dense": n,
        "constexpr_sparse_blockwise_shift_scale": n,
    },
}


_PALETT_EXPECTED_OPS = {
    ExportBackend.CoreAI: lambda n: {"lut_to_dense": n, "sparse_to_dense": n},
    ExportBackend.CoreML: lambda n: {"constexpr_lut_to_sparse": n, "constexpr_sparse_to_dense": n},
}


class TestJointQuantizationCompression:
    """PTQ + PTS (post-training quantization + sparsity) across the dtype/qscheme/
    granularity matrix.

    CoreAI's sparse_to_dense scatters raw quantized codes before dequantizing
    the whole reconstructed tensor, so a nonzero zero_point would misrepresent
    pruned positions -- it must reject those configs outright. CoreML's sparse
    constexpr chain dequantizes the compact nonzero_data first and only then
    scatters (real float 0.0 padding), which is correct for any zero_point --
    so it always takes the joint sparse chain when sparsity is set, with no
    zero_point-dependent gating at all.
    """

    # int4/int8 x symmetric/asymmetric x per-tensor/per-channel. Symmetric always
    # has zero_point == 0 for a signed dtype; asymmetric has a data-dependent,
    # essentially-never-zero zero_point on a real trained weight -- so this
    # matrix also happens to split cleanly into "CoreAI accepts" / "CoreAI
    # rejects" (CoreML accepts and joint-chains both groups identically).
    QUANT_VALID_CONFIGS: list[tuple[str, torch.dtype, QuantizationGranularity]] = [
        ("int8_symmetric_per_tensor", torch.int8, PerTensorGranularity()),
        ("int8_symmetric_per_channel", torch.int8, PerChannelGranularity(axis=0)),
        ("int4_symmetric_per_tensor", torch.int4, PerTensorGranularity()),
        ("int4_symmetric_per_channel", torch.int4, PerChannelGranularity(axis=0)),
    ]
    QUANT_INVALID_CONFIGS: list[tuple[str, torch.dtype, QuantizationGranularity]] = [
        ("int8_asymmetric_per_tensor", torch.int8, PerTensorGranularity()),
        ("int8_asymmetric_per_channel", torch.int8, PerChannelGranularity(axis=0)),
        ("int4_asymmetric_per_tensor", torch.int4, PerTensorGranularity()),
        ("int4_asymmetric_per_channel", torch.int4, PerChannelGranularity(axis=0)),
    ]

    @staticmethod
    def _build_quantizer(
        model: nn.Module,
        dtype: torch.dtype,
        qscheme: QuantizationScheme,
        granularity: QuantizationGranularity,
    ) -> Quantizer:
        config = QuantizerConfig(
            global_config=ModuleQuantizerConfig(
                op_state_spec={
                    "weight": QuantizationSpec(
                        dtype=dtype,
                        qscheme=qscheme,
                        granularity=granularity,
                        _sparsity=_SPARSITY,
                    )
                },
                op_input_spec=None,
                op_output_spec=None,
            ),
            execution_mode=ExecutionMode.GRAPH,
        )
        return Quantizer(model, config)

    @classmethod
    def _run(
        cls,
        backend: ExportBackend,
        model: nn.Module,
        input_data: torch.Tensor,
        dtype: torch.dtype,
        qscheme: QuantizationScheme,
        granularity: QuantizationGranularity,
        expected_ops: dict[str, int],
    ) -> None:
        model.eval()
        quantizer = cls._build_quantizer(model, dtype, qscheme, granularity)
        prepared_model = quantizer.prepare((input_data,))

        with torch.no_grad():
            prepared_model_output = prepared_model(input_data)

        finalized_model = quantizer.finalize(backend=backend)

        export_utils.convert_and_verify(
            finalized_model=finalized_model,
            input_data=input_data,
            expected_ops=expected_ops,
            export_backend=backend,
            prepared_model_output=prepared_model_output,
        )

    @classmethod
    def _run_rejects(
        cls,
        model: nn.Module,
        input_data: torch.Tensor,
        dtype: torch.dtype,
        granularity: QuantizationGranularity,
    ) -> None:
        model.eval()
        quantizer = cls._build_quantizer(model, dtype, QuantizationScheme.ASYMMETRIC, granularity)
        prepared_model = quantizer.prepare((input_data,))

        with torch.no_grad():
            prepared_model(input_data)

        with pytest.raises((RuntimeError, ValueError)):
            quantizer.finalize(backend=ExportBackend.CoreAI)

    @pytest.mark.parametrize("backend", _BACKENDS, ids=["coreai", "coreml"])
    @pytest.mark.parametrize(
        "dtype,granularity",
        [c[1:] for c in QUANT_VALID_CONFIGS],
        ids=[c[0] for c in QUANT_VALID_CONFIGS],
    )
    def test_accepts_zero_preserving_mnist(
        self, backend, dtype, granularity, custom_test_mnist_model, mnist_example_input
    ):
        self._run(
            backend,
            custom_test_mnist_model,
            mnist_example_input,
            dtype,
            QuantizationScheme.SYMMETRIC,
            granularity,
            _QUANT_EXPECTED_OPS[backend](_MNIST_LAYER_COUNT),
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("backend", _BACKENDS, ids=["coreai", "coreml"])
    @pytest.mark.parametrize(
        "dtype,granularity",
        [c[1:] for c in QUANT_VALID_CONFIGS],
        ids=[c[0] for c in QUANT_VALID_CONFIGS],
    )
    def test_accepts_zero_preserving_resnet(
        self, backend, dtype, granularity, resnet50_model, resnet_example_input
    ):
        self._run(
            backend,
            resnet50_model,
            resnet_example_input,
            dtype,
            QuantizationScheme.SYMMETRIC,
            granularity,
            _QUANT_EXPECTED_OPS[backend](_RESNET_LAYER_COUNT),
        )

    @pytest.mark.parametrize(
        "dtype,granularity",
        [c[1:] for c in QUANT_INVALID_CONFIGS],
        ids=[c[0] for c in QUANT_INVALID_CONFIGS],
    )
    def test_coreai_rejects_nonzero_zero_point_mnist(
        self, dtype, granularity, custom_test_mnist_model, mnist_example_input
    ):
        self._run_rejects(custom_test_mnist_model, mnist_example_input, dtype, granularity)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "dtype,granularity",
        [c[1:] for c in QUANT_INVALID_CONFIGS],
        ids=[c[0] for c in QUANT_INVALID_CONFIGS],
    )
    def test_coreai_rejects_nonzero_zero_point_resnet(
        self, dtype, granularity, resnet50_model, resnet_example_input
    ):
        self._run_rejects(resnet50_model, resnet_example_input, dtype, granularity)

    @pytest.mark.parametrize(
        "dtype,granularity",
        [c[1:] for c in QUANT_INVALID_CONFIGS],
        ids=[c[0] for c in QUANT_INVALID_CONFIGS],
    )
    def test_coreml_accepts_nonzero_zero_point_mnist(
        self, dtype, granularity, custom_test_mnist_model, mnist_example_input
    ):
        # Unlike CoreAI, CoreML's sparse constexpr chain dequantizes the
        # compact nonzero_data before scattering it into the padded dense
        # tensor, so the padding is a real float 0.0 rather than a raw int
        # later reinterpreted through zero_point -- safe for any zero_point,
        # so the joint chain always applies here, same op counts as symmetric.
        self._run(
            ExportBackend.CoreML,
            custom_test_mnist_model,
            mnist_example_input,
            dtype,
            QuantizationScheme.ASYMMETRIC,
            granularity,
            _QUANT_EXPECTED_OPS[ExportBackend.CoreML](_MNIST_LAYER_COUNT),
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "dtype,granularity",
        [c[1:] for c in QUANT_INVALID_CONFIGS],
        ids=[c[0] for c in QUANT_INVALID_CONFIGS],
    )
    def test_coreml_accepts_nonzero_zero_point_resnet(
        self, dtype, granularity, resnet50_model, resnet_example_input
    ):
        # See test_coreml_accepts_nonzero_zero_point_mnist.
        self._run(
            ExportBackend.CoreML,
            resnet50_model,
            resnet_example_input,
            dtype,
            QuantizationScheme.ASYMMETRIC,
            granularity,
            _QUANT_EXPECTED_OPS[ExportBackend.CoreML](_RESNET_LAYER_COUNT),
        )


class TestJointPalettizationCompression:
    """PTP + PTS (post-training palettization + sparsity) across the n_bits/
    cluster_dim/granularity matrix.

    Masking flattens indices to rank 1 before the LUT lookup. CoreAI supports
    scalar (cluster_dim=1) palettization with per-tensor or axis-0 grouped LUTs:
    the mask is balanced per group, so each group's nonzero indices form one
    contiguous, equal-length block that ``lut_to_dense`` maps to that group's LUT.
    It rejects other configs outright.

    CoreML is more permissive: per-tensor and per-grouped-channel granularity
    both keep one index per weight element, so coremltools' own sparse LUT op
    can flatten indices via the mask and still recover per-position group
    context -- confirmed directly against coremltools' own
    ``palettize_weights(joint_compression=True)``, so coreai_opt no longer
    gates on granularity for CoreML. A quantized LUT (``lut_qspec``) is safe
    too, since it only touches the small, fixed-size LUT array, independent of
    the sparsity mask. Vector palettization (``cluster_dim>1``) and
    ``enable_per_channel_scale`` are still rejected outright for CoreML: both
    are genuine coremltools limitations (an index-count mismatch and a
    scale/data rank mismatch, respectively), confirmed directly.
    """

    # Scalar (cluster_dim=1) palettization gets the joint sparse op chain on both backends.
    PALETT_VALID_CONFIGS: list[tuple[str, dict]] = [
        ("4bit", {"n_bits": 4}),
        ("6bit", {"n_bits": 6}),
        ("8bit", {"n_bits": 8}),
        (
            "grouped_channel",
            {"n_bits": 4, "granularity": PerGroupedChannelGranularity(axis=0, group_size=2)},
        ),
    ]
    # CoreAI rejects vector palettization outright; CoreML rejects it too (see class docstring).
    PALETT_INVALID_CONFIGS: list[tuple[str, dict]] = [
        ("vector_ndim", {"n_bits": 4, "cluster_dim": 2}),
    ]
    PALETT_COREML_REJECTS_CONFIGS: list[tuple[str, dict]] = [
        ("vector_ndim", {"n_bits": 4, "cluster_dim": 2}),
        ("per_channel_scale", {"n_bits": 4, "enable_per_channel_scale": True}),
    ]

    @staticmethod
    def _build_palettizer(model: nn.Module, **spec_kwargs) -> KMeansPalettizer:
        spec_kwargs.setdefault("granularity", PalettPerTensorGranularity())
        spec_kwargs.setdefault("_sparsity", _SPARSITY)
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
        model.eval()
        palettizer = cls._build_palettizer(model, **spec_kwargs)
        prepared_model = palettizer.prepare((input_data,))

        with torch.no_grad():
            prepared_model_output = prepared_model(input_data)

        finalized_model = palettizer.finalize(backend=backend)

        export_utils.convert_and_verify(
            finalized_model=finalized_model,
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
        model.eval()
        palettizer = cls._build_palettizer(model, **spec_kwargs)
        prepared_model = palettizer.prepare((input_data,))

        with torch.no_grad():
            prepared_model(input_data)

        with pytest.raises((RuntimeError, ValueError)):
            palettizer.finalize(backend=backend)

    @pytest.mark.parametrize("backend", _BACKENDS, ids=["coreai", "coreml"])
    @pytest.mark.parametrize(
        "spec_kwargs",
        [c[1] for c in PALETT_VALID_CONFIGS],
        ids=[c[0] for c in PALETT_VALID_CONFIGS],
    )
    def test_accepts_scalar_mnist(
        self, backend, spec_kwargs, custom_test_mnist_model, mnist_example_input
    ):
        self._run(
            backend,
            custom_test_mnist_model,
            mnist_example_input,
            spec_kwargs,
            _PALETT_EXPECTED_OPS[backend](_MNIST_LAYER_COUNT),
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("backend", _BACKENDS, ids=["coreai", "coreml"])
    @pytest.mark.parametrize(
        "spec_kwargs",
        [c[1] for c in PALETT_VALID_CONFIGS],
        ids=[c[0] for c in PALETT_VALID_CONFIGS],
    )
    def test_accepts_scalar_resnet(
        self, backend, spec_kwargs, resnet50_model, resnet_example_input
    ):
        self._run(
            backend,
            resnet50_model,
            resnet_example_input,
            spec_kwargs,
            _PALETT_EXPECTED_OPS[backend](_RESNET_LAYER_COUNT),
        )

    @pytest.mark.parametrize(
        "granularity",
        [PalettPerTensorGranularity(), PerGroupedChannelGranularity(axis=0, group_size=2)],
        ids=["per_tensor", "grouped_channel"],
    )
    def test_pruned_weights_stay_zero_with_balanced_groups(
        self, granularity, custom_test_mnist_model, mnist_example_input
    ):
        # At 2 bits and 25% sparsity, a zero centroid fit on pruned and unpruned weights
        # alike drifts off 0.
        sparsity = 0.25
        palettizer = self._build_palettizer(
            custom_test_mnist_model.eval(), n_bits=2, granularity=granularity, _sparsity=sparsity
        )
        prepared_model = palettizer.prepare((mnist_example_input,))
        group_size = getattr(granularity, "group_size", None)
        for name in ["conv1", "conv2", "conv_transpose1", "conv_transpose2", "dense1", "dense2"]:
            with torch.no_grad():
                weight = getattr(prepared_model, name).weight
            groups = weight.split(group_size or weight.shape[0], dim=0)
            num_zeros = {int((group == 0).sum()) for group in groups}
            assert num_zeros == {int(groups[0].numel() * sparsity)}, name
            assert weight.unique().numel() <= 2**2 * len(groups), name

    @pytest.mark.parametrize(
        "spec_kwargs",
        [c[1] for c in PALETT_INVALID_CONFIGS],
        ids=[c[0] for c in PALETT_INVALID_CONFIGS],
    )
    def test_coreai_rejects_vector_mnist(
        self, spec_kwargs, custom_test_mnist_model, mnist_example_input
    ):
        self._run_rejects(
            ExportBackend.CoreAI, custom_test_mnist_model, mnist_example_input, spec_kwargs
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "spec_kwargs",
        [c[1] for c in PALETT_INVALID_CONFIGS],
        ids=[c[0] for c in PALETT_INVALID_CONFIGS],
    )
    def test_coreai_rejects_vector_resnet(self, spec_kwargs, resnet50_model, resnet_example_input):
        self._run_rejects(ExportBackend.CoreAI, resnet50_model, resnet_example_input, spec_kwargs)

    @pytest.mark.parametrize(
        "spec_kwargs",
        [c[1] for c in PALETT_COREML_REJECTS_CONFIGS],
        ids=[c[0] for c in PALETT_COREML_REJECTS_CONFIGS],
    )
    def test_coreml_rejects_vector_or_per_channel_scale_mnist(
        self, spec_kwargs, custom_test_mnist_model, mnist_example_input
    ):
        self._run_rejects(
            ExportBackend.CoreML, custom_test_mnist_model, mnist_example_input, spec_kwargs
        )

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "spec_kwargs",
        [c[1] for c in PALETT_COREML_REJECTS_CONFIGS],
        ids=[c[0] for c in PALETT_COREML_REJECTS_CONFIGS],
    )
    def test_coreml_rejects_vector_or_per_channel_scale_resnet(
        self, spec_kwargs, resnet50_model, resnet_example_input
    ):
        self._run_rejects(ExportBackend.CoreML, resnet50_model, resnet_example_input, spec_kwargs)
