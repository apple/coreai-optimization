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
    granularity matrix, on both export backends.
    """

    # int4/int8 x symmetric/asymmetric x per-tensor/per-channel. Symmetric always
    # has zero_point == 0 for a signed dtype; asymmetric has a data-dependent,
    # essentially-never-zero zero_point on a real trained weight -- so this
    # matrix also happens to split cleanly into "accepted" / "rejected".
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
    def _run_accepts(
        cls,
        backend: ExportBackend,
        model: nn.Module,
        input_data: torch.Tensor,
        dtype: torch.dtype,
        granularity: QuantizationGranularity,
        expected_count: int,
    ) -> None:
        model.eval()
        quantizer = cls._build_quantizer(model, dtype, QuantizationScheme.SYMMETRIC, granularity)
        prepared_model = quantizer.prepare((input_data,))

        with torch.no_grad():
            prepared_model_output = prepared_model(input_data)

        finalized_model = quantizer.finalize(backend=backend)

        export_utils.convert_and_verify(
            finalized_model=finalized_model,
            input_data=input_data,
            expected_ops=_QUANT_EXPECTED_OPS[backend](expected_count),
            export_backend=backend,
            prepared_model_output=prepared_model_output,
        )

    @classmethod
    def _run_rejects(
        cls,
        backend: ExportBackend,
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
            quantizer.finalize(backend=backend)

    @pytest.mark.parametrize("backend", _BACKENDS, ids=["coreai", "coreml"])
    @pytest.mark.parametrize(
        "dtype,granularity",
        [c[1:] for c in QUANT_VALID_CONFIGS],
        ids=[c[0] for c in QUANT_VALID_CONFIGS],
    )
    def test_accepts_zero_preserving_mnist(
        self, backend, dtype, granularity, custom_test_mnist_model, mnist_example_input
    ):
        self._run_accepts(
            backend,
            custom_test_mnist_model,
            mnist_example_input,
            dtype,
            granularity,
            _MNIST_LAYER_COUNT,
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
        self._run_accepts(
            backend, resnet50_model, resnet_example_input, dtype, granularity, _RESNET_LAYER_COUNT
        )

    @pytest.mark.parametrize("backend", _BACKENDS, ids=["coreai", "coreml"])
    @pytest.mark.parametrize(
        "dtype,granularity",
        [c[1:] for c in QUANT_INVALID_CONFIGS],
        ids=[c[0] for c in QUANT_INVALID_CONFIGS],
    )
    def test_rejects_nonzero_zero_point_mnist(
        self, backend, dtype, granularity, custom_test_mnist_model, mnist_example_input
    ):
        self._run_rejects(backend, custom_test_mnist_model, mnist_example_input, dtype, granularity)

    @pytest.mark.slow
    @pytest.mark.parametrize("backend", _BACKENDS, ids=["coreai", "coreml"])
    @pytest.mark.parametrize(
        "dtype,granularity",
        [c[1:] for c in QUANT_INVALID_CONFIGS],
        ids=[c[0] for c in QUANT_INVALID_CONFIGS],
    )
    def test_rejects_nonzero_zero_point_resnet(
        self, backend, dtype, granularity, resnet50_model, resnet_example_input
    ):
        self._run_rejects(backend, resnet50_model, resnet_example_input, dtype, granularity)


class TestJointPalettizationCompression:
    """PTP + PTS (post-training palettization + sparsity) across the n_bits/
    cluster_dim/granularity matrix, on both export backends.
    """

    # Per-tensor, scalar (cluster_dim=1) palettization: the only combination
    # joint-sparsity export supports, at a few n_bits.
    PALETT_VALID_CONFIGS: list[tuple[str, dict]] = [
        ("4bit", {"n_bits": 4}),
        ("6bit", {"n_bits": 6}),
        ("8bit", {"n_bits": 8}),
    ]
    # Each violates exactly one of the two allowances (per-tensor, scalar).
    PALETT_INVALID_CONFIGS: list[tuple[str, dict]] = [
        ("vector_ndim", {"n_bits": 4, "cluster_dim": 2}),
        (
            "grouped_channel",
            {"n_bits": 4, "granularity": PerGroupedChannelGranularity(axis=0, group_size=2)},
        ),
    ]

    @staticmethod
    def _build_palettizer(model: nn.Module, **spec_kwargs) -> KMeansPalettizer:
        spec_kwargs.setdefault("granularity", PalettPerTensorGranularity())
        config = KMeansPalettizerConfig(
            global_config=ModuleKMeansPalettizerConfig(
                op_state_spec={"weight": PalettizationSpec(_sparsity=_SPARSITY, **spec_kwargs)},
                # Required whenever cluster_dim > 1 is among the configs under test.
                enable_fast_kmeans_mode=False,
            )
        )
        return KMeansPalettizer(model, config)

    @classmethod
    def _run_accepts(
        cls,
        backend: ExportBackend,
        model: nn.Module,
        input_data: torch.Tensor,
        spec_kwargs: dict,
        expected_count: int,
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
            expected_ops=_PALETT_EXPECTED_OPS[backend](expected_count),
            export_backend=backend,
            prepared_model_output=prepared_model_output,
        )

    @classmethod
    def _run_rejects(
        cls, backend: ExportBackend, model: nn.Module, input_data: torch.Tensor, spec_kwargs: dict
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
    def test_accepts_scalar_per_tensor_mnist(
        self, backend, spec_kwargs, custom_test_mnist_model, mnist_example_input
    ):
        self._run_accepts(
            backend, custom_test_mnist_model, mnist_example_input, spec_kwargs, _MNIST_LAYER_COUNT
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("backend", _BACKENDS, ids=["coreai", "coreml"])
    @pytest.mark.parametrize(
        "spec_kwargs",
        [c[1] for c in PALETT_VALID_CONFIGS],
        ids=[c[0] for c in PALETT_VALID_CONFIGS],
    )
    def test_accepts_scalar_per_tensor_resnet(
        self, backend, spec_kwargs, resnet50_model, resnet_example_input
    ):
        self._run_accepts(
            backend, resnet50_model, resnet_example_input, spec_kwargs, _RESNET_LAYER_COUNT
        )

    @pytest.mark.parametrize("backend", _BACKENDS, ids=["coreai", "coreml"])
    @pytest.mark.parametrize(
        "spec_kwargs",
        [c[1] for c in PALETT_INVALID_CONFIGS],
        ids=[c[0] for c in PALETT_INVALID_CONFIGS],
    )
    def test_rejects_non_scalar_or_non_per_tensor_mnist(
        self, backend, spec_kwargs, custom_test_mnist_model, mnist_example_input
    ):
        self._run_rejects(backend, custom_test_mnist_model, mnist_example_input, spec_kwargs)

    @pytest.mark.slow
    @pytest.mark.parametrize("backend", _BACKENDS, ids=["coreai", "coreml"])
    @pytest.mark.parametrize(
        "spec_kwargs",
        [c[1] for c in PALETT_INVALID_CONFIGS],
        ids=[c[0] for c in PALETT_INVALID_CONFIGS],
    )
    def test_rejects_non_scalar_or_non_per_tensor_resnet(
        self, backend, spec_kwargs, resnet50_model, resnet_example_input
    ):
        self._run_rejects(backend, resnet50_model, resnet_example_input, spec_kwargs)
