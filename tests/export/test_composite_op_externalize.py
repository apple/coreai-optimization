# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Externalize-specific structural tests for _patch_model_for_externalization
in presence of coreai-opt graph mode quantization.

Test structural assertions: after ``_patch_model_for_externalization``
patches a composite submodule's forward into a ``torch.library.custom_op``,
the resulting opaque call_function node survives Graph-mode ``prepare`` + ``finalize``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from coreai_torch import ExternalizeSpec, _patch_model_for_externalization

from coreai_opt import ExportBackend
from coreai_opt.quantization import (
    Quantizer,
    QuantizerConfig,
)
from coreai_opt.quantization.spec import (
    PerTensorGranularity,
    QuantizationScheme,
    QuantizationSpec,
)
from tests.fixtures.quantization import (
    COMPOSITE_BOUNDARY_ACT_DTYPE,
    make_graph_mode_composite_boundary_config,
    make_graph_mode_ptq_config,
)
from tests.models.composite import (
    CompositeRMSNormModel,
    CompositeRMSNormOnlyModel,
    CompositeSDPAModel,
    rmsnorm_externalize_spec,
    sdpa_externalize_spec,
)
from tests.test_utils.general import (
    assert_single_call_function_node,
    get_quantize_dtype,
    is_coreai_dequantize,
    is_coreai_quantize,
)

_RMSNORM_SPEC = rmsnorm_externalize_spec()
_SDPA_SPEC = sdpa_externalize_spec()


@pytest.mark.parametrize(
    "quantize_activations",
    [
        pytest.param(False, id="w8-weight-only"),
        pytest.param(True, id="w8a8"),
    ],
)
def test_composite_op_survives_prepare_and_finalize(
    composite_rmsnorm_model,
    composite_rmsnorm_input,
    quantize_activations: bool,
) -> None:
    """The externalized composite must remain a single opaque
    call_function node end-to-end, under both w8 and w8a8.
    """
    model = composite_rmsnorm_model
    sample = composite_rmsnorm_input

    _patch_model_for_externalization(model, [_RMSNORM_SPEC])
    op_name = model.norm._externalize_op_name
    target_substr = f"coreai_torch_ext.{op_name}"

    quantizer = Quantizer(
        model, make_graph_mode_ptq_config(quantize_activations=quantize_activations)
    )
    prepared = quantizer.prepare((sample,))
    assert_single_call_function_node(prepared, target_substr, stage="prepared")

    finalized = quantizer.finalize(backend=ExportBackend.CoreAI)
    assert_single_call_function_node(finalized, target_substr, stage="finalized")


# Composite op I/O boundary quantization

# (model class, externalize spec, submodule attribute name, tensor input count).
# All three models default to dim=32 and accept the same rank-3 fp16 sample.
_BOUNDARY_CASES = [
    pytest.param(CompositeRMSNormOnlyModel, _RMSNORM_SPEC, "norm", 1, id="rmsnorm-only"),
    pytest.param(CompositeRMSNormModel, _RMSNORM_SPEC, "norm", 1, id="rmsnorm-mixed"),
    pytest.param(CompositeSDPAModel, _SDPA_SPEC, "composite", 3, id="sdpa-qkv"),
]


class TestCompositeOpIOQuantization:
    """Ensure externalized composite's I/O boundary can be quantized
    via a module-level config, by name and by type.

    A global config quantizes the rest of the model with the default activation
    dtype (int8) and the composite config provides a distinct
    ``_COMPOSITE_ACT_DTYPE`` (uint8) on the composite's edges. Module config
    outranks global, so the composite boundary must carry the composite dtype
    while every other quantized edge carries the global dtype.
    """

    _COMPOSITE_ACT_DTYPE = COMPOSITE_BOUNDARY_ACT_DTYPE

    @classmethod
    def _composite_act_spec(cls) -> QuantizationSpec:
        return QuantizationSpec(
            dtype=cls._COMPOSITE_ACT_DTYPE,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerTensorGranularity(),
        )

    @classmethod
    def _config(
        cls,
        spec: ExternalizeSpec,
        module_name: str,
        target_by: str,
        module_input_spec: dict | None = None,
    ) -> QuantizerConfig:
        return make_graph_mode_composite_boundary_config(
            module_name=module_name if target_by == "name" else None,
            module_type=spec.target_class if target_by == "type" else None,
            module_input_spec=module_input_spec,
        )

    def _finalize(
        self,
        model: nn.Module,
        sample: torch.Tensor,
        spec: ExternalizeSpec,
        module_name: str,
        target_by: str,
        module_input_spec: dict | None = None,
    ) -> tuple[torch.fx.GraphModule, str]:
        _patch_model_for_externalization(model, [spec])
        op_name = model.get_submodule(module_name)._externalize_op_name
        target_substr = f"coreai_torch_ext.{op_name}"

        quantizer = Quantizer(model, self._config(spec, module_name, target_by, module_input_spec))
        prepared = quantizer.prepare((sample,))
        assert_single_call_function_node(prepared, target_substr, stage="prepared")

        finalized = quantizer.finalize(backend=ExportBackend.CoreAI)
        assert_single_call_function_node(finalized, target_substr, stage="finalized")
        return finalized, target_substr

    def _assert_boundary_quantized(
        self,
        finalized: torch.fx.GraphModule,
        target_substr: str,
        num_tensor_inputs: int,
    ) -> None:
        composite = assert_single_call_function_node(finalized, target_substr, stage="finalized")

        # A composite's non-tensor captured attributes appear either as baked-in
        # constants (SDPA's scale / is_causal / window_size) or as a get_attr arg
        # (RMSNorm's scale), and neither is a quantized activation edge, so
        # filter get_attr out rather than indexing fixed arg positions.
        tensor_inputs = [
            a for a in composite.args if isinstance(a, torch.fx.Node) and a.op != "get_attr"
        ]
        assert len(tensor_inputs) == num_tensor_inputs, (
            f"Expected {num_tensor_inputs} tensor inputs to {composite.name}, "
            f"got {[n.name for n in tensor_inputs]}"
        )
        for act_input in tensor_inputs:
            assert is_coreai_dequantize(act_input.target)
            assert get_quantize_dtype(act_input.args[0]) == self._COMPOSITE_ACT_DTYPE

        users = list(composite.users)
        assert len(users) == 1
        assert is_coreai_quantize(users[0].target)
        assert get_quantize_dtype(users[0]) == self._COMPOSITE_ACT_DTYPE

        composite_dtype_quant = [
            n
            for n in finalized.graph.nodes
            if is_coreai_quantize(n.target) and get_quantize_dtype(n) == self._COMPOSITE_ACT_DTYPE
        ]
        assert len(composite_dtype_quant) == num_tensor_inputs + 1

    @pytest.mark.parametrize("target_by", ["name", "type"])
    @pytest.mark.parametrize("model_cls, spec, module_name, num_tensor_inputs", _BOUNDARY_CASES)
    def test_composite_boundary_quantized(
        self,
        model_cls: type[nn.Module],
        spec: ExternalizeSpec,
        module_name: str,
        num_tensor_inputs: int,
        target_by: str,
    ) -> None:
        model = model_cls().eval().half()
        sample = torch.randn(2, 4, 32, dtype=torch.float16)
        finalized, target_substr = self._finalize(model, sample, spec, module_name, target_by)
        self._assert_boundary_quantized(finalized, target_substr, num_tensor_inputs)

    @pytest.mark.parametrize("target_by", ["name", "type"])
    def test_composite_boundary_input_index_selects_those_args(self, target_by: str) -> None:
        """Integer keys in ``module_input_spec`` quantize exactly those positional args
        for composite ops.

        The unselected input is left  unquantized rather than falling
        back to the global spec, because the composite is opaque to the
        op-pattern annotator and only a module-level config reaches its edges.
        """
        quantized_indices = (0, 2)
        num_tensor_inputs = 3
        model = CompositeSDPAModel().eval().half()
        sample = torch.randn(2, 4, 32, dtype=torch.float16)

        finalized, target_substr = self._finalize(
            model,
            sample,
            _SDPA_SPEC,
            "composite",
            target_by,
            module_input_spec={i: self._composite_act_spec() for i in quantized_indices},
        )

        composite = assert_single_call_function_node(finalized, target_substr, stage="finalized")
        tensor_inputs = [
            a for a in composite.args if isinstance(a, torch.fx.Node) and a.op != "get_attr"
        ]
        assert len(tensor_inputs) == num_tensor_inputs, (
            f"Expected {num_tensor_inputs} tensor inputs to {composite.name}, "
            f"got {[n.name for n in tensor_inputs]}"
        )

        for index, act_input in enumerate(tensor_inputs):
            if index in quantized_indices:
                assert is_coreai_dequantize(act_input.target), (
                    f"input {index} was selected by module_input_spec but is not fed by "
                    f"a dequantize: {act_input.target}"
                )
                input_dtype = get_quantize_dtype(act_input.args[0])
                assert input_dtype == self._COMPOSITE_ACT_DTYPE, (
                    f"input {index} was selected by module_input_spec but is quantized as "
                    f"{input_dtype}, expected the composite dtype {self._COMPOSITE_ACT_DTYPE}"
                )
            else:
                assert not is_coreai_dequantize(act_input.target), (
                    f"input {index} was not selected by module_input_spec but is fed by "
                    f"a dequantize: {act_input.target}"
                )

        # module_output_spec stays the wildcard, so the composite's consumer is
        # quantized with the composite dtype regardless of which inputs were selected.
        consumers = list(composite.users)
        assert len(consumers) == 1, (
            f"expected the composite to have exactly one consumer, got "
            f"{[n.name for n in consumers]}"
        )
        consumer = consumers[0]
        assert is_coreai_quantize(consumer.target), (
            f"the composite's consumer is not a quantize node: {consumer.target}"
        )
        consumer_dtype = get_quantize_dtype(consumer)
        assert consumer_dtype == self._COMPOSITE_ACT_DTYPE, (
            f"the composite's output is quantized as {consumer_dtype}, expected the "
            f"composite dtype {self._COMPOSITE_ACT_DTYPE}"
        )
