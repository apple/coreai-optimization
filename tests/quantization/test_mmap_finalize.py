# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Mmap backed finalize, tested for both eager and graph execution modes."""

import copy

import pytest
import torch
from safetensors.torch import load_file, save_file

from coreai_opt import ExportBackend
from coreai_opt.quantization import Quantizer
from coreai_opt.quantization.config import ExecutionMode
from tests.fixtures.fp4 import ParametrizedFP4Configs
from tests.fixtures.quantization import make_quant_config

MODES = [ExecutionMode.EAGER, ExecutionMode.GRAPH]


def weight_only_config(execution_mode, kind="int4"):
    """Weight only config for either mode"""
    if kind == "fp4":
        configs = ParametrizedFP4Configs.from_fp4_params(
            with_activation_quant=False, per_block_weights=True
        )
        return configs.eager if execution_mode is ExecutionMode.EAGER else configs.pt2e
    return make_quant_config(weight_dtype=torch.int4, act_dtype=None, execution_mode=execution_mode)


def prepare_and_finalize(model, config, example_input, mmap_dir):
    """Prepare and finalize ``model`` for CoreAI, with or without ``mmap_dir``."""
    quantizer = Quantizer(model, config)
    quantizer.prepare((example_input,))
    return quantizer.finalize(backend=ExportBackend.CoreAI, mmap_dir=mmap_dir)


def _resolve_live_buffer(finalized, stem, key, execution_mode):
    """Return the live buffer/parameter behind safetensors ``key`` in the file whose
    stem is ``stem``. This is the only mode specific step: graph keeps flat buffers
    named exactly ``key``, eager keeps them inside the weight's dequant
    parametrization, whose file stem is ``<module>.<param>``."""
    if execution_mode is ExecutionMode.GRAPH:
        return getattr(finalized, key)
    module_name, _, param_name = stem.rpartition(".")
    parent = finalized.get_submodule(module_name) if module_name else finalized
    dequant = next(p for p in parent.parametrizations[param_name] if hasattr(p, key))
    return getattr(dequant, key)


def _expected_file_names(weight_fqns, execution_mode):
    """Safetensors file name each weight FQN is written to: eager keeps the dotted
    ``<module>.<param>`` stem, graph mangles the dots to underscores."""
    if execution_mode is ExecutionMode.GRAPH:
        return sorted(f"{fqn.replace('.', '_')}.safetensors" for fqn in weight_fqns)
    return sorted(f"{fqn}.safetensors" for fqn in weight_fqns)


@pytest.mark.parametrize("execution_mode", MODES)
@pytest.mark.parametrize("kind", ["int4", "fp4"])
def test_finalize_mmap_is_file_backed_and_output_preserving(
    execution_mode, kind, simple_linear_model, simple_linear_model_input, tmp_path
):
    """With ``mmap_dir`` set, test every quantized weight and its qparams live in a safetensors
    file and are read back as mmap views, and finalized model output remains unchanged with
    and without mmap.
    """
    config = weight_only_config(execution_mode, kind)
    example_input = simple_linear_model_input

    model = simple_linear_model.eval()
    model_ref = copy.deepcopy(model)

    finalized_mmap = prepare_and_finalize(model, config, example_input, str(tmp_path))
    finalized_ref = prepare_and_finalize(model_ref, config, example_input, None)

    files = sorted(tmp_path.glob("*.safetensors"))
    assert [path.name for path in files] == _expected_file_names(
        ["l1.weight", "l2.weight"], execution_mode
    ), "finalize wrote an unexpected set of safetensors files"
    for path in files:
        for key, file_tensor in load_file(path).items():
            live = _resolve_live_buffer(finalized_mmap, path.stem, key, execution_mode)
            live = getattr(live, "elem", live)
            assert torch.equal(file_tensor, live), f"{key} in {path.name} does not match the buffer"
            assert not live.untyped_storage().resizable(), (
                f"{key} storage is resizable, so it is a direct allocation on {live.device}."
            )

    with torch.no_grad():
        assert torch.equal(finalized_ref(example_input), finalized_mmap(example_input)), (
            "mmap backed quantized weights changed the finalized model's output"
        )


@pytest.mark.parametrize("execution_mode", MODES)
def test_finalize_mmap_preserves_weight_sharing(
    execution_mode, shared_params_model, shared_params_model_input, tmp_path
):
    """A weight shared across layers is quantized once and written once, so mmap does
    not rewrite a file that has already been mapped, and output is unchanged."""
    example_input = shared_params_model_input
    config = weight_only_config(execution_mode)

    model = shared_params_model.eval()
    model_ref = copy.deepcopy(model)

    finalized_mmap = prepare_and_finalize(model, config, example_input, str(tmp_path))
    finalized_ref = prepare_and_finalize(model_ref, config, example_input, None)

    expected = {
        ExecutionMode.EAGER: [
            "input_layer.weight.safetensors",
            "output.weight.safetensors",
            "shared_linear.weight.safetensors",
        ],
        ExecutionMode.GRAPH: [
            "input_layer_weight.safetensors",
            "layer2_weight.safetensors",
            "output_weight.safetensors",
        ],
    }[execution_mode]
    files = sorted(path.name for path in tmp_path.glob("*.safetensors"))
    assert files == expected, f"a shared weight must be written once, got {files}"

    if execution_mode is ExecutionMode.EAGER:
        assert (
            finalized_mmap.layer1.parametrizations["weight"][0]
            is finalized_mmap.layer2.parametrizations["weight"][0]
        ), "mmap finalize did not preserve sharing for weight tied modules"
    else:
        # torch.export collapses the tied pair to a single buffer
        quantized_buffers = sorted(
            name for name, _ in finalized_mmap.named_buffers() if name.endswith("_weight_quantized")
        )
        assert quantized_buffers == [
            "input_layer_weight_quantized",
            "layer2_weight_quantized",
            "output_weight_quantized",
        ], f"tied weight was not collapsed to one graph buffer, got {quantized_buffers}"

    with torch.no_grad():
        assert torch.equal(finalized_ref(example_input), finalized_mmap(example_input))


@pytest.mark.parametrize("execution_mode", MODES)
@pytest.mark.parametrize("backend", [ExportBackend.CoreML, ExportBackend._TORCH])
def test_finalize_mmap_rejects_non_coreai_backend(
    execution_mode, backend, simple_linear_model, simple_linear_model_input, tmp_path
):
    """``mmap_dir`` is a CoreAI only feature in both execution modes."""
    quantizer = Quantizer(simple_linear_model.eval(), weight_only_config(execution_mode))
    quantizer.prepare((simple_linear_model_input,))

    with pytest.raises(
        ValueError, match="mmap_dir is only supported with backend=ExportBackend.CoreAI"
    ):
        quantizer.finalize(backend=backend, mmap_dir=str(tmp_path))


def test_eager_finalize_state_dict_safetensors_roundtrip(
    simple_linear_model, simple_linear_model_input, tmp_path
):
    """Test that an eager mmap finalized model survives a state_dict -> save_file -> load_file ->
    load_state_dict(assign=True) round trip with identical forward output."""

    example_input = simple_linear_model_input
    config = make_quant_config(
        weight_dtype=torch.int8, act_dtype=None, execution_mode=ExecutionMode.EAGER
    )
    finalized = prepare_and_finalize(
        simple_linear_model.eval(),
        config,
        example_input,
        str(tmp_path / "per_layer_mmap"),
    )

    with torch.no_grad():
        out_before = finalized(example_input)

    bundled = tmp_path / "full_state.safetensors"
    save_file(
        {
            k: v.contiguous()
            for k, v in finalized.state_dict().items()
            if isinstance(v, torch.Tensor)
        },
        str(bundled),
    )
    finalized.load_state_dict(load_file(str(bundled), device="cpu"), assign=True)

    with torch.no_grad():
        assert torch.equal(out_before, finalized(example_input))
