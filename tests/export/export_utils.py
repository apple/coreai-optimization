# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Utilities for converting and verifying PyTorch models for export testing."""

import asyncio
import sys
import tempfile
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import coreai_torch
import coremltools as ct
import pytest
import torch
from coreai.authoring import AIProgram
from coreai.runtime import NDArray
from coremltools import ComputeUnit

from coreai_opt import CoreMLExportError, ExportBackend
from tests.test_utils.general import verify_snr_psnr as _verify_snr_psnr

# Substring of the dtype guard message raised by the CoreML export validation. Shared so
# test files asserting the rejection don't drift from one another.
COREML_DTYPE_REJECTION_MATCH = "CoreML export does not support"


def assert_coreml_finalize_rejects_unsupported_dtype(finalizer: Any) -> None:
    """Assert ``finalizer.finalize(backend=CoreML)`` rejects an unsupported dtype.

    CoreML does not support FP4, FP8, INT2, or UINT2 quantization or
    palettization dtypes, so finalize must raise a ``CoreMLExportError`` rather
    than emit an invalid model.

    Args:
        finalizer (Any): A prepared ``Quantizer`` or ``KMeansPalettizer``.
    """
    with pytest.raises(CoreMLExportError, match=COREML_DTYPE_REJECTION_MATCH):
        finalizer.finalize(backend=ExportBackend.CoreML)


class ModelConverter(ABC):
    """Base class for model conversion strategies."""

    # Public abstract methods

    @abstractmethod
    def trace(
        self,
        pytorch_model: torch.nn.Module | torch.fx.GraphModule,
        input_data: torch.Tensor,
        expected_ops: Mapping[str, int],
    ) -> Any:
        """Trace PyTorch model for conversion to specific format"""
        msg = "Subclasses must implement trace()"
        raise NotImplementedError(msg)

    @abstractmethod
    def convert(
        self,
        traced_model: Any,
        input_data: torch.Tensor,
        **kwargs,
    ) -> Any:
        """Convert traced PyTorch model to target format."""
        msg = "Subclasses must implement convert()"
        raise NotImplementedError(msg)

    # Public concrete methods

    def verify(
        self,
        converted_model: Any,
        input_data: torch.Tensor,
        expected_ops: Mapping[str, int],
        prepared_model_output: torch.Tensor | tuple[torch.Tensor, ...],
        finalized_model_output: torch.Tensor | tuple[torch.Tensor, ...] | None = None,
        snr_thresh: float = 20.0,
        psnr_thresh: float = 22.0,
        **kwargs,
    ) -> None:
        """Verify converted model outputs and operations.

        Template method that handles complete verification workflow:
        1. Verify pre-computed finalized model output against prepared model output
           (skipped if finalized_model_output is None)
        2. Run inference on converted model (calls _run_inference)
        3. Verify output shapes and metrics (SNR/PSNR)
        4. Verify expected operations are present

        Args:
            converted_model: Converted model (post-export, backend-specific format)
            input_data: Input tensor
            expected_ops: Expected operation counts in converted model
            prepared_model_output: Pre-computed reference output from the prepared
                PyTorch model (single tensor or tuple).
            finalized_model_output: Pre-computed output from the finalized PyTorch
                model (pre-export). Must be computed before tracing, since
                torch.export may mutate the model. Pass None to skip this
                verification.
            snr_thresh: Minimum acceptable SNR value
            psnr_thresh: Minimum acceptable PSNR value
            **kwargs: Additional backend-specific arguments

        """
        _ = kwargs

        # Normalize single tensor to tuple for unified processing
        if not isinstance(prepared_model_output, tuple):
            prepared_model_output = (prepared_model_output,)

        # Verify finalized model output
        if finalized_model_output is not None:
            finalized_model_outputs = (
                finalized_model_output
                if isinstance(finalized_model_output, tuple)
                else (finalized_model_output,)
            )
            self._verify_outputs(
                finalized_model_outputs,
                prepared_model_output,
                snr_thresh,
                psnr_thresh,
            )

        # Run inference on converted model (backend-specific)
        converted_outputs = self._run_inference(converted_model, input_data)

        # Verify operations
        self._verify_ops(converted_model, expected_ops)

        # Verify outputs
        self._verify_outputs(converted_outputs, prepared_model_output, snr_thresh, psnr_thresh)

    # Private abstract methods

    @abstractmethod
    def _run_inference(
        self,
        converted_model: Any,
        input_data: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Run inference on converted model and return outputs as torch tensors.

        Args:
            converted_model: The converted model
            input_data: Input tensor for inference

        Returns:
            Model output(s) as tuple of torch tensor(s)

        """
        msg = "Subclasses must implement _run_inference()"
        raise NotImplementedError(msg)

    @abstractmethod
    def _get_op_counts(
        self,
        converted_model: Any,
        op_names: Sequence[str],
    ) -> dict[str, int] | None:
        """Extract operation counts from converted model for specified operations.

        Args:
            converted_model: The converted model
            op_names: Sequence of operation names to count

        Returns:
            Dictionary mapping operation names to their actual counts, or None if
            op counting is not yet implemented for this backend

        """
        msg = "Subclasses must implement _get_op_counts()"
        raise NotImplementedError(msg)

    # Private concrete methods

    def _verify_outputs(
        self,
        target_outputs: tuple[torch.Tensor, ...],
        ref_outputs: tuple[torch.Tensor, ...],
        snr_thresh: float,
        psnr_thresh: float,
    ) -> None:
        """Verify target outputs match reference outputs.

        Args:
            target_outputs: Outputs from the model under verification
                (e.g., converted model or finalized model)
            ref_outputs: Reference outputs from the prepared PyTorch model
            snr_thresh: Minimum acceptable SNR value
            psnr_thresh: Minimum acceptable PSNR value

        """
        # Validate output count matches with `strict=True`
        # Validate dtypes, shapes, and metrics
        for i, (target_tensor, ref_tensor) in enumerate(
            zip(target_outputs, ref_outputs, strict=True),
        ):
            assert target_tensor.dtype == ref_tensor.dtype, (
                f"Dtype mismatch for output {i}: "
                f"target {target_tensor.dtype} "
                f"vs reference {ref_tensor.dtype}"
            )

            assert target_tensor.shape == ref_tensor.shape, (
                f"Shape mismatch for output {i}: "
                f"target {target_tensor.shape} "
                f"vs reference {ref_tensor.shape}"
            )

            _verify_snr_psnr(
                target_tensor,
                ref_tensor,
                snr_thresh,
                psnr_thresh,
                label=f"Output {i}",
            )

    def _verify_ops(
        self,
        converted_model: Any,
        expected_ops: Mapping[str, int],
    ) -> None:
        """Verify expected operations are present in converted model.

        Template method that gets actual op counts via _get_op_counts() and compares
        them against expected counts.

        Args:
            converted_model: The converted model
            expected_ops: Map of operation names to expected counts

        """
        if not expected_ops:
            return

        actual_op_counts = self._get_op_counts(
            converted_model,
            list(expected_ops.keys()),
        )

        # Skip verification if None (indicates unimplemented backend)
        if actual_op_counts is None:
            return

        # Ensure all expected ops exist in actual counts (fill missing with 0)
        normalized_actual = {op: actual_op_counts.get(op, 0) for op in expected_ops}

        assert normalized_actual == expected_ops, (
            f"Op count mismatch: expected {dict(expected_ops)}, found {normalized_actual}"
        )


class MILConverter(ModelConverter):
    """Handles conversion to CoreML MIL format."""

    # Public methods

    def trace(
        self,
        pytorch_model: torch.nn.Module | torch.fx.GraphModule,
        input_data: torch.Tensor,
        expected_ops: Mapping[str, int],
    ) -> torch.jit.ScriptModule:
        _ = expected_ops

        pytorch_model.eval()

        with torch.no_grad():
            traced_model = torch.jit.trace(pytorch_model, example_inputs=(input_data,))

        # API contract ensures the returned model is a torch.jit.ScriptModule
        return traced_model

    def convert(
        self,
        traced_model: torch.jit.ScriptModule,
        input_data: torch.Tensor,
        compute_unit: ComputeUnit = ComputeUnit.CPU_ONLY,
        pass_pipeline: ct.PassPipeline | None = None,
        minimum_deployment_target: ct.target = ct.target.iOS18,
        **kwargs: Any,
    ) -> ct.models.MLModel:
        _ = kwargs
        try:
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(shape=input_data.shape)],
                compute_units=compute_unit,
                pass_pipeline=pass_pipeline,
                minimum_deployment_target=minimum_deployment_target,
            )
        except Exception as err:
            msg = f"CoreML conversion failed: {err}"
            raise RuntimeError(msg) from err

        # `ct.convert` always return `ct.models.MLModel`.
        return coreml_model

    def _run_inference(
        self,
        converted_model: ct.models.MLModel,
        input_data: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        # CoreML runtime only available on Apple platforms
        if sys.platform == "linux":
            pytest.skip(
                "CoreML runtime not available on Linux - CoreML export verification requires macOS"
            )

        # Validate MIL program exists
        assert converted_model._mil_program is not None, "MIL model is not available"  # noqa: SLF001

        # Prepare input dictionary
        ml_model_input_names = [str(x) for x in converted_model.input_description]
        assert len(ml_model_input_names) == 1, (
            f"Expected single input, got {len(ml_model_input_names)}"
        )

        coreml_input_dict = {ml_model_input_names[0]: input_data.cpu().detach().numpy()}

        # Run CoreML prediction
        output_dict = converted_model.predict(coreml_input_dict)

        # Convert numpy outputs to torch tensors and return as tuple
        output_tensors = tuple(torch.from_numpy(v) for v in output_dict.values())
        return output_tensors

    def _get_op_counts(
        self,
        converted_model: ct.models.MLModel,
        op_names: Sequence[str],
    ) -> dict[str, int]:
        assert converted_model._mil_program is not None, "MIL model is not available"  # noqa: SLF001

        main_function = converted_model._mil_program.functions["main"]  # noqa: SLF001

        op_counts = {}
        for op_name in op_names:
            compressed_ops = main_function.find_ops(op_type=op_name)
            op_counts[op_name] = len(compressed_ops)

        return op_counts


class MLIRConverter(ModelConverter):
    """Handles conversion to Core AI MLIR format."""

    def trace(
        self,
        pytorch_model: torch.nn.Module | torch.fx.GraphModule,
        input_data: torch.Tensor,
        expected_ops: Mapping[str, int],
    ) -> torch.export.ExportedProgram:
        pytorch_model.eval()

        with torch.no_grad():
            exported_program = torch.export.export(
                pytorch_model,
                (input_data,),
                strict=False,
            )
            exported_program = exported_program.run_decompositions()

        # Verify right compression ops were inserted in the model
        self._verify_custom_ops_in_torch_program(exported_program, expected_ops)

        return exported_program

    def convert(
        self,
        traced_model: torch.export.ExportedProgram,
        input_data: torch.Tensor,
        **kwargs: Any,
    ) -> AIProgram:
        _, _ = input_data, kwargs
        coreai_program = self._lower_to_coreai(traced_model)
        assert type(coreai_program) is AIProgram

        return coreai_program

    def _run_inference(
        self,
        converted_model: AIProgram,
        input_data: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return asyncio.run(self._run_inference_async(converted_model, input_data))

    async def _run_inference_async(
        self,
        converted_model: AIProgram,
        input_data: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Async implementation of MLIR inference."""
        with tempfile.TemporaryDirectory(
            prefix="mlir_converter_inference",
            suffix=".aimodel",
        ) as tmpdir:
            asset = converted_model.save_asset(Path(tmpdir))
            async with asset.executable() as ai_model:
                rt_func = ai_model.load_function("main")

                input_names = rt_func.desc.input_names
                assert len(input_names) == 1, (
                    f"Expected 1 input, got {len(input_names)}: {input_names}"
                )
                input_name = input_names[0]

                coreai_outputs = await rt_func(
                    inputs={input_name: NDArray(input_data.cpu())},
                )

                # TODO: replace with coreai's public NDArray torch() conversion
                # once that API is available.
                return tuple(
                    torch.from_dlpack(v._tensor.to_dlpack())  # noqa: SLF001
                    for v in coreai_outputs.values()
                )

    def _get_op_counts(
        self,
        converted_model: AIProgram,
        op_names: Sequence[str],
    ) -> dict[str, int] | None:
        # TODO: implement op extraction from AIProgram.
        _, _ = converted_model, op_names
        return None

    def _verify_custom_ops_in_torch_program(
        self,
        exported_program: torch.export.ExportedProgram,
        expected_ops: Mapping[str, int],
    ) -> None:
        """Verify expected compression operations in exported program."""
        if not expected_ops:
            return

        expected_mlir_ops = {}
        for k, v in expected_ops.items():
            mlir_op_key = "coreai::" + k
            expected_mlir_ops[mlir_op_key] = v

        found_ops = defaultdict(int)
        for node in exported_program.graph.nodes:
            if hasattr(node.target, "name") and node.target.name() in expected_mlir_ops:
                found_ops[node.target.name()] += 1

        for op_name, op_count in expected_mlir_ops.items():
            # Skip verification for ops with expected count of 0
            if op_count == 0:
                continue

            assert op_name in found_ops, f"{op_name} not found in exported model!"

            assert found_ops[op_name] == op_count, (
                f"Expected {op_count} occurrences of {op_name}, found {found_ops[op_name]}"
            )

    @staticmethod
    def _lower_to_coreai(
        exported_program: torch.export.ExportedProgram,
    ) -> AIProgram:
        """Lower exported program to Core AI."""
        converter = coreai_torch.TorchConverter()
        converter.add_exported_program(exported_program)
        return converter.to_coreai()


def create_converter(backend: ExportBackend) -> ModelConverter:
    """Create the appropriate converter for the backend."""
    converter_map: dict[ExportBackend, type[ModelConverter]] = {
        ExportBackend.CoreML: MILConverter,
        ExportBackend.CoreAI: MLIRConverter,
    }

    converter_cls = converter_map.get(backend)
    if converter_cls is None:
        msg = f"Unsupported export backend: {backend}"
        raise ValueError(msg)

    return converter_cls()


def convert_and_verify(
    finalized_model: torch.nn.Module,
    input_data: torch.Tensor,
    expected_ops: Mapping[str, int],
    export_backend: ExportBackend,
    prepared_model_output: torch.Tensor | tuple[torch.Tensor, ...],
    snr_thresh: float = 20.0,
    psnr_thresh: float = 22.0,
    skip_finalized_model_verify: bool = False,
    **converter_kwargs: Any,
) -> Any:
    """Convert a PyTorch model to the specified inference stack and verify its outputs.

    Args:
        finalized_model: The finalized PyTorch model to convert
        input_data: Input tensor for the model
        expected_ops: Map of ops expected in the converted model and the expected count
        export_backend: Target inference stack (CoreML or CoreAI)
        prepared_model_output: Pre-computed reference output from the prepared
            PyTorch model (single tensor or tuple).
        snr_thresh: Minimum acceptable SNR value
        psnr_thresh: Minimum acceptable PSNR value
        skip_finalized_model_verify: If True, skip forward pass verification on
            finalized_model (e.g., when finalized model contains unsupported ops)
        converter_kwargs: Backend-specific converter arguments
            (e.g., compute_unit for MIL)

    Returns:
        The converted model in the specified format

    """
    converter = create_converter(export_backend)

    # Run finalized model forward pass BEFORE tracing. torch.export.export()
    # (called in trace) may mutate the model (e.g., strip parametrizations on
    # older PyTorch versions), so the forward pass must happen first.
    finalized_model_output = None
    if not skip_finalized_model_verify:
        finalized_model.eval()
        with torch.no_grad():
            finalized_model_output = finalized_model(input_data)

    traced_model = converter.trace(finalized_model, input_data, expected_ops)
    converted_model = converter.convert(
        traced_model,
        input_data,
        **converter_kwargs,
    )
    converter.verify(
        converted_model,
        input_data,
        expected_ops,
        prepared_model_output=prepared_model_output,
        finalized_model_output=finalized_model_output,
        snr_thresh=snr_thresh,
        psnr_thresh=psnr_thresh,
    )

    return converted_model
