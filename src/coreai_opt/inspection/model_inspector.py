# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Public API for model operation inspection."""

from __future__ import annotations

import re
import warnings
from typing import Any

import torch

from coreai_opt._utils.python_utils import fqn as _fqn
from coreai_opt._utils.torch_utils import export_model as _export_model
from coreai_opt.base_model_compressor import _BaseModelCompressor
from coreai_opt.palettization import KMeansPalettizer
from coreai_opt.quantization import Quantizer
from coreai_opt.quantization.config.quantization_config import ExecutionMode

from ._eager_mode import parse_ops_for_eager as _parse_ops_for_eager
from ._formatting import format_model_summary as _format_model_summary
from ._graph_mode import parse_ops_for_graph as _parse_ops_for_graph
from .types import ModelSummary, OpInfo


class ModelInspector:
    """Inspect operations in a PyTorch model for compression configuration.

    Accepts an ``nn.Module`` with example inputs, auto-exports the model
    (for graph mode), and provides query methods for discovering operation
    names, types, and module hierarchy.

    Attributes:
        summary (ModelSummary): The underlying operation summary.

    Args:
        model: The model to inspect.
        example_inputs: Example inputs for tracing.
        execution_mode: Execution mode to use for model inspection.
        compressor: A compressor class (e.g., ``Quantizer``) to filter
            ops to only those supported by that compression algorithm.
            When ``None``, all ops in the model are included.
        dynamic_shapes: Only relevant for graph execution mode.
            Optional dynamic shapes specification for torch.export.
        export_with_no_grad: Only relevant for "graph" execution mode.
            Whether to call torch.export.export within a
            torch.no_grad() context. Defaults to True.

    Raises:
        TypeError: If *model* is not an ``nn.Module``, or if *model* is a
            ``GraphModule`` and *execution_mode* is ``"eager"``.
        RuntimeError: If model export fails (graph mode).
        ValueError: If example_inputs is None without the right model/execution_mode combination, or
            if execution_mode is not either "eager" or "graph".

    Example:
        >>> import torch
        >>> import torch.nn as nn
        >>> from coreai_opt.inspection import ModelInspector
        >>> from coreai_opt.quantization import Quantizer
        >>> model = nn.Sequential(nn.Linear(10, 5))

        Inspect all compressable ops for the Quantizer compressor:

        >>> inspector = ModelInspector(model, (torch.randn(1, 10),),
        ...     execution_mode="graph", compressor=Quantizer)

        Query all ops in the model:

        >>> ops = inspector.summary.model.all_ops()

        Pretty print color coded summary of model inspection:

        >>> print(inspector.format_summary())

        Navigate the module hierarchy:

        >>> root = inspector.summary.model
        >>> for name, child in root.named_children():
        ...     print(f"{name}: {child.module_type}, {len(child.ops)} ops")

        Look up a specific submodule by fully-qualified name:

        >>> linear_mod = root.get_submodule("0")
        >>> print(linear_mod.module_type)  # torch.nn.modules.linear.Linear
        >>> print(linear_mod.ops)          # ops directly owned by this module

        Get all ops under a subtree (the module and all its descendants):

        >>> subtree_ops = linear_mod.all_ops()

        Filter ops by type, name pattern, or module with the same filtering logic which Quantizer
        uses:

        >>> inspector.get_matched_ops_for_op_type("linear")
        >>> inspector.get_matched_ops_for_op_name(".*linear.*")
        >>> inspector.get_matched_ops_for_module_type(nn.Linear)
    """

    _summary: ModelSummary

    def __init__(
        self,
        model: torch.fx.GraphModule | torch.nn.Module,
        example_inputs: tuple[Any, ...] | None,
        execution_mode: ExecutionMode,
        compressor: type[_BaseModelCompressor] | None = None,
        dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None = None,
        export_with_no_grad: bool = True,
    ) -> None:
        self._validate_args(
            model, example_inputs, execution_mode, compressor, dynamic_shapes, export_with_no_grad
        )

        if execution_mode == ExecutionMode.GRAPH:
            gm = model
            if not isinstance(gm, torch.fx.GraphModule):
                gm = _export_model(model, example_inputs, dynamic_shapes, export_with_no_grad)
            self._summary = _parse_ops_for_graph(gm, compressor)
        else:
            self._summary = _parse_ops_for_eager(model, example_inputs, compressor)

    @staticmethod
    def _validate_args(
        model: torch.fx.GraphModule | torch.nn.Module,
        example_inputs: tuple[Any, ...] | None,
        execution_mode: ExecutionMode,
        compressor: type[_BaseModelCompressor] | None,
        dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None,
        export_with_no_grad: bool,
    ) -> None:
        """Validate constructor arguments."""
        if not isinstance(model, (torch.fx.GraphModule, torch.nn.Module)):
            msg = f"Expected a torch.fx.GraphModule or torch.nn.Module, got {type(model).__name__}"
            raise TypeError(msg)

        if execution_mode not in (ExecutionMode.GRAPH, ExecutionMode.EAGER):
            msg = f"Unknown execution_mode {execution_mode}. Expected 'graph' or 'eager'."
            raise ValueError(msg)

        if example_inputs is None and not (
            isinstance(model, torch.fx.GraphModule) and execution_mode == ExecutionMode.GRAPH
        ):
            msg = (
                "example_inputs can only be None when model is a GraphModule and "
                "execution_mode is ExecutionMode.GRAPH"
            )
            raise ValueError(msg)

        if compressor is not None and not issubclass(compressor, (Quantizer, KMeansPalettizer)):
            msg = (
                f"Unsupported compressor class {compressor.__name__}. "
                "Supported compressors: Quantizer, KMeansPalettizer."
            )
            raise ValueError(msg)

        if execution_mode == ExecutionMode.GRAPH:
            if compressor is not None and not issubclass(compressor, Quantizer):
                msg = (
                    f"Compressor {compressor.__name__} is not supported in graph mode. "
                    "Only Quantizer is supported for graph mode inspection."
                )
                raise ValueError(msg)
        else:
            if isinstance(model, torch.fx.GraphModule):
                msg = (
                    "Expected a torch.nn.Module for Eager execution_mode, got torch.fx.GraphModule"
                )
                raise TypeError(msg)
            if dynamic_shapes is not None:
                warnings.warn(
                    "dynamic_shapes is only supported in graph mode and will be ignored.",
                    UserWarning,
                    stacklevel=3,
                )
            if not export_with_no_grad:
                warnings.warn(
                    "export_with_no_grad is only supported in graph mode and will be ignored.",
                    UserWarning,
                    stacklevel=3,
                )

    @property
    def summary(self) -> ModelSummary:
        """The underlying operation summary."""
        return self._summary

    def get_matched_ops_for_op_type(self, op_type: str) -> tuple[OpInfo, ...]:
        """Return operations matching the given op type.

        Args:
            op_type (str): The operation type to filter by (e.g.,
                ``"conv2d"``, ``"linear"``).

        Returns:
            tuple[OpInfo, ...]: Matching operations.
        """
        return tuple(op for op in self._summary.model.all_ops() if op.op_type == op_type)

    def get_matched_ops_for_op_name(self, pattern: str) -> tuple[OpInfo, ...]:
        """Return operations whose name matches the given regex pattern.

        Uses ``re.fullmatch``, consistent with how ``op_name_config``
        patterns are matched in Graph mode.

        Args:
            pattern (str): A regex pattern to match against op names.

        Returns:
            tuple[OpInfo, ...]: Matching operations.

        Raises:
            ValueError: If *pattern* is not a valid regex.
        """
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}") from e
        return tuple(op for op in self._summary.model.all_ops() if compiled.fullmatch(op.op_name))

    def get_matched_ops_for_module_name(self, module_name: str) -> tuple[OpInfo, ...]:
        """Return operations whose module stack contains the given module name.

        Uses ``re.fullmatch`` against each module FQN in the op's module
        stack, consistent with how ``module_name_configs`` patterns are
        matched in Graph mode.

        Args:
            module_name (str): A regex pattern to match against module FQNs
                (e.g., ``"encoder.layer1"``, ``"encoder\\..*"``).

        Returns:
            tuple[OpInfo, ...]: Matching operations.

        Raises:
            ValueError: If *module_name* is not a valid regex.
        """
        try:
            compiled = re.compile(module_name)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{module_name}': {e}") from e
        return tuple(
            op
            for op in self._summary.model.all_ops()
            if any(compiled.fullmatch(m.module_name) for m in op.module_stack)
        )

    def get_matched_ops_for_module_type(self, module_type: type | str) -> tuple[OpInfo, ...]:
        """Return operations whose module stack contains the given type.

        Matches using exact string equality against the fully-qualified
        type name, consistent with how ``module_type_configs`` keys are
        resolved in the quantizer.  Accepts either a class (converted via
        :func:`~coreai_opt._utils.python_utils.fqn`) or a fully-qualified
        type string (e.g., ``"torch.nn.modules.conv.Conv2d"``).

        Args:
            module_type (type | str): Module type to filter by.

        Returns:
            tuple[OpInfo, ...]: Matching operations.
        """
        type_fqn = _fqn(module_type) if isinstance(module_type, type) else module_type
        return tuple(
            op
            for op in self._summary.model.all_ops()
            if any(m.module_type == type_fqn for m in op.module_stack)
        )

    def format_summary(self, colorize: bool | None = None) -> str:
        """Format discovered operations as a module-hierarchy tree string.

        Args:
            colorize (bool | None): Whether to include ANSI color codes in the
                output. ``None`` (default) auto-detects based on terminal
                capabilities. Pass ``False`` when writing to files or logs.

        Returns:
            str: The formatted tree.
        """
        return _format_model_summary(self._summary, colorize=colorize)
