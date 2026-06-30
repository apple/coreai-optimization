# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import warnings
from contextlib import contextmanager
from os import PathLike
from typing import Any

import torch.nn as nn
from torch import fx
from torchao.quantization.pt2e import (
    disable_fake_quant as _torchao_disable_fake_quant,
    disable_observer as _torchao_disable_observer,
    enable_fake_quant as _torchao_enable_fake_quant,
    enable_observer as _torchao_enable_observer,
)

from coreai_opt._utils.config_utils import ConfigLevel as _ConfigLevel
from coreai_opt._utils.export_utils import (
    validate_mmap_backend_and_device as _validate_mmap_backend_and_device,
)
from coreai_opt._utils.torch_utils import get_module_name
from coreai_opt.common import ExportBackend
from coreai_opt.quantization._eager import EagerQuantizer as _EagerQuantizer
from coreai_opt.quantization._graph import GraphQuantizer as _GraphQuantizer
from coreai_opt.quantization.base_quantizer import _BaseQuantizer
from coreai_opt.quantization.config.quantization_config import (
    ExecutionMode,
    QATSchedule,
    QuantizerConfig,
)
from coreai_opt.quantization.spec.fake_quantize import FakeQuantizeImplBase
from coreai_opt.quantization.spec.qparams_calculator import StatelessQParamsCalculatorBase


class Quantizer(_BaseQuantizer):
    """
    Unified quantizer API that provides a single entry point for various quantization
    workflows, including:

    - **Data Types**: Integer (e.g. int8, int4) and floating-point
      (e.g. float8_e4m3fn, float8_e5m2) quantization
    - **Quantization Workflows**: Post-training quantization (PTQ) and
      quantization-aware training (QAT)
    - **Execution Modes**: Graph mode (built on torchao's PT2E) or eager mode

    The quantizer automatically selects the appropriate underlying implementation based
    on the `execution_mode` specified in the configuration. Defaults to graph mode. Some
    of the key differences between the execution modes are summarized below:

    +-----------------------+---------------------------------+----------------------------+
    | Feature               | Graph Mode (Default)            | Eager Mode                 |
    +=======================+=================================+============================+
    | Input/Output Types    | nn.Module                       | nn.Module -> nn.Module     |
    |                       | -> fx.GraphModule.              |                            |
    +-----------------------+---------------------------------+----------------------------+
    | Module Fusion         | Automatic pattern-based fusion  | Manual fusion required     |
    |                       | (e.g., conv+bn+relu)            |                            |
    +-----------------------+---------------------------------+----------------------------+
    | Control Flow          | Static graph only;              | Supports dynamic           |
    |                       | Requires torch.export           | control flow               |
    |                       | compatible model                | (if/else, loops)           |
    +-----------------------+---------------------------------+----------------------------+
    | Shared Observer Ops   | Handled correctly; ops like     | Not supported; Ops like    |
    |                       | MaxPool that share the same     | MaxPool have independent   |
    |                       | observer across inputs and      | observers for input vs     |
    |                       | outputs are detected and        | output, which can cause    |
    |                       | deduplicated on the graph.      | incorrect quantization.    |
    +-----------------------+---------------------------------+----------------------------+
    | FQ Node Deduplication | Back-to-back fake-quantize      | No deduplication; if both  |
    |                       | nodes on the same tensor are    | the output of one op and   |
    |                       | collapsed into a single node,   | the input of the next are  |
    |                       | avoiding redundant quantization | quantized, two consecutive |
    |                       | on intermediate edges.          | FQ nodes are inserted on   |
    |                       |                                 | that intermediate edge.    |
    +-----------------------+---------------------------------+----------------------------+

    As a result of above mentioned differences, the total number of fake-quantize nodes
    inserted by graph and eager mode can differ for the same ``QuantizerConfig``. This
    means the two modes are **not guaranteed to produce equivalent quantized models**,
    and final model performance (accuracy and latency) may differ between modes even
    when using identical configurations.

    Args:
        model: The PyTorch model to quantize.
        config: Quantization configuration. If None, a default configuration
            with int8 weight and activation quantization is created.

    Example:
        >>> from coreai_opt.quantization import Quantizer, QuantizerConfig, ExecutionMode
        >>>
        >>> # PTQ with calibration (default int8, graph mode)
        >>> config = QuantizerConfig()
        >>> quantizer = Quantizer(model, config)
        >>> prepared_model = quantizer.prepare((example_input,))
        >>> with quantizer.calibration_mode():
        ...     for data in calibration_loader:
        ...         prepared_model(data)
        >>> quantized_model = quantizer.finalize()
        >>>
        >>> # QAT workflow (default schedule — observers and fake_quant enabled throughout)
        >>> prepared_model = quantizer.prepare((example_input,))
        >>> with quantizer.training_mode():
        ...     for epoch in range(num_epochs):
        ...         for data, target in train_loader:
        ...             optimizer.zero_grad()
        ...             output = prepared_model(data)
        ...             loss = criterion(output, target)
        ...             loss.backward()
        ...             optimizer.step()
        >>> quantized_model = quantizer.finalize()
        >>>
        >>> # QAT workflow with schedule
        >>> from coreai_opt.quantization import ModuleQuantizerConfig
        >>> from coreai_opt.quantization.config import QATSchedule
        >>> # Enable observers from the start, enable fake quant at the 100th step,
        >>> # and disable observers at the 500th step.
        >>> schedule = QATSchedule(
        ...     enable_observer=0, enable_fake_quant=100, disable_observer=500
        ... )
        >>> config = QuantizerConfig(
        ...     global_config=ModuleQuantizerConfig(qat_schedule=schedule)
        ... )
        >>> quantizer = Quantizer(model, config)
        >>> prepared_model = quantizer.prepare((example_input,))
        >>> with quantizer.training_mode():
        ...     for data, target in train_loader:
        ...         optimizer.zero_grad()
        ...         loss = criterion(prepared_model(data), target)
        ...         loss.backward()
        ...         optimizer.step()
        ...         quantizer.step()
        >>> quantized_model = quantizer.finalize()
    """

    def __init__(
        self,
        model: nn.Module,
        config: QuantizerConfig | None = None,
    ):
        if config is None:
            config = QuantizerConfig()

        execution_mode = config.execution_mode
        self._execution_mode = execution_mode

        # Create the underlying quantizer based on execution mode
        if execution_mode == ExecutionMode.GRAPH:
            self._quantizer = _GraphQuantizer(model, config)
        elif execution_mode == ExecutionMode.EAGER:
            self._quantizer = _EagerQuantizer(model, config)
        else:
            raise ValueError(f"Unsupported execution mode: {execution_mode}")

        super().__init__(model, config)

        # QAT schedule state
        self._step_count: int = 0
        self._in_training_mode: bool = False
        # Mapping of FakeQuantize module to its corresponding QATSchedule
        self._fq_to_schedule: dict[FakeQuantizeImplBase, QATSchedule] = {}
        # Cached module-name → FQ-modules map, populated after prepare()
        self._module_to_fqs: dict[str, list[FakeQuantizeImplBase]] = {}

    @property
    def _model(self):
        """Delegate to the underlying quantizer's model."""
        return self._quantizer._model

    @_model.setter
    def _model(self, value):
        """Delegate model setting to the underlying quantizer."""
        self._quantizer._model = value

    def _get_fake_quantize_modules(self) -> dict[str, list]:
        """Delegate to the underlying execution-mode quantizer."""
        return self._quantizer._get_fake_quantize_modules()

    def _resolve_schedule(self, module_name: str) -> QATSchedule | None:
        """Look up the QAT schedule for a module via the config hierarchy."""
        for level in _ConfigLevel.priority_order():
            config = self._module_config_dict[level].get(module_name)
            if config is not None:
                return config.qat_schedule
        return None

    def _build_fq_to_schedule(self) -> None:
        """Build ``_fq_to_schedule`` from cached config dict + FQ modules.

        Must be called after ``prepare()`` so that FQ modules exist.
        Requires ``_module_config_dict`` to have been populated before
        ``prepare()`` (since prepare may modify the module types in Eager
        mode).
        """
        if self._fq_to_schedule:
            return

        for module_name, fq_list in self._module_to_fqs.items():
            schedule = self._resolve_schedule(module_name)
            if schedule is not None:
                for fq_mod in fq_list:
                    if fq_mod in self._fq_to_schedule:
                        warnings.warn(
                            f"FakeQuantize module under '{module_name}' is shared "
                            f"with another module that already has a qat_schedule "
                            f"assigned. The existing schedule will be kept.",
                            UserWarning,
                            stacklevel=2,
                        )
                    else:
                        self._fq_to_schedule[fq_mod] = schedule

    def _maybe_apply_qat_schedule(self) -> None:
        """Apply observer/fake-quant state for the current step count."""
        for fq_module, schedule in self._fq_to_schedule.items():
            state = schedule._compute_state(self._step_count)
            fq_module.enable_observer(state.obs_on)
            fq_module.enable_fake_quant(state.fq_on)

    def _validate_no_schedule_configured(self) -> None:
        """Raise RuntimeError if any FQ modules have a qat_schedule."""
        if self._fq_to_schedule:
            raise RuntimeError(
                "Enable/disable APIs for observers or fake quantization cannot be "
                "used with a qat_schedule configured. To use these APIs, make sure "
                "there are no global or module-level qat_schedule configured. For "
                "using the QAT schedule, refer to the step() API."
            )

    def step(self) -> None:
        """
        Advance the QAT schedule by one step and apply observer/fake_quant
        transitions after the step has been incremented.

        Must be called inside a training_mode() context. Increments _step_count
        (monotonically; never reset between training loops), then applies the
        absolute observer/fake_quant state corresponding to the new step count.

        Raises:
            RuntimeError: If called outside a training_mode() context.

        Warns:
            UserWarning: If no qat_schedule is configured on any module.
        """
        if not self._in_training_mode:
            raise RuntimeError("step() must be called inside a training_mode() context.")

        self._step_count += 1

        if not self._fq_to_schedule:
            warnings.warn(
                "step() was called but no qat_schedule is configured on any module. "
                "step() has no effect. Configure a QATSchedule on at least one "
                "ModuleQuantizerConfig to use QAT scheduling.",
                UserWarning,
                stacklevel=2,
            )
            return

        self._maybe_apply_qat_schedule()

    def _maybe_apply_fn_to_fqs(self, fn: callable, module: nn.Module | None = None) -> None:
        """Apply fn to FQ modules if no QAT schedule is configured.

        Validates that no schedule is set (raises RuntimeError otherwise).
        If module is None, applies to the entire model. If module is given,
        finds its FQs via the cached ``_module_to_fqs`` by resolving the module's
        name and walking its children.

        Args:
            fn: A torchao function (e.g. ``enable_observer``) to apply.
            module: If None, applies to the entire model.
                    Otherwise, applies only to FQs associated with the given
                    module and its children.

        Raises:
            RuntimeError: If any ModuleQuantizerConfig has qat_schedule configured.
            ValueError: If the given module is not found in the model.
        """
        self._validate_no_schedule_configured()
        if module is None:
            self._quantizer._model.apply(fn)
            return

        prefix = get_module_name(self._quantizer._model, module)
        if prefix is None:
            raise ValueError(f"Module {module} is not a submodule of the prepared model.")

        for child_name, _ in module.named_modules():
            full_name = f"{prefix}.{child_name}" if child_name else prefix
            for fq in self._module_to_fqs.get(full_name, []):
                fq.apply(fn)

    def enable_observer(self, module: nn.Module | None = None) -> None:
        """Enable observers on the model or a specific module."""
        self._maybe_apply_fn_to_fqs(_torchao_enable_observer, module)

    def disable_observer(self, module: nn.Module | None = None) -> None:
        """Disable observers on the model or a specific module."""
        self._maybe_apply_fn_to_fqs(_torchao_disable_observer, module)

    def enable_fake_quant(self, module: nn.Module | None = None) -> None:
        """Enable fake quantization on the model or a specific module."""
        self._maybe_apply_fn_to_fqs(_torchao_enable_fake_quant, module)

    def disable_fake_quant(self, module: nn.Module | None = None) -> None:
        """Disable fake quantization on the model or a specific module."""
        self._maybe_apply_fn_to_fqs(_torchao_disable_fake_quant, module)

    def prepare(
        self,
        example_inputs: tuple[Any, ...],
        dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None = None,
        export_with_no_grad: bool = True,
    ) -> nn.Module | fx.GraphModule:
        """
        Prepare the model for quantization by inserting fake quantization modules.

        **Graph Mode:**
        Exports the model using torch.export, applies quantization annotations, and
        sets up fake quantization modules. Returns an fx.GraphModule.

        **Eager Mode:**
        Uses `__torch_function__` to trace model execution and insert fake quantizers
        during the forward pass. Returns an nn.Module.

        **Important Notes:**

        - For weight-only PTQ: The prepared model can be directly finalized
          (prepare() → finalize() workflow).
        - For activation quantization: The prepared model should be calibrated using
          calibration_mode() before finalization to collect statistics and achieve
          good accuracy.

        Args:
            example_inputs: Tuple of example inputs for model tracing. When
                activation quantization is in use, these should be
                representative of the data the model would typically see.
            dynamic_shapes: Dynamic shapes specification (graph mode only).
                Ignored in EAGER mode.
            export_with_no_grad: Whether to export with no_grad (graph mode
                only). Ignored in EAGER mode.

        Returns:
            The prepared model with fake quantization modules inserted, ready for
            calibration or training. This is a data-free PTQ compressed model.

        Note:
            In graph mode, the returned ``fx.GraphModule`` supports calling
            ``.train()`` and ``.eval()``, but with limited effect: only dropout
            and batchnorm ops are affected via FX graph rewriting. User code
            branching on the ``training`` flag and other ops with
            mode-dependent behavior are not affected.
        """
        # Cache config dict before prepare() so that module_type_configs can
        # match original types. After prepare, modules can be modified such
        # that the types no longer match what is given in the config.
        self._module_config_dict = self._config.build_module_config_dict(self._quantizer._model)

        if self._execution_mode == ExecutionMode.EAGER:
            if dynamic_shapes is not None:
                warnings.warn(
                    "dynamic_shapes is only supported in graph mode and will be ignored.",
                    UserWarning,
                    stacklevel=2,
                )
            if not export_with_no_grad:
                warnings.warn(
                    "export_with_no_grad is only supported in graph mode and will be ignored.",
                    UserWarning,
                    stacklevel=2,
                )
            prepared_model = self._quantizer.prepare(example_inputs)
        else:
            prepared_model = self._quantizer.prepare(
                example_inputs,
                dynamic_shapes=dynamic_shapes,
                export_with_no_grad=export_with_no_grad,
            )

        self._module_to_fqs = self._get_fake_quantize_modules()
        self._build_fq_to_schedule()

        return prepared_model

    def _validate_mmap_dir_constraints(
        self,
        model: nn.Module | fx.GraphModule | None,
        backend: ExportBackend,
        mmap_dir: str | PathLike[str] | None,
    ) -> None:
        """Validate that ``mmap_dir`` is compatible with the current execution mode,
        target backend, and model device. No-op when ``mmap_dir is None``.
        """
        if mmap_dir is None:
            return
        if self._execution_mode != ExecutionMode.EAGER:
            raise ValueError(
                "mmap_dir is only supported in eager execution mode, "
                f"got execution_mode={self._execution_mode}."
            )
        model_to_check = model if model is not None else self._model
        _validate_mmap_backend_and_device(model_to_check, backend, mmap_dir)

    def _validate_no_persistent_observer_calculators(
        self,
        model: nn.Module | fx.GraphModule | None,
        backend: ExportBackend,
    ) -> None:
        """Reject CoreAI/CoreML export when any qparams calculator is a
        ``StatelessQParamsCalculatorBase`` (e.g. dynamic quantization).
        """
        if backend == ExportBackend._TORCH:
            return
        model_to_check = model if model is not None else self._model
        stateless_fq_names = [
            name
            for name, mod in model_to_check.named_modules()
            if isinstance(mod, FakeQuantizeImplBase)
            and isinstance(mod.qparams_calculator, StatelessQParamsCalculatorBase)
        ]
        if stateless_fq_names:
            raise NotImplementedError(
                f"backend={backend} does not yet support qparams calculators that "
                f"recompute every forward (e.g. dynamic quantization). "
                f"Affected FakeQuantize modules: {stateless_fq_names}. Use "
                f"backend=ExportBackend._TORCH for torch-only inference."
            )

    def finalize(
        self,
        model: nn.Module | fx.GraphModule | None = None,
        backend: ExportBackend = ExportBackend.CoreAI,
        *,
        mmap_dir: str | PathLike[str] | None = None,
    ) -> nn.Module | fx.GraphModule:
        """Convert quantized model to backend-specific representations.

        Converts fake quantization modules into backend-specific quantization ops.
        Only call ``finalize`` when exporting to a target backend. For torch-based evaluation, use
        the model returned by ``prepare()`` directly rather than calling ``finalize``.

        Backend-specific processing:

        - CoreAI: Prepares for CoreAI export by replacing fake quantization modules
          with Core AI specific PyTorch custom ops.
        - CoreML: Prepares for CoreML export by registering compression metadata
          as buffers and removes fake quantization modules.

        Args:
            model: Optional model to finalize. If None, uses the internal
                prepared model.
            backend: Target export backend for the quantized model. Supports
                CoreAI (default), CoreML, and _TORCH backends.
            mmap_dir (str | None): If provided, serialize finalized quantized
                weights to safetensors files under this directory and re-load
                them via mmap. Only supported in eager execution mode with the
                CoreAI backend; raises ``ValueError`` otherwise. The files in
                ``mmap_dir`` must remain in place for the lifetime of the
                returned model; removing them invalidates the mmap-backed
                weights.

        Returns:
            The finalized quantized model ready for deployment on the target backend.

        Note:
            In graph mode, the returned ``fx.GraphModule`` supports calling
            ``.train()`` and ``.eval()``, but with limited effect: only dropout
            and batchnorm ops are affected via FX graph rewriting. User code
            branching on the ``training`` flag and other ops with
            mode-dependent behavior are not affected.

        Note:
            When ``backend=ExportBackend.CoreAI`` in execution_mode=ExecutionMode.EAGER,
            finalize frees the original dense weights.
        """
        self._validate_mmap_dir_constraints(model, backend, mmap_dir)
        self._validate_no_persistent_observer_calculators(model, backend)
        return self._quantizer.finalize(model, backend, mmap_dir=mmap_dir)

    @contextmanager
    def calibration_mode(self, model: nn.Module | fx.GraphModule | None = None):
        """
        Context manager for calibration-based post-training quantization.

        When entering this context, observers are enabled to collect statistics
        from calibration data, and fake quantization is disabled to get accurate
        statistics. When exiting, observers are disabled and fake quantization
        is re-enabled for evaluation.

        **When to use:**

        - Required for activation quantization to achieve good accuracy.
          The model post prepare() may have poor accuracy for activation
          quantization until calibrated with representative data
        - Not needed for weight-only PTQ (prepare() → finalize() is sufficient)

        Args:
            model: Optional model to setup for calibration. If None, uses
                the internal prepared model.

        Example:
            >>> quantizer = Quantizer(model, config)
            >>> prepared_model = quantizer.prepare(example_inputs)
            >>> # For activation quantization, calibrate to improve accuracy:
            >>> with quantizer.calibration_mode():
            ...     for batch in calibration_dataloader:
            ...         prepared_model(batch)
            >>> finalized_model = quantizer.finalize()

        Raises:
            RuntimeError: If the model has not been prepared.
        """
        with self._quantizer.calibration_mode(model):
            yield

    @contextmanager
    def training_mode(self, model: nn.Module | fx.GraphModule | None = None):
        """
        Context manager for quantization-aware training (QAT) workflow.

        When entering this context, the model is configured for training with both
        observers and fake quantization enabled (default behavior), or with the state
        determined by the current step count if a QATSchedule is configured.
        This allows the model to:

        1. Set the model in training mode (model.training is set to True)
        2. Enable the observers and activate the fake quantization
        3. Using the observers, simulate quantization during forward/backward passes

        When exiting the context, observers are disabled and fake quantization is
        enabled (regardless of schedule).

        The step count is not reset when re-entering training_mode() — it resumes
        from the last value, so schedule state is restored from the accumulated count.

        Nested calls to training_mode() are not allowed and will raise a RuntimeError.

        **When to use:**

        - For quantization-aware training (QAT) to fine-tune a prepared model
        - The prepared model from prepare() may have poor accuracy for
          weight-only quantization. Fine-tuning the model with the quantization
          enabled will help the weights adapt to the effects of quantization.
        - Upon calibrating an activation-quantized model, there wasn't enough
          improvement in model accuracy. Fine-tuning the weights to adapt to
          the effect of activation (and weight) quantization can help recover
          the lost accuracy.

        Args:
            model: Optional model to setup for training. If None, uses
                the internal prepared model.

        Example:
            >>> quantizer = Quantizer(model, config)
            >>> prepared_model = quantizer.prepare(example_inputs)
            >>> # Fine-tune with quantization-aware training:
            >>> with quantizer.training_mode():
            ...     # Model is put in training mode
            ...     for epoch in range(num_epochs):
            ...         for batch in train_dataloader:
            ...             # Perform training step
            ...             optimizer.zero_grad()
            ...             loss = loss_fn(prepared_model(batch), targets)
            ...             loss.backward()
            ...             optimizer.step()
            ...             quantizer.step()
            ...
            >>> finalized_model = quantizer.finalize()

        Raises:
            RuntimeError: If the model has not been prepared.
            RuntimeError: If called while already inside a training_mode() context.
            TypeError: If the provided model is not a torch.fx.GraphModule (graph mode).
        """
        if self._in_training_mode:
            raise RuntimeError(
                "Cannot enter training_mode() while already inside a "
                "training_mode() context. Nested training_mode() calls are not "
                "supported."
            )

        self._in_training_mode = True
        try:
            with self._quantizer.training_mode(model):
                # Inner training_mode enables obs+fq by default.
                # Override with schedule state if configured.
                self._maybe_apply_qat_schedule()
                yield
        finally:
            self._in_training_mode = False
