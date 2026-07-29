# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import itertools
import warnings
from collections import defaultdict
from collections.abc import Callable, Generator
from contextlib import contextmanager
from os import PathLike

import torch
import torch.nn as nn
from torch.nn.utils.parametrize import ParametrizationList
from torchao.quantization.pt2e import (
    disable_fake_quant,
    disable_observer,
    enable_fake_quant,
    enable_observer,
)

from coreai_opt._utils.config_utils import ALL_TENSORS
from coreai_opt._utils.eager_utils import EagerCompressionComponentBuilderMixin
from coreai_opt._utils.insertion.torch_function import (
    TorchFunctionEagerHandler,
)
from coreai_opt._utils.spec_utils import PartialConstructor
from coreai_opt._utils.torch_utils import move_model_to_eval, move_model_to_train
from coreai_opt.common import ExportBackend
from coreai_opt.config.compression_config import ModuleCompressionConfig
from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.config.spec.base import CompressionSpec
from coreai_opt.quantization._axis_defaults import (
    apply_weight_axis_defaults_eager as _apply_weight_axis_defaults,
    validate_activation_axes,
)
from coreai_opt.quantization._fake_quant_utils import (
    disable_activation_fake_quant,
    enable_weight_fake_quant,
)
from coreai_opt.quantization.base_quantizer import _BaseQuantizer
from coreai_opt.quantization.config import (
    ModuleQuantizerConfig,
    QuantizerConfig,
)
from coreai_opt.quantization.config.quantization_config import _ACTIVATION_SPEC_DICT
from coreai_opt.quantization.spec.factory import QuantizationComponentFactory
from coreai_opt.quantization.spec.fake_quantize import (
    FakeQuantizeImplBase,
)

from ._prepare_for_export import prepare_for_mlir_export
from ._prepare_for_mil_export import prepare_for_mil_export
from ._utils import remove_act_fq_from_reference_tracker, remove_fake_quant_modules
from .supported_ops_registry import (
    EagerQuantizerSupportedOpsRegistry,
)


class EagerQuantizer(_BaseQuantizer, EagerCompressionComponentBuilderMixin):
    """Eager mode quantization-aware training (QAT) quantizer.

    Uses `__torch_function__` to trace model execution and insert fake quantizers
    during forward pass. Supports calibration and finalization for deployment.

    Example:
        >>> import torch.nn as nn
        >>> model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
        >>> quantizer = EagerQuantizer(model)
        >>>
        >>> # When activation quantization is in use, example_inputs should be
        >>> # representative of the data the model would typically see.
        >>> example_inputs = ...
        >>>
        >>> # Prepare for QAT
        >>> prepared_model = quantizer.prepare(example_inputs)
        >>>
        >>> # Calibration (optional)
        >>> with quantizer.calibration_mode():
        ...     for data in calibration_loader:
        ...         prepared_model(data)
        >>>
        >>> # Finalize for deployment
        >>> final_model = quantizer.finalize()
    """

    def __init__(
        self,
        model: nn.Module,
        config: QuantizerConfig,
    ):
        self._validate_config(config)
        super().__init__(model, config)

        module_components_dict, module_priority_dict = (
            self._get_module_compression_components_and_priority(model, config)
        )

        # TODO: Add a iterative eager handler and provide a flag to toggle
        self._handler = TorchFunctionEagerHandler(
            compression_config=config,
            module_components_dict=module_components_dict,
            module_priority_dict=module_priority_dict,
            supported_ops_registry=EagerQuantizerSupportedOpsRegistry,
            optimization_type_name="quantize",
        )

    @classmethod
    def get_op_type_resolver(cls) -> Callable[[Callable], str | None]:
        """Return a function that maps a torch function to its quantizable op type."""
        return EagerQuantizerSupportedOpsRegistry.get_func_type

    @staticmethod
    def _fill_op_input_output_specs_to_check_for_module_config(
        op_input_specs_to_check: list[_ACTIVATION_SPEC_DICT],
        op_output_specs_to_check: list[_ACTIVATION_SPEC_DICT],
        module_config: ModuleQuantizerConfig,
    ):
        """
        Helper function to populate lists of op_input_specs and op_output_specs to
        validate.
        """
        # Combine all configs: module config + op_name_config + op_type_config
        all_configs = itertools.chain(
            [module_config],
            module_config.op_name_config.values(),
            module_config.op_type_config.values(),
        )

        # Extract specs from all configs
        for config in all_configs:
            op_input_specs_to_check.append(config.op_input_spec)
            op_output_specs_to_check.append(config.op_output_spec)

    @staticmethod
    def _validate_config(config: QuantizerConfig) -> None:
        """
        Validate the current state of support for config. This function will be
        continually updated as support for different aspects of QuantizerConfig is
        implemented.
        """
        op_input_specs_to_check = []
        op_output_specs_to_check = []
        for module_config in itertools.chain(
            [config.global_config],
            config.module_type_configs.values(),
            config.module_name_configs.values(),
        ):
            EagerQuantizer._fill_op_input_output_specs_to_check_for_module_config(
                op_input_specs_to_check, op_output_specs_to_check, module_config
            )

        # Check for currently unsupported string based input/output tensor
        # identification.
        for spec in op_input_specs_to_check + op_output_specs_to_check:
            for key in spec:
                if isinstance(key, str) and key != ALL_TENSORS:
                    error_msg = (
                        "Only integer indices or '*' are supported for op_input_spec "
                        f"and op_output_spec currently. Got {spec}"
                    )
                    raise NotImplementedError(error_msg)

        # preserved attributes
        if config.preserved_attributes is not None:
            warnings.warn(
                "'preserved_attributes' is only supported in graph mode quantization and "
                "will be ignored in eager mode.",
                UserWarning,
                stacklevel=2,
            )

    def prepare(self, example_inputs: tuple[torch.Tensor]) -> nn.Module:
        """Prepare model for quantization by inserting fake quantizers.

        Args:
            example_inputs: Sample inputs used to trace the model and configure
                quantizers. When activation quantization is in use, these should
                be representative of the data the model would typically see.

        Returns:
            Model with fake quantizers inserted
        """
        if self._is_model_prepared(self._model):
            warnings.warn(
                "Model has already been prepared for quantization. "
                "This call to prepare will be a no-op",
                stacklevel=2,
            )
            return self._model

        # Move model to eval mode and disable gradients during the initial forward pass to insert
        # quantization layers. The model will be put back into its original mode when exiting
        # move_model_to_eval context manager.
        with (
            move_model_to_eval(self._model),
            torch.no_grad(),
        ):
            self._model = self._handler.prepare(self._model, example_inputs=example_inputs)

            self._postprocess_prepared_model(self._model)

            self._model.apply(enable_observer)
            self._model.apply(disable_fake_quant)

            self._model(*example_inputs)

            # Remove FakeQuantize modules that were disabled during the forward
            # pass due to incompatible tensor shapes (e.g., non-divisible block sizes).
            self._remove_disabled_fake_quant_modules()

        self._model.apply(enable_fake_quant)
        self._model.apply(disable_observer)

        self._mark_model_as_prepared(self._model)
        return self._model

    @staticmethod
    def _postprocess_prepared_model(model: nn.Module) -> None:
        """Apply post-processing fixes after eager prepare.

        Args:
            model (nn.Module): The model after eager prepare().
        """
        _apply_weight_axis_defaults(model)
        validate_activation_axes(model)

    def finalize(
        self,
        model: nn.Module | None = None,
        backend: ExportBackend = ExportBackend.CoreAI,
        *,
        mmap_dir: str | PathLike[str] | None = None,
    ) -> nn.Module:
        """Convert quantized model to backend-specific representations.

        Converts fake quantization modules into backend-specific quantization ops.
        Only call ``finalize`` when exporting to a target backend. For torch-based evaluation, use
        the model returned by ``prepare()`` directly rather than calling ``finalize``.

        Args:
            model: Model to finalize (uses internal model if None).
            backend: Target export backend for the quantized model. Supports
                CoreAI (default) and CoreML.
            mmap_dir (str | None): If provided, serialize finalized quantized
                weights to safetensors files under this directory and re-load
                them via mmap. Only supported with the CoreAI backend. The
                files in ``mmap_dir`` must remain in place for the lifetime
                of the returned model; removing them invalidates the
                mmap-backed weights.

        Returns:
            The finalized quantized model ready for deployment on the target backend.

        """
        if model is None:
            model = self._model

        if not self._is_model_prepared(model):
            raise RuntimeError(
                "Model has not been prepared. Run the prepare step first before "
                "finalizing the model."
            )

        if mmap_dir is not None and backend != ExportBackend.CoreAI:
            raise ValueError(
                "mmap_dir is only supported with backend=ExportBackend.CoreAI, "
                f"got backend={backend}."
            )

        # Backend specific logic
        match backend:
            case ExportBackend._TORCH:
                finalized_model = model
            case ExportBackend.CoreAI:
                finalized_model = prepare_for_mlir_export(model, mmap_dir=mmap_dir)
            case ExportBackend.CoreML:
                finalized_model = prepare_for_mil_export(model)
            case _:
                msg = f"Unsupported backend {backend}"
                raise ValueError(msg)

        return finalized_model

    @contextmanager
    def calibration_mode(self, model: nn.Module | None = None) -> Generator:
        """Context manager for calibration phase.

        Enables observers and disables activation fake quantization for
        calibration data collection. Weight fake quantization stays enabled so
        that activation observers see the effect of quantized weights when
        computing activation ranges.

        Args:
            model: Model to calibrate (uses internal model if None)

        Yields:
            Context for running calibration data through the model
        """
        if model is not None:
            self._model = model

        if not self._is_model_prepared(self._model):
            raise RuntimeError(
                "Model must be prepared before entering calibration mode. Call prepare() first."
            )
        with move_model_to_eval(self._model):
            self._model.apply(enable_observer)
            self._model.apply(enable_weight_fake_quant)
            self._model.apply(disable_activation_fake_quant)
            try:
                yield
            finally:
                # Restore the state: disable observers, enable fake quantization
                self._model.apply(disable_observer)
                self._model.apply(enable_fake_quant)

    @contextmanager
    def training_mode(self, model: nn.Module | None = None) -> Generator:
        """Context manager for quantization-aware training (QAT) phase.

        Enables observers and fake quantization for QAT.

        Args:
            model: Model to train (uses internal model if None)

        Yields:
            Context for running QAT on the model
        """
        if model is not None:
            self._model = model

        if not self._is_model_prepared(self._model):
            raise RuntimeError(
                "Model must be prepared before entering training mode. Call prepare() first."
            )

        with move_model_to_train(self._model):
            self._model.apply(enable_observer)
            self._model.apply(enable_fake_quant)
            try:
                yield
            finally:
                self._model.apply(disable_observer)
                self._model.apply(enable_fake_quant)

    def _get_fake_quantize_modules(
        self,
    ) -> dict[str, list[FakeQuantizeImplBase]]:
        """Map original module names to their fake quantize modules.

        Walks the module tree of the prepared model and finds the fake
        quantize modules present within each module by collecting
        ``FakeQuantizeImplBase`` child modules. Weight fake quantize
        modules are handled specially since they reside inside a
        ``ParametrizationList`` within the owning module.

        Returns:
            Dict mapping original module name to the list of FQ module
            instances that belong to that module.
        """
        mapping: dict[str, list[FakeQuantizeImplBase]] = defaultdict(list)
        for name, module in self._model.named_modules():
            fq_list = [c for c in module.children() if isinstance(c, FakeQuantizeImplBase)]
            if fq_list:
                # Weight FQs live inside ParametrizationList; map back
                # to the owning module (strip ".parametrizations.<param>")
                if isinstance(module, ParametrizationList):
                    key = name.rsplit(".", 2)[0]
                else:
                    key = name
                mapping[key] += fq_list
        return dict(mapping)

    def _remove_disabled_fake_quant_modules(self) -> None:
        """Remove FakeQuantize modules that were disabled during the forward pass."""
        disabled_fq = {
            name: m
            for name, m in self._model.named_modules(remove_duplicate=True)
            if isinstance(m, FakeQuantizeImplBase) and m.is_disabled()
        }
        if not disabled_fq:
            return

        # Scrub stale activation optimizer names from the reference tracker
        # before removing the modules from the model.
        if self._handler.act_handler is not None:
            remove_act_fq_from_reference_tracker(
                disabled_fq, self._handler.act_handler.reference_tracker
            )

        remove_fake_quant_modules(self._model, set(disabled_fq.values()))

    @staticmethod
    def _spec_to_partial(
        spec: CompressionSpec | None,
        target: CompressionTargetTensor,
        module_config: ModuleCompressionConfig,
    ) -> PartialConstructor | None:
        return QuantizationComponentFactory.construct_partial(spec=spec, target=target)
