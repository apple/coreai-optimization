# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import logging
import os
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from os import PathLike

import torch
import torch.multiprocessing as mp
import torch.nn.utils.parametrize as P
from tqdm import tqdm

from coreai_opt._utils.eager_utils import (
    EagerCompressionComponentBuilderMixin as _EagerCompressionComponentBuilderMixin,
)
from coreai_opt._utils.export_utils import (
    validate_mmap_backend_and_device as _validate_mmap_backend_and_device,
)
from coreai_opt._utils.insertion.torch_function import (
    TorchFunctionEagerHandler as _TorchFunctionEagerHandler,
)
from coreai_opt._utils.spec_utils import PartialConstructor as _PartialConstructor
from coreai_opt._utils.torch_utils import (
    move_model_to_eval as _move_model_to_eval,
    remove_compression_parametrizations as _remove_compression_parametrizations,
)
from coreai_opt.common import ExportBackend
from coreai_opt.config.compression_config import ModuleCompressionConfig
from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.config.spec.base import CompressionSpec
from coreai_opt.palettization.base_palettizer import _BasePalettizer
from coreai_opt.palettization.config.palettization_config import (
    KMeansPalettizerConfig,
)
from coreai_opt.palettization.spec.fake_palettize import (
    _disable_fake_palett,
    _disable_observer,
    _enable_fake_palett,
    _enable_observer,
)

from ._prepare_for_export import (
    prepare_for_mil_export as _prepare_for_mil_export,
    prepare_for_mlir_export as _prepare_for_mlir_export,
)
from .kmeans_fake_palettize import _KMeansFakePalettize
from .supported_ops_registry import _KMeansPalettizerSupportedOpsRegistry

logger = logging.getLogger(__name__)

# Threshold to clip very small sensitivity values to stabilize k-means
_SENSITIVITY_CLIP_THR: float = 1e-12


@dataclass
class _FakePalettInfo:
    """Metadata about _KMeansFakePalettize module and its associated module / param"""

    module: torch.nn.Module
    module_name: str
    attr_name: str
    idx: int
    fp_module: _KMeansFakePalettize
    weight: torch.Tensor

    @property
    def layer_name(self) -> str:
        return f"{self.module_name}.{self.attr_name}" if self.module_name else self.attr_name


def _calculate_centroids_for_module(
    args: tuple[_KMeansFakePalettize, torch.Tensor, str],
) -> _KMeansFakePalettize:
    """Compute centroids for a single _KMeansFakePalettize module.

    Worker entry point for cross-layer parallel centroid calculation. Defined
    at module level so it is picklable by ``torch.multiprocessing`` workers
    using the ``spawn`` start method.

    Invokes ``fp_module.forward(weight)`` to mirror the sequential path.

    Args:
        args (tuple[_KMeansFakePalettize, torch.Tensor, str]): Tuple of
            ``(fp_module, weight, layer_name)``.

    Returns:
        _KMeansFakePalettize: The mutated module, ready to be swapped into
            the parent's parametrization slot.
    """
    fp_module, weight, layer_name = args

    try:
        fp_module(weight)
    except Exception as e:
        raise RuntimeError(f"Centroid calculation failed for layer {layer_name!r}") from e

    if fp_module._disabled:
        fp_module._disabled_reason = f"layer {layer_name!r}"

    return fp_module


class KMeansPalettizer(_BasePalettizer, _EagerCompressionComponentBuilderMixin):
    """K-means palettizer with integrated supported operations strategy."""

    def __init__(self, model: torch.nn.Module, config: KMeansPalettizerConfig | None = None):
        """
        Initialize the KMeans palettizer.

        Args:
            model: The PyTorch model to palettize.
            config: Optional palettization configuration. If None, default configuration
                   will be used.
        """
        if not isinstance(model, torch.nn.Module):
            raise TypeError("Model must be a torch.nn.Module")

        if not config:
            # Use default config
            config = KMeansPalettizerConfig()

        super().__init__(model, config)

        module_components_dict, module_priority_dict = (
            self._get_module_compression_components_and_priority(model, config)
        )

        # Use _KMeansPalettizerSupportedOpsRegistry to only process
        # palettization-supported ops
        self._handler = _TorchFunctionEagerHandler(
            compression_config=config,
            module_components_dict=module_components_dict,
            module_priority_dict=module_priority_dict,
            supported_ops_registry=_KMeansPalettizerSupportedOpsRegistry,
            optimization_type_name="palettize",
        )

        # Store example inputs for sensitivity-based centroid recomputation
        self._example_inputs = None

        self._num_workers = 1

    @classmethod
    def get_op_type_resolver(cls) -> Callable[[Callable], str | None]:
        """Return a function that maps a torch function to its palettizable op type."""
        return _KMeansPalettizerSupportedOpsRegistry.get_func_type

    def prepare(
        self,
        example_inputs: tuple[torch.Tensor],
        sensitivity_path: str | None = None,
        num_workers: int = 1,
    ) -> torch.nn.Module:
        """
        Prepare the model for palettization.

        Args:
            example_inputs: Sample inputs to trace the model and configure
                palettizers
            sensitivity_path: Optional path to precomputed sensitivity values for
                weighted k-means clustering. These sensitivity values indicate the
                importance of each weight element and can be computed using
                calibration_mode(). When provided, k-means clustering will place
                centroids closer to more sensitive weight values. If None (default),
                vanilla (non-weighted) k-means clustering is used.
            num_workers: ``1`` runs clustering sequentially. Values greater than
                ``1`` use ``torch.multiprocessing`` to parallelize clustering
                across layers. It is recommended to use more than one worker
                process to parallelize the clustering, especially when multiple
                CPUs are available. Defaults to ``1``.

        Returns:
            The prepared nn.Module with fake palettization
            modules inserted. This is a data-free PTP compressed model.

        Raises:
            RuntimeError: If the model has already been prepared.
            ValueError: If ``num_workers`` is less than 1.
        """
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")

        if self._is_model_prepared(self._model):
            raise RuntimeError(
                "Model has already been prepared. Cannot re-prepare a prepared model. "
            )

        # Save so calibration_mode's recompute can use the same parallelism.
        self._num_workers = num_workers

        # Prepare the model
        logger.info("Preparing model for palettization")
        prepared_model = self._handler.prepare(self._model, example_inputs=example_inputs)

        # Save example inputs for later use in calibration
        self._example_inputs = tuple([ip.detach().clone() for ip in example_inputs])

        # Load precomputed sensitivities if provided
        if sensitivity_path is not None:
            logger.info(
                f"Loading precomputed sensitivities from {sensitivity_path} "
                "for weighted k-means clustering"
            )
            sensitivities = torch.load(sensitivity_path, weights_only=True)
            self._set_sensitivities_in_fake_palettize_modules(sensitivities)

        self._model.apply(_enable_observer)
        self._model.apply(_disable_fake_palett)

        if self._num_workers > 1:
            self._calculate_centroids_parallel(num_workers)
        else:
            self._calculate_centroids_sequential(example_inputs)

        # Remove FakePalettize modules that were disabled during the forward
        # pass due to incompatible granularity or cluster dimensions.
        self._remove_disabled_fake_palett_modules(self._model)

        self._model.apply(_enable_fake_palett)
        self._model.apply(_disable_observer)

        # Mark the model as prepared to prevent re-preparation
        self._mark_model_as_prepared(prepared_model)

        # Update internal model reference
        self._model = prepared_model

        return self._model

    @contextmanager
    def calibration_mode(
        self,
        model: torch.nn.Module | None = None,
        *,
        loss_fn: Callable,
        sensitivity_path: str | None = None,
    ):
        """Context manager for calibration using Sensitive K-Means clustering.

        This method implements sensitivity-based palettization as described in
        "SqueezeLLM: Dense-and-Sparse Quantization"
        (https://arxiv.org/pdf/2306.07629.pdf). The loss function is used to compute
        gradients via backpropagation, and the squared gradients are collected as
        sensitivity values for each weight element.

        These sensitivity values indicate how sensitive a given weight element is:
        the more sensitive an element, the larger the impact palettizing it has on the
        model's loss function. This means that weighted k-means moves the clusters
        closer to the sensitive weight values, allowing them to be represented more
        exactly. This leads to a lower degradation in model performance after
        palettization.

        Args:
            loss_fn: Loss function that takes (output, target) and returns a scalar
                    loss. The loss is used for gradient computation, where the squared
                    gradients serve as sensitivity weights for kmeans clustering.
            sensitivity_path: Optional path for saving the sensitivity
                of weights. Defaults to None.
            model: Optional model to calibrate. If None, uses self._model.


        Example:
            >>> import torch.nn.functional as F
            >>> with palettizer.calibration_mode(loss_fn=F.cross_entropy) as skm:
            ...     for input, label in calibration_dataset:
            ...         out = model(input)
            ...         skm.step(out, label)  # Computes loss + backward
        """
        if model is not None:
            if not isinstance(model, torch.nn.Module):
                raise TypeError("Provided model must be a torch.nn.Module")
            self._model = model

        if not self._is_model_prepared(self._model):
            raise RuntimeError(
                "Model must be prepared before entering calibration mode. Call prepare() first."
            )

        # Save model checkpoint before modifying gradients
        checkpoint_path = self._save_model_checkpoint(self._model)
        self._model.zero_grad()

        # Helper class for loss computation
        class CalibrationHelper:
            def __init__(self, loss_fn):
                self.loss_fn = loss_fn
                self.step_called = False

            def step(self, output: torch.Tensor, target: torch.Tensor):
                """Compute loss and backward pass."""
                loss = self.loss_fn(output, target)
                loss.backward()
                self.step_called = True

        # Disable observers and fake palettization for sensitivity computation
        self._model.apply(_disable_observer)
        self._model.apply(_disable_fake_palett)

        calibration_helper = CalibrationHelper(loss_fn)

        with self._register_grad_square_hooks(self._model):
            try:
                yield calibration_helper
            finally:
                # Ensure step() was called at least once
                if not calibration_helper.step_called:
                    raise RuntimeError(
                        "calibration_mode requires at least one call to step(). "
                        "No calibration data was processed."
                    )

                # Construct sensitivities
                sensitivities = self._construct_sensitivities(sensitivity_path)

                # Restore model from checkpoint
                self._load_model_checkpoint(self._model, checkpoint_path)

                # Set sensitivities in fake palettize modules
                self._set_sensitivities_in_fake_palettize_modules(sensitivities)

                # Zero out gradients to clean up squared gradient values from hooks
                self._model.zero_grad()

                # Enable observers to recompute LUTs with sensitivities
                self._model.apply(_enable_observer)

                # Recompute centroids with sensitivities, matching the
                # parallelism the user opted into at prepare() time.
                if self._num_workers > 1:
                    self._calculate_centroids_parallel(self._num_workers)
                else:
                    self._calculate_centroids_sequential(self._example_inputs)

                # Restore normal operation
                self._model.apply(_enable_fake_palett)
                self._model.apply(_disable_observer)

    def _validate_mmap_dir_constraints(
        self,
        model: torch.nn.Module | None,
        backend: ExportBackend,
        mmap_dir: str | PathLike[str] | None,
    ) -> None:
        """Validate that ``mmap_dir`` is compatible with the target backend and
        model device. No-op when ``mmap_dir is None``. Falls back to
        ``self._model`` when ``model is None`` so the check is self-contained.
        """
        model_to_check = model if model is not None else self._model
        _validate_mmap_backend_and_device(model_to_check, backend, mmap_dir)

    def finalize(
        self,
        model: torch.nn.Module | None = None,
        backend: ExportBackend = ExportBackend.CoreAI,
        *,
        mmap_dir: str | PathLike[str] | None = None,
    ) -> torch.nn.Module:
        """Convert palettized model to backend-specific representations.

        Only call ``finalize`` when exporting to a target backend. For torch-based
        evaluation, use the model returned by ``prepare()`` directly rather than
        calling ``finalize``.

        Args:
            model (nn.Module | None): Model to finalize. If None, uses the
                internal prepared model.
            backend (ExportBackend): Target export backend for the palettized
                model. Supports CoreAI (default) and CoreML backends.
            mmap_dir (str | None): If provided, finalized palettized weights are
                written under this directory and re-loaded as mmap-backed
                tensors so they don't have to be held in RAM. Only supported
                with the CoreAI backend; raises ``ValueError`` otherwise. The
                files in ``mmap_dir`` must remain in place for the lifetime of
                the returned model; removing them invalidates the mmap-backed
                weights.

        Returns:
            torch.nn.Module: The finalized palettized model ready for deployment.

        Note:
            When ``backend=ExportBackend.CoreAI``, finalize frees the original
            dense weights in place: on each parametrized weight,
            ``parametrizations[...].original`` is replaced with a zero-size
            placeholder so its storage can be released.
        """
        if model is None:
            model = self._model
        elif not isinstance(model, torch.nn.Module):
            raise TypeError("Provided model must be a torch.nn.Module")

        if not self._is_model_prepared(model):
            raise RuntimeError("Model must be prepared before finalization. Call prepare() first.")

        self._validate_mmap_dir_constraints(model, backend, mmap_dir)

        logger.info(f"Finalizing model for backend: {backend}")
        finalized_model = model

        # Backend-specific processing
        match backend:
            case ExportBackend._TORCH:
                pass

            case ExportBackend.CoreAI:
                finalized_model = _prepare_for_mlir_export(finalized_model, mmap_dir=mmap_dir)

            case ExportBackend.CoreML:
                finalized_model = _prepare_for_mil_export(finalized_model)

            case _:
                msg = f"Unsupported backend: {backend}"
                raise NotImplementedError(msg)

        # Clear the prepared flag so the finalized model can be further processed
        # (e.g., by Quantizer for activation quantization after weight palettization)
        if hasattr(finalized_model, "_coreai_opt_prepared"):
            delattr(finalized_model, "_coreai_opt_prepared")

        return finalized_model

    @staticmethod
    def _spec_to_partial(
        spec: CompressionSpec | None,
        target: CompressionTargetTensor,
        module_config: ModuleCompressionConfig,
    ) -> _PartialConstructor | None:
        if spec is None:
            return None
        # Serialize the spec, then layer in the owning module's compressor-specific
        # settings (e.g. enable_fast_kmeans_mode, rounding_precision).
        args = spec.model_dump_preserve_objects()
        args.update(module_config._get_compressor_specific_settings())
        return _KMeansFakePalettize.with_args(**args)

    def _calculate_centroids_sequential(self, example_inputs: tuple[torch.Tensor]) -> None:
        """Run a forward pass to calculate centroids, with a per-layer progress bar."""
        fp_modules: list[_KMeansFakePalettize] = []
        for _, module in self._model.named_modules(remove_duplicate=True):
            if not P.is_parametrized(module):
                continue
            for parametrizations in module.parametrizations.values():
                for p in parametrizations:
                    if isinstance(p, _KMeansFakePalettize):
                        fp_modules.append(p)
                        break

        progress = tqdm(total=len(fp_modules), desc="Palettizing layers (num_workers=1)")
        seen: set[int] = set()

        def _tick(module, _inputs, _output):
            if id(module) not in seen:
                seen.add(id(module))
                progress.update(1)

        handles = [m.register_forward_hook(_tick) for m in fp_modules]
        try:
            with _move_model_to_eval(self._model):
                with torch.no_grad():
                    self._model(*example_inputs)
        finally:
            for h in handles:
                h.remove()
            progress.close()

    def _calculate_centroids_parallel(self, num_workers: int) -> None:
        """Compute centroids for all _KMeansFakePalettize modules in parallel."""
        # Track parametrization slot (module, attr_name, idx) so the worker's
        # mutated module can be swapped back in. Whole-module swap means every
        # buffer and plain attribute round-trips automatically.
        fp_info: list[_FakePalettInfo] = []
        for module_name, module in self._model.named_modules(remove_duplicate=True):
            if not P.is_parametrized(module):
                continue
            for attr_name, parametrizations in module.parametrizations.items():
                for idx, p in enumerate(parametrizations):
                    if isinstance(p, _KMeansFakePalettize):
                        weight = parametrizations.original.detach().cpu()
                        fp_info.append(
                            _FakePalettInfo(
                                module=module,
                                module_name=module_name,
                                attr_name=attr_name,
                                idx=idx,
                                fp_module=p,
                                weight=weight,
                            )
                        )
                        break

        if not fp_info:
            return

        # Cap worker count at the number of layers to avoid idle processes.
        effective_workers = min(num_workers, len(fp_info))
        logger.info(
            f"Calculating centroids for {len(fp_info)} layers "
            f"in parallel with {effective_workers} workers"
        )

        # spawn (not fork) so workers don't inherit the parent's CUDA context
        # or other process-global state.
        pool_args = [(info.fp_module, info.weight, info.layer_name) for info in fp_info]
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=effective_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(_calculate_centroids_for_module, pool_args),
                    total=len(fp_info),
                    desc=f"Palettizing layers (num_workers={num_workers})",
                )
            )

        for info, new_fp in zip(fp_info, results, strict=True):
            if getattr(new_fp, "_disabled", False):
                logger.warning(
                    f"Disabling palettization for a module: "
                    f"{getattr(new_fp, '_disabled_reason', '')}"
                )
            # ParametrizationList supports item assignment; this swaps the
            # worker's mutated module into the live model without touching
            # the surrounding parametrization registration.
            info.module.parametrizations[info.attr_name][info.idx] = new_fp

    @staticmethod
    @contextmanager
    def _register_grad_square_hooks(model: torch.nn.Module):
        """
        Context manager for registering gradient squaring hooks within the context
        and unregistering them on exit.
        """
        hook_handles = []
        for param in model.parameters():
            if param.requires_grad:
                hook_handles.append(param.register_hook(lambda grad: torch.square(grad)))
        try:
            yield model
        finally:
            for handle in hook_handles:
                handle.remove()

    def _normalize_sensitivities(
        self, raw_sensitivity_dict: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Normalize raw sensitivity values for weighted k-means clustering.

        Applies normalization to ensure numerical stability:
        - Negates and scales by 100 (to convert from gradient hooks)
        - Normalizes to [0, 1] range per parameter
        - Clips zero/small values to prevent k-means divergence

        Args:
            raw_sensitivity_dict: Dictionary mapping parameter names to raw
                sensitivity tensors (typically negative gradients)

        Returns:
            Dictionary with normalized sensitivity values ready for k-means
        """
        normalized_sensitivity_dict = {}
        for key, val in raw_sensitivity_dict.items():
            # Since optimizer sets param value as: p <= p - learning_rate * (grad**2),
            # we need to negate the values to get grad**2
            val = 100 * -val
            if len(val.nonzero()) == 0:
                val[val == 0] = 1.0

            # normalize sensitivity between 0 and 1
            val = val / torch.max(val)

            # Clipping very small or zero sensitivity values stabilizes k-means,
            # they can lead to divergence otherwise
            val[val == 0] = torch.min(val[val != 0])
            val[val < _SENSITIVITY_CLIP_THR] = _SENSITIVITY_CLIP_THR

            normalized_sensitivity_dict[key] = val

        return normalized_sensitivity_dict

    def _construct_sensitivities(self, sensitivity_path: str | None) -> dict[str, torch.Tensor]:
        """
        Construct sensitivities from model gradients during calibration.

        Extracts gradients from model parameters, optionally saves them,
        and returns normalized sensitivities for weighted k-means.

        Args:
            sensitivity_path: Optional path to save normalized sensitivity values

        Returns:
            Dictionary with normalized sensitivity values
        """
        sensitivity_dict = {}
        for name, param in self._model.named_parameters(remove_duplicate=True):
            if param.requires_grad and param.grad is not None:
                sensitivity_dict[name] = -param.grad.cpu()

        normalized_sensitivities = self._normalize_sensitivities(sensitivity_dict)

        if sensitivity_path is not None:
            logger.info(f"Saving sensitivities to {sensitivity_path}")
            torch.save(normalized_sensitivities, sensitivity_path)

        return normalized_sensitivities

    def _set_sensitivities_in_fake_palettize_modules(
        self, sensitivity_dict: dict[str, torch.Tensor]
    ) -> None:
        """Set sensitivity buffers in fake palettize modules via parametrizations."""
        for module_name, module in self._model.named_modules(remove_duplicate=True):
            if P.is_parametrized(module):
                for attr_name, parametrizations in module.parametrizations.items():
                    # Find FakePalett parametrization
                    for p in parametrizations:
                        if isinstance(p, _KMeansFakePalettize):
                            param_name = ".".join(
                                [module_name, "parametrizations", attr_name, "original"]
                            )
                            if param_name not in sensitivity_dict:
                                logger.error(f"No sensitivity value found for {param_name}")
                                break

                            param = parametrizations.original
                            assert param.shape == sensitivity_dict[param_name].shape, (
                                f"A param's sensitivity shape: "
                                f"{sensitivity_dict[param_name].shape} must match the "
                                f"shape of the param: {param.shape}"
                            )
                            p.sensitivities = sensitivity_dict[param_name]
                            logger.debug(f"Set sensitivities for parameter: {param_name}")
                            break

        logger.info("Updated sensitivities in fake palettize modules")

    def save_sensitivities(self, path: str) -> None:
        """
        Save sensitivity values from the prepared model to a file.

        This method extracts the sensitivity values currently set in the model's
        _KMeansFakePalettize modules and saves them to the specified path. This is
        useful when sensitivities were computed via calibration_mode() but not
        saved at that time.

        The saved sensitivities can later be loaded using prepare(sensitivity_path=...)
        to apply the same weighted k-means clustering to a fresh model.

        Args:
            path: File path where sensitivities will be saved

        Raises:
            RuntimeError: If the model has not been prepared yet
            ValueError: If no sensitivities are found in the model

        Example:
            >>> palettizer = KMeansPalettizer(model, config)
            >>> prepared_model = palettizer.prepare(example_inputs)
            >>> with palettizer.calibration_mode(loss_fn=loss_fn) as skm:
            ...     output = prepared_model(input)
            ...     skm.step(output, target)
            >>> # Save sensitivities for later use
            >>> palettizer.save_sensitivities("sensitivities.pt")
        """
        if not self._is_model_prepared(self._model):
            raise RuntimeError(
                "Model must be prepared before saving sensitivities. Call prepare() first."
            )

        sensitivity_dict = self._get_sensitivities_from_fake_palettize_modules()

        if not sensitivity_dict:
            raise ValueError(
                "No sensitivities found in the model. Run calibration_mode() first "
                "to compute sensitivity values, or load precomputed sensitivities "
                "via prepare(sensitivity_path=...)."
            )

        logger.info(f"Saving sensitivities to {path}")
        torch.save(sensitivity_dict, path)

    def _get_sensitivities_from_fake_palettize_modules(
        self,
    ) -> dict[str, torch.Tensor]:
        """
        Extract sensitivity values from fake palettize modules.

        Returns:
            Dictionary mapping parameter names to sensitivity tensors
        """
        sensitivity_dict = {}

        for module_name, module in self._model.named_modules(remove_duplicate=True):
            if P.is_parametrized(module):
                for attr_name, parametrizations in module.parametrizations.items():
                    for p in parametrizations:
                        if not isinstance(p, _KMeansFakePalettize):
                            continue
                        if p.sensitivities is not None:
                            param_name = ".".join(
                                [module_name, "parametrizations", attr_name, "original"]
                            )
                            sensitivity_dict[param_name] = p.sensitivities.cpu()
                        break

        return sensitivity_dict

    @staticmethod
    def _save_model_checkpoint(model: torch.nn.Module) -> str:
        """Save model checkpoint to a temporary file and return the path."""
        # Create a temporary file for the checkpoint
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pt", prefix="palettizer_calibration_"
        ) as tmp_file:
            checkpoint_path = tmp_file.name
            logger.debug(f"Saving model checkpoint to {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)
            return checkpoint_path

    @staticmethod
    def _load_model_checkpoint(model: torch.nn.Module, checkpoint_path: str):
        """Restore model checkpoint from specified checkpoint path."""
        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            logger.debug(
                f"Restoring model from checkpoint {checkpoint_path} "
                "before setting sensitivities and recomputing centroids"
            )
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            # Clean up temporary checkpoint file
            logger.debug(f"Removing temporary checkpoint {checkpoint_path}")
            os.unlink(checkpoint_path)
        else:
            logger.error(f"Failed to load model checkpoint from path: {checkpoint_path}")

    @staticmethod
    def _remove_disabled_fake_palett_modules(model: torch.nn.Module) -> None:
        """Remove FakePalettize modules that were disabled during the forward pass."""
        disabled_fp = {
            m for m in model.modules() if isinstance(m, _KMeansFakePalettize) and m.is_disabled()
        }
        if disabled_fp:
            _remove_compression_parametrizations(model, disabled_fp)
