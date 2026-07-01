# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

import warnings
from abc import abstractmethod

import torch
import torch.nn as nn
from torchao.quantization.quant_primitives import (
    choose_qparams_affine_with_min_max,
)

from coreai_opt._utils.registry_utils import (
    ClassRegistryMixin as _ClassRegistryMixin,
)
from coreai_opt._utils.torch_utils import (
    E8M0_EXPONENT_BIAS as _E8M0_EXPONENT_BIAS,
    F32_MIN_NORMAL as _F32_MIN_NORMAL,
    FP_DTYPE_TO_MAX_POW2 as _FP_DTYPE_TO_MAX_POW2,
)

from .granularity import QuantizationGranularity
from .qscheme import QuantizationScheme
from .range_calculator import RangeCalculatorBase


class QParamsCalculatorBase(_ClassRegistryMixin, nn.Module):
    """Abstract base for qparams calculators — common configuration and helpers.

    Concrete subclasses inherit from either:

    - ``StatefulQParamsCalculatorBase`` — has scale/zp/minval buffers; used by
      Static, MovingAverage, GlobalMinMax.
    - ``StatelessQParamsCalculatorBase`` — no buffers, qparams recomputed per
      forward; used by Dynamic. ``FakeQuantizeImplBase`` and ``Quantizer``
      detect this subclass to keep the observer always on and to reject
      export.

    Subclasses must implement ``forward(tensor) -> (scale, zero_point, minval)``.
    """

    def __init__(
        self,
        dtype: torch.dtype,
        qscheme: QuantizationScheme,
        granularity: QuantizationGranularity,
        target_dtype: torch.dtype,
        quant_min: int,
        quant_max: int,
        range_calculator: RangeCalculatorBase,
        float_range: tuple[float | None, float | None],
        scale_dtype: torch.dtype | None = None,  # primarily for FP4
        **kwargs,
    ):
        super().__init__()
        self.scale_dtype = scale_dtype
        self.dtype = dtype
        self.qscheme = qscheme
        self._granularity = granularity
        self.target_dtype = target_dtype
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.range_calculator = range_calculator
        self.float_range = float_range

        self._initialized = False

        # This is added to address MLIR limitation where
        # tensor after q-dq op is not casted to incoming tensor dtype
        self._compute_dtype_for_export = torch.float32

        # Resolved non-negative axis for per-channel export paths.
        # Ellipsis sentinel means unresolved; resolved on first forward() call.
        self._resolved_axis: int | None = ...  # type: ignore[assignment]

    @property
    def granularity(self) -> QuantizationGranularity:
        """Getter for granularity."""
        return self._granularity

    @granularity.setter
    def granularity(self, granularity: QuantizationGranularity) -> None:
        """Update granularity for this calculator and its range calculator.

        Can only be performed before the first forward pass.
        """
        if self._initialized:
            raise RuntimeError(
                "Cannot change granularity after observer has been initialized. "
                "Granularity must be set before the first forward pass."
            )
        self._granularity = granularity
        self.range_calculator.granularity = granularity

    def _resolve_axis(self, tensor_ndim: int) -> None:
        """Resolve axis to non-negative on first call, delegating to granularity.

        Caches the result in ``_resolved_axis``. Each QParamsCalculator instance
        is per-node, so tensor rank is consistent across calls.

        Args:
            tensor_ndim: Rank of the input tensor.

        """
        if self._resolved_axis is not ...:
            return
        self._resolved_axis = QuantizationGranularity._resolve_axis(self.granularity, tensor_ndim)

    def _get_tensor_with_granularity_from_scalar(
        self, scalar: float, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Return a tensor with dimensions equal to num blocks in each dimension, comprised
        of values equal to scalar.
        """
        block_size_list = self.granularity.get_block_size(input_tensor.shape)
        num_blocks_list = [
            inp_size // block_size
            for inp_size, block_size in zip(input_tensor.shape, block_size_list, strict=True)
        ]
        return torch.full(size=num_blocks_list, fill_value=scalar, device=input_tensor.device)

    def _get_min_and_max_val(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return min and max tensors computed from range calculator statistics and/or
        taken from float_range setting.
        """
        min_val = (
            torch.clamp(
                self._get_tensor_with_granularity_from_scalar(self.float_range[0], tensor), max=0
            )
            if self.float_range[0] is not None
            else None
        )

        max_val = (
            torch.clamp(
                self._get_tensor_with_granularity_from_scalar(self.float_range[1], tensor), min=0
            )
            if self.float_range[1] is not None
            else None
        )

        if min_val is None or max_val is None:
            computed_min, computed_max = self.range_calculator(tensor)
            if min_val is None:
                min_val = torch.clamp(computed_min, max=0)
            if max_val is None:
                max_val = torch.clamp(computed_max, min=0)
            # A one-sided float_range pairs a float_range bound (built at float32)
            # with a range-calculator bound at the input dtype. Cast both to the
            # input dtype so this mixed case has matching dtypes for the qparams op.
            min_val = min_val.to(tensor.dtype)
            max_val = max_val.to(tensor.dtype)
        return min_val, max_val

    def _compute_e8m0_scale(self, max_abs: torch.Tensor) -> torch.Tensor:
        """
        Compute power-of-2 scale in e8m0 format using FLOOR mode.
        References:
            - OCP Microscaling Formats (MX) Specification:
              https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
            - torchao implementation:
              https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/mx_tensor.py
        """
        target_max_pow2 = _FP_DTYPE_TO_MAX_POW2.get(self.dtype)
        if target_max_pow2 is None:
            raise ValueError(
                f"Unsupported dtype for e8m0 scale computation: {self.dtype}. "
                f"Supported: {list(_FP_DTYPE_TO_MAX_POW2.keys())}"
            )

        max_abs_fp32 = max_abs.to(torch.float32)
        max_abs_int32 = max_abs_fp32.view(torch.int32)

        # Extract biased exponent from float32 (bits 23-30)
        extracted_pow2 = ((max_abs_int32 >> 23) & 0xFF) - _E8M0_EXPONENT_BIAS

        # Scale exponent = data exponent - target_max_pow2
        scale_e8m0_unbiased = extracted_pow2 - target_max_pow2

        # Clamp to e8m0 representable range
        scale_e8m0_unbiased = torch.clamp(
            scale_e8m0_unbiased,
            min=-_E8M0_EXPONENT_BIAS,
            max=_E8M0_EXPONENT_BIAS + 1,
        )

        # Convert biased e8m0 back to float32: scale = 2^(unbiased_exponent)
        scale_e8m0_biased = scale_e8m0_unbiased + _E8M0_EXPONENT_BIAS
        scale_fp32 = (scale_e8m0_biased.to(torch.int32) << 23).view(torch.float32)

        # Clamp to minimum normal float32 to avoid denormals
        scale_fp32 = torch.clamp(scale_fp32, min=_F32_MIN_NORMAL)

        return scale_fp32

    def _compute_scale_zero_point_minval(
        self, tensor: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Supports two scale computation modes based on ``scale_dtype``:

        1. **Default (``scale_dtype=None``)**: Uses torchao's
        ``choose_qparams_affine_with_min_max`` custom op to compute scale and
        zero point. Supports integer dtypes and FP8 dtypes.

        2. **e8m0 scales** (``scale_dtype=torch.float8_e8m0fnu``): Scales are
        constrained to powers of 2 following the OCP Microscaling (MX)
        specification (FLOOR mode):
            scale = 2^(floor(log2(max_abs)) - target_max_pow2)
            where ``target_max_pow2`` is the largest power-of-2 component of the
            target dtype's maximum representable value:
                - FP4 E2M1:  max = 6.0     = 1.5  * 2^2,  target_max_pow2 = 2
                - FP8 E4M3:  max = 448.0   = 1.75 * 2^8,  target_max_pow2 = 8
                - FP8 E5M2:  max = 57344.0 = 1.75 * 2^15, target_max_pow2 = 15
        """

        # e8m0 path: power-of-2 scales
        if self.scale_dtype == torch.float8_e8m0fnu:
            max_abs = torch.maximum(torch.abs(min_val), torch.abs(max_val))
            return self._compute_e8m0_scale(max_abs), None, None

        # Default path: torchao handles integer and FP8 dtypes
        scale, zero_point = choose_qparams_affine_with_min_max(
            min_val=min_val,
            max_val=max_val,
            mapping_type=QuantizationScheme._to_mapping_type(self.qscheme),
            block_size=self.granularity.get_block_size(tensor.shape),
            target_dtype=self.target_dtype,
            quant_min=self.quant_min,
            quant_max=self.quant_max,
            eps=torch.finfo(torch.float32).eps,
            zero_point_dtype=torch.int32,
        )

        # ``minval`` is the minimum representable float value for the
        # observed range. The fake-quantize / export layer decides whether
        # the ZP or MINVAL formulation is in use and selects between
        # ``zero_point`` and ``minval`` accordingly; the calculator just
        # provides both.
        if self.qscheme in [
            QuantizationScheme.SYMMETRIC,
            QuantizationScheme.SYMMETRIC_WITH_CLIPPING,
        ]:
            minval = -torch.max(torch.abs(min_val), torch.abs(max_val))
        else:
            # Asymmetric: ``minval`` is ``min(min_val, 0)``. This is a
            # defensive no-op — ``min_val`` is always ``<= 0`` here because:
            #   - the spec validator rejects ``float_range[0] > 0`` at
            #     construction (see ``QuantizationSpec.validate_float_range``);
            #   - the computed-range path clamps to ``<= 0`` in
            #     ``_get_min_and_max_val``.
            minval = torch.min(min_val, torch.zeros_like(min_val))

        # For FP dtypes, neither zero_point nor minval is used (symmetric
        # quantization with no offset).
        if self.dtype.is_floating_point:
            zero_point = None
            minval = None

        return scale, zero_point, minval

    def _initialize_state(
        self,
        tensor: torch.Tensor,
        min_val: torch.Tensor,
        max_val: torch.Tensor,
    ) -> None:
        """Hook for subclass-specific initialization on the first forward pass.

        Tensor shape and device are not known at construction time, so stateful
        subclasses override this to resize their buffers (e.g.
        ``running_min``/``running_max``) to match the observed input."""

    def compute_qparams(
        self,
        tensor: torch.Tensor,
        min_val: torch.Tensor,
        max_val: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Given the observed min/max range, return ``(scale, zero_point, minval)``.

        Default implementation: pure function of the supplied range, no running state.
        ``RunningRangeMixin`` overrides this to apply a running-range smoothing rule
        before computing qparams.
        """
        return self._compute_scale_zero_point_minval(tensor, min_val, max_val)

    @abstractmethod
    def forward(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Compute and return ``(scale, zero_point, minval)`` for ``tensor``."""


class StatefulQParamsCalculatorBase(QParamsCalculatorBase):
    """Stateful base: maintains scale/zero_point/minval as nn.Module buffers
    across forwards.

    Buffer shapes are allocated on first forward and must remain stable
    (``copy_`` requires shape compatibility) — use
    ``StatelessQParamsCalculatorBase`` for variable-shape scales.
    """

    scale: torch.Tensor
    zero_point: torch.Tensor | None
    minval: torch.Tensor | None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.register_buffer("scale", torch.empty(0))

        if self.dtype.is_floating_point:
            self.register_buffer("zero_point", None)
            self.register_buffer("minval", None)
        else:
            self.register_buffer("zero_point", torch.empty(0, dtype=torch.int32))
            self.register_buffer("minval", torch.empty(0))

        self._export_mode = False

    def forward(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Compute qparams from ``tensor``; cache to buffers; return.

        On the first forward pass, initializes internal buffers using the
        observed tensor shape and device. Delegates the actual qparams
        calculation to ``compute_qparams``.
        """
        self._resolve_axis(tensor.ndim)

        scale = self.scale.clone()
        zero_point = self.zero_point.clone() if self.zero_point is not None else None
        minval = self.minval.clone() if self.minval is not None else None

        # Optimization to skip range updates if both min and max are frozen.
        if (
            self._initialized
            and self.float_range[0] is not None
            and self.float_range[1] is not None
        ):
            return scale, zero_point, minval

        if not self._export_mode:
            min_val, max_val = self._get_min_and_max_val(tensor)

            if not self._initialized:
                self._compute_dtype_for_export = tensor.dtype
                self._initialize_state(tensor, min_val, max_val)

            scale, zero_point, minval = self.compute_qparams(tensor, min_val, max_val)

            if not self._initialized:
                self.scale = torch.empty(scale.shape, dtype=self.scale.dtype, device=tensor.device)

                # Only resize zero_point if it exists (None for FP4/FP8)
                if zero_point is not None:
                    self.zero_point = torch.empty(
                        zero_point.shape,
                        dtype=self.zero_point.dtype,
                        device=tensor.device,
                    )

                if minval is not None:
                    self.minval = torch.empty(
                        minval.shape,
                        dtype=self.minval.dtype,
                        device=tensor.device,
                    )

                self._initialized = True

            self.scale.copy_(scale.detach())
            if zero_point is not None:
                self.zero_point.copy_(zero_point.detach())

            if minval is not None:
                self.minval.copy_(minval.detach())

        return scale, zero_point, minval

    def get_qparams(self) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """
        Return the computed scale, zero point and minval.
        For FP4/FP8/floating-point quantization, zero_point and minval are None.
        """
        if not self._initialized:
            warnings.warn(
                "Using default scale and zero point. Call forward pass to compute actual values.",
                stacklevel=1,
            )
        return self.scale, self.zero_point, self.minval

    def set_export_mode(self, enabled: bool = True) -> None:
        self._export_mode = enabled

    def extra_repr(self) -> str:
        return (
            f"scale={self.scale}, zero_point={self.zero_point}, "
            f"minval={self.minval}, export_mode={self._export_mode}"
        )


class StatelessQParamsCalculatorBase(QParamsCalculatorBase):
    """Stateless base: no cached qparams; recomputed every forward.

    Used for dynamic quantization where activations vary per inference and the
    scale shape may change across forwards (e.g. LLM token-wise with variable
    sequence length). ``self.scale``/``zero_point``/``minval`` are assigned in
    forward as plain Python attributes (not buffers) for debugging visibility
    only — they reflect the most recent forward and are not in ``state_dict``.

    - ``FakeQuantizeImplBase`` keeps ``observer_enabled = 1`` for this subclass
      so the recompute path stays live.
    - ``get_qparams`` is undefined; ``set_export_mode(True)`` raises;
      ``float_range=[None, None]`` is required.
    """

    def __init__(self, **kwargs):
        float_range = kwargs.get("float_range", (None, None))
        if float_range[0] is not None or float_range[1] is not None:
            raise ValueError(
                f"StatelessQParamsCalculatorBase requires float_range=[None, None]; "
                f"got {float_range}. Bounded ranges contradict the per-forward "
                f"recompute contract."
            )
        super().__init__(**kwargs)

    def forward(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Compute qparams from ``tensor`` and return; no buffer state."""
        self._resolve_axis(tensor.ndim)

        min_val, max_val = self._get_min_and_max_val(tensor)

        if not self._initialized:
            self._compute_dtype_for_export = tensor.dtype
            self._initialized = True

        scale, zero_point, minval = self.compute_qparams(tensor, min_val, max_val)

        # Plain-attribute (not buffer) assignment for debugging visibility.
        # Shape-mobile across forwards; not in state_dict.
        self.scale = scale.detach()
        self.zero_point = zero_point.detach() if zero_point is not None else None
        self.minval = minval.detach() if minval is not None else None

        return scale, zero_point, minval

    def set_export_mode(self, enabled: bool = True) -> None:
        if enabled:
            raise NotImplementedError(
                "Stateless quantization (e.g. dynamic) does not support export mode; "
                "qparams are input-dependent and cannot be frozen for export."
            )
        # ``enabled=False`` is a deliberate no-op: stateless calculators have no
        # ``_export_mode`` attribute (defined only on StatefulQParamsCalculatorBase),
        # and there is nothing to disable.


@QParamsCalculatorBase.register("default")
class _DefaultQParamsCalculator(QParamsCalculatorBase):
    """
    Marker class for context-dependent qparam calculator resolution.

    This class should not be used directly. When "default" is specified for
    qparam_calculator_cls, the QuantizationComponentFactory will resolve it to:
    - StaticQParamsCalculator for weight quantization
    - MovingAverageQParamsCalculator for activation quantization

    Raises:
        RuntimeError: If __init__ is called, indicating the factory didn't resolve it
    """

    def __init__(self, **kwargs):
        raise RuntimeError(
            "_DefaultQParamsCalculator is a marker class and must be resolved by "
            "QuantizationComponentFactory before use. This error indicates the factory "
            "did not properly resolve 'default' to the appropriate qparam calculator "
            "based on quantization target (weight or activation)."
        )

    def forward(
        self, tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        raise RuntimeError("_DefaultQParamsCalculator.forward() should never be called")


@QParamsCalculatorBase.register("static")
class StaticQParamsCalculator(StatefulQParamsCalculatorBase):
    """
    Computes scale/zero-point/minval using min/max values from the current tensor.

    This QParamsCalculator directly uses the min/max range from each forward pass to compute
    quantization parameters. So in that sense, it does not maintain any "history" and
    only computes the min/max based off of the current (most recent) tensor input.

    This QParamsCalculator is typically used for weight quantization. In case of PTQ based
    workflows the weights are fixed and during QAT, the min/max range is calculated using the
    most recent weight tensor value.

    Uses the base-class default ``compute_qparams`` which
    directly delegates to ``_compute_scale_zero_point_minval`` without any running state.
    """

    # Inherits base-class' default: compute_qparams
    # which directly computes qparams from current min/max with no running state.


@QParamsCalculatorBase.register("dynamic")
class DynamicQParamsCalculator(StatelessQParamsCalculatorBase):
    """
    Dynamically computes scale/zero-point/minval from the current tensor every forward.

    Typically used for activation quantization where activations vary per
    inference and there is no calibration phase. Supports variable-shape scales
    (e.g. LLM token-wise quantization with variable sequence length) since no
    nn.Module buffers are allocated — see ``StatelessQParamsCalculatorBase``
    for the full stateless contract.
    """


# ``# type: ignore`` comments are used where the mixin accesses
# attributes and methods provided by ``QParamsCalculatorBase`` /
# ``nn.Module``, which mypy cannot resolve from the mixin class alone.
class RunningRangeMixin:
    """Mixin for calculators that maintain running min/max range buffers.

    Provides ``running_min`` and ``running_max`` buffers, first-forward
    initialization via ``_initialize_state``, and a
    ``compute_qparams`` implementation that delegates
    the range-update rule to the abstract ``update_running_range`` hook.

    Subclasses that want to re-use the logic of computing quantization
    parameters but with different ways of updating the running statistics
    can override the ``update_running_range`` method.

    Must appear before ``StatefulQParamsCalculatorBase`` in the MRO so that its
    ``compute_qparams`` and ``_initialize_state``
    take precedence over the base-class defaults.
    """

    running_min: torch.Tensor
    running_max: torch.Tensor

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer("running_min", torch.empty(0))  # type: ignore[attr-defined]
        self.register_buffer("running_max", torch.empty(0))  # type: ignore[attr-defined]

    def _initialize_state(
        self,
        tensor: torch.Tensor,
        min_val: torch.Tensor,
        max_val: torch.Tensor,
    ) -> None:
        """Resize and move running_min/running_max buffers to match input
        tensor shape and device on the first forward pass."""
        self.running_min = self.running_min.to(device=tensor.device, dtype=min_val.dtype)
        self.running_min.resize_(min_val.shape)

        self.running_max = self.running_max.to(device=tensor.device, dtype=max_val.dtype)
        self.running_max.resize_(max_val.shape)

    def compute_qparams(
        self,
        tensor: torch.Tensor,
        min_val: torch.Tensor,
        max_val: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Update running range, persist to buffers, then compute qparams."""
        if self._initialized:  # type: ignore[attr-defined]
            running_min, running_max = self.update_running_range(min_val, max_val)
        else:
            running_min, running_max = min_val, max_val

        self.running_min.data.copy_(running_min)
        self.running_max.data.copy_(running_max)
        return self._compute_scale_zero_point_minval(tensor, running_min, running_max)  # type: ignore[attr-defined, no-any-return]

    @abstractmethod
    def update_running_range(
        self, min_val: torch.Tensor, max_val: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(updated_min, updated_max)`` using subclass-specific rule."""

    def extra_repr(self) -> str:
        return super().extra_repr() + (  # type: ignore[misc, no-any-return]
            f"\nrunning_min={self.running_min}, running_max={self.running_max}"
        )


@QParamsCalculatorBase.register("moving_average")
class MovingAverageQParamsCalculator(RunningRangeMixin, StatefulQParamsCalculatorBase):
    """
    Computes the scale and zero point using a moving average of the range.

    Maintains ``running_min`` and ``running_max`` buffers that are updated each
    forward pass using exponential moving average (EMA):

        a_{i} = c * x_{i} + (1 - c) * a_{i-1}

    where ``c`` is the ``averaging_constant``.
    """

    def __init__(
        self,
        dtype: torch.dtype,
        qscheme: QuantizationScheme,
        granularity: QuantizationGranularity,
        target_dtype: torch.dtype,
        quant_min: int,
        quant_max: int,
        range_calculator: RangeCalculatorBase,
        float_range: list[float | None],
        averaging_constant: float = 1e-2,
        **kwargs,
    ):
        super().__init__(
            dtype=dtype,
            qscheme=qscheme,
            granularity=granularity,
            target_dtype=target_dtype,
            quant_min=quant_min,
            quant_max=quant_max,
            range_calculator=range_calculator,
            float_range=float_range,
            **kwargs,
        )
        self.averaging_constant = averaging_constant

    def update_running_range(
        self, min_val: torch.Tensor, max_val: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Exponential moving average of the min and max values
        # a_{i} = c * x_{i} + (1-c) * a_{i-1}
        # Reference: https://en.wikipedia.org/wiki/Exponential_smoothing
        running_min = (
            self.averaging_constant * min_val + (1 - self.averaging_constant) * self.running_min
        )
        running_max = (
            self.averaging_constant * max_val + (1 - self.averaging_constant) * self.running_max
        )
        return running_min, running_max


@QParamsCalculatorBase.register("global_minmax")
class GlobalMinMaxQParamsCalculator(RunningRangeMixin, StatefulQParamsCalculatorBase):
    """Computes scale and zero point by tracking the running min/max.

    Maintains ``running_min`` and ``running_max`` buffers that are updated each
    forward pass via element-wise minimum and maximum:

        running_min = min(running_min, x_min)
        running_max = max(running_max, x_max)
    """

    def update_running_range(
        self, min_val: torch.Tensor, max_val: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        running_min = torch.minimum(self.running_min.detach(), min_val)
        running_max = torch.maximum(self.running_max.detach(), max_val)
        return running_min, running_max
