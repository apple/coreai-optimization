# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from functools import cached_property
from typing import Annotated, Any, ClassVar

import torch
from pydantic import (
    BeforeValidator,
    Field,
    PrivateAttr,
    computed_field,
    field_validator,
    model_validator,
)

from coreai_opt._utils.torch_utils import (
    get_n_bits_from_dtype as _get_n_bits_from_dtype,
    is_float4_dtype as _is_float4_dtype,
)
from coreai_opt.config.spec import CompressionSpec, CompressionType

from .fake_quantize import FakeQuantizeImplBase
from .granularity import (
    PerChannelGranularity,
    PerTensorGranularity,
    QuantizationGranularity,
)
from .qformulation import QuantizationFormulation
from .qparams_calculator import QParamsCalculatorBase
from .qscheme import QuantizationScheme
from .range_calculator import RangeCalculatorBase


class QuantizationSpec(CompressionSpec):
    """
    Specification for quantizing tensors in neural networks.

    This class defines all the parameters needed to quantize a tensor, including the
    target data type, quantization scheme, granularity, and the algorithms used for fake
    quantization, quantization parameter calculation, and range calculation.

    Attributes:
        dtype (str | torch.dtype): Target data type for quantization.
            Valid inputs:

            - Integer dtypes: torch.int8, torch.uint8, torch.int4, torch.uint4, etc.
            - Floating-point dtypes: torch.float8_e4m3fn, torch.float8_e5m2,
              torch.float4_e2m1fn_x2.
              For FP8 dtypes, the notation specifies the format (e.g., in
              torch.float8_e4m3fn, 'e4m3' indicates 4 exponent bits and 3
              mantissa bits, 'f' stands for finite values only, and 'n' stands
              for non-standard NaN representation). For more details on FP8
              dtypes, see https://arxiv.org/pdf/2209.05433
            - String names: "int8", "int4", "float8_e4m3fn", etc. Must correspond to
              an existing torch dtype

            Default: torch.int8

        qscheme (str | coreai_opt.quantization.QuantizationScheme):
            Quantization scheme determining how values are mapped to the quantized
            range.
            Valid inputs:
             - "symmetric" (default), "symmetric_with_clipping", "asymmetric"

            On how it affects the quantization and dequantization formulae,
            please refer to the `qformulation` description below.

        qformulation (str | coreai_opt.quantization.QuantizationFormulation):
            Quantization formula determining how values are mapped between the
            quantized and dequantized domains.
            Valid inputs:

            - ``"zp"`` (default), ``"minval"``
            - ``QuantizationFormulation.ZP``, ``QuantizationFormulation.MINVAL``

            Notation used in the formulae below:

            - ``x``: unquantized data.
            - ``q``: quantized data (dtype as specified by ``QuantizationSpec.dtype``).
            - ``x'``: dequantized data (same dtype as ``x``).
            - ``scale``: for INT quantization, defaults to the same dtype as
              ``x``. For FP quantization, see the ``scale_dtype`` description
              below.

            Formulae:

            - ``"zp"`` — Zero-point formulation. ``zero_point`` has the same
              dtype as ``q``.

              - ``q  = clamp(round(x / scale) + zero_point, quant_min, quant_max)``
              - ``x' = (q - zero_point) * scale``

            - ``"minval"`` — Min-value formulation. ``minval`` has the same
              dtype as ``x``.

              - ``q  = clamp(round((x - minval) / scale) + quant_min, quant_min, quant_max)``
              - ``x' = (q - quant_min) * scale + minval``

            Default: ``QuantizationFormulation.ZP``

            The tables below illustrate how the joint settings across
            ``QuantizationSpec.dtype``, ``QuantizationSpec.qscheme``,
            ``QuantizationSpec.qformulation`` manifest in the formulae above.
            (Note that the min and max values of "x" assumed below are the ones
            which will be calculated based on observer settings, as specified in
            ``QuantizationSpec.qparam_calculator_cls``,
            ``QuantizationSpec.range_calculator_cls``,
            ``QuantizationSpec.float_range``.)

            Derived quantities used in the tables:

            - ``max_abs     = max(|x|)``
            - ``max_val_pos = max(0, max(x))``
            - ``min_val_neg = min(0, min(x))``
            - ``range       = max_val_pos - min_val_neg``

            For per-channel / per-block granularity, the reductions above are
            taken over each quantization unit (channel slice or block) rather
            than the full tensor.

            **ZP formulation**, e.g. with 8 bit signed and unsigned fixed point types:

            +-------+------------+-------------+-----------------+--------------------------------+
            | dtype | qscheme    | quant range | scale           | zero_point                     |
            +=======+============+=============+=================+================================+
            | INT8  | SYMMETRIC  | [-128, 127] | max_abs / 127.5 | 0                              |
            +-------+------------+-------------+-----------------+--------------------------------+
            | INT8  | SYM_W_CLIP | [-127, 127] | max_abs / 127   | 0                              |
            +-------+------------+-------------+-----------------+--------------------------------+
            | INT8  | ASYMMETRIC | [-128, 127] | range / 255     | clip(-128-round(               |
            |       |            |             |                 | min_val_neg/scale), -128, 127) |
            +-------+------------+-------------+-----------------+--------------------------------+
            | UINT8 | SYMMETRIC  | [0, 255]    | max_abs / 127.5 | 128                            |
            +-------+------------+-------------+-----------------+--------------------------------+
            | UINT8 | SYM_W_CLIP | [0, 255]    | max_abs / 127.5 | 128                            |
            +-------+------------+-------------+-----------------+--------------------------------+
            | UINT8 | ASYMMETRIC | [0, 255]    | range / 255     | clip(-round(                   |
            |       |            |             |                 | min_val_neg/scale), 0, 255)    |
            +-------+------------+-------------+-----------------+--------------------------------+

            And for FP4/FP8 dtypes, zero-point is always set to 0 (FP supports only the
            symmetric qscheme). The scale formula depends on ``scale_dtype``:

            - ``scale_dtype=None`` (FP8 only): ``scale = max_abs / fp_max``, where
              ``fp_max`` is the largest representable value for the target FP dtype
              (448.0 for FP8 E4M3, 57344.0 for FP8 E5M2).
            - ``scale_dtype=torch.float8_e8m0fnu`` (FP4 and FP8):
              power-of-2 scale per OCP MX spec —
              ``scale = 2^(floor(log2(max_abs)) - target_max_pow2)``, with
              ``target_max_pow2`` of 2 for FP4 E2M1, 8 for FP8 E4M3, 15 for FP8 E5M2.

            **MINVAL formulation**, e.g. with 8 bit signed and unsigned fixed point types:

            ======  =============  =============  ===============  ===========  =============
            dtype   qscheme        quant range    scale            minval       quant_offset
            ======  =============  =============  ===============  ===========  =============
            INT8    SYMMETRIC      [-128, 127]    max_abs / 127.5  -max_abs     -128
            INT8    SYM_W_CLIP     [-127, 127]    max_abs / 127    -max_abs     -127
            INT8    ASYMMETRIC     [-128, 127]    range / 255      min_val_neg  -128
            UINT8   SYMMETRIC      [0, 255]       max_abs / 127.5  -max_abs     0
            UINT8   SYM_W_CLIP     [0, 255]       max_abs / 127.5  -max_abs     0
            UINT8   ASYMMETRIC     [0, 255]       range / 255      min_val_neg  0
            ======  =============  =============  ===============  ===========  =============

            ``quant_offset`` equals ``q_min`` (the lower bound of the "quant range" column).

            This formulation is not allowed with FP4/FP8 dtypes.

            Note:
                Export-backend constraints:

                - CoreML export only supports ``ZP``. Specs with ``qformulation=MINVAL``
                  are rejected during finalize with CoreML Export-backend.
                - CoreAI export supports both ``ZP`` and ``MINVAL``.

        granularity (dict | coreai_opt.quantization.QuantizationGranularity):
            Quantization granularity determining the scope of
            quantization parameters.
            Valid inputs:

            - Dictionary format:

              - ``{"type": "per_tensor"}`` - Single scale/zero-point for entire
                tensor
              - ``{"type": "per_channel", "axis": <int>}`` - Per-channel
                quantization along axis
              - ``{"type": "per_block", "axis": <int>, "block_size": <tuple>}`` -
                Block-wise quantization along axis with specified block size

            - coreai_opt.quantization.QuantizationGranularity instances:
              PerTensorGranularity(), PerChannelGranularity(axis=1), etc.

            Default: PerTensorGranularity()

        fake_quantize_cls (str | type[coreai_opt.quantization.fake_quantize.FakeQuantizeImplBase]):
            Fake quantization implementation class for simulating quantization.
            This entity makes use of the scale and zero point computed from
            qparam_calculator_cls in order to perform fake quantization (back to back
            quantize/dequantize) to simulate quantization by adding quantization error
            to tensors in the model.
            Users may define their own fake_quantize_cls by inheriting from
            coreai_opt.quantization.fake_quantize.FakeQuantizeImplBase and register
            the class using the decorator
            @FakeQuantizeImplBase.register("<identifier>"), where <identifier> is a
            string which can be used to refer to the registered class in
            fake_quantize_cls.
            Valid inputs:

            - String key: "default" or custom registered class string name
            - Class type:
              coreai_opt.quantization.fake_quantize._DefaultFakeQuantizeImpl
              or custom registered class type

            Default: "default"

        qparam_calculator_cls
            (str | type[QParamsCalculatorBase]):
            Algorithm for calculating quantization parameters (scale and zero
            point).
            Users may define their own qparam_calculator_cls by inheriting from
            coreai_opt.quantization.qparams_calculator.QParamsCalculatorBase
            and register the class using the decorator
            @QParamsCalculatorBase.register("<identifier>"), where
            <identifier> is a string which can be used to refer to the
            registered class in qparam_calculator_cls.
            If float_range is provided, the "default", "static", and
            "moving_average" qparam calculators will take it into account when
            computing scale and zero point.
            Valid inputs:

            - "default": Context-aware default:

              * For weights → StaticQParamsCalculator
              * For activations → MovingAverageQParamsCalculator

            - "static": Direct min/max quantization parameter calculation based on
              most recent calibration sample only
            - "moving_average": Uses exponential moving average for stability
            - "global_minmax": Tracks running min/max across all calibration samples
            - "dynamic": Computes scale/zero/minval point on each forward pass from the
              current tensor — no calibration. Only valid for activation quantization
              (rejected by the factory for weights/LUT).
            - Custom registered class string name
            - coreai_opt.quantization.qparams_calculator.QParamsCalculatorBase
              class type: StaticQParamsCalculator,
              MovingAverageQParamsCalculator, or custom registered class type

            Default: "default"

        range_calculator_cls
            (str | type[RangeCalculatorBase]):
            Algorithm for calculating the min/max range of values to quantize.
            Users may define their own range_calculator_cls by inheriting from
            coreai_opt.quantization.range_calculator.RangeCalculatorBase and
            register the class using the decorator
            @RangeCalculatorBase.register("identifier"), where <identifier>
            is a string which can be used to refer to the registered class in
            range_calculator_cls.
            Valid inputs:

            - "minmax": Uses actual min/max values from the tensor
            - Custom registered class string name
            - coreai_opt.quantization.range_calculator.RangeCalculatorBase
              class type: MinMaxRangeCalculator or custom registered class type

            Default: "minmax"

        float_range (list[float | int | None]): Custom floating-point
            range [min, max] to set for quantization.
            This can be used to set ranges for functions with known bounds (ReLU, Tanh,
            Sigmoid, Softmax, etc.) as well as constraining certain tensors in the model
            to be within a specified range if users want to exclude outliers.
            float_range is used by qparams_calculator_cls. Predefined qparam classes
            "default", "static", and "moving_average" handle float_range. If the
            user defines a custom qparam_calculator_cls, float_range would need to be
            handled properly within the implementation.
            Default: [None, None] (no constraints, allow qparam_calculator_cls
            to determine range)
            Valid inputs:

            - [None, None]: No range constraints (default)
            - [None, float_max]: Fix float max while allowing float min to be
              determined
            - [float_min, None]: Fix float min while allowing float max to be
              determined
            - [float_min, float_max]: Fix both float min and max to a specific
              range

            Constraints:

            - Must be a list or tuple of length 2
            - float_min must be <= 0
            - float_max must be >= 0
            - float_min < float_max

        scale_dtype (torch.dtype | None): Data type for quantization scale factors.
            Controls whether scales are constrained to power-of-2 values (e8m0 format)
            or allowed to be arbitrary floating-point values.
            Valid inputs:

            - None: Use default scale computation via torchao's
              choose_qparams_affine_with_min_max (integer and FP8 dtypes).
              For FP4, None is resolved to torch.float8_e8m0fnu automatically.
            - torch.float8_e8m0fnu: Power-of-2 scales following OCP Microscaling (MX)
              spec. Required for FP4 quantization, optional for FP8.

            Constraints:

            - FP4 (float4_e2m1fn_x2): scale_dtype must be torch.float8_e8m0fnu or None
              (defaults to e8m0)
            - FP8 (float8_e4m3fn, float8_e5m2): scale_dtype must be
              torch.float8_e8m0fnu or None (defaults to None)
            - Integer dtypes: scale_dtype must be None (defaults to None)

            Default: None

    Example:
        >>> # Minimal config using defaults (int8, symmetric, per-tensor)
        >>> spec = QuantizationSpec()
        >>>
        >>> # Quantization with per-channel granularity
        >>> spec = QuantizationSpec(
        ...     dtype=torch.int8,
        ...     qscheme="symmetric",
        ...     granularity={"type": "per_channel", "axis": 1},
        ...     fake_quantize_cls="default",
        ...     qparam_calculator_cls="default",
        ...     range_calculator_cls="minmax",
        ... )
        >>>
        >>> # Quantization with per-tensor granularity and specific float range
        >>> spec = QuantizationSpec(
        ...     dtype="int8",
        ...     qscheme="symmetric",
        ...     granularity={"type": "per_tensor"},
        ...     fake_quantize_cls="default",
        ...     qparam_calculator_cls="moving_average",
        ...     range_calculator_cls="minmax",
        ...     float_range=[-1.0, 1.0]
        ... )

    Notes:
        - All fields have defaults and are optional
        - The qparam_calculator_cls "default" is context-aware and resolved by the
          factory based on whether it's used for weight or activation quantization
        - String inputs are automatically converted to their corresponding types if
          present in corresponding registries.
        - The spec is immutable (frozen=True) once created
        - Custom implementations can be registered and used via string keys
    """

    dtype: torch.dtype = torch.int8
    qscheme: QuantizationScheme = QuantizationScheme.SYMMETRIC
    qformulation: QuantizationFormulation = QuantizationFormulation.ZP
    granularity: Annotated[
        QuantizationGranularity,
        BeforeValidator(QuantizationGranularity.maybe_build_from_dict),
    ] = Field(default_factory=PerTensorGranularity)
    fake_quantize_cls: type[FakeQuantizeImplBase] = Field(default="default", validate_default=True)
    qparam_calculator_cls: type[QParamsCalculatorBase] = Field(
        default="default", validate_default=True
    )
    range_calculator_cls: type[RangeCalculatorBase] = Field(default="minmax", validate_default=True)
    float_range: list[float | int | None] = Field(default_factory=lambda: [None, None])
    scale_dtype: torch.dtype | None = None

    # Private attribute for compression type
    _compression_type: CompressionType = PrivateAttr(default=CompressionType.QUANTIZATION)

    # Supported dtypes for quantization (class attribute for testing extensibility)
    SUPPORTED_DTYPES: ClassVar[set[torch.dtype]] = {
        # Signed integer types
        torch.int8,
        torch.int4,
        torch.int2,
        # Unsigned integer types
        torch.uint8,
        torch.uint4,
        torch.uint2,
        # FP8 types (standard formats)
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        # FP4 types
        torch.float4_e2m1fn_x2,
    }

    # String aliases for convenience (e.g. "float4_e2m1fn" → torch.float4_e2m1fn_x2,
    #                                       "float8_e4m3"   → torch.float8_e4m3fn)
    _DTYPE_ALIASES: ClassVar[dict[str, torch.dtype]] = {
        "float4_e2m1fn": torch.float4_e2m1fn_x2,
        "float8_e4m3": torch.float8_e4m3fn,
        "float8_e8m0": torch.float8_e8m0fnu,
    }

    # Field Validators
    @classmethod
    def _resolve_str_dtype(cls, name: str) -> torch.dtype:
        """Resolve a string to a torch.dtype via aliases or ``torch.<name>``."""
        dtype = cls._DTYPE_ALIASES.get(name) or getattr(torch, name, None)
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {name!r}")
        return dtype

    @field_validator("dtype", mode="before")
    @classmethod
    def convert_dtype(cls, data: Any) -> torch.dtype:
        if isinstance(data, str):
            return cls._resolve_str_dtype(data)
        return data

    @field_validator("dtype", mode="after")
    @classmethod
    def validate_dtype(cls, dtype: torch.dtype) -> torch.dtype:
        """Validate that dtype is supported for quantization."""
        if dtype not in cls.SUPPORTED_DTYPES:
            allowed_names = sorted([str(dt) for dt in cls.SUPPORTED_DTYPES])
            error_msg = f"Unsupported dtype: {dtype}. Allowed dtypes: {', '.join(allowed_names)}"
            raise ValueError(error_msg)
        return dtype

    @field_validator("range_calculator_cls", mode="before")
    @classmethod
    def convert_range_calculator(cls, data: Any) -> type[RangeCalculatorBase]:
        return RangeCalculatorBase.resolve(data)

    @field_validator("float_range", mode="before")
    @classmethod
    def validate_float_range(
        cls, data: list[float | int | None] | tuple[float | int | None]
    ) -> list[float | None]:
        if not isinstance(data, (tuple | list)):
            raise ValueError("Float range must be a list or tuple.")
        if len(data) != 2:
            raise ValueError("Float range must have length 2.")
        if not isinstance(data[0], (type(None) | int | float)) or not isinstance(
            data[1], (type(None) | int | float)
        ):
            raise ValueError("Float range entries must be ints, floats or None.")
        if isinstance(data[0], bool) or isinstance(data[1], bool):
            # This is needed since bool is a subclass of int and will pass the previous
            # check.
            raise ValueError("Float range entries must be ints, floats or None.")
        if data[0] is not None and data[1] is not None and data[0] >= data[1]:
            raise ValueError("Float range [float_min, float_max] expects float_min < float_max.")
        if data[0] is not None and data[0] > 0.0:
            raise ValueError("Float range min value must be less than or equal to 0.")
        if data[1] is not None and data[1] < 0.0:
            raise ValueError("Float range max value must be greater than or equal to 0.")
        # Standardize tuples to lists and ints to floats
        return [
            None if data[0] is None else float(data[0]),
            None if data[1] is None else float(data[1]),
        ]

    @field_validator("qparam_calculator_cls", mode="before")
    @classmethod
    def convert_qparam_calculator(cls, data: Any) -> type[QParamsCalculatorBase]:
        return QParamsCalculatorBase.resolve(data)

    @field_validator("fake_quantize_cls", mode="before")
    @classmethod
    def convert_fake_quantize(cls, data: Any) -> type[FakeQuantizeImplBase]:
        return FakeQuantizeImplBase.resolve(data)

    @model_validator(mode="before")
    @classmethod
    def _strip_computed_fields(cls, data: Any) -> Any:
        """Strip computed fields when deserializing from dict.

        Computed fields (n_bits, target_dtype, _quant_range, quant_min,
        quant_max) are included in model_dump() output but rejected on
        construction since the model uses extra="forbid". We dynamically
        strip any keys that are not declared model fields so round-tripping
        via model_dump works.
        """
        if isinstance(data, dict):
            declared = set(cls.model_fields.keys())
            return {k: v for k, v in data.items() if k in declared}
        return data

    @model_validator(mode="before")
    @classmethod
    def resolve_scale_dtype(cls, data: Any) -> Any:
        """Resolve scale_dtype: convert string to torch.dtype and default to e8m0 for FP4."""
        if isinstance(data, dict):
            dtype = data.get("dtype")
            scale_dtype = data.get("scale_dtype")
            if isinstance(dtype, str):
                dtype = cls._resolve_str_dtype(dtype)
                data["dtype"] = dtype
            if isinstance(scale_dtype, str):
                data["scale_dtype"] = cls._resolve_str_dtype(scale_dtype)
            if _is_float4_dtype(dtype) and data.get("scale_dtype") is None:
                data["scale_dtype"] = torch.float8_e8m0fnu
        return data

    @model_validator(mode="after")
    def validate_qscheme_for_fp_quant(self) -> QuantizationSpec:
        """
        Validate that FP quantization uses symmetric quantization scheme.
        """
        if self.dtype.is_floating_point:
            if self.qscheme != QuantizationScheme.SYMMETRIC:
                error_msg = (
                    f"FP quantization (dtype={self.dtype}) requires "
                    f"symmetric quantization scheme, got "
                    f"qscheme={self.qscheme}. Valid option: 'symmetric'"
                )
                raise ValueError(error_msg)
        return self

    @model_validator(mode="after")
    def validate_qformulation_for_fp_quant(self) -> QuantizationSpec:
        """
        Validate that FP quantization uses zero-point quantization formulation.
        """
        if self.dtype.is_floating_point:
            if self.qformulation != QuantizationFormulation.ZP:
                error_msg = (
                    f"FP quantization (dtype={self.dtype}) requires "
                    f"zero-point quantization formulation, got "
                    f"qformulation={self.qformulation}. Valid option: 'zp'"
                )
                raise ValueError(error_msg)
        return self

    @model_validator(mode="after")
    def validate_scale_dtype(self) -> QuantizationSpec:
        """
        Validate scale_dtype based on element dtype.

        Rules:
            - Only None or torch.float8_e8m0fnu are supported.
            - Integer dtypes: scale_dtype must be None.
            - FP8 dtypes: scale_dtype may be None or torch.float8_e8m0fnu.
            - FP4 dtypes: scale_dtype is resolved to torch.float8_e8m0fnu
              by resolve_scale_dtype (before validator).
        """
        if self.scale_dtype is not None and self.scale_dtype != torch.float8_e8m0fnu:
            raise ValueError(
                f"Unsupported scale_dtype: {self.scale_dtype}. "
                f"Only None or torch.float8_e8m0fnu are supported."
            )

        if not self.dtype.is_floating_point and self.scale_dtype is not None:
            raise ValueError(
                f"scale_dtype must be None for integer dtypes, "
                f"got scale_dtype={self.scale_dtype} with dtype={self.dtype}."
            )

        return self

    def get_extra_args(self) -> dict[str, Any]:
        """
        Automatically detect and return fields beyond base QuantizationSpec.

        This method introspects the current instance to find any additional fields
        that have been added in subclasses, allowing the factory to automatically
        pass them to component constructors.

        Returns:
            Dict[str, Any]: Dictionary of extra field names and their values

        Example:
            >>> class ExtraArgQuantizationSpec(QuantizationSpec):
            ...     eps: float
            ...     temperature: float = 1.0
            >>>
            >>> spec = ExtraArgQuantizationSpec(eps=0.1, ...)
            >>> extra_args = spec.get_extra_args()
            >>> # Returns: {'eps': 0.1, 'temperature': 1.0}
        """
        # Get base class field names
        base_field_names = set(QuantizationSpec.model_fields.keys())

        # Get current instance's field values, excluding base fields
        extra_args = {}
        for field_name in self.__class__.model_fields:
            if field_name not in base_field_names:
                extra_args[field_name] = getattr(self, field_name)

        return extra_args

    # Factory Methods
    @classmethod
    def get_n_bits_from_dtype(cls, dtype: torch.dtype) -> int:
        """
        Extract the number of bits from a torch dtype.

        Args:
            dtype: The torch dtype to extract bits from

        Returns:
            Number of bits for the dtype

        Raises:
            RuntimeError: If unable to extract bits from the dtype
        """
        return _get_n_bits_from_dtype(dtype)

    @classmethod
    def get_target_dtype(cls, dtype: torch.dtype) -> torch.dtype:
        """
        Returns the target dtype for quantization, mapping custom dtypes
        to concrete ones.

        Custom integer dtypes (int1-int7, uint1-uint7) are mapped to their 8-bit
        equivalents, since PyTorch doesn't have native support for
        sub-byte integer types.

        FP4 (float4_e2m1fn_x2) is mapped to float8_e4m3fn, since
        PyTorch support is minimal for float4_e2m1fn_x2. All FP4
        representable values are exactly representable in FP8.

        Args:
            dtype: The source dtype

        Returns:
            The target dtype for quantization:
            - int1, int2, ..., int7 → int8
            - uint1, uint2, ..., uint7 → uint8
            - float4_e2m1fn_x2 → float8_e4m3fn
            - int8, uint8, float16, float32, etc. → unchanged
        """
        n_bits = cls.get_n_bits_from_dtype(dtype)
        if not dtype.is_floating_point and n_bits <= 8:
            return torch.int8 if dtype.is_signed else torch.uint8
        if dtype == torch.float4_e2m1fn_x2:
            return torch.float8_e4m3fn
        return dtype

    @classmethod
    def get_quant_range(
        cls, dtype: torch.dtype, qscheme: QuantizationScheme
    ) -> tuple[int | float, int | float]:
        """
        Calculate quantization range (quant_min, quant_max) for the given
        dtype and scheme.

        Args:
            dtype: The quantization dtype
            qscheme: The quantization scheme (symmetric, asymmetric, etc.)

        Returns:
            Tuple of (quant_min, quant_max) values. Returns floats for
            floating-point dtypes and ints for integer dtypes.

        Examples:
            - int8 symmetric: (-128, 127)
            - int8 symmetric_with_clipping: (-127, 127)
            - int4 symmetric: (-8, 7)
            - int4 symmetric_with_clipping: (-7, 7)
            - uint8: (0, 255)
            - uint8 symmetric_with_clipping: (0, 255) (same as symmetric)
            - float4_e2m1fn_x2: (-6.0, 6.0)
            - float8_e4m3fn: (-448.0, 448.0)
            - float8_e5m2: (-57344.0, 57344.0)
        """
        # Handle FP4, FP8 and other floating-point dtypes
        if dtype.is_floating_point:
            # Special handling for FP4 as torch.finfo() is not implemented yet
            if dtype == torch.float4_e2m1fn_x2:
                # FP4 E2M1 format: 1 sign + 2 exp + 1 mantissa
                # Max value: 2^(3-1) * (1 + 1/2) = 4 * 1.5 = 6.0
                # Range is symmetric: [-6.0, 6.0]
                return -6.0, 6.0

            finfo = torch.finfo(dtype)
            return finfo.min, finfo.max

        # Integer quantization logic
        n_bits = cls.get_n_bits_from_dtype(dtype)
        max_q = 2**n_bits
        if not dtype.is_signed:
            quant_min = 0
            quant_max = max_q - 1
        else:
            quant_min = -max_q / 2
            quant_max = max_q / 2 - 1

        # Apply clipping for SYMMETRIC_WITH_CLIPPING
        return QuantizationScheme._maybe_clip_bounds(qscheme, dtype, int(quant_min), int(quant_max))

    # Computed Properties
    @computed_field(repr=False)  # type: ignore[misc]
    @cached_property
    def n_bits(self) -> int:
        return self.get_n_bits_from_dtype(self.dtype)

    @computed_field(repr=False)  # type: ignore[misc]
    @cached_property
    def target_dtype(self) -> torch.dtype:
        return self.get_target_dtype(self.dtype)

    @computed_field(repr=False)
    @cached_property
    def _quant_range(self) -> tuple[int | float, int | float]:
        return self.get_quant_range(self.dtype, self.qscheme)

    @computed_field(repr=False)
    @cached_property
    def quant_min(self) -> int | float:
        return self._quant_range[0]

    @computed_field(repr=False)
    @cached_property
    def quant_max(self) -> int | float:
        return self._quant_range[1]


def default_weight_quantization_spec() -> QuantizationSpec:
    return QuantizationSpec(
        dtype=torch.int8,
        qscheme="symmetric",
        granularity=PerChannelGranularity(axis=0),
        fake_quantize_cls="default",
        qparam_calculator_cls="static",
        range_calculator_cls="minmax",
    )


def default_activation_quantization_spec() -> QuantizationSpec:
    return QuantizationSpec(
        dtype=torch.int8,
        qscheme="symmetric",
        granularity=PerTensorGranularity(),
        fake_quantize_cls="default",
        qparam_calculator_cls="moving_average",
        range_calculator_cls="minmax",
    )
