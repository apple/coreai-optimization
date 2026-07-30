# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import pytest
import torch
import yaml
from pydantic import ValidationError
from torchao.quantization import MappingType as TorchAOMappingType

from coreai_opt.config.spec import CompressionTargetTensor
from coreai_opt.quantization import QuantizationSpec
from coreai_opt.quantization.spec import (
    PerBlockGranularity,
    PerChannelGranularity,
    PerTensorGranularity,
    QuantizationFormulation,
    QuantizationScheme,
    default_activation_quantization_spec,
    default_weight_quantization_spec,
)
from coreai_opt.quantization.spec.errors import _BlockSizeMismatchError
from coreai_opt.quantization.spec.fake_quantize import (
    _DefaultFakeQuantizeImpl,
)
from coreai_opt.quantization.spec.qparams_calculator import (
    GlobalMinMaxQParamsCalculator,
    MovingAverageQParamsCalculator,
    StaticQParamsCalculator,
    _DefaultQParamsCalculator,
)
from coreai_opt.quantization.spec.range_calculator import MinMaxRangeCalculator


@pytest.fixture
def expanded_dtype_allowlist(monkeypatch):
    """
    Temporarily expand the dtype allowlist to include additional int/uint types for
    testing. This is necessary because test dtypes and config combinations are tightly
    coupled, and some tests need to verify functionality with dtypes that aren't
    officially supported for production use (e.g., int1, uint1, int3, int5, int6,
    int7).
    """
    # Start with production-supported dtypes and add test-only types
    original_dtypes = QuantizationSpec.SUPPORTED_DTYPES.copy()
    expanded_dtypes = original_dtypes.copy()
    expanded_dtypes.update(
        {
            # Additional signed integer types for testing
            # (int2, int4, int8 already included)
            torch.int1,
            torch.int3,
            torch.int5,
            torch.int6,
            torch.int7,
            torch.int16,
            torch.int32,
            # Additional unsigned integer types for testing
            # (uint2, uint4, uint8 already included)
            torch.uint1,
            torch.uint3,
            torch.uint5,
            torch.uint6,
            torch.uint7,
        }
    )

    # Temporarily replace the class attribute
    monkeypatch.setattr(QuantizationSpec, "SUPPORTED_DTYPES", expanded_dtypes)
    yield
    # Restore original
    monkeypatch.setattr(QuantizationSpec, "SUPPORTED_DTYPES", original_dtypes)


def test_qspec_basic():
    spec_dict = {
        "dtype": "int4",
        "qscheme": "symmetric",
        "qformulation": "zp",
        "granularity": {"type": "per_tensor"},
        "fake_quantize_cls": "default",
        "qparam_calculator_cls": "moving_average",
        "range_calculator_cls": "minmax",
    }
    spec = QuantizationSpec(**spec_dict)
    assert spec.dtype == torch.int4
    assert spec.qscheme == QuantizationScheme.SYMMETRIC
    assert spec.qformulation == QuantizationFormulation.ZP
    assert isinstance(spec.granularity, PerTensorGranularity)


def test_qspec_objects():
    spec = QuantizationSpec(
        dtype=torch.int2,
        qscheme="symmetric",
        qformulation=QuantizationFormulation.ZP,
        granularity=PerTensorGranularity(),
        fake_quantize_cls=_DefaultFakeQuantizeImpl,
        qparam_calculator_cls=StaticQParamsCalculator,
        range_calculator_cls=MinMaxRangeCalculator,
    )
    assert spec.dtype == torch.int2
    assert spec.qscheme == QuantizationScheme.SYMMETRIC
    assert spec.qformulation == QuantizationFormulation.ZP
    assert isinstance(spec.granularity, PerTensorGranularity)


def test_qspec_global_minmax_calculator_string():
    """Test that 'global_minmax' string resolves to GlobalMinMaxQParamsCalculator in spec."""
    spec = QuantizationSpec(qparam_calculator_cls="global_minmax")
    assert spec.qparam_calculator_cls == GlobalMinMaxQParamsCalculator


def test_qspec_global_minmax_calculator_class():
    """Test that GlobalMinMaxQParamsCalculator class is accepted directly."""
    spec = QuantizationSpec(qparam_calculator_cls=GlobalMinMaxQParamsCalculator)
    assert spec.qparam_calculator_cls == GlobalMinMaxQParamsCalculator


def test_qspec_with_options():
    spec_dict = {
        "dtype": "int8",
        "qscheme": "asymmetric",
        "qformulation": "minval",
        "granularity": {"type": "per_channel", "axis": 0},
        "fake_quantize_cls": "default",
        "qparam_calculator_cls": "moving_average",
        "range_calculator_cls": "minmax",
    }
    spec = QuantizationSpec(**spec_dict)
    assert spec.dtype == torch.int8
    assert spec.qscheme == QuantizationScheme.ASYMMETRIC
    assert spec.qformulation == QuantizationFormulation.MINVAL
    assert isinstance(spec.granularity, PerChannelGranularity)
    assert spec.granularity.axis == 0


def test_weight_qspec():
    spec = default_weight_quantization_spec()
    assert spec.dtype == torch.int8
    assert spec.qscheme == QuantizationScheme.SYMMETRIC
    assert spec.qformulation == QuantizationFormulation.ZP
    assert isinstance(spec.granularity, PerChannelGranularity)
    assert spec.granularity.axis == 0
    assert spec.fake_quantize_cls == _DefaultFakeQuantizeImpl
    assert spec.qparam_calculator_cls == StaticQParamsCalculator
    assert spec.range_calculator_cls == MinMaxRangeCalculator
    assert spec.float_range == [None, None]


def test_activation_qspec():
    spec = default_activation_quantization_spec()
    assert spec.dtype == torch.int8
    assert spec.qscheme == QuantizationScheme.SYMMETRIC
    assert spec.qformulation == QuantizationFormulation.ZP
    assert isinstance(spec.granularity, PerTensorGranularity)
    assert spec.fake_quantize_cls == _DefaultFakeQuantizeImpl
    assert spec.qparam_calculator_cls == MovingAverageQParamsCalculator
    assert spec.range_calculator_cls == MinMaxRangeCalculator
    assert spec.float_range == [None, None]


def test_invalid_dtype_string():
    with pytest.raises(ValueError, match="Unsupported dtype: 'invalid_dtype'"):
        QuantizationSpec(
            dtype="invalid_dtype",
            qscheme="symmetric",
            granularity="per_tensor",
        )


def test_yaml_config():
    d = yaml.safe_load(
        """
dtype: int8
qscheme: asymmetric
qformulation: minval
granularity:
  type: per_block
  axis: 1
  block_size: 128
fake_quantize_cls:
    default
qparam_calculator_cls:
    default
range_calculator_cls:
    minmax
"""
    )
    spec = QuantizationSpec(**d)
    assert spec.dtype == torch.int8
    assert spec.qscheme == QuantizationScheme.ASYMMETRIC
    assert spec.qformulation == QuantizationFormulation.MINVAL
    assert isinstance(spec.granularity, PerBlockGranularity)
    assert spec.granularity.axis == 1
    assert spec.granularity.block_size == 128


@pytest.mark.parametrize(
    "qscheme,mapping_type",
    [
        ("symmetric", TorchAOMappingType.SYMMETRIC),
        ("symmetric_with_clipping", TorchAOMappingType.SYMMETRIC),
        ("asymmetric", TorchAOMappingType.ASYMMETRIC),
    ],
)
def test_mapping_type(qscheme, mapping_type):
    qspec = QuantizationSpec(
        dtype=torch.int8,
        qscheme=qscheme,
        granularity=PerTensorGranularity(),
        fake_quantize_cls=_DefaultFakeQuantizeImpl,
        qparam_calculator_cls=StaticQParamsCalculator,
        range_calculator_cls=MinMaxRangeCalculator,
    )
    assert QuantizationScheme._to_mapping_type(qspec.qscheme) == mapping_type


def test_invalid_qformulation_string():
    with pytest.raises(ValueError, match="Input should be 'minval' or 'zp'"):
        QuantizationSpec(
            dtype="int4",
            qscheme="symmetric",
            qformulation="invalid_qformulation",
            granularity="per_tensor",
        )


@pytest.mark.parametrize(
    "granularity_type,granularity_params",
    [
        ("per_channel", {"axis": -2, "block_size": 0}),
        (
            "per_channel",
            {
                "axis": -4,
                "block_size": (
                    1,
                    2,
                    3,
                ),
            },
        ),
        (
            "per_channel",
            {
                "axis": None,
                "block_size": (
                    1,
                    2,
                    3,
                ),
            },
        ),
        ("per_block", {"axis": 1, "block_size": 0}),
        (
            "per_block",
            {
                "axis": None,
                "block_size": (
                    2,
                    0,
                ),
            },
        ),
    ],
)
def test_invalid_axis_block_size(granularity_type, granularity_params):
    granularity_config = {"type": granularity_type, **granularity_params}
    with pytest.raises(ValueError):
        _ = QuantizationSpec(
            dtype="int8",
            qscheme="symmetric",
            granularity=granularity_config,
            fake_quantize_cls=_DefaultFakeQuantizeImpl,
            qparam_calculator_cls=StaticQParamsCalculator,
            range_calculator_cls=MinMaxRangeCalculator,
        )


@pytest.mark.parametrize(
    "granularity_type,granularity_params,tensor_shape,expected_block_size",
    [
        # Per tensor
        ("per_tensor", {}, (10, 20, 30), (10, 20, 30)),
        ("per_tensor", {}, (5,), (5,)),
        # Per channel
        ("per_channel", {"axis": 0}, (10, 20, 30), (1, 20, 30)),
        ("per_channel", {"axis": 1}, (10, 20, 30), (10, 1, 30)),
        ("per_channel", {"axis": 0}, (5,), (1,)),
        ("per_channel", {"axis": -1}, (10, 20, 30), (10, 20, 1)),
        ("per_channel", {"axis": -2}, (10, 20, 30), (10, 1, 30)),
        ("per_channel", {"axis": 2}, (10, 20, 30), (10, 20, 1)),
        # Per block
        ("per_block", {"axis": 0, "block_size": 5}, (10, 20), (5, 1)),
        ("per_block", {"axis": 1, "block_size": 4}, (10, 20), (1, 4)),
        ("per_block", {"axis": 0, "block_size": 2}, (8, 16), (2, 1)),
        ("per_block", {"axis": 1, "block_size": 8}, (4, 16), (1, 8)),
        ("per_block", {"axis": 0, "block_size": 1}, (7, 3), (1, 1)),
        ("per_block", {"axis": 1, "block_size": 3}, (7, 3), (1, 3)),
        ("per_block", {"axis": 0, "block_size": 4}, (8, 16, 3), (4, 1, 3)),
        ("per_block", {"axis": 1, "block_size": 8}, (7, 16, 3, 3), (1, 8, 3, 3)),
        # Per block - block axis >= 2 and negative axes. Default target is WEIGHT,
        # so only the two leading channel axes collapse to 1; trailing dims are
        # left at full size. See test_get_block_size_per_block_target for how the
        # ACTIVATION target differs.
        ("per_block", {"axis": 2, "block_size": 16}, (1, 10, 32), (1, 1, 16)),
        ("per_block", {"axis": -1, "block_size": 16}, (1, 10, 32), (1, 1, 16)),
        ("per_block", {"axis": -1, "block_size": 4}, (10, 20), (1, 4)),
        ("per_block", {"axis": -2, "block_size": 5}, (10, 20), (5, 1)),
        ("per_block", {"axis": None, "block_size": (2,)}, (10,), (2,)),
        (
            "per_block",
            {"axis": None, "block_size": (4, 2, -1, 1)},
            (8, 16, 3, 3),
            (4, 2, 3, 1),
        ),
    ],
)
def test_get_block_size_valid_conditions(
    granularity_type, granularity_params, tensor_shape, expected_block_size
):
    granularity_config = {"type": granularity_type, **granularity_params}

    spec = QuantizationSpec(
        dtype="int8",
        qscheme="symmetric",
        granularity=granularity_config,
        fake_quantize_cls=_DefaultFakeQuantizeImpl,
        qparam_calculator_cls=StaticQParamsCalculator,
        range_calculator_cls=MinMaxRangeCalculator,
    )

    result = spec.granularity.get_block_size(torch.Size(tensor_shape))
    assert result == expected_block_size


@pytest.mark.parametrize(
    "axis,block_size,tensor_shape,weight_block_size,activation_block_size",
    [
        (1, 16, (1, 32, 10), (1, 16, 10), (1, 16, 1)),
        (-1, 16, (1, 10, 32), (1, 1, 16), (1, 1, 16)),
        (1, 16, (1, 32, 10, 64), (1, 16, 10, 64), (1, 16, 1, 1)),
        (0, 16, (32, 64, 3, 3), (16, 1, 3, 3), (16, 1, 1, 1)),
        (3, 16, (1, 32, 10, 64), (1, 1, 10, 16), (1, 1, 1, 16)),
        (-1, 16, (1, 32, 10, 64), (1, 1, 10, 16), (1, 1, 1, 16)),
    ],
)
def test_get_block_size_per_block_target(
    axis, block_size, tensor_shape, weight_block_size, activation_block_size
):
    """Per-block single-axis blocking resolves differently per quantization target.

    Weights block across the channel axes only; activations give every non-block
    axis its own scale.
    """
    granularity = PerBlockGranularity(axis=axis, block_size=block_size)
    shape = torch.Size(tensor_shape)

    assert granularity.get_block_size(shape, CompressionTargetTensor.WEIGHT) == weight_block_size
    assert (
        granularity.get_block_size(shape, CompressionTargetTensor.ACTIVATION)
        == activation_block_size
    )
    # Omitting the target keeps the default weight behavior.
    assert granularity.get_block_size(shape) == weight_block_size


@pytest.mark.parametrize(
    "granularity_type,granularity_params,tensor_shape",
    [
        # Per channel - axis out of bounds
        ("per_channel", {"axis": 1}, (5,)),
        ("per_channel", {"axis": -2}, (5,)),
        ("per_channel", {"axis": -3}, (5, 10)),
        # Per block - axis out of bounds
        ("per_block", {"axis": 1, "block_size": 3}, (5,)),
        ("per_block", {"axis": 2, "block_size": 5}, (10, 20)),
        ("per_block", {"axis": 3, "block_size": 5}, (8, 16, 3)),
        ("per_block", {"axis": -3, "block_size": 4}, (10, 20)),
        # Per block - None axis with integer block_size
        ("per_block", {"axis": None, "block_size": 5}, (10, 20)),
        # Per block - integer axis with list block_size
        ("per_block", {"axis": 1, "block_size": [2, 2]}, (10, 20)),
        # Per block - integer axis with tuple block_size
        (
            "per_block",
            {
                "axis": 0,
                "block_size": (
                    2,
                    2,
                ),
            },
            (10, 20),
        ),
        # Per block - tuple block_size rank mismatch with input tensor_shape
        ("per_block", {"axis": None, "block_size": (2, 4)}, (10, 20, 30)),
        ("per_block", {"axis": None, "block_size": (2, 4)}, (10,)),
    ],
)
def test_get_block_size_failure_conditions(
    granularity_type,
    granularity_params,
    tensor_shape,
):
    granularity_config = {"type": granularity_type, **granularity_params}

    spec = QuantizationSpec(
        dtype="int8",
        qscheme="symmetric",
        granularity=granularity_config,
        fake_quantize_cls=_DefaultFakeQuantizeImpl,
        qparam_calculator_cls=StaticQParamsCalculator,
        range_calculator_cls=MinMaxRangeCalculator,
    )

    with pytest.raises(ValueError):
        spec.granularity.get_block_size(torch.Size(tensor_shape))


@pytest.mark.parametrize(
    "granularity_type,granularity_params,tensor_shape",
    [
        # Per block - single axis divisibility check
        ("per_block", {"axis": 0, "block_size": 3}, (10, 20, 30)),
        ("per_block", {"axis": 1, "block_size": 7}, (10, 20, 30)),
        ("per_block", {"axis": 0, "block_size": 10}, (5,)),
        # Per block - multi axis divisibility check
        ("per_block", {"axis": None, "block_size": [2, 7, -1]}, (10, 20, 30)),
    ],
)
def test_get_block_size_divisibility_failure(
    granularity_type,
    granularity_params,
    tensor_shape,
):
    granularity_config = {"type": granularity_type, **granularity_params}

    spec = QuantizationSpec(
        dtype="int8",
        qscheme="symmetric",
        granularity=granularity_config,
        fake_quantize_cls=_DefaultFakeQuantizeImpl,
        qparam_calculator_cls=StaticQParamsCalculator,
        range_calculator_cls=MinMaxRangeCalculator,
    )

    with pytest.raises(_BlockSizeMismatchError):
        spec.granularity.get_block_size(torch.Size(tensor_shape))


@pytest.mark.parametrize(
    "dtype,n_bits",
    [
        (torch.int1, 1),
        (torch.uint1, 1),
        (torch.int2, 2),
        (torch.uint2, 2),
        (torch.int4, 4),
        (torch.uint4, 4),
        (torch.int8, 8),
        (torch.uint8, 8),
        (torch.float8_e4m3fn, 8),
        ("float8_e4m3", 8),
        (torch.float8_e5m2, 8),
        (torch.float4_e2m1fn_x2, 4),
        ("float4_e2m1fn", 4),
    ],
)
def test_n_bits(dtype, n_bits, expanded_dtype_allowlist):
    # Test class method
    assert QuantizationSpec.get_n_bits_from_dtype(dtype) == n_bits

    # Test instance property
    spec = QuantizationSpec(
        dtype=dtype,
        qscheme="symmetric",
        granularity=PerTensorGranularity(),
        fake_quantize_cls=_DefaultFakeQuantizeImpl,
        qparam_calculator_cls=StaticQParamsCalculator,
        range_calculator_cls=MinMaxRangeCalculator,
    )

    assert spec.n_bits == n_bits


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
    ],
)
def test_unsupported_dtypes(dtype):
    """Test that unsupported dtypes raise ValueError."""
    with pytest.raises(
        ValueError,
        match="Unsupported dtype.*Allowed dtypes:",
    ):
        QuantizationSpec(
            dtype=dtype,
            qscheme="symmetric",
            granularity=PerTensorGranularity(),
            fake_quantize_cls=_DefaultFakeQuantizeImpl,
            qparam_calculator_cls=_DefaultQParamsCalculator,
            range_calculator_cls=MinMaxRangeCalculator,
        )


@pytest.mark.parametrize(
    "dtype,qscheme",
    [
        (torch.float8_e4m3fn, "asymmetric"),
        (torch.float8_e5m2, "asymmetric"),
        (torch.float4_e2m1fn_x2, "asymmetric"),
        (torch.float8_e4m3fn, "symmetric_with_clipping"),
        (torch.float8_e5m2, "symmetric_with_clipping"),
        (torch.float4_e2m1fn_x2, "symmetric_with_clipping"),
    ],
)
def test_fp_quant_requires_symmetric_qscheme(dtype, qscheme):
    """Test that FP4/FP8 dtypes only work with symmetric qscheme."""
    with pytest.raises(
        ValueError,
        match="FP quantization.*requires symmetric quantization scheme",
    ):
        QuantizationSpec(
            dtype=dtype,
            qscheme=qscheme,
            granularity=PerTensorGranularity(),
            fake_quantize_cls=_DefaultFakeQuantizeImpl,
            qparam_calculator_cls=_DefaultQParamsCalculator,
            range_calculator_cls=MinMaxRangeCalculator,
        )


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float4_e2m1fn_x2,
    ],
)
def test_fp_quant_requires_zp_qformulation(dtype):
    """Test that FP4/FP8 dtypes only work with ZP qformulation."""
    with pytest.raises(
        ValueError,
        match="FP quantization.*requires zero-point quantization formulation",
    ):
        QuantizationSpec(
            dtype=dtype,
            qscheme="symmetric",
            qformulation="minval",
            granularity=PerTensorGranularity(),
            fake_quantize_cls=_DefaultFakeQuantizeImpl,
            qparam_calculator_cls=_DefaultQParamsCalculator,
            range_calculator_cls=MinMaxRangeCalculator,
        )


@pytest.mark.parametrize(
    "dtype,target_dtype",
    [
        (torch.int1, torch.int8),
        (torch.uint1, torch.uint8),
        (torch.int2, torch.int8),
        (torch.uint2, torch.uint8),
        (torch.int8, torch.int8),
        (torch.uint8, torch.uint8),
        (torch.float8_e4m3fn, torch.float8_e4m3fn),
        (torch.float8_e5m2, torch.float8_e5m2),
        (torch.float4_e2m1fn_x2, torch.float8_e4m3fn),
    ],
)
def test_target_dtype(dtype, target_dtype, expanded_dtype_allowlist):
    # Test class method
    assert QuantizationSpec.get_target_dtype(dtype) == target_dtype

    # Test instance property
    spec = QuantizationSpec(
        dtype=dtype,
        qscheme="symmetric",
        granularity=PerTensorGranularity(),
        fake_quantize_cls=_DefaultFakeQuantizeImpl,
        qparam_calculator_cls=StaticQParamsCalculator,
        range_calculator_cls=MinMaxRangeCalculator,
    )
    assert spec.target_dtype == target_dtype


@pytest.mark.parametrize(
    "dtype,qscheme,range",
    [
        # SYMMETRIC (full range)
        (torch.int8, QuantizationScheme.SYMMETRIC, (-128, 127)),
        (torch.int4, QuantizationScheme.SYMMETRIC, (-8, 7)),
        (torch.int2, QuantizationScheme.SYMMETRIC, (-2, 1)),
        (torch.uint8, QuantizationScheme.SYMMETRIC, (0, 255)),
        (torch.uint4, QuantizationScheme.SYMMETRIC, (0, 15)),
        (torch.uint2, QuantizationScheme.SYMMETRIC, (0, 3)),
        # SYMMETRIC_WITH_CLIPPING (clipped range for signed)
        (torch.int8, QuantizationScheme.SYMMETRIC_WITH_CLIPPING, (-127, 127)),
        (torch.int4, QuantizationScheme.SYMMETRIC_WITH_CLIPPING, (-7, 7)),
        (torch.int2, QuantizationScheme.SYMMETRIC_WITH_CLIPPING, (-1, 1)),
        # Unsigned dtypes unchanged
        (torch.uint8, QuantizationScheme.SYMMETRIC_WITH_CLIPPING, (0, 255)),
        (torch.uint4, QuantizationScheme.SYMMETRIC_WITH_CLIPPING, (0, 15)),
        (torch.uint2, QuantizationScheme.SYMMETRIC_WITH_CLIPPING, (0, 3)),
        # ASYMMETRIC (same as SYMMETRIC for range)
        (torch.int4, QuantizationScheme.ASYMMETRIC, (-8, 7)),
        (torch.int8, QuantizationScheme.ASYMMETRIC, (-128, 127)),
        (torch.int2, QuantizationScheme.ASYMMETRIC, (-2, 1)),
        # FP8 dtypes (scheme doesn't affect FP ranges)
        (torch.float8_e4m3fn, QuantizationScheme.SYMMETRIC, (-448.0, 448.0)),
        (torch.float8_e5m2, QuantizationScheme.SYMMETRIC, (-57344.0, 57344.0)),
        # FP4 dtype
        (torch.float4_e2m1fn_x2, QuantizationScheme.SYMMETRIC, (-6.0, 6.0)),
    ],
)
def test_quant_range(dtype, qscheme, range):
    # Test class method
    assert QuantizationSpec.get_quant_range(dtype, qscheme) == range

    # Test instance property
    spec = QuantizationSpec(
        dtype=dtype,
        qscheme=qscheme,
        granularity=PerTensorGranularity(),
        fake_quantize_cls=_DefaultFakeQuantizeImpl,
        qparam_calculator_cls=StaticQParamsCalculator,
        range_calculator_cls=MinMaxRangeCalculator,
    )

    assert spec.quant_min == range[0]
    assert spec.quant_max == range[1]


def test_symmetric_with_clipping_equal_bins():
    """
    Test that SYMMETRIC_WITH_CLIPPING produces equal bins on each side
    of zero for signed dtypes.
    """
    test_cases = [
        (torch.int8, -127, 127),  # 127 bins on each side
        (torch.int4, -7, 7),  # 7 bins on each side
        (torch.int2, -1, 1),  # 1 bin on each side
    ]

    for dtype, expected_min, expected_max in test_cases:
        spec = QuantizationSpec(
            dtype=dtype,
            qscheme=QuantizationScheme.SYMMETRIC_WITH_CLIPPING,
            granularity=PerTensorGranularity(),
            fake_quantize_cls=_DefaultFakeQuantizeImpl,
            qparam_calculator_cls=_DefaultQParamsCalculator,
            range_calculator_cls=MinMaxRangeCalculator,
        )

        # Verify equal number of bins on each side
        assert abs(spec.quant_min) == abs(spec.quant_max), (
            f"{dtype}: Expected equal bins, got [{spec.quant_min}, {spec.quant_max}]"
        )
        assert spec.quant_min == expected_min
        assert spec.quant_max == expected_max


@pytest.mark.parametrize(
    "float_range", [[None, None], [0.0, 5.0], [-5.0, 0.0], (-5.0, 5), [-5, 5], "empty"]
)
def test_valid_float_range_setting(float_range):
    """
    Validate various float_ranges which all should be valid.
    """
    if float_range == "empty":
        spec = QuantizationSpec(
            dtype=torch.int8,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerTensorGranularity(),
            fake_quantize_cls=_DefaultFakeQuantizeImpl,
            qparam_calculator_cls=StaticQParamsCalculator,
            range_calculator_cls=MinMaxRangeCalculator,
        )
    else:
        spec = QuantizationSpec(
            dtype=torch.int8,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerTensorGranularity(),
            fake_quantize_cls=_DefaultFakeQuantizeImpl,
            qparam_calculator_cls=StaticQParamsCalculator,
            range_calculator_cls=MinMaxRangeCalculator,
            float_range=float_range,
        )

    if float_range == "empty":
        assert spec.float_range == [None, None]
    else:
        # We define spec's float_range to use 'list' in Pydantic so tuple is made into
        # a list.
        assert spec.float_range == list(float_range)


@pytest.mark.parametrize(
    "float_range",
    [None, 0.0, [0.0, 4.0, 5.0], [0.0, True], [5.0, -5.0], [1.0, 5.0], [-3.0, -1.0], [0.0, 0.0]],
)
def test_invalid_float_range_checks(float_range):
    """
    Test various invalid float_ranges.
    """
    with pytest.raises(ValidationError):
        _ = QuantizationSpec(
            dtype=torch.int8,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerTensorGranularity(),
            fake_quantize_cls=_DefaultFakeQuantizeImpl,
            qparam_calculator_cls=StaticQParamsCalculator,
            range_calculator_cls=MinMaxRangeCalculator,
            float_range=float_range,
        )


def test_instantiate_from_dict():
    """
    Basic test checking instantiation from dictionary functionality.
    """
    spec_dict = {
        "dtype": "int8",
        "qscheme": "symmetric",
        "qformulation": "minval",
        "granularity": {"type": "per_block", "axis": 1, "block_size": 10},
        "fake_quantize_cls": "default",
        "qparam_calculator_cls": "moving_average",
        "range_calculator_cls": "minmax",
        "float_range": (0, 10.0),
    }
    spec_from_dict = QuantizationSpec(**spec_dict)
    reference_spec = QuantizationSpec(
        dtype=torch.int8,
        qscheme=QuantizationScheme.SYMMETRIC,
        qformulation=QuantizationFormulation.MINVAL,
        granularity=PerBlockGranularity(axis=1, block_size=10),
        fake_quantize_cls=_DefaultFakeQuantizeImpl,
        qparam_calculator_cls=MovingAverageQParamsCalculator,
        range_calculator_cls=MinMaxRangeCalculator,
        float_range=(0, 10.0),
    )

    for attr in [
        "dtype",
        "qscheme",
        "qformulation",
        "granularity",
        "fake_quantize_cls",
        "qparam_calculator_cls",
        "range_calculator_cls",
        "float_range",
    ]:
        assert getattr(spec_from_dict, attr) == getattr(reference_spec, attr)


@pytest.mark.parametrize(
    "invalid_attr_name, invalid_attr_val",
    [
        ("dtype", "int9"),
        ("qscheme", "symmetricc"),
        ("qformulation", "min_val"),
        ("granularity", "per_tensor"),
        ("fake_quantize_cls", "defaultt"),
        ("qparam_calculator_cls", "defaultt"),
        ("range_calculator_cls", "minmaxx"),
        ("float_range", [1.0, 2.0]),
    ],
)
def test_from_dict_invalid_attrs(invalid_attr_name, invalid_attr_val):
    """
    Test that invalid attributes provided for different spec fields throw
    ValidationErrors.
    """
    good_starting_spec_dict = {
        "dtype": "int8",
        "qscheme": "symmetric",
        "qformulation": "minval",
        "granularity": {"type": "per_block", "axis": 1, "block_size": 10},
        "fake_quantize_cls": "default",
        "qparam_calculator_cls": "moving_average",
        "range_calculator_cls": "minmax",
        "float_range": (0, 10.0),
    }

    # Sanity check for a good starting spec dict
    _ = QuantizationSpec(**good_starting_spec_dict)

    good_starting_spec_dict[invalid_attr_name] = invalid_attr_val
    with pytest.raises(ValidationError):
        _ = QuantizationSpec(**good_starting_spec_dict)


def test_default_values():
    """
    Test that QuantizationSpec can be created with all default values.
    """
    spec = QuantizationSpec()

    # Check that defaults are set correctly
    assert spec.dtype == torch.int8
    assert spec.qscheme == QuantizationScheme.SYMMETRIC
    assert spec.qformulation == QuantizationFormulation.ZP
    assert isinstance(spec.granularity, PerTensorGranularity)
    assert spec.fake_quantize_cls == _DefaultFakeQuantizeImpl
    assert spec.qparam_calculator_cls == _DefaultQParamsCalculator
    assert spec.range_calculator_cls == MinMaxRangeCalculator
    assert spec.float_range == [None, None]


def test_partial_defaults():
    """
    Test that QuantizationSpec can be created with some fields overridden.
    """
    spec = QuantizationSpec(
        dtype=torch.int4,
        granularity=PerChannelGranularity(axis=1),
    )

    # Check overridden values
    assert spec.dtype == torch.int4
    assert isinstance(spec.granularity, PerChannelGranularity)
    assert spec.granularity.axis == 1

    # Check defaults for non-overridden fields
    assert spec.qscheme == QuantizationScheme.SYMMETRIC
    assert spec.qformulation == QuantizationFormulation.ZP
    assert spec.fake_quantize_cls == _DefaultFakeQuantizeImpl
    assert spec.qparam_calculator_cls == _DefaultQParamsCalculator
    assert spec.range_calculator_cls == MinMaxRangeCalculator
    assert spec.float_range == [None, None]


@pytest.mark.parametrize(
    "dtype,scale_dtype,expected_scale_dtype",
    [
        (torch.int8, None, None),
        (torch.int4, None, None),
        (torch.float8_e4m3fn, None, None),
        (torch.float8_e5m2, None, None),
        (
            torch.float8_e4m3fn,
            torch.float8_e8m0fnu,
            torch.float8_e8m0fnu,
        ),
        (
            torch.float8_e4m3fn,
            "float8_e8m0fnu",
            torch.float8_e8m0fnu,
        ),
        (
            torch.float8_e4m3fn,
            "float8_e8m0",
            torch.float8_e8m0fnu,
        ),
        (torch.float4_e2m1fn_x2, None, torch.float8_e8m0fnu),
        (torch.float4_e2m1fn_x2, torch.float8_e8m0fnu, torch.float8_e8m0fnu),
    ],
)
def test_valid_scale_dtype(dtype, scale_dtype, expected_scale_dtype):
    spec = QuantizationSpec(
        dtype=dtype,
        qscheme="symmetric",
        granularity=PerTensorGranularity(),
        scale_dtype=scale_dtype,
    )
    assert spec.scale_dtype == expected_scale_dtype


@pytest.mark.parametrize(
    "dtype,scale_dtype",
    [
        (torch.float8_e4m3fn, torch.float32),
        (torch.int8, torch.float16),
        (torch.float8_e4m3fn, "float8_typo"),
        (torch.float4_e2m1fn_x2, torch.float32),
        (
            torch.int4,
            torch.float8_e8m0fnu,
        ),
    ],
)
def test_invalid_scale_dtype(dtype, scale_dtype):
    with pytest.raises(ValueError):
        QuantizationSpec(
            dtype=dtype,
            qscheme="symmetric",
            granularity=PerChannelGranularity(axis=0),
            scale_dtype=scale_dtype,
        )


@pytest.mark.parametrize(
    "spec",
    [
        QuantizationSpec(),
        QuantizationSpec(
            dtype=torch.int8,
            qscheme=QuantizationScheme.ASYMMETRIC,
            granularity=PerChannelGranularity(axis=0),
        ),
        QuantizationSpec(
            dtype=torch.uint8,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerTensorGranularity(),
        ),
        QuantizationSpec(
            dtype=torch.int4,
            qscheme=QuantizationScheme.SYMMETRIC_WITH_CLIPPING,
            granularity=PerBlockGranularity(axis=1, block_size=32),
        ),
        QuantizationSpec(
            dtype=torch.float8_e4m3fn,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerTensorGranularity(),
        ),
        QuantizationSpec(
            dtype=torch.uint4,
            qscheme=QuantizationScheme.SYMMETRIC,
            qformulation=QuantizationFormulation.MINVAL,
            granularity=PerTensorGranularity(),
        ),
        QuantizationSpec(
            dtype=torch.int4,
            qscheme=QuantizationScheme.ASYMMETRIC,
            qformulation=QuantizationFormulation.ZP,
            granularity=PerChannelGranularity(axis=0),
        ),
    ],
)
def test_model_dump_round_trip(spec):
    """Test that QuantizationSpec round-trips through model_dump."""
    dumped = spec.model_dump()

    # Computed fields should be present in the dump
    assert "n_bits" in dumped
    assert "target_dtype" in dumped
    assert "quant_min" in dumped
    assert "quant_max" in dumped

    # Granularity should include the registry type key
    assert "type" in dumped["granularity"]

    # Round-trip: reconstruct from the dumped dict
    reconstructed = QuantizationSpec(**dumped)
    assert reconstructed == spec
