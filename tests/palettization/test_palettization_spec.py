# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import pytest
import torch
import yaml
from pydantic import ValidationError

from coreai_opt.palettization import PalettizationSpec
from coreai_opt.palettization.spec import (
    PerGroupedChannelGranularity,
    PerTensorGranularity,
    default_weight_palettization_spec,
)
from coreai_opt.palettization.spec.spec import _SUPPORTED_LUT_DTYPES
from coreai_opt.quantization.spec import (
    PerChannelGranularity as QuantPerChannelGranularity,
    QuantizationFormulation,
    QuantizationScheme,
    QuantizationSpec,
)


def test_palettization_spec_basic():
    """Test basic PalettizationSpec creation with default values."""
    spec = PalettizationSpec()
    assert spec.n_bits == 4
    assert spec.lut_qspec is None
    assert isinstance(spec.granularity, PerTensorGranularity)
    assert spec.cluster_dim == 1
    assert spec.enable_per_channel_scale is False


def test_palettization_spec_custom():
    """Test PalettizationSpec creation with custom values."""
    spec = PalettizationSpec(
        n_bits=8,
        lut_qspec=QuantizationSpec(dtype=torch.int8, qscheme=QuantizationScheme.SYMMETRIC),
        granularity=PerGroupedChannelGranularity(axis=0, group_size=4),
        cluster_dim=1,
        enable_per_channel_scale=True,
    )
    assert spec.n_bits == 8
    assert spec.lut_qspec is not None
    assert spec.lut_qspec.dtype == torch.int8
    assert spec.lut_qspec.qscheme == QuantizationScheme.SYMMETRIC
    assert isinstance(spec.granularity, PerGroupedChannelGranularity)
    assert spec.granularity.axis == 0
    assert spec.granularity.group_size == 4
    assert spec.cluster_dim == 1
    assert spec.enable_per_channel_scale is True


def test_default_weight_palettization_spec():
    """Test default weight palettization spec function."""
    spec = default_weight_palettization_spec()
    assert spec.n_bits == 4
    assert spec.lut_qspec is None
    assert isinstance(spec.granularity, PerTensorGranularity)
    assert spec.cluster_dim == 1
    assert spec.enable_per_channel_scale is False


def test_palettization_spec_from_dict_per_channel():
    """Test creating PalettizationSpec from dictionary with per-channel granularity."""
    spec_dict = {
        "n_bits": 2,
        "granularity": {"type": "per_grouped_channel", "axis": 1, "group_size": 8},
    }
    spec = PalettizationSpec(**spec_dict)
    assert spec.n_bits == 2
    assert isinstance(spec.granularity, PerGroupedChannelGranularity)
    assert spec.granularity.axis == 1
    assert spec.granularity.group_size == 8


def test_palettization_spec_from_dict_with_lut_qspec():
    """Test creating PalettizationSpec from dictionary with lut_qspec."""
    spec_dict = {
        "n_bits": 6,
        "lut_qspec": QuantizationSpec(dtype=torch.int8, qscheme=QuantizationScheme.ASYMMETRIC),
        "granularity": {"type": "per_tensor"},
        "cluster_dim": 1,
        "enable_per_channel_scale": True,
    }
    spec = PalettizationSpec(**spec_dict)
    assert spec.n_bits == 6
    assert spec.lut_qspec is not None
    assert spec.lut_qspec.dtype == torch.int8
    assert spec.lut_qspec.qscheme == QuantizationScheme.ASYMMETRIC
    assert isinstance(spec.granularity, PerTensorGranularity)
    assert spec.cluster_dim == 1
    assert spec.enable_per_channel_scale is True


def test_yaml_config():
    """Test creating PalettizationSpec from YAML configuration."""
    yaml_content = """
n_bits: 3
lut_qspec:
    dtype: int8
    qscheme: asymmetric
granularity:
  type: per_grouped_channel
  axis: 0
  group_size: 16
cluster_dim: 1
enable_per_channel_scale: true
"""
    config_dict = yaml.safe_load(yaml_content)
    spec = PalettizationSpec(**config_dict)
    assert spec.n_bits == 3
    assert isinstance(spec.lut_qspec, QuantizationSpec)
    assert spec.lut_qspec.dtype == torch.int8
    assert spec.lut_qspec.qscheme == QuantizationScheme.ASYMMETRIC
    assert isinstance(spec.granularity, PerGroupedChannelGranularity)
    assert spec.granularity.axis == 0
    assert spec.granularity.group_size == 16
    assert spec.cluster_dim == 1
    assert spec.enable_per_channel_scale is True


@pytest.mark.parametrize(
    "invalid_field,invalid_value",
    [
        ("n_bits", -1),
        ("n_bits", 0),
        ("cluster_dim", -1),
        ("enable_per_channel_scale", "not_a_bool"),
    ],
)
def test_palettization_spec_invalid_values(invalid_field, invalid_value):
    """Test that invalid values raise appropriate errors."""
    spec_dict = {
        "n_bits": 4,
        "granularity": PerTensorGranularity(),
        "cluster_dim": 1,
        "enable_per_channel_scale": False,
    }
    spec_dict[invalid_field] = invalid_value

    with pytest.raises((ValidationError, ValueError, TypeError)):
        PalettizationSpec(**spec_dict)


@pytest.mark.parametrize("group_size", [-1, 0])
def test_per_grouped_channel_rejects_non_positive_group_size(group_size):
    """Test that a non-positive group_size raises at construction."""
    with pytest.raises(ValidationError):
        PerGroupedChannelGranularity(axis=0, group_size=group_size)


def test_model_dump():
    """Test model serialization."""
    lut_qspec = QuantizationSpec(dtype=torch.uint8, qscheme=QuantizationScheme.ASYMMETRIC)
    spec = PalettizationSpec(
        n_bits=8,
        lut_qspec=lut_qspec,
        granularity=PerGroupedChannelGranularity(axis=1, group_size=4),
        cluster_dim=2,
        enable_per_channel_scale=False,
    )

    # serialize
    dumped = spec.model_dump()

    assert dumped["n_bits"] == spec.n_bits
    assert dumped["granularity"]["type"] == "per_grouped_channel"
    assert dumped["granularity"]["axis"] == spec.granularity.axis
    assert dumped["granularity"]["group_size"] == spec.granularity.group_size
    assert dumped["enable_per_channel_scale"] == spec.enable_per_channel_scale
    assert dumped["cluster_dim"] == spec.cluster_dim

    # lut_qspec should be serialized as a dict
    assert isinstance(dumped["lut_qspec"], dict)

    # deserialize
    assert PalettizationSpec(**dumped) == spec


def test_spec_immutability():
    """Test that PalettizationSpec is immutable after creation."""
    spec = PalettizationSpec(n_bits=4)

    # Test that we can't modify fields after creation
    with pytest.raises((AttributeError, ValidationError)):
        spec.n_bits = 8


def test_compression_type():
    """Test that compression type is properly set."""
    spec = PalettizationSpec()
    # Access private attribute for testing
    assert hasattr(spec, "_compression_type")
    print(spec._compression_type.value)


def test_enable_per_channel_scale_with_cluster_dim():
    """Test that enable_per_channel_scale works with any cluster_dim."""
    # cluster_dim = 1
    spec = PalettizationSpec(cluster_dim=1, enable_per_channel_scale=True)
    assert spec.enable_per_channel_scale is True
    assert spec.cluster_dim == 1

    # cluster_dim > 1 with enable_per_channel_scale=True
    spec = PalettizationSpec(cluster_dim=2, enable_per_channel_scale=True)
    assert spec.enable_per_channel_scale is True
    assert spec.cluster_dim == 2

    # cluster_dim > 1 with enable_per_channel_scale=False
    spec = PalettizationSpec(cluster_dim=2, enable_per_channel_scale=False)
    assert spec.cluster_dim == 2
    assert spec.enable_per_channel_scale is False


def test_lut_qspec_validation():
    """Test validation of lut_qspec configurations."""
    # Valid: lut_qspec is None
    spec = PalettizationSpec(lut_qspec=None)
    assert spec.lut_qspec is None

    # Valid: all supported dtype/qscheme combinations
    # FP8 dtypes only support symmetric; integer dtypes support both.
    for dtype in _SUPPORTED_LUT_DTYPES:
        spec = PalettizationSpec(
            lut_qspec=QuantizationSpec(dtype=dtype, qscheme=QuantizationScheme.SYMMETRIC)
        )
        assert spec.lut_qspec.dtype == dtype
        if not dtype.is_floating_point:
            spec = PalettizationSpec(
                lut_qspec=QuantizationSpec(dtype=dtype, qscheme=QuantizationScheme.ASYMMETRIC)
            )
            assert spec.lut_qspec.dtype == dtype

    # Invalid: FP8 dtypes with asymmetric quantization
    for fp8_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        with pytest.raises(ValidationError, match="symmetric"):
            PalettizationSpec(
                lut_qspec=QuantizationSpec(dtype=fp8_dtype, qscheme=QuantizationScheme.ASYMMETRIC)
            )

    # Invalid: unsupported dtype (int4 is valid for QuantizationSpec but not for LUT)
    with pytest.raises(ValidationError, match="lut_qspec.dtype must be one of"):
        PalettizationSpec(
            lut_qspec=QuantizationSpec(dtype=torch.int4, qscheme=QuantizationScheme.SYMMETRIC)
        )

    # Invalid: non-PerTensor granularity
    with pytest.raises(ValidationError, match="lut_qspec.granularity must be PerTensorGranularity"):
        PalettizationSpec(
            lut_qspec=QuantizationSpec(
                dtype=torch.int8,
                qscheme=QuantizationScheme.SYMMETRIC,
                granularity=QuantPerChannelGranularity(axis=0),
            )
        )

    # Invalid: MINVAL qformulation (palettization export drops minval)
    with pytest.raises(ValidationError, match="qformulation=MINVAL is not supported"):
        PalettizationSpec(
            lut_qspec=QuantizationSpec(
                dtype=torch.int8,
                qscheme=QuantizationScheme.ASYMMETRIC,
                qformulation=QuantizationFormulation.MINVAL,
            )
        )


@pytest.mark.parametrize(
    "granularity", [PerTensorGranularity(), PerGroupedChannelGranularity(axis=1, group_size=8)]
)
def test_model_dump_preserve_objects(granularity):
    """Test model_dump_preserve_objects preserves Pydantic BaseModel instances."""

    lut_qspec = QuantizationSpec(dtype=torch.int8, qscheme=QuantizationScheme.SYMMETRIC)
    spec = PalettizationSpec(
        n_bits=4,
        lut_qspec=lut_qspec,
        granularity=granularity,
        cluster_dim=1,
        enable_per_channel_scale=True,
    )

    result = spec.model_dump_preserve_objects()

    # Verify non-Pydantic fields are serialized normally
    assert result["n_bits"] == 4
    assert result["cluster_dim"] == 1
    assert result["enable_per_channel_scale"] is True

    # Verify Pydantic BaseModel fields are preserved as objects
    assert isinstance(result["granularity"], granularity.__class__)
    assert result["granularity"] is spec.granularity  # Same object reference

    # lut_qspec is a Pydantic BaseModel, so it should be preserved as object
    assert isinstance(result["lut_qspec"], QuantizationSpec)
    assert result["lut_qspec"] is spec.lut_qspec

    # Compare with normal model_dump to show the difference
    normal_dump = spec.model_dump()
    assert isinstance(normal_dump["granularity"], dict)  # Serialized to dict
    assert normal_dump["granularity"] == granularity.model_dump()
