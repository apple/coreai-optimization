# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests verifying public exports of the coreai_opt.coreai_utils package."""

import inspect

import coreai_opt.coreai_utils as cu
import coreai_opt.coreai_utils.common as cuc
from coreai_opt.coreai_utils import (
    CompressionGranularity,
    DType,
    QScheme,
    quantize_weights,
)


def test_qscheme_exported() -> None:
    """Verify QScheme is re-exported at package root and matches common.QScheme."""
    assert hasattr(cu, "QScheme")
    assert cu.QScheme is cuc.QScheme
    assert hasattr(cu.QScheme, "SYMMETRIC")
    assert hasattr(cu.QScheme, "ASYMMETRIC")


def test_all_public_symbols_exported() -> None:
    """Verify all public symbols are declared in __all__ and resolvable."""
    expected_symbols = {
        "CompressionGranularity",
        "DType",
        "QScheme",
        "palettize_weights",
        "quantize_weights",
        "sparsify_weights",
    }
    assert set(cu.__all__) == expected_symbols
    for name in expected_symbols:
        assert hasattr(cu, name), f"Symbol {name} listed in __all__ but missing from package"


def test_docs_coreai_compression_import_snippet() -> None:
    """Verify the exact import from docs/src/utils/coreai_compression.md#L92-L97 succeeds."""
    assert CompressionGranularity is cuc.CompressionGranularity
    assert DType is cuc.DType
    assert QScheme is cuc.QScheme
    assert callable(quantize_weights)


def test_quantize_weights_signature_default_qscheme() -> None:
    """Verify quantize_weights default parameter aligns with exported QScheme."""
    sig = inspect.signature(cu.quantize_weights)
    assert sig.parameters["qscheme"].default is cu.QScheme.SYMMETRIC
