# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Errors shared across compression domains (quantization, pruning, ...)."""


class _BlockSizeMismatchError(ValueError):
    """Raised when a tensor dimension is not divisible by the block size."""
