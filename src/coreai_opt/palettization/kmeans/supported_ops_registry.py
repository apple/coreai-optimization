# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Registry for KMeans palettization operations."""

from collections.abc import Callable
from typing import Any, TypeVar, cast

import torch.nn.functional as F

from coreai_opt._utils.insertion.torch_function import (
    BaseSupportedOpsRegistry as _BaseSupportedOpsRegistry,
)
from coreai_opt.palettization.kmeans.kmeans_support_mixins import (
    _ConvPalettizationMixin,
    _ConvTransposePalettizationMixin,
    _LinearPalettizationMixin,
    _PalettizationSupportMixin,
)

# Use the same TypeVar as the parent class for type compatibility
_T = TypeVar("_T")


class _KMeansPalettizerSupportedOpsRegistry(_BaseSupportedOpsRegistry):
    """
    Registry for KMeans palettization operations.

    This registry contains only operations that support palettization,
    i.e., those with _PalettizationSupportMixin implementations.
    """

    @classmethod
    def register(cls, key: Any) -> Callable[[_T], _T]:
        """
        A decorator that validates the class inherits from _PalettizationSupportMixin
        and registers it.

        Raises:
            TypeError: If the class does not inherit from _PalettizationSupportMixin.
        """

        def inner_wrapper(wrapped_class: _T) -> _T:
            # Runtime validation - ensure registered class subclasses
            # _PalettizationSupportMixin
            if not issubclass(wrapped_class, _PalettizationSupportMixin):
                raise TypeError(
                    f"Class {wrapped_class.__name__} must inherit from "
                    f"_PalettizationSupportMixin to be registered in "
                    f"{cls.__name__}. Current MRO: "
                    f"{[c.__name__ for c in wrapped_class.__mro__]}"
                )

            # Call parent's register method
            result = super(_KMeansPalettizerSupportedOpsRegistry, cls).register(key)(wrapped_class)
            return cast(_T, result)

        return inner_wrapper


# Register operations that support palettization
@_KMeansPalettizerSupportedOpsRegistry.register("conv1d")
class _Conv1dSupport(_ConvPalettizationMixin):
    ops = [F.conv1d]


@_KMeansPalettizerSupportedOpsRegistry.register("conv2d")
class _Conv2dSupport(_ConvPalettizationMixin):
    ops = [F.conv2d]


@_KMeansPalettizerSupportedOpsRegistry.register("conv3d")
class _Conv3dSupport(_ConvPalettizationMixin):
    ops = [F.conv3d]


@_KMeansPalettizerSupportedOpsRegistry.register("conv_transpose1d")
class _ConvTranspose1dSupport(_ConvTransposePalettizationMixin):
    ops = [F.conv_transpose1d]


@_KMeansPalettizerSupportedOpsRegistry.register("conv_transpose2d")
class _ConvTranspose2dSupport(_ConvTransposePalettizationMixin):
    ops = [F.conv_transpose2d]


@_KMeansPalettizerSupportedOpsRegistry.register("conv_transpose3d")
class _ConvTranspose3dSupport(_ConvTransposePalettizationMixin):
    ops = [F.conv_transpose3d]


@_KMeansPalettizerSupportedOpsRegistry.register("linear")
class _LinearSupport(_LinearPalettizationMixin):
    ops = [F.linear]


@_KMeansPalettizerSupportedOpsRegistry.register("multi_head_attention_forward")
class _MultiHeadAttentionSupport(_LinearPalettizationMixin):
    ops = [F.multi_head_attention_forward]
