# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Registry for pruning-supported operations."""

import torch.nn.functional as F

from coreai_opt._utils.insertion.torch_function import (
    BaseSupportedOpsRegistry as _BaseSupportedOpsRegistry,
)


class _PrunerSupportedOpsRegistry(_BaseSupportedOpsRegistry):
    """Registry of operations that support pruning."""


@_PrunerSupportedOpsRegistry.register("linear")
class _LinearSupport:
    ops = [F.linear]


@_PrunerSupportedOpsRegistry.register("conv1d")
class _Conv1dSupport:
    ops = [F.conv1d]


@_PrunerSupportedOpsRegistry.register("conv2d")
class _Conv2dSupport:
    ops = [F.conv2d]


@_PrunerSupportedOpsRegistry.register("conv3d")
class _Conv3dSupport:
    ops = [F.conv3d]


@_PrunerSupportedOpsRegistry.register("conv_transpose1d")
class _ConvTranspose1dSupport:
    ops = [F.conv_transpose1d]


@_PrunerSupportedOpsRegistry.register("conv_transpose2d")
class _ConvTranspose2dSupport:
    ops = [F.conv_transpose2d]


@_PrunerSupportedOpsRegistry.register("conv_transpose3d")
class _ConvTranspose3dSupport:
    ops = [F.conv_transpose3d]


@_PrunerSupportedOpsRegistry.register("multi_head_attention")
class _MultiHeadAttentionSupport:
    ops = [F.multi_head_attention_forward]
