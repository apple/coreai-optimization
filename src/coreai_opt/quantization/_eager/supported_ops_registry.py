# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Registry for eager mode quantization operations."""

import torch
import torch.nn.functional as F

from coreai_opt._utils.insertion.torch_function import (
    BaseSupportedOpsRegistry,
)


class EagerQuantizerSupportedOpsRegistry(BaseSupportedOpsRegistry):
    """Registry for eager mode quantization operations.

    This registry contains all operations supported by eager quantization,
    including convolutions, linear layers, and pooling operations.
    """


# Register convolution operations
@EagerQuantizerSupportedOpsRegistry.register("conv1d")
class Conv1dQuantizationSupport:
    ops = [F.conv1d]


@EagerQuantizerSupportedOpsRegistry.register("conv2d")
class Conv2dQuantizationSupport:
    ops = [F.conv2d]


@EagerQuantizerSupportedOpsRegistry.register("conv3d")
class Conv3dQuantizationSupport:
    ops = [F.conv3d]


# Register transposed convolution operations
@EagerQuantizerSupportedOpsRegistry.register("conv_transpose1d")
class ConvTranspose1dQuantizationSupport:
    ops = [F.conv_transpose1d]


@EagerQuantizerSupportedOpsRegistry.register("conv_transpose2d")
class ConvTranspose2dQuantizationSupport:
    ops = [F.conv_transpose2d]


@EagerQuantizerSupportedOpsRegistry.register("conv_transpose3d")
class ConvTranspose3dQuantizationSupport:
    ops = [F.conv_transpose3d]


# Register linear operations
@EagerQuantizerSupportedOpsRegistry.register("linear")
class LinearQuantizationSupport:
    ops = [F.linear]


# Register embedding operations
@EagerQuantizerSupportedOpsRegistry.register("embedding")
class EmbeddingQuantizationSupport:
    ops = [F.embedding]


# Register pooling operations (no weight parameter)
@EagerQuantizerSupportedOpsRegistry.register("max_pool2d")
class MaxPool2dQuantizationSupport:
    ops = [F.max_pool2d]


@EagerQuantizerSupportedOpsRegistry.register("adaptive_avg_pool2d")
class AdaptiveAvgPool2dQuantizationSupport:
    ops = [F.adaptive_avg_pool2d]


@EagerQuantizerSupportedOpsRegistry.register("add")
class AddQuantizationSupport:
    ops = [torch.add, torch._C.TensorBase.add, torch._C.TensorBase.add_]


@EagerQuantizerSupportedOpsRegistry.register("matmul")
class MatMulQuantizationSupport:
    ops = [torch.matmul, torch._C.TensorBase.matmul]


@EagerQuantizerSupportedOpsRegistry.register("mul")
class MulQuantizationSupport:
    ops = [torch.mul, torch._C.TensorBase.mul, torch._C.TensorBase.mul_]


@EagerQuantizerSupportedOpsRegistry.register("sub")
class SubQuantizationSupport:
    ops = [torch.sub, torch._C.TensorBase.sub, torch._C.TensorBase.sub_]
