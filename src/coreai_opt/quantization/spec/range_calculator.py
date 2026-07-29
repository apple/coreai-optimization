# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from abc import abstractmethod

import torch
import torch.nn as nn
from torchao.quantization.quant_primitives import _get_reduction_params

from coreai_opt._utils.registry_utils import ClassRegistryMixin as _ClassRegistryMixin

from .granularity import QuantizationGranularity


class RangeCalculatorBase(_ClassRegistryMixin, nn.Module):
    """
    Base class and registry for classes used to compute the range
    of a given tensor.
    """

    def __init__(self, granularity: QuantizationGranularity, **kwargs):
        super().__init__()
        self.granularity = granularity

    def _reshape_min_max(self, range_tensor: torch.Tensor, input_shape: torch.Size):
        """
        Reshape range_tensor to have the same number of dimensions as input shape,
        taking block size into account.
        """
        block_size_list = self.granularity.get_block_size(input_shape)

        # While reducing, each dimension with block size other than 1 or the original
        # dimension size will be split into 2 dimensions of num_blocks and block_size.
        # At the end, min and max val tensors should be reshaped back to combine split
        # dimensions into single dimensions again.
        # For example, given a tensor of shape [1, 10, 8, 8] with block size 2 and
        # axis 1, shape_for_reduction would come out to be [1, 5, 2, 8, 8].
        # Post-reduction, the min/max tensors would have shape [1, 5, 1, 1, 1]. To
        # align to the original tensor with 4 dimensions, we need to combine axes 1 and
        # 2 to get [1, 5, 1, 1].
        # In the end, each dimension in scale should have size equal to the number of
        # blocks for that dimension.
        range_tensor_shape = [input_shape[i] // block_size_list[i] for i in range(len(input_shape))]
        return range_tensor.reshape(range_tensor_shape)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute range statistics on an input and return the min/max bounds.

        Calls _generate_min_max to compute range statistics and validates that
        the returned min/max shapes match the original tensor number of dimensions.

        Args:
            x (:py:class:`torch.Tensor`): Tensor to compute range statistics upon.
        """
        min_tensor, max_tensor = self._generate_min_max(x)
        min_tensor = self._reshape_min_max(min_tensor, x.shape)
        max_tensor = self._reshape_min_max(max_tensor, x.shape)
        return min_tensor, max_tensor

    @abstractmethod
    def _generate_min_max(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the lower and upper bound of the range.

        Args:
            x (:py:class:`torch.Tensor`): Tensor to compute range statistics upon.
        """
        pass


@RangeCalculatorBase.register("minmax")
class MinMaxRangeCalculator(RangeCalculatorBase):
    """
    Range calculator that computes the range of a given tensor as the min and max
    values of the tensor.
    """

    def _generate_min_max(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        block_size_list = self.granularity.get_block_size(tensor.shape)
        shape_for_reduction, reduction_dims = _get_reduction_params(block_size_list, tensor.size())

        # If tensor is already the shape required, no minmaxing is needed.
        if len(reduction_dims) == 0:
            error_msg = (
                f"With no reduction dims, tensor shape {tensor.shape} is "
                f"expected to match shape_for_reduction {shape_for_reduction}."
            )
            assert list(tensor.shape) == shape_for_reduction, error_msg
            return tensor, tensor

        tensor = tensor.view(shape_for_reduction)
        min_val = torch.amin(tensor, dim=reduction_dims, keepdim=True)
        max_val = torch.amax(tensor, dim=reduction_dims, keepdim=True)
        return min_val, max_val
