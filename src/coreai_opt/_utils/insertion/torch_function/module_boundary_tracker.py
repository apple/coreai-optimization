# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tracker for module inputs and output tensors during eager mode compression preparation."""

import weakref
from collections import deque
from dataclasses import dataclass
from typing import Literal, NamedTuple

import torch

from coreai_opt._utils.torch_utils import NamedModule, flatten_tensors_to_list


@dataclass(frozen=True)
class ModuleBoundaryInfo:
    """Information about a tensor at a module boundary."""

    named_module: NamedModule
    index: int


class TensorIdVersion(NamedTuple):
    """NamedTuple representing a tensor object using id and version."""

    id: int
    version: int


class ModuleBoundaryTracker:
    """
    Tracks tensors at module input/output boundaries.

    Uses counter as keys, where counter is a monotonically incrementing integer assigned per unique
    tensor object.
    The counter is stored in a plain dict keyed by object id and version, with a weakref finalizer
    to remove the entry when the tensor is GC'd.

    This composite key gives two correctness guarantees:
    - GC reuse: when a tensor is GC'd its counter entry is removed before the memory can be
      reused by a new tensor, which will receive a fresh counter. So the counter is
      unique even if two tensors share the same id and version at different points in time.
    - In-place mutation: the same tensor object before and after an in-place op has the same
      id but an incremented _version. Since we identify tensors through both id as
      well as version, we still associate a distinct counter for each mutation state.

    A tensor can be an input/output to multiple nested modules, so we track all of them
    to enable module-level specs at any nesting level.
    """

    def __init__(self):
        # Maps a tensor's counter (unique integer ID) to its boundary information.
        # Each entry maps boundary type ("input" / "output") to a list of ModuleBoundaryInfo,
        # because the same tensor can appear at the boundary of multiple nested modules.
        #
        # Example: given a model with nested modules:
        #
        #   class Inner(nn.Module):
        #       def forward(self, x): return x * 2
        #
        #   class Outer(nn.Module):
        #       def __init__(self): self.inner = Inner()
        #       def forward(self, x): return self.inner(x)
        #
        # The tensor `x` passed to Outer.forward is also passed to Inner.forward.
        # So for that tensor's counter, the "input" list would contain two entries:
        #   [ModuleBoundaryInfo(Inner, idx=0), ModuleBoundaryInfo(Outer, idx=0)]
        #
        # A tensor can also appear at different boundary types for different modules.
        # For example, in a sequential model:
        #
        #   class Model(nn.Module):
        #       def __init__(self):
        #           self.linear1 = nn.Linear(10, 10)
        #           self.linear2 = nn.Linear(10, 10)
        #       def forward(self, x): return self.linear2(self.linear1(x))
        #
        # The intermediate tensor (output of linear1) is both the output of linear1
        # and the input of linear2. Its entry would look like:
        #   {
        #       "input":  [ModuleBoundaryInfo(linear2, idx=0)],
        #       "output": [ModuleBoundaryInfo(linear1, idx=0)],
        #   }
        #
        # This allows module-level specs to be applied at any nesting level.
        # Lists are ordered from innermost module to outermost module.
        self.tensor_boundaries: dict[
            int, dict[Literal["input", "output"], deque[ModuleBoundaryInfo]]
        ] = {}

        # Maps object id -> counter value. Entries are removed via weakref finalizer when GC'd.
        self._id_version_to_counter: dict[TensorIdVersion, int] = {}
        self._next_counter: int = 0

    def get_or_assign_counter(self, tensor: torch.Tensor) -> int:
        """
        Return the counter value assigned to this tensor object, assigning a new one if not
        yet seen. The counter identifies object identity; pair it with tensor._version to form
        a key that also distinguishes in-place mutation states.

        Args:
            tensor: The tensor to get or assign a counter for.

        Returns:
            The integer counter value uniquely identifying this tensor object.
        """
        tensor_id_version = TensorIdVersion(id(tensor), tensor._version)
        existing = self._id_version_to_counter.get(tensor_id_version)
        if existing is not None:
            return existing

        counter = self._next_counter
        self._next_counter += 1
        self._id_version_to_counter[tensor_id_version] = counter
        # Remove the entry when the tensor is GC'd so the id can't alias a future tensor.
        weakref.finalize(tensor, self._id_version_to_counter.pop, tensor_id_version, None)
        return counter

    def record_module_boundary_tensors(
        self,
        named_module: NamedModule,
        boundary_tensors: torch.Tensor | tuple | dict,
        boundary_type: Literal["input", "output"],
    ) -> None:
        """Record tensors at module boundary."""
        flattened = flatten_tensors_to_list(boundary_tensors)
        for idx, tensor in enumerate(flattened):
            if isinstance(tensor, torch.Tensor):
                key = self.get_or_assign_counter(tensor)
                if key not in self.tensor_boundaries:
                    self.tensor_boundaries[key] = {"input": deque(), "output": deque()}

                # When dealing with module inputs, these will be added starting from the top
                # most module down to nested children due to the pre-hook hitting the higher
                # level modules first. Insert the ModuleBoundaryInfo by prepending each new
                # entry to the front in order to keep the overall order of the list going from
                # innermost module to outermost module.

                # For module outputs, these entries will naturally come in the
                # order of innermost module to outermost module since the forward hook will
                # trigger when exiting the innermost module first. Thus, unlike module inputs,
                # we can simply append each new entry to the end of the list to maintain the
                # ordering of innermost module to outermost module.
                if boundary_type == "input":
                    self.tensor_boundaries[key][boundary_type].appendleft(
                        ModuleBoundaryInfo(named_module=named_module, index=idx)
                    )
                else:
                    self.tensor_boundaries[key][boundary_type].append(
                        ModuleBoundaryInfo(named_module=named_module, index=idx)
                    )

    def get_module_boundaries_for_tensor(
        self, tensor_counter: int, boundary_type: Literal["input", "output"]
    ) -> list[ModuleBoundaryInfo]:
        """
        For a given tensor identified by its counter, return a list of
        ModuleBoundaryInfo for the boundary_type for the tensor.

        Args:
            tensor_counter: Counter uniquely identifying the tensor object.
            boundary_type: Either "input" or "output" to signal which module boundary to look for.

        Returns:
            A list of ModuleBoundaryInfo for the tensor and boundary type.
        """
        module_boundaries_for_tensor = self.tensor_boundaries.get(tensor_counter)
        if module_boundaries_for_tensor is not None:
            return list(module_boundaries_for_tensor[boundary_type])
        return []
