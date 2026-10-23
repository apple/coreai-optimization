# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from abc import abstractmethod

import torch

from coreai_opt.base_model_compressor import _BaseModelCompressor
from coreai_opt.quantization.spec.fake_quantize import FakeQuantizeImplBase
from coreai_opt.quantization.spec.qparams_calculator import StatefulQParamsCalculatorBase


class _BaseQuantizer(_BaseModelCompressor):
    @abstractmethod
    def _get_fake_quantize_modules(self) -> dict[str, list[FakeQuantizeImplBase]]:
        """Return a mapping of module name to its fake quantize modules.

        Collects all fake quantization modules (weight and activation) in the
        prepared model and groups them by the original module name they belong
        to.

        Returns:
            Dict mapping module name to the list of fake quantize module
            instances.
        """
        ...

    def _uninitialized_compression_modules(self, model: torch.nn.Module) -> list[str]:
        """Return the fake quantizers whose stored qparams are still empty."""
        return [
            name
            for name, module in model.named_modules()
            if isinstance(module, FakeQuantizeImplBase)
            and isinstance(module.qparams_calculator, StatefulQParamsCalculatorBase)
            and module.qparams_calculator.scale.numel() == 0
        ]
