# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Record which parameter each fake-palettize module palettizes.

``_FakePalettizeImplBase`` warns and disables itself from inside ``forward``, where a
module cannot know its own position in the model. The pass here runs during
``prepare()``, before centroids are computed, so the warning can name the weight.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.utils.parametrize as P

from coreai_opt.palettization.spec.fake_palettize import _FakePalettizeImplBase


def record_weight_source_names(model: nn.Module) -> None:
    """Stamp fake-palettize modules with the FQN of the parameter they palettize.

    Args:
        model (nn.Module): The prepared model.
    """
    for module_name, module in model.named_modules(remove_duplicate=True):
        if not P.is_parametrized(module):
            continue
        # A fake palettize lives in the ParametrizationList of the parameter it
        # compresses.
        for param_name, parametrizations in module.parametrizations.items():
            for fake_palett in parametrizations:
                if isinstance(fake_palett, _FakePalettizeImplBase):
                    fake_palett.set_source_name(module_name, param_name)
