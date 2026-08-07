# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Record which model tensor each fake-palettize module palettizes.

``_FakePalettizeImplBase`` warns and disables itself from inside ``forward``
when a tensor is incompatible with the configured granularity or cluster
dimension, but a module cannot discover its own position in the model at that
point. The pass here runs during ``prepare()``, before centroids are computed,
and stamps each fake-palettize module with the FQN of the parameter it
palettizes so the warning can name the offending weight.

Palettization targets weights only and is eager-only, so a single walk over
``named_modules()`` covers every case.
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.utils.parametrize as P

from coreai_opt.palettization.spec.fake_palettize import _FakePalettizeImplBase


def record_weight_source_names(model: nn.Module) -> None:
    """Stamp fake-palettize modules with the FQN of the parameter they palettize.

    Fake-palettize modules live in the ``ParametrizationList`` registered for the
    parameter they compress, so the owning module name and the parameter name
    both come straight from ``named_modules()``.

    Args:
        model (nn.Module): The prepared model.
    """
    for module_name, module in model.named_modules(remove_duplicate=True):
        if not P.is_parametrized(module):
            continue
        for param_name, parametrizations in module.parametrizations.items():
            for fake_palett in parametrizations:
                if isinstance(fake_palett, _FakePalettizeImplBase):
                    fake_palett.set_source_name(module_name, param_name)
