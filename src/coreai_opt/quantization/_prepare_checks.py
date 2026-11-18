# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable

from torch import nn

PrepareCheck = Callable[[nn.Module], None]

_PREPARE_CHECKS: list[PrepareCheck] = []


def register_prepare_check(check: PrepareCheck) -> None:
    """Run ``check`` on every model ``Quantizer.prepare`` prepares from now on.

    The registry is process-global. Registering the same check twice has no effect.

    Args:
        check (PrepareCheck): Called with the prepared model; raises to reject it.
    """
    if check not in _PREPARE_CHECKS:
        _PREPARE_CHECKS.append(check)


def run_prepare_checks(model: nn.Module) -> None:
    """Run every registered check on the prepared ``model``, in registration order.

    In eager mode ``prepare`` has already modified ``model`` in place, so a rejected model must be
    prepared again from the original.

    Args:
        model (nn.Module): The prepared model (an ``fx.GraphModule`` in graph mode).

    Raises:
        Exception: Whatever the first failing check raises.
    """
    for check in _PREPARE_CHECKS:
        check(model)
