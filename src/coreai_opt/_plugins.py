# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Discovery of extension packages that adjust what ``coreai_opt`` supports.

Each extension can call some initialization code, in its own ``pyproject.toml``, which
will change ``coreai_opt`` default behavior.

    [project.entry-points."coreai_opt.plugins"]
    my_extension = "my_extension._register:register_all"

The motivating case is block-wise activation quantization, which ``coreai_opt`` reports as
unsupported on its own. An extension that registers an activation export handler for that
granularity turns it on just by being installed, so code written against a package that
already supports it keeps working after switching to ``coreai_opt`` -- the user imports no
extension and adds no registration call of their own.

``coreai_opt`` advertises no entry point itself, so with no extension installed this is a
no-op.
"""

from __future__ import annotations

import warnings
from importlib.metadata import entry_points

PLUGIN_GROUP = "coreai_opt.plugins"


def load_plugins() -> None:
    """Call every registration callable advertised in the ``coreai_opt.plugins`` group.

    A plugin that fails to load warns and is skipped, so one broken extension cannot make
    ``import coreai_opt`` fail.
    """
    for entry_point in entry_points(group=PLUGIN_GROUP):
        # Recovering, not just re-raising: the point is to skip this plugin and keep
        # loading the rest, so the handler has to sit inside the loop.
        try:
            register = entry_point.load()
            register()
        except Exception as error:  # noqa: BLE001
            msg = (
                f"The {entry_point.name!r} plugin ({entry_point.value}) failed to load: "
                f"{error!r}. coreai_opt keeps its defaults, so whatever that plugin adds "
                f"support for will be reported as unsupported."
            )
            warnings.warn(msg, RuntimeWarning, stacklevel=2)
