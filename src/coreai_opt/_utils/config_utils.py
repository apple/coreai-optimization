# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Utilities for config related items"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from enum import Enum, auto
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Constant representing all tensors for an input/output/state spec
ALL_TENSORS = "*"


def get_last_matching_spec(
    identifiers: Iterable[int | str],
    spec_dict: Mapping[int | str, Any],
) -> tuple[Any, bool]:
    """Return the last value in ``spec_dict`` whose key matches any identifier.

    Iterates ``spec_dict`` keys in declaration order, tracking the last match.
    Later-declared keys take precedence over earlier ones. Falls back to the
    ``"*"`` wildcard key if no specific match is found. Warns when multiple
    keys match. Returns ``(value, True)`` on match — value may be ``None`` if
    the entry is explicit-None to disable. Returns ``(None, False)`` if no key
    matched at all.
    """
    identifiers_set = set(identifiers)
    matching_keys: list[int | str] = []
    last_value = None
    found = False
    for key, value in spec_dict.items():
        if key in identifiers_set:
            matching_keys.append(key)
            last_value = value
            found = True
    if found:
        if len(matching_keys) > 1:
            logger.warning(
                "Multiple spec keys matched for identifiers %s against spec keys %s: "
                "%s. Using the last matching key '%s'.",
                list(identifiers_set),
                list(spec_dict.keys()),
                matching_keys,
                matching_keys[-1],
            )
        return last_value, True
    if ALL_TENSORS in spec_dict:
        return spec_dict[ALL_TENSORS], True
    return None, False


def is_yaml_file(file_path: Path) -> bool:
    """
    Returns True if file_path points to a file ending in .yaml or .yml suffix, False
    otherwise.
    """
    return file_path.is_file() and file_path.suffix.lower() in [".yaml", ".yml"]


class ConfigLevel(Enum):
    """
    Enum to specify the config type.

    Enum entries should be defined in order of highest priority to lowest priority.

    - MODULE_NAME: Applied to specific module names (e.g., "layer1.conv")
    - MODULE_TYPE: Applied to specific module types (e.g., all Conv2d)
    - GLOBAL: Applied to all modules
    """

    MODULE_NAME = auto()
    MODULE_TYPE = auto()
    GLOBAL = auto()

    @classmethod
    def priority_order(cls) -> list[ConfigLevel]:
        """Return config levels in priority order (highest to lowest)."""
        return list(cls)
