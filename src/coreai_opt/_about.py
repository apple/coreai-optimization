# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Version information for coreai_opt package.

``latest_released_version`` is the last tagged release (e.g. ``"0.2.1"``).
Hard-code it right after tagging a new release. ``__version__`` is
computed from it: add one to its last number and add ``.dev0`` at the end
(e.g. ``"0.2.2.dev0"``) — this is the release ``main`` is working toward. A
pre-commit hook checks that ``__version__`` matches this rule and that
``latest_released_version`` matches the repo's latest release tag.

Keep ``__version__`` a plain string, not an expression, so setuptools can
read it at build time without importing the package (which would pull in a
lot of dependencies).
"""

latest_released_version = "0.2.1"
__version__ = "0.2.2.dev0"
