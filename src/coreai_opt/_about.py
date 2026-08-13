# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Version information for coreai_opt package.

``latest_released_version`` is the OSS release this tree is anchored to (e.g.
``"0.2.1"``) — on ``main`` the most recently cut release, on a release branch
the release that branch produces.
``__version__`` is the release ``main`` is working toward (e.g.
``"0.2.2.dev0"``). Both are set together when a release branch is cut. A
pre-commit hook checks that ``__version__`` raises exactly one of
``latest_released_version``'s numbers by one, zeroes the rest, and ends in
``.dev0`` — so a minor or major can be declared, not just a patch.

Keep ``__version__`` a plain string, not an expression, so setuptools can
read it at build time without importing the package (which would pull in a
lot of dependencies).
"""

latest_released_version = "0.2.1"
__version__ = "0.2.2.dev0"
