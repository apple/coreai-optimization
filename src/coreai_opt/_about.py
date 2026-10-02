# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Version information for coreai_opt package.

``latest_released_version`` is the OSS release this tree is built from (e.g.
``"0.2.1"``) — on ``main`` the most recently cut release, on a release branch the
release that branch produces. ``__version__`` is the release ``main`` is working toward
(e.g. ``"0.2.2.dev0"``); on a release branch it equals ``latest_released_version``.
Both are set together when a release branch is cut.

Keep ``__version__`` a plain string, not an expression, so setuptools can
read it at build time without importing the package (which would pull in a
lot of dependencies).
"""

latest_released_version = "0.3.0"
__version__ = "0.3.1.dev0"
