# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""
Version information for coreai_opt package.

``main`` carries the next planned release with a ``.dev0`` suffix (e.g.
``0.2.2.dev0``). A clean release is cut on a release branch by dropping the
``.dev`` suffix and tagging. Keep this a plain string literal so setuptools can
read it via ``ast.literal_eval`` at build time without importing the package
(which would pull in torch).
"""

__version__ = "0.2.2.dev0"
