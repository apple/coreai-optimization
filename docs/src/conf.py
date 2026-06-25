# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-Clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import enum
import functools
import inspect
import pydoc
import re
import sys
import textwrap
from pathlib import Path

from sphinx.application import Sphinx

# Resolve two paths so this conf.py works in both build contexts:
#   - Internal-repo testing: built from <repo>/external/docs/src/
#   - OSS (after export):    built from <oss-root>/docs/src/
#
# In both contexts, conf.py sits at <somewhere>/docs/src/conf.py and the
# scripts directory lives at <somewhere>/docs/scripts/. We add <somewhere>/docs/
# to sys.path so the autodoc/mermaid extensions can be imported as
# ``scripts.generate_api_index`` and ``scripts.mermaid_compat``.
_docs_dir = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(_docs_dir))

# Find the package root by walking up until we hit ``src/coreai_opt``. In both
# layouts that root is the directory immediately above docs/: the OSS root after
# export, or ``external/`` in the internal repo.
_search = _docs_dir
while _search != _search.parent:
    if (_search / "src" / "coreai_opt").exists():
        coreai_opt_root = _search
        break
    _search = _search.parent
else:
    msg = "Could not locate src/coreai_opt from external/docs/src/conf.py"
    raise RuntimeError(msg)
sys.path.insert(0, str(coreai_opt_root))

from scripts.generate_api_index import generate_api_index  # noqa: E402

# -- Project information -----------------------------------------------------

project = "CoreAI-Opt"
version = "main"
project_copyright = "2026, Apple, Inc. All rights reserved"
author = "Apple CoreAI-Opt Team"
release = "main"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_parser",
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.autodoc_pydantic",
    "sphinx_copybutton",
    "sphinx_llm.txt",
    # Order between mermaid extensions doesn't matter — see mermaid_compat.py docstring.
    "sphinxcontrib.mermaid",
    "scripts.mermaid_compat",
]

# -- Autodoc / Autosummary configuration -------------------------------------

autosummary_generate = True

# Extra context exposed to autosummary Jinja templates. ``has_presets`` gates
# the "Presets" section in the class-page templates — classes without presets
# get ``False`` and no extra rendering. The lambda defers name resolution
# since ``_has_presets`` is defined later in this file.
autosummary_context = {
    "has_presets": lambda fullname: _has_presets(fullname),
}

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "inherited-members": False,
    "exclude-members": "REGISTRY, model_config, model_fields, model_computed_fields",
}

# Move type annotations from signatures into the description body
autodoc_typehints = "description"

# Group members by type (classes, methods, attributes) for better readability
autodoc_member_order = "groupwise"

# -- Autodoc-Pydantic configuration ------------------------------------------

# Show Pydantic models as clean class docs instead of raw __init__ signatures
autodoc_pydantic_model_show_config_summary = False
autodoc_pydantic_model_show_validator_summary = False
autodoc_pydantic_model_show_validator_members = False
autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_model_show_json = False
autodoc_pydantic_field_list_validators = False
autodoc_pydantic_field_show_constraints = False
autodoc_pydantic_field_show_alias = False
autodoc_pydantic_model_signature_prefix = "class"
autodoc_pydantic_model_undoc_members = False
autodoc_pydantic_model_members = True
autodoc_pydantic_settings_show_json = False
autodoc_pydantic_settings_show_config_summary = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", ".DS_Store", "README.md"]

# Suppress duplicate object warnings from enum members appearing on multiple pages.
# These occur because ExportBackend and ExecutionMode enum members are documented
# both on their own page and referenced from other pages.
suppress_warnings = ["app.add_object"]

# The suffix(es) of source filenames.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- MyST-Parser configuration -----------------------------------------------

# MyST extensions setting
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
    "attrs_inline",
]

# Enable Mermaid diagrams in fenced code blocks
myst_fence_as_directive = ["mermaid"]

# Render Mermaid diagrams to SVG at build time via mermaid-cli (mmdc),
# so the published site serves static SVGs without client-side JavaScript.
# The white SVG background hardcoded by mermaid is stripped post-build by the
# `scripts.mermaid_compat` extension so diagrams blend with the page theme.
mermaid_output_format = "svg"
mermaid_cmd = "mmdc"

myst_heading_anchors = 4

# -- nbsphinx configuration -------------------------------------------------

# The tutorial notebooks ship with stored outputs, so nbsphinx renders them
# as-is rather than re-running the cells (which would require torch + training).
nbsphinx_execute = "never"
nbsphinx_widgets_path = ""


# -- Napoleon settings -------------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- llms.txt configuration --------------------------------------------------
# Reference: https://github.com/NVIDIA/sphinx-llm

llms_txt_description = (
    "CoreAI-Opt is an Apple library for model compression, quantization, and related optimizations."
)
llms_txt_build_parallel = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.
html_theme = "shibuya"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Add a small JS file that inserts a zero-width space after each `.` in
# sidebar entries so long dotted FQNs wrap at segment boundaries, not
# mid-word. See _static/sidebar-wrap.js.
# copy-page-button.js is the runtime companion for the copy-page-button
# template override at _templates/components/copy-page-button.html.
html_js_files = ["sidebar-wrap.js", "copy-page-button.js"]

# Theme options
html_theme_options = {
    # Placeholder accent color; matches Radix UI color names
    # (tomato, red, ruby, crimson, pink, plum, purple, violet, iris, indigo, blue, ...)
    "accent_color": "indigo",
    "dark_code": False,
    "globaltoc_expand_depth": 0,
    "show_ai_links": True,
    "nav_links": [
        {"title": "API Reference", "url": "api/index"},
        {
            "title": "GitHub Repo",
            "url": "https://github.com/apple/coreai-optimization/",
            "external": True,
        },
    ],
}

html_show_sourcelink = False

# Pygments (syntax highlighting) style
pygments_style = "friendly"

# -- Custom setup ------------------------------------------------------------

# Internal class attributes that should never appear in public API docs
_AUTODOC_SKIP_NAMES = frozenset(
    {
        "REGISTRY",
        "model_config",
        "model_computed_fields",
        "model_fields",
        "model_post_init",
    }
)

# Methods inherited from str that clutter StrEnum-based classes (ExportBackend, etc.)
_STR_INHERITED_METHODS = frozenset(dir(str)) - {"__doc__", "__module__", "__class__"}


@functools.cache
def _has_presets(fullname: str) -> bool:
    """Return True if the class at ``fullname`` exposes a ``.presets`` namespace.

    Drives the gate in the class-page Jinja templates so only classes that
    actually have presets get the hidden toctree to their presets index page.
    """
    cls = pydoc.locate(fullname)
    return cls is not None and hasattr(cls, "presets")


def _method_summary(method_obj: object) -> str:
    """Return the first line of the method's docstring, or ``""``."""
    doc = inspect.getdoc(method_obj) or ""
    return doc.split("\n", 1)[0] if doc else ""


def _public_preset_names(presets: object) -> list[str]:
    """Return sorted public method names on a preset namespace instance."""
    return sorted(m for m in dir(presets) if not m.startswith("_"))


def _discover_preset_owners() -> list[str]:
    """Return FQNs of every class in ``api/index.md`` with a ``.presets`` attribute.

    Keyed off the generated API index so only classes that will appear in the
    docs get stubs. New preset-bearing classes picked up automatically once
    they're added to the index.
    """
    api_index = Path(__file__).parent / "api" / "index.md"
    if not api_index.exists():
        return []
    entries = re.findall(r"^\s+(coreai_opt\.[\w.]+)\s*$", api_index.read_text(), re.MULTILINE)
    return [fqn for fqn in entries if _has_presets(fqn)]


def _write_if_changed(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` only when it differs from the existing file.

    Avoids mtime churn on rebuilds where the generated stub hasn't actually
    changed, which would otherwise force Sphinx to re-render the page.
    """
    if path.exists() and path.read_text() == content:
        return
    path.write_text(content)


def _write_preset_stubs(out_dir: Path) -> None:
    """Write one stub ``.rst`` per preset method at its user-facing usage path.

    The stub filename and H1 title use ``{OwnerClass.presets.method}`` (the Python
    expression a user would write), not the underscore-prefixed namespace class or
    the filename where the method is defined. The body uses a manual
    ``.. py:method::`` directive with the signature + docstring introspected from
    the bound method, so the rendered signature line also shows the usage path
    instead of the implementation path.

    The Google-style docstring is converted to RST via ``GoogleDocstring`` so
    continuation-line indentation doesn't collide with the directive body indent.
    """
    from sphinx.ext.napoleon import Config as NapoleonConfig  # noqa: PLC0415
    from sphinx.ext.napoleon.docstring import GoogleDocstring  # noqa: PLC0415

    napoleon_cfg = NapoleonConfig(
        napoleon_google_docstring=True,
        napoleon_numpy_docstring=False,
        napoleon_use_admonition_for_examples=True,
        napoleon_use_admonition_for_notes=True,
        napoleon_use_param=True,
        napoleon_use_rtype=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for owner_fqn in _discover_preset_owners():
        cls = pydoc.locate(owner_fqn)
        if cls is None:
            continue
        presets = cls.presets
        method_names = _public_preset_names(presets)

        for method_name in method_names:
            method_obj = getattr(presets, method_name)
            # Strip type annotations from signatures to match the project-wide
            # ``autodoc_typehints = "description"`` behaviour — types stay
            # documented via the Napoleon ``:param:`` / ``:rtype:`` fields.
            raw_sig = inspect.signature(method_obj)
            clean_sig = raw_sig.replace(
                parameters=[
                    p.replace(annotation=inspect.Parameter.empty)
                    for p in raw_sig.parameters.values()
                ],
                return_annotation=inspect.Signature.empty,
            )
            rst_doc = str(GoogleDocstring(inspect.getdoc(method_obj) or "", napoleon_cfg))
            content = (
                f"{method_name}\n{'=' * len(method_name)}\n\n"
                f".. py:method:: {method_name}{clean_sig}\n"
                f"   :noindex:\n\n"
                f"{textwrap.indent(rst_doc, '   ')}\n"
            )
            _write_if_changed(out_dir / f"{owner_fqn}.presets.{method_name}.rst", content)

        table_rows = "".join(
            f"   * - :doc:`{m}() </api/generated/{owner_fqn}.presets.{m}>`\n"
            f"     - {_method_summary(getattr(presets, m))}\n"
            for m in method_names
        )
        toctree = "".join(
            f"   {m} </api/generated/{owner_fqn}.presets.{m}>\n" for m in method_names
        )
        index_content = (
            "presets\n=======\n\n"
            "Convenient factories for common compression recipes. Each preset\n"
            "returns a ready-to-use config that can be further refined by chaining\n"
            ":meth:`~coreai_opt.config.CompressionConfig.set_module_type`,\n"
            ":meth:`~coreai_opt.config.CompressionConfig.set_module_name`,\n"
            ":meth:`~coreai_opt.config.CompressionConfig.only_for`, or\n"
            ":meth:`~coreai_opt.config.CompressionConfig.without`.\n\n"
            ".. list-table::\n"
            "   :header-rows: 1\n"
            "   :class: preset-table\n\n"
            "   * - Preset\n"
            "     - Description\n"
            f"{table_rows}\n"
            ".. toctree::\n"
            "   :hidden:\n\n"
            f"{toctree}"
        )
        _write_if_changed(out_dir / f"{owner_fqn}.presets.rst", index_content)


def _autodoc_skip_member(
    app: Sphinx, what: str, name: str, obj: object, skip: bool, options: object
) -> bool | None:
    """Skip internal attributes inherited from mixins and Pydantic internals."""
    if name in _AUTODOC_SKIP_NAMES:
        return True
    # Skip private members that start with underscore (unless explicitly included)
    if name.startswith("_") and not name.startswith("__"):
        return True
    # Skip inherited dunder methods from object/ABC (keep only __init__)
    if name.startswith("__") and name.endswith("__") and name != "__init__":
        return True
    # Skip str methods inherited by StrEnum classes (encode, split, replace, etc.)
    if name in _STR_INHERITED_METHODS:
        return True
    # Skip enum member values — already documented via Napoleon's Attributes: section
    # in the class docstring. Without this, autodoc renders them again as raw
    # "MEMBER = 'value'" blocks at the bottom of the page.
    if isinstance(obj, enum.Enum):
        return True
    # Skip methods/attributes inherited from non-coreai-opt base classes (torch, torchao,
    # pydantic, etc.). Keeps __init__ unconditionally since the dunder check above already
    # decides whether to show it, and Pydantic-generated __init__ has __module__ from pydantic.
    if name != "__init__":
        obj_module = getattr(obj, "__module__", None) or ""
        if obj_module and not obj_module.startswith("coreai_opt"):
            return True
    return skip


def setup(app: Sphinx) -> None:
    """Custom setup function to add CSS files, autodoc filters, and generate API index."""
    app.add_css_file("custom.css")
    app.connect("autodoc-skip-member", _autodoc_skip_member)

    # Generate api/index.md from the package tree before Sphinx reads sources.
    api_index = Path(__file__).parent / "api" / "index.md"
    api_index.parent.mkdir(parents=True, exist_ok=True)
    api_index.write_text(generate_api_index())

    # Auto-generate preset stubs at their user-facing usage paths. Must run
    # after api/index.md exists (discovery reads it) and before Sphinx reads
    # sources (stubs need to be on disk to be included in the build).
    _write_preset_stubs(api_index.parent / "generated")
