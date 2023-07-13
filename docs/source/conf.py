# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys
from datetime import datetime
from pathlib import Path

from cherab.lhd import __file__ as lhd_file
from cherab.lhd import __version__

sys.path.insert(0, os.path.abspath("."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "cherab-lhd"
author = "Koyo Munechika"
copyright = f"2020-{datetime.now().year}, {author}"
# The short X.Y version.
version = __version__
# The full version, including alpha/beta/rc tags.
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx-prompt",
    "sphinx_copybutton",
    "nbsphinx",
    "sphinx_design",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx_codeautolink",
]

default_role = "obj"

# autodoc config
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_member_order = "bysource"

# autosummary config
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = True
autosummary_ignore_module_all = False

# napoleon config
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_ivar = False

# todo config
todo_include_todos = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# This is added to the end of RST files — a good place to put substitutions to
# be used globally.
rst_epilog = ""
with open("common_links.rst") as cl:
    rst_epilog += cl.read()

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "common_links.rst",
]

# The suffix of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

# html_logo = "_static/images/cherab_lhd_logo.svg"

# html_favicon = "_static/favicon/favicon.ico"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/munechika-koyo/cherab_lhd",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
    ],
    "pygment_light_style": "default",
    "pygment_dark_style": "native",
}

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "Cherab-LHD Documentation"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]

# === Intersphinx configuration ===============================================
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "raysect": ("http://www.raysect.org", None),
    "cherab": ("https://www.cherab.info", None),
    "plotly": ("https://plotly.com/python-api-reference/", None),
}

intersphinx_timeout = 10

# === NB Sphinx configuration ============================================
nbsphinx_allow_errors = True
nbsphinx_prolog = """
{% set docname = env.doc2path(env.docname, base=None) %}
.. only:: html

    .. role:: raw-html(raw)
        :format: html

    .. note::
        This page was generated from `{{ docname }}`__.
    __ https://github.com/munechika-koyo/cherab_lhd/blob/main/docs/{{ docname }}
"""
nbsphinx_thumbnails = {
    "notebooks/machine/visualize_LHD_plotly": "_static/images/LHD_machine_thumbnails.png",
    "notebooks/observer/bolos_config": "_static/images/plotting/bolos_config.png",
}

# -----------------------------------------------------------------------------
# Source code links
# -----------------------------------------------------------------------------
link_github = True
# You can add build old with link_github = False

if link_github:
    import inspect

    from packaging.version import parse

    extensions.append("sphinx.ext.linkcode")

    def linkcode_resolve(domain, info):
        """Determine the URL corresponding to Python object."""
        if domain != "py":
            return None

        modname = info["module"]
        fullname = info["fullname"]

        submod = sys.modules.get(modname)
        if submod is None:
            return None

        obj = submod
        for part in fullname.split("."):
            try:
                obj = getattr(obj, part)
            except AttributeError:
                return None

        if inspect.isfunction(obj):
            obj = inspect.unwrap(obj)
        try:
            fn = inspect.getsourcefile(obj)
        except TypeError:
            fn = None
        if not fn or fn.endswith("__init__.py"):
            try:
                fn = inspect.getsourcefile(sys.modules[obj.__module__])
            except (TypeError, AttributeError, KeyError):
                fn = None
        if not fn:
            return None

        try:
            source, lineno = inspect.getsourcelines(obj)
        except (OSError, TypeError):
            lineno = None

        linespec = f"#L{lineno:d}-L{lineno + len(source) - 1:d}" if lineno else ""

        startdir = Path(lhd_file).parent.parent.parent
        try:
            fn = os.path.relpath(fn, start=startdir).replace(os.path.sep, "/")
        except ValueError:
            return None

        if not fn.startswith(("cherab")):
            return None

        parse(__version__)
        # tag is temporarily tied to main
        tag = "main"
        # tag = "main" if version.is_devrelease else f"v{version.public}"
        return f"https://github.com/munechika-koyo/cherab_lhd/blob/{tag}/{fn}{linespec}"

else:
    extensions.append("sphinx.ext.viewcode")
