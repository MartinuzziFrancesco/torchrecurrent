# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

project = "torchrecurrent"
html_title = "torchrecurrent"
copyright = "2025, Francesco Martinuzzi"
author = "Francesco Martinuzzi"
release = "0.1.3"
html_logo = "_static/logo.png"
html_favicon = "_static/favicon.ico"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    # "sphinx_autodoc_typehints",
    "sphinx.ext.viewcode",
]

napoleon_google_docstring = True
napoleon_numpy_docstring = False
autodoc_typehints = "none"
autodoc_typehints_format = "short"
autosummary_generate = True

napoleon_custom_sections = [
    ("Inputs", "params_style"),
    ("Outputs", "params_style"),
    ("Variables", "params_style"),
]

sys.path.insert(0, os.path.abspath(".."))

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "logo": {
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo.png",
    },
    "use_edit_page_button": True,
    "show_nav_level": 2,
}

html_context = {
    "github_user": "MartinuzziFrancesco",
    "github_repo": "torchrecurrent",
    "github_version": "main",
    "doc_path": "docs",
}
