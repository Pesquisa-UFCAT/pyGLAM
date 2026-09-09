"""Sphinx configuration adapted from the documentation branch."""

import doctest
import sys
from importlib.metadata import version as package_version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

project = "pyglam"
copyright = "2026, Wanderlei M. Pereira Junior"
author = "pyGLAM contributors"
release = package_version("pyglam")
version = release
language = "en"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
]
exclude_patterns = []
autodoc_member_order = "bysource"
autodoc_typehints = "signature"
doctest_default_flags = doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE
html_theme = "sphinx_rtd_theme"
html_title = f"pyGLAM {release} documentation"
