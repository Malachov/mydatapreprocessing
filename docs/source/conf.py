
# Generate rst files with
# sphinx-apidoc -f -e -o source/ ../mydatapreprocessing
# Only other important file is index.rst

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import sys
import pathlib
import datetime
from recommonmark.parser import CommonMarkParser

script_dir = pathlib.Path(__file__).resolve()
lib_path = script_dir.parents[2].as_posix()
sys.path.insert(0, lib_path)

# Delete one.. its foru source extension
sys.path.insert(1, script_dir.as_posix())
lib_path2 = script_dir.parents[2] / 'mydatapreprocessing'
sys.path.insert(2, lib_path2.as_posix())


# -- Project information -----------------------------------------------------

project = 'mydatapreprocessing'
copyright = '2020, Daniel Malachov'
author = 'Daniel Malachov'

# The full version, including alpha/beta/rc tags
release = datetime.datetime.now().strftime('%d-%m-%Y')

master_doc = 'index'

source_parsers = {".md": CommonMarkParser}
source_suffix = ['.rst', '.md']

# -- General configuration ---------------------------------------------------
html_theme_options = {
    'github_user': 'Malachov',
    'github_repo': 'mydatapreprocessing',
    'github_banner': True,
    'logo': 'logo.png'
}

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
                'sphinx.ext.autodoc',
                'sphinx.ext.napoleon',
                'sphinx.ext.intersphinx',
                'sphinx.ext.viewcode',
                'sphinx.ext.githubpages',
                'sphinx.ext.imgmath',
                'sphinx.ext.autosectionlabel',
                'recommonmark',
]

# 'about.html'
html_sidebars = {'**': ['navi.html', 'searchbox.html']}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
