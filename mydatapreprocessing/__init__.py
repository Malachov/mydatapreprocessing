# -*- coding: utf-8 -*-

"""

.. image:: https://img.shields.io/pypi/pyversions/mydatapreprocessing.svg
    :target: https://pypi.python.org/pypi/mydatapreprocessing/
    :alt: Py versions

.. image:: https://badge.fury.io/py/mydatapreprocessing.svg
    :target: https://badge.fury.io/py/mydatapreprocessing
    :alt: PyPI package

.. image:: https://img.shields.io/lgtm/grade/python/github/Malachov/mydatapreprocessing.svg
    :target: https://lgtm.com/projects/g/Malachov/mydatapreprocessing/context:python
    :alt: Language grade: Python

.. image:: https://readthedocs.org/projects/mydatapreprocessing/badge/?version=latest
    :target: https://mydatapreprocessing.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :target: https://opensource.org/licenses/MIT
    :alt: License: MIT

.. image:: https://codecov.io/gh/Malachov/mydatapreprocessing/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/Malachov/mydatapreprocessing
    :alt: Codecov

Library contain 3 modules - preprocessing, inputs and generatedata.

Links
=====

Repo on github - https://github.com/Malachov/mydatapreprocessing

Readthedocs documentation - https://mydatapreprocessing.readthedocs.io

Installation
============

Python >=3.6 (Python 2 is not supported).

Install just with::

    pip install mydatapreprocessing

There are some libraries that not every user will be using (for some data inputs).
If you want to be sure to have all libraries, you can download ``requirements_advanced.txt`` and then install
advanced requirements with ``pip install -r requirements_advanced.txt``.

"""
__version__ = "1.1.35"
__author__ = "Daniel Malachov"
__license__ = "MIT"
__email__ = "malachovd@seznam.cz"

__all__ = ["preprocessing", "inputs", "generatedata", "database"]

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("once")

    from . import preprocessing
    from . import inputs
    from . import generatedata
    from . import database
