# -*- coding: utf-8 -*-

"""
mydatapreprocessing
===================

.. image:: https://img.shields.io/pypi/pyversions/mydatapreprocessing.svg
    :target: https://pypi.python.org/pypi/mydatapreprocessing/
    :alt: Py versions

.. image:: https://badge.fury.io/py/mydatapreprocessing.svg
    :target: https://badge.fury.io/py/mydatapreprocessing
    :alt: PyPI package

.. image:: https://img.shields.io/lgtm/grade/python/g/Malachov/mydatapreprocessing.svg?logo=lgtm&logoWidth=18
    :target: https://lgtm.com/projects/g/Malachov/mydatapreprocessing/context:python
    :alt: Language grade: Python

.. image:: https://travis-ci.com/Malachov/mydatapreprocessing.svg?branch=master
    :target: https://travis-ci.com/Malachov/mydatapreprocessing
    :alt: Build Status

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
"""
__version__ = "1.1.29"
__author__ = "Daniel Malachov"
__license__ = "MIT"
__email__ = "malachovd@seznam.cz"

__all__ = ["preprocessing", "inputs", "generatedata"]

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("once")

    from . import preprocessing
    from . import inputs
    from . import generatedata
