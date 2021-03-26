# -*- coding: utf-8 -*-

"""
mydatapreprocessing
===================

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/mydatapreprocessing.svg)](https://pypi.python.org/pypi/mydatapreprocessing/)
[![PyPI version](https://badge.fury.io/py/mydatapreprocessing.svg)](https://badge.fury.io/py/mydatapreprocessing)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/Malachov/mydatapreprocessing.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/Malachov/mydatapreprocessing/context:python)
[![Build Status](https://travis-ci.com/Malachov/mydatapreprocessing.svg?branch=master)](https://travis-ci.com/Malachov/mydatapreprocessing)
[![Documentation Status](https://readthedocs.org/projects/mydatapreprocessing/badge/?version=latest)](https://mydatapreprocessing.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/Malachov/mydatapreprocessing/branch/master/graph/badge.svg)](https://codecov.io/gh/Malachov/mydatapreprocessing)

Library contain 3 modules - **preprocessing**, **inputs** and **generatedata**
"""
__version__ = "1.1.27"
__author__ = "Daniel Malachov"
__license__ = "MIT"
__email__ = "malachovd@seznam.cz"

__all__ = ['preprocessing', 'inputs', 'generatedata']

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings('once')

    from . import preprocessing
    from . import inputs
    from . import generatedata
