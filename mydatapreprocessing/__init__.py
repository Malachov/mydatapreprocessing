# -*- coding: utf-8 -*-

"""Load, consolidate and preprocess data in simplest possible way.

.. image:: https://img.shields.io/pypi/pyversions/mydatapreprocessing.svg
    :target: https://pypi.python.org/pypi/mydatapreprocessing/
    :alt: Py versions

.. image:: https://badge.fury.io/py/mydatapreprocessing.svg
    :target: https://badge.fury.io/py/mydatapreprocessing
    :alt: PyPI package

.. image:: https://pepy.tech/badge/mydatapreprocessing
    :target: https://pepy.tech/project/mydatapreprocessing
    :alt: Downloads

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/Malachov/mydatapreprocessing/HEAD?filepath=demo.ipynb
    :alt: Jupyter MyBinder

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

Load data from web link or local file (json, csv, excel file, parquet, h5...), consolidate it (resample data,
clean NaN values, do string embedding) derive new features via columns derivation and do preprocessing like
standardization or smoothing. If you want to see how functions works, check it's docstrings - working examples
with printed results are also in tests - visual.py.

Links
=====

Repo on github - https://github.com/Malachov/mydatapreprocessing

Readthedocs documentation - https://mydatapreprocessing.readthedocs.io

Installation
============

Python >= 3.6 (Python 2 is not supported).

Install just with::

    pip install mydatapreprocessing

There are some libraries that not every user will be using (for some specific data inputs for example). If you
want to be sure to have all libraries, you can provide extras requirements like::

    pip install mydatapreprocessing[datatypes]

Available extras are ["all", "datasets", "datatypes"]

Examples:
=========

    >>> import mydatapreprocessing as mdp

    **Load data**

    You can use

        - python formats (numpy.ndarray, pd.DataFrame, list, tuple, dict)
        - local files
        - web urls

    Supported path formats are:

        - csv
        - xlsx and xls
        - json
        - parquet
        - h5

    You can load more data at once in list.

    Syntax is always the same.

    >>> data = mdp.load_data.load_data(
    ...     "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
    ... )
    >>> # data2 = mdp.load_data.load_data([PATH_TO_FILE.csv, PATH_TO_FILE2.csv])

    **Consolidation**

    If you want to use data for some machine learning models, you will probably want to remove Nan values,
    convert string columns to numeric if possible, do encoding or keep only numeric data and resample.

    Consolidation is working with pandas DataFrame as column names matters here.

    There are many functions, but there is main function pipelining other functions `consolidate_data`

    >>> data = mdp.load_data.load_data(r"https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv")
    ...
    >>> consolidation_config = mdp.consolidation.consolidation_config.default_consolidation_config.do.copy()
    >>> consolidation_config.datetime.datetime_column = 'Date'
    >>> consolidation_config.resample.resample = 'M'
    >>> consolidation_config.resample.resample_function = "mean"
    >>> consolidation_config.dtype = 'float32'
    ...
    >>> consolidated = mdp.consolidation.consolidate_data(data, consolidation_config)
    >>> consolidated.head()
                     Temp
    Date                 
    1981-01-31  17.712904
    1981-02-28  17.678572
    1981-03-31  13.500000
    1981-04-30  12.356667
    1981-05-31   9.490322

    In config, you can use shorter update dict syntax as all values names are unique.

    **Feature engineering**
    
    Create new columns that can be for example used as another machine learning model input.

    >>> import mydatapreprocessing as mdp
    >>> import mydatapreprocessing.feature_engineering as mdpf
    >>> import pandas as pd
    ...
    >>> data = pd.DataFrame(
    ...     [mdp.datasets.sin(n=30), mdp.datasets.ramp(n=30)]
    ... ).T
    ...
    >>> extended = mdpf.add_derived_columns(data, differences=True, rolling_means=10)
    >>> extended.columns
    Index([                      0,                       1,
                  '0 - Difference',        '1 - Difference',
           '0 - Second difference', '1 - Second difference',
            'Multiplicated (0, 1)',      '0 - Rolling mean',
                '1 - Rolling mean',       '0 - Rolling std',
                 '1 - Rolling std',     '0 - Mean distance',
               '1 - Mean distance'],
          dtype='object')
    >>> len(extended)
    21

    Functions in `feature_engineering` and `preprocessing` expects that data are in form
    (n_samples, n_features). n_samples are usually much bigger and therefore transformed in `consolidate_data`
    if necessary.
    
    **Preprocessing**

    Preprocessing can be used on pandas DataFrame as well as on numpy array. Column names are not important
    as it's just matrix with defined dtype.

    There is many functions, but there is main function pipelining other functions `preprocess_data`
    Preprocessed data can be converted back with `preprocess_data_inverse`

    >>> import numpy as np
    >>> import pandas as pd
    ...
    >>> from mydatapreprocessing import preprocessing as mdpp
    ...
    >>> df = pd.DataFrame(np.array([range(5), range(20, 25), np.random.randn(5)]).astype("float32").T)
    >>> df.iloc[2, 0] = 500
    ...
    >>> config = mdpp.preprocessing_config.default_preprocessing_config.do.copy()
    >>> config.do.update({"remove_outliers": None, "difference_transform": True, "standardize": "standardize"})
    ...
    >>> data_preprocessed, inverse_config = mdpp.preprocess_data(df.values, config)
    >>> data_preprocessed
    array([[ 0.       ,  0.       ,  0.2571587],
           [ 1.4142135,  0.       , -0.633448 ],
           [-1.4142135,  0.       ,  1.5037845],
           [ 0.       ,  0.       , -1.1274952]], dtype=float32)
    
    
    If using for prediction, default last value is used for inverse transform. Here for test using first value
    is used to check whether original data will be restored.    

    >>> inverse_config.difference_transform = df.iloc[0, 0]
    >>> data_preprocessed_inverse = mdpp.preprocess_data_inverse(
    ...     data_preprocessed[:, 0], inverse_config
    ... )
    >>> data_preprocessed_inverse
    array([  1., 500.,   3.,   4.], dtype=float32)
    >>> np.allclose(df.values[1:, 0], data_preprocessed_inverse, atol=1.0e-5)
    True
"""
from . import (
    consolidation,
    database,
    datasets,
    feature_engineering,
    helpers,
    load_data,
    misc,
    preprocessing,
    types,
)

__version__ = "3.0.3"
__author__ = "Daniel Malachov"
__license__ = "MIT"
__email__ = "malachovd@seznam.cz"

__all__ = [
    "consolidation",
    "database",
    "datasets",
    "feature_engineering",
    "helpers",
    "load_data",
    "misc",
    "preprocessing",
    "types",
]
