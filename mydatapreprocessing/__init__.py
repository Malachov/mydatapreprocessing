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

Load data from web link or local file (json, csv, excel file, parquet, h5...), consolidate it (resample data,
clean NaN values, do string embedding) derive new featurs via columns derivation and do preprocessing like
standardization or smoothing. If you want to see how functions works, check it's docstrings - working examples
with printed results are also in tests - visual.py.

Links
=====

Repo on github - https://github.com/Malachov/mydatapreprocessing

Readthedocs documentation - https://mydatapreprocessing.readthedocs.io

Installation
============

Python >=3.6 (Python 2 is not supported).

Install just with::

    pip install mydatapreprocessing

There are some libraries that not every user will be using (for some data inputs). If you want to be sure to
have all libraries, you can download ``requirements_advanced.txt`` and then install advanced requirements with
``pip install -r requirements_advanced.txt``.


Examples:
=========

    >>> import mydatapreprocessing as mdp

    **Load data**

    You can use

        - python formats (numpy.ndarray, pd.DataFrame, list, tuple, dict)
        - local files
        - web urls

    You can load more data at once in list.

    Syntax is always the same.

    >>> data = mdp.load_data.load_data(
    ...     "https://blockchain.info/unconfirmed-transactions?format=json",
    ...     request_datatype_suffix=".json",
    ...     data_orientation="index",
    ...     predicted_table="txs",
    ... )
    >>> # data2 = mdp.load_data.load_data([PATH_TO_FILE.csv, PATH_TO_FILE2.csv])

    **Consolidation**

    If you want to use data for some machine learning models, you will probably want to remove Nan values, convert
    string columns to numeric if possible, do encoding or keep only numeric data and resample.

    >>> data = mdp.preprocessing.data_consolidation(
    ...     data, predicted_column="weight", remove_nans_threshold=0.9, remove_nans_or_replace="interpolate"
    ... )

    **Feature engineering**

    Functions in `feature_engineering` and `preprocessing` expects that data are in form (n_samples, n_features).
    n_samples are ususally much bigger and therefore transformed in `data_consolidation` if necessary.

    >>> data = mdp.feature_engineering.add_derived_columns(data, differences=True, rolling_means=32)

    **Preprocessing**

    ``preprocess_data`` returns preprocessed data, but also last undifferenced value and scaler for inverse
    transformation, so unpack it with `_`

    >>> data_preprocessed, _, _ = mdp.preprocessing.preprocess_data(
    ...     data,
    ...     remove_outliers=True,
    ...     smoothit=False,
    ...     correlation_threshold=False,
    ...     data_transform=False,
    ...     standardizeit="standardize",
    ... )

    **Creating inputs**
    >>> seqs, Y, x_input, test_inputs = mdp.create_model_inputs.make_sequences(
    ...     data_preprocessed.values, predicts=7, repeatit=3, n_steps_in=6, n_steps_out=1, constant=1
    ... )

"""
__version__ = "2.0.2"
__author__ = "Daniel Malachov"
__license__ = "MIT"
__email__ = "malachovd@seznam.cz"

__all__ = [
    "create_model_inputs",
    "database",
    "feature_engineering",
    "generate_data",
    "load_data",
    "misc",
    "preprocessing",
]

from . import (
    create_model_inputs,
    database,
    feature_engineering,
    generate_data,
    load_data,
    misc,
    preprocessing,
)
