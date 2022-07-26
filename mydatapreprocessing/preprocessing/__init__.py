"""Subpackage for data preprocessing.

Preprocessing means for example standardization, data smoothing, outliers removal or binning.

There are many small functions that you can use separately, but there is main function `preprocess_data` that
call all the functions based on input params for you. For inverse preprocessing use `preprocess_data_inverse`.

Functions are available for pd.DataFrame as well as numpy array. Output is usually of the same type as
an input. Functions can be use inplace or copy can be created.

Note:
    In many functions, there is main column necessary for correct functioning. It's supposed, that this column
    is on index 0 as first column. If using consolidation, use `first_column` param. Or use
    `move_on_first_column` manually.
"""
from mydatapreprocessing.preprocessing.preprocessing_pipeline_internal import (
    preprocess_data,
    preprocess_data_inverse,
)
from . import preprocessing_functions
from . import preprocessing_config
from .preprocessing_config import default_preprocessing_config

__all__ = [
    "preprocess_data",
    "preprocess_data_inverse",
    "preprocessing_functions",
    "preprocessing_config",
    "default_preprocessing_config",
]
