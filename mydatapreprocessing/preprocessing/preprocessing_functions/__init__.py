"""Module for data preprocessing.

Preprocessing means for example standardization, data smoothing, outliers removal or binning.

There are many small functions that you can use separately, but there is main function `preprocess_data` that
call all the functions based on input params for you. For inverse preprocessing use `preprocess_data_inverse`.

Note:
    In many functions, there is main column necessary for correct functioning. It's supposed, that this column
    is on index 0 as first column. If using consolidation, use `first_column` param. Or use
    `move_on_first_column` manually.
"""

from mydatapreprocessing.preprocessing.preprocessing_functions.preprocessing_functions_internal import (
    binning,
    do_difference,
    fitted_power_transform,
    inverse_difference,
    remove_the_outliers,
    smooth,
    standardize_one_way,
    standardize,
)

__all__ = [
    "binning",
    "do_difference",
    "fitted_power_transform",
    "inverse_difference",
    "remove_the_outliers",
    "smooth",
    "standardize_one_way",
    "standardize",
]
