"""Extract new features from available data.

You can add new derived columns. This new generated data can help to machine learning models to better
results.

In `add_derived_columns` you add first and second derivations, multiplication of columns, rolling means and
rolling standard deviation.

In `add_frequency_columns` you can add fast fourier transform results maximums on running window.
"""

from mydatapreprocessing.feature_engineering.feature_engineering_internal import (
    add_derived_columns,
    add_frequency_columns,
    keep_correlated_data,
)

__all__ = ["add_derived_columns", "add_frequency_columns", "keep_correlated_data"]
