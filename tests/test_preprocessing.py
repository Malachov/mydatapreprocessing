"""Tests for preprocessing package."""

import numpy as np
import pandas as pd

from mypythontools_cicd import tests

tests.setup_tests()


import mydatapreprocessing.preprocessing as mdpp
import mydatapreprocessing.preprocessing.preprocessing_functions as mdppf

# pylint: disable=missing-function-docstring


def test_preprocessing():

    df = pd.DataFrame(
        np.array([range(5), range(20, 25), np.random.randn(5)]).astype("float32").T,
    )

    df.iloc[2, 0] = 500

    array = df.values.copy()

    config = mdpp.preprocessing_config.default_preprocessing_config
    config.remove_outliers = 1
    config.difference_transform = True
    config.standardize = "standardize"

    # Predicted column moved to index 0, but for test reason test, use different one
    processed_df, inverse_preprocessing_config_df = mdpp.preprocess_data(df, config)

    processed_array, inverse_preprocessing_config_arr = mdpp.preprocess_data(array, config)

    inverse_preprocessing_config_df.difference_transform = df.iloc[0, 0]
    inverse_preprocessing_config_arr.difference_transform = df.iloc[0, 0]

    inverse_processed_df = mdpp.preprocess_data_inverse(
        processed_df.iloc[:, 0].values, inverse_preprocessing_config_df
    )

    inverse_processed_array = mdpp.preprocess_data_inverse(
        processed_array[:, 0], inverse_preprocessing_config_arr
    )

    correct_preprocessing = np.array(
        [
            [-0.7071068, -0.7071068, 0.37706152],
            [1.4142137, 1.4142137, 0.99187946],
            [-0.7071068, -0.7071068, -1.3689411],
        ]
    )

    check_1 = np.allclose(processed_df.values, correct_preprocessing)
    check_2 = np.allclose(processed_array, correct_preprocessing)

    correct_inverse_preprocessing = np.array([1, 3, 4])

    check_3 = np.allclose(inverse_processed_df, correct_inverse_preprocessing)
    check_4 = np.allclose(inverse_processed_array, correct_inverse_preprocessing)

    assert all([check_1, check_2, check_3, check_4])


def test_removing_outliers():
    data = pd.DataFrame(
        [
            [1, 7],
            [66, 3],
            [5, 5],
            [2, 3],
            [2, 3],
            [3, 9],
        ]
    )
    processed = mdppf.remove_the_outliers(data, threshold=2)
    should_be = pd.DataFrame(
        [
            [1, 7],
            [5, 5],
            [2, 3],
            [2, 3],
            [3, 9],
        ]
    )

    assert np.allclose(processed.values, should_be), "Outliers not removed."


def test_binning():
    mdppf.binning(np.array(range(10)), bins=3, binning_type="cut")


def test_fit_power_transform():
    mdppf.fitted_power_transform(np.array(range(100)), fitted_stdev=2, mean=9)


if __name__ == "__main__":

    # test_preprocessing()

    pass
