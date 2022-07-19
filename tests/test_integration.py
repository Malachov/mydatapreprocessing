"""Tests for integration package."""

import numpy as np

from mypythontools_cicd import tests

tests.setup_tests()

import mydatapreprocessing as mdp

# pylint: disable=missing-function-docstring


def test_integration():
    # Load data from file or URL
    data_loaded = mdp.load_data.load_data(
        "https://raw.githubusercontent.com/Malachov/mydatapreprocessing/master/tests/test_files/list.json"
    )

    # Transform various data into defined format - pandas DataFrame - convert to numeric if possible, keep
    # only numeric data and resample ifg configured. It return array, DataFrame
    config = mdp.consolidation.consolidation_config.default_consolidation_config.copy()
    config.update(
        {"first_column": 0, "remove_all_column_with_nans_threshold": 0.7, "remove_nans_type": "interpolate"}
    )
    data_consolidated = mdp.consolidation.consolidate_data(data_loaded, config)

    # Preprocess data. It return preprocessed data, but also last undifferenced value and scaler for inverse
    # transformation, so unpack it with _
    config = mdp.preprocessing.preprocessing_config.default_preprocessing_config.copy()
    config.update(
        {
            "remove_outliers": 3,
            "smooth": (11, 2),
            "difference_transform": True,
            "standardize": "standardize",
        }
    )
    data_preprocessed, inverse_config = mdp.preprocessing.preprocess_data(data_consolidated, config)

    inverse_preprocessed_prediction = mdp.preprocessing.preprocess_data_inverse(
        data_preprocessed.values, inverse_config
    )

    # TODO test results not only no error
    assert not np.any(np.isnan(np.min(inverse_preprocessed_prediction)))


if __name__ == "__main__":
    # test_integration()

    pass
