"""Tests for consolidation package."""

import numpy as np
import pandas as pd

from mypythontools_cicd import tests

tests.setup_tests()

import mydatapreprocessing.consolidation as mdpc
import mydatapreprocessing.consolidation.consolidation_functions as mdpcf

from tests.helpers import compare_values

# pylint: disable=missing-function-docstring


def test_consolidation():

    # May delete all test when doctest finished
    df = pd.DataFrame(
        np.array([range(4), range(20, 24), np.random.randn(4)]).T,
        columns=["Column", "First", "Random"],
    )

    df.iloc[2, 2] = np.nan
    config = mdpc.consolidation_config.default_consolidation_config.do.copy()

    config.first_column = "First"
    config.datetime.datetime_column = None
    config.remove_missing_values.remove_all_column_with_nans_threshold = 0.6

    consolidated_df = mdpc.consolidate_data(df, config)

    assert consolidated_df.columns[0] == "First", "Column not moved on first index"

    # TODO test results - datetime etc
    # assert consolidated_df


def test_remove_nans():
    # fmt: off
    arr = np.array(
        [
            [np.nan, np.nan, 1     ],
            [np.nan, 2,      np.nan],        # noqa
            [30,     30,     30    ],        # noqa
        ]
    ).astype("float")
    # fmt: on

    df = pd.DataFrame(arr)

    for data in [arr, df]:
        assert (
            mdpc.consolidation_functions.remove_nans(
                data.copy(), remove_all_column_with_nans_threshold=0.4
            ).shape[1]
            == 3
        ), "Column deleted"
        result = mdpc.consolidation_functions.remove_nans(
            data.copy(), remove_all_column_with_nans_threshold=0.9
        )
        assert result.shape[1] == 2, "Column not deleted"

        # fmt: off
        expected_results = {
            "interpolate": np.array([[2,  1], [2, 15.5], [30, 30]]),
            "mean":        np.array([[16, 1], [2, 15.5], [30, 30]]),
            "neighbor":    np.array([[2,  1], [2, 1   ], [30, 30]]),
            "remove":      np.array([[30, 30]]),
            0:             np.array([[0,  1], [2, 0   ], [30, 30]]),
        }
        # fmt: on

        for i, j in expected_results.items():
            removed = mdpc.consolidation_functions.remove_nans(
                data.copy(), remove_all_column_with_nans_threshold=0.9, remove_nans_type=i
            )

            assert np.allclose(removed, j)


def test_embedding():
    data = pd.DataFrame([[1, "e", "e"], [2, "e", "l"], [3, "r", "v"], [4, "e", "r"], [5, "r", "r"]])

    embedded_one_hot = mdpcf.categorical_embedding(data, embedding="one-hot", unique_threshold=0.5)
    embedded_label = mdpcf.categorical_embedding(data, embedding="label", unique_threshold=0.5)

    label_supposed_result = np.array([[1, 0], [2, 0], [3, 1], [4, 0], [5, 1]])
    one_hot_supposed_result = np.array([[1, 1, 0], [2, 1, 0], [3, 0, 1], [4, 1, 0], [5, 0, 1]])

    embedded_label_shorter = mdpcf.categorical_embedding(data, embedding="label", unique_threshold=0.99)

    assert all(
        [
            np.array_equal(embedded_label.values, label_supposed_result),
            np.array_equal(embedded_one_hot.values, one_hot_supposed_result),
            embedded_label_shorter.shape[1] == 1,
        ]
    )


def test_resample():
    from datetime import datetime

    df = pd.DataFrame(
        {
            "date": [datetime(2022, 1, 1), datetime(2022, 1, 2), datetime(2022, 2, 1), datetime(2022, 5, 1)],
            "col_1": [1] * 4,
            "col_2": [2] * 4,
        }
    )
    df = df.set_index("date")

    compare_values(
        mdpc.consolidation_functions.resample(df, "M", "mean"),
        np.array([[1.0, 2.0], [1.0, 2.0], [np.nan, np.nan], [np.nan, np.nan], [1.0, 2.0]]),
    )


if __name__ == "__main__":
    pass
