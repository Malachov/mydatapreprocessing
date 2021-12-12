import numpy as np
import pandas as pd

import mypythontools

mypythontools.tests.setup_tests()


import mydatapreprocessing.preprocessing as mdpp
import mydatapreprocessing as mdp

np.random.seed(2)


def test_preprocessing():

    ### Column with nan should be removed, row with outlier big value should be removed.
    ### Preprocessing and inverse will be made and than just compare with good results
    np.random.seed(2)

    test_df = pd.DataFrame(
        np.array([range(5), range(20, 25), range(25, 30), np.random.randn(5)]).T,
        columns=["First", "Predicted", "Ignored", "Ignored 2"],
    )

    test_df.iloc[2, 1] = 500
    test_df.iloc[2, 2] = np.nan

    df_df = mdpp.data_consolidation(
        test_df, predicted_column=1, other_columns=1, datetime_column=None, remove_nans_threshold=0.9,
    )
    data_df = df_df.values.copy()

    # Predicted column moved to index 0, but for test reason test, use different one
    processed_df, _, final_scaler_df = mdpp.preprocess_data(
        df_df,
        remove_outliers=1,
        correlation_threshold=0.9,
        data_transform="difference",
        standardizeit="standardize",
    )

    inverse_processed_df = mdpp.preprocess_data_inverse(
        processed_df["Predicted"].iloc[1:].values,
        data_transform="difference",
        last_undiff_value=test_df["Predicted"][0],
        standardizeit="standardize",
        final_scaler=final_scaler_df,
    )

    processed_df_2, _, final_scaler_df_2 = mdpp.preprocess_data(
        data_df,
        remove_outliers=1,
        correlation_threshold=0.9,
        data_transform="difference",
        standardizeit="standardize",
    )

    inverse_processed_df_2 = mdpp.preprocess_data_inverse(
        processed_df_2[1:, 0],
        final_scaler=final_scaler_df_2,
        last_undiff_value=test_df["Predicted"][0],
        standardizeit="standardize",
        data_transform="difference",
    )

    correct_preprocessing = np.array([[-0.707107, -0.707107], [1.414214, 1.414214], [-0.707107, -0.707107]])

    check_1 = np.allclose(processed_df.values, correct_preprocessing)
    check_2 = np.allclose(processed_df_2, correct_preprocessing)

    correct_inverse_preprocessing = np.array([22.0, 23.0])

    check_3 = np.allclose(inverse_processed_df, correct_inverse_preprocessing)
    check_4 = np.allclose(inverse_processed_df_2, correct_inverse_preprocessing)

    assert all([check_1, check_2, check_3, check_4])


# NOTE Consolidation
def test_remove_nans():
    data = np.random.randn(50, 10)
    data[0, :] = np.nan
    data[data < 0] = np.nan

    for i in ["mean", "neighbor", "remove", 0]:
        removed = mdpp.data_consolidation(data, remove_nans_or_replace=i)
        if np.isnan(removed.values).any():
            raise ValueError("Nan in results")

    not_removed = mdpp.data_consolidation(data, remove_nans_or_replace=np.nan)

    assert (
        np.isnan(not_removed.values).any()
        and mdpp.data_consolidation(data, remove_nans_threshold=0.5).shape[1]
        > mdpp.data_consolidation(data, remove_nans_threshold=0.8).shape[1]
    )


def test_binnig():
    mdpp.binning(np.array(range(10)), bins=3, binning_type="cut")


def test_embedding():
    data = pd.DataFrame([[1, "e", "e"], [2, "e", "l"], [3, "r", "v"], [4, "e", "r"], [5, "r", "r"]])

    embedded_one_hot = mdpp.categorical_embedding(data, embedding="one-hot", unique_threshold=0.5)
    embedded_label = mdpp.categorical_embedding(data, embedding="label", unique_threshold=0.5)

    label_supposed_result = np.array([[1, 0], [2, 0], [3, 1], [4, 0], [5, 1]])
    one_hot_supposed_result = np.array([[1, 1, 0], [2, 1, 0], [3, 0, 1], [4, 1, 0], [5, 0, 1]])

    embedded_label_shorter = mdpp.categorical_embedding(data, embedding="label", unique_threshold=0.99)

    assert all(
        [
            np.array_equal(embedded_label.values, label_supposed_result),
            np.array_equal(embedded_one_hot.values, one_hot_supposed_result),
            embedded_label_shorter.shape[1] == 1,
        ]
    )


def test_resample():
    data = mdp.load_data.load_data(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
        csv_style={"sep": ",", "decimal": "."},
    )
    resampled = mdpp.data_consolidation(data, datetime_column="Date", resample_freq="M")
    assert len(data) > len(resampled) > 1


def test_fit_power_transform():
    mdpp.fitted_power_transform(np.array(range(100)), fitted_stdev=2, mean=9)


if __name__ == "__main__":
    pass
