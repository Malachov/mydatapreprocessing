""" Test module. Auto pytest that can be started in IDE or with

    >>> python -m pytest

in terminal in tests folder.
"""
#%%

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import inspect
import os
import json
import requests
import urllib

import mylogging

# Find paths and add to sys.path to be able to import local modules
test_dir_path = Path(
    os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename)
).parent
root_path = test_dir_path.parent

if test_dir_path.as_posix() not in sys.path:
    sys.path.insert(0, test_dir_path.as_posix())

if root_path.as_posix() not in sys.path:
    sys.path.insert(0, root_path.as_posix())

import mydatapreprocessing.preprocessing as mdpp
import mydatapreprocessing as mdp

from visual import visual_test

mylogging.config.COLOR = 0

np.random.seed(2)

#########################
### SECTION Integration tests
#########################


def test_integration():

    # Load data from file or URL
    data_loaded = mdpp.load_data(
        "https://blockchain.info/unconfirmed-transactions?format=json",
        request_datatype_suffix=".json",
        predicted_table="txs",
        data_orientation="index",
    )

    # Transform various data into defined format - pandas dataframe - convert to numeric if possible, keep
    # only numeric data and resample ifg configured. It return array, dataframe
    data_consolidated = mdpp.data_consolidation(
        data_loaded,
        predicted_column="weight",
        remove_nans_threshold=0.9,
        remove_nans_or_replace="interpolate",
    )

    # Preprocess data. It return preprocessed data, but also last undifferenced value and scaler for inverse
    # transformation, so unpack it with _
    data_preprocessed_df, _, _ = mdpp.preprocess_data(
        data_consolidated,
        remove_outliers=True,
        smoothit=(11, 2),
        correlation_threshold=False,
        data_transform=True,
        standardizeit="standardize",
    )

    data_preprocessed, _, _ = mdpp.preprocess_data(
        data_consolidated.values,
        remove_outliers=True,
        smoothit=(11, 2),
        correlation_threshold=0.9,
        data_transform="difference",
        standardizeit="01",
    )

    assert not np.isnan(np.min(data_preprocessed_df.values)) and not np.isnan(
        np.min(data_preprocessed)
    )


def test_preprocessing():

    ### Column with nan should be removed, row with outlier big value should be removed.
    ### Preprocessing and inverse will be made and than just  compare with good results

    test_df = pd.DataFrame(
        np.array([range(5), range(20, 25), range(25, 30), np.random.randn(5)]).T,
        columns=["First", "Predicted", "Ignored", "Ignored 2"],
    )
    test_df.iloc[2, 1] = 500
    test_df.iloc[2, 2] = np.nan

    df_df = mdpp.data_consolidation(
        test_df,
        predicted_column=1,
        other_columns=1,
        datetime_column=False,
        remove_nans_threshold=0.9,
    )
    data_df = df_df.values.copy()

    # Predicted column moved to index 0, but for test reason test, use different one
    processed_df, last_undiff_value_df, final_scaler_df = mdpp.preprocess_data(
        df_df,
        remove_outliers=1,
        correlation_threshold=0.9,
        data_transform="difference",
        standardizeit="standardize",
    )

    inverse_processed_df = mdpp.preprocess_data_inverse(
        processed_df["Predicted"].iloc[1:],
        final_scaler=final_scaler_df,
        last_undiff_value=test_df["Predicted"][0],
        standardizeit="standardize",
        data_transform="difference",
    )

    processed_df_2, last_undiff_value_df_2, final_scaler_df_2 = mdpp.preprocess_data(
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

    correct_preprocessing = np.array(
        [[-0.707107, -0.707107], [1.414214, 1.414214], [-0.707107, -0.707107]]
    )

    check_1 = np.allclose(processed_df.values, correct_preprocessing)
    check_2 = np.allclose(processed_df_2, correct_preprocessing)

    correct_inveerse_preprocessing = np.array([22.0, 23.0])

    check_3 = np.allclose(inverse_processed_df, correct_inveerse_preprocessing)
    check_4 = np.allclose(inverse_processed_df_2, correct_inveerse_preprocessing)

    assert all([check_1, check_2, check_3, check_4])


def test_visual():
    visual_test(print_preprocessing=1, print_postprocessing=1)


# !SECTION
##################
### SECTION Unit tests
##################

### ANCHOR preprocessing module

# NOTE Data load


def test_exceptions():

    exceptions = []

    try:
        mdpp.load_data("testfile")
    except Exception as e:
        exceptions.append(isinstance(e, FileNotFoundError))

    try:
        test_file = test_dir_path / "data_test"
        mdpp.load_data(test_file)
    except Exception as e:
        exceptions.append(isinstance(e, TypeError))

    try:
        mdpp.load_data("https://blockchain.info/unconfirmed-transactions?format=json")
    except Exception as e:
        exceptions.append(isinstance(e, TypeError))

    assert all(exceptions)


def test_test_data():
    assert mdpp.load_data("test").ndim


def test_numpy_and_dataframe():
    assert (
        mdpp.load_data(np.random.randn(100, 3)).ndim
        and mdpp.load_data(pd.DataFrame(np.random.randn(100, 3))).ndim
    )


def test_numpys_and_pandas():
    assert (
        mdpp.load_data([np.random.randn(20, 3), np.random.randn(25, 3)]).ndim
        and mdpp.load_data(
            (pd.DataFrame(np.random.randn(20, 3)), pd.DataFrame(np.random.randn(25, 3)))
        ).ndim
    )


def test_dict():
    assert mdpp.load_data(
        {"col_1": [3, 2, 1, 0], "col_2": [3, 2, 1, 0]}, data_orientation="index"
    ).ndim


def test_list_of_dicts():
    assert mdpp.load_data(
        [
            {"col_1": [3, 2, 1, 0], "col_2": [3, 2, 1, 0]},
            {"col_1": [4, 2, 1, 0], "col_2": [3, 2, 1, 0]},
            {"col_1": [5, 2, 1, 0], "col_2": [3, 2, 1, 0]},
        ],
        data_orientation="columns",
    ).ndim


def test_list():
    assert mdpp.load_data(
        [["Jon", "Smith", 21], ["Mark", "Brown", 38], ["Maria", "Lee", 42]]
    ).ndim


def test_tuple():
    assert mdpp.load_data(
        (("Jon", "Smith", 21), ("Mark", "Brown", 38), ("Maria", "Lee", 42))
    ).ndim


def test_more_files():

    data_loaded = mdpp.load_data(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
        request_datatype_suffix=".json",
        predicted_table="txs",
        data_orientation="index",
    )
    data_loaded2 = mdpp.load_data(
        [
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
        ],
        request_datatype_suffix=".json",
        predicted_table="txs",
        data_orientation="index",
    )

    assert len(data_loaded2) == 2 * len(data_loaded)


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

    mdpp.data_consolidation(data, remove_nans_threshold=0.5).shape[
        1
    ] > mdpp.data_consolidation(data, remove_nans_threshold=0.8).shape[1]

    assert (
        np.isnan(not_removed.values).any()
        and mdpp.data_consolidation(data, remove_nans_threshold=0.5).shape[1]
        > mdpp.data_consolidation(data, remove_nans_threshold=0.8).shape[1]
    )


def test_add_none_to_gaps():

    data = pd.DataFrame([[0, 1]] * 7, index=[0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 2.0])
    df_gaps = mdpp.add_none_to_gaps(data)

    assert df_gaps.iloc[:, 0].isnull().sum() == 2


def test_local_files():

    test_files = Path(__file__).parent / "test_files"
    xls = mdpp.load_data(test_files / "file_example_xls.xls")
    xlsx = mdpp.load_data(test_files / "file_example_xlsx.xlsx")

    df_imported = pd.read_csv(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    )
    df_part = df_imported.iloc[:10, :]

    df_imported.to_csv("tested.csv")
    df_part.to_csv("tested2.csv")
    df_imported.to_parquet("tested.parquet", compression="gzip")
    df_imported.to_hdf("tested.h5", key="df")

    loaded_data = requests.get(
        "https://blockchain.info/unconfirmed-transactions?format=json"
    ).content
    with open("tested.json", "w") as outfile:
        json.dump(json.loads(loaded_data), outfile)

    try:
        df_csv = mdpp.load_data("tested.csv")
        df_csv_joined = mdpp.load_data(["tested.csv", "tested2.csv"])
        df_json = mdpp.data_consolidation(
            mdpp.load_data(
                "tested.json", request_datatype_suffix=".json", predicted_table="txs"
            )
        )
        df_parquet = mdpp.load_data("tested.parquet")
        df_hdf = mdpp.load_data("tested.h5")

    except Exception:
        pass

    finally:
        os.remove("tested.csv")
        os.remove("tested2.csv")
        os.remove("tested.json")
        os.remove("tested.parquet")
        os.remove("tested.h5")

    assert all(
        [
            xls.ndim,
            xlsx.ndim,
            df_csv.ndim,
            df_json.ndim,
            df_parquet.ndim,
            df_hdf.ndim,
            len(df_csv_joined) == len(df_csv) + 10,
        ]
    )


# NOTE Preprocessing


def test_resample():
    data = mdpp.load_data(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
        csv_style={"sep": ",", "decimal": "."},
    )
    resampled = mdpp.data_consolidation(data, datetime_column="Date", freq="M")
    assert len(resampled) < len(data) and len(resampled) > 1


def test_add_frequency_columns():
    data = np.array([[0, 2] * 64, [0, 0, 0, 5] * 32]).T
    mdpp.add_frequency_columns(data, window=8)


def test_add_derived_columns():
    data = pd.DataFrame([range(30), range(30, 60)]).T
    mdpp.add_derived_columns(
        data,
        differences=True,
        second_differences=True,
        multiplications=True,
        rolling_means=True,
        rolling_stds=True,
        mean_distances=True,
        window=10,
    )


def test_split():
    data = pd.DataFrame([range(30), range(30, 60)]).T
    mdpp.split(data)


def test_embedding():
    data = pd.DataFrame(
        [[1, "e", "e"], [2, "e", "l"], [3, "r", "v"], [4, "e", "r"], [5, "r", "r"]]
    )

    embedded_one_hot = mdpp.categorical_embedding(
        data, embedding="one-hot", unique_threshlold=0.5
    )
    embedded_label = mdpp.categorical_embedding(
        data, embedding="label", unique_threshlold=0.5
    )

    label_supposed_result = np.array([[1, 0], [2, 0], [3, 1], [4, 0], [5, 1]])
    one_hot_supposed_result = np.array(
        [[1, 1, 0], [2, 1, 0], [3, 0, 1], [4, 1, 0], [5, 0, 1]]
    )

    embedded_label_shorter = mdpp.categorical_embedding(
        data, embedding="label", unique_threshlold=0.99
    )

    assert all(
        [
            np.array_equal(embedded_label.values, label_supposed_result),
            np.array_equal(embedded_one_hot.values, one_hot_supposed_result),
            embedded_label_shorter.shape[1] == 1,
        ]
    )


def test_fit_power_transform():
    mdpp.fitted_power_transform(np.array(range(100)), fitted_stdev=2, mean=9)


### ANCHOR inputs module


def test_make_sequences():
    data = np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13, 14, 15, 16],
            [17, 18, 19, 20, 21, 22, 23, 24],
        ]
    ).T
    X, y, x_input, _ = mdp.inputs.make_sequences(
        data, n_steps_in=2, n_steps_out=3, constant=1
    )

    X_res = np.array(
        [
            [1.0, 1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0, 17.0, 18.0, 19.0, 20.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 13.0, 18.0, 19.0, 20.0, 21.0],
        ]
    )
    y_res = np.array([[5, 6, 7], [6, 7, 8]])
    x_inpu_res = np.array(
        [[1.0, 5.0, 6.0, 7.0, 8.0, 13.0, 14.0, 15.0, 16.0, 21.0, 22.0, 23.0, 24.0]]
    )

    data2 = np.array(
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]
    ).T
    X2, y2, x_input2, test_inputs2 = mdp.inputs.make_sequences(
        data2, n_steps_in=2, n_steps_out=1, constant=0, predicts=3, repeatit=2
    )

    X2_res = np.array(
        np.array(
            [
                [1, 2, 11, 12],
                [2, 3, 12, 13],
                [3, 4, 13, 14],
                [4, 5, 14, 15],
                [5, 6, 15, 16],
                [6, 7, 16, 17],
                [7, 8, 17, 18],
                [8, 9, 18, 19],
            ]
        )
    )
    y2_res = np.array(([[3], [4], [5], [6], [7], [8], [9], [10]]))
    x_input2_res = np.array(([[9, 10, 19, 20]]))
    test_inputs2_res = np.array([[[5, 6, 15, 16]], [[6, 7, 16, 17]]])

    assert all(
        [
            np.allclose(X, X_res),
            np.allclose(y, y_res),
            np.allclose(x_input, x_inpu_res),
            np.allclose(X2, X2_res),
            np.allclose(y2, y2_res),
            np.allclose(x_input2, x_input2_res),
            np.allclose(test_inputs2, test_inputs2_res),
        ]
    )


if __name__ == "__main__":

    pass


# a = mdpp.load_data(
#     "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
#     header=0,
#     csv_style={"decimal": ".", "separator": ","},
#     predicted_table=None,
#     max_imported_length=300,
#     request_datatype_suffix=None,
#     data_orientation=None,
# )


# b = 8