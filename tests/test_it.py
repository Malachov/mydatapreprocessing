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

import mylogging

sys.path.insert(0, Path(os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename)).parents[0].as_posix())
from visual import visual_test

sys.path.insert(0, Path(os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename)).parents[1].as_posix())
import mydatapreprocessing.preprocessing as mdpp
import mydatapreprocessing as mdp

mylogging._COLORIZE = 0

np.random.seed(2)


#########################
### Integration tests ###
#########################

def test_integration():

    # Load data from file or URL
    data_loaded = mdpp.load_data("https://blockchain.info/unconfirmed-transactions?format=json", request_datatype_suffix=".json", predicted_table='txs', data_orientation="index")

    # Transform various data into defined format - pandas dataframe - convert to numeric if possible, keep
    # only numeric data and resample ifg configured. It return array, dataframe
    data_consolidated = mdpp.data_consolidation(
        data_loaded, predicted_column="weight", remove_nans_threshold=0.9, remove_nans_or_replace='interpolate')

    # Preprocess data. It return preprocessed data, but also last undifferenced value and scaler for inverse
    # transformation, so unpack it with _
    data_preprocessed_df, _, _ = mdpp.preprocess_data(data_consolidated, remove_outliers=True, smoothit=(11, 2),
                                                      correlation_threshold=False, data_transform=True, standardizeit='standardize')

    data_preprocessed, _, _ = mdpp.preprocess_data(data_consolidated.values, remove_outliers=True, smoothit=(11, 2),
                                                   correlation_threshold=0.9, data_transform='difference', standardizeit='01')

    assert not np.isnan(np.min(data_preprocessed_df.values)) and not np.isnan(np.min(data_preprocessed))


def test_preprocessing():

    ### Column with nan should be removed, row with outlier big value should be removed.
    ### Preprocessing and inverse will be made and than just  compare with good results

    test_df = pd.DataFrame(np.array([range(5), range(20, 25), range(25, 30), np.random.randn(5)]).T, columns=["First", "Predicted", "Ignored", "Ignored 2"])
    test_df.iloc[2, 1] = 500
    test_df.iloc[2, 2] = np.nan

    df_df = mdpp.data_consolidation(test_df, predicted_column=1, other_columns=1, datetime_column=False, remove_nans_threshold=0.9)
    data_df = df_df.values.copy()

    # Predicted column moved to index 0, but for test reason test, use different one
    processed_df, last_undiff_value_df, final_scaler_df = mdpp.preprocess_data(
        df_df, remove_outliers=1, correlation_threshold=0.9, data_transform='difference', standardizeit='standardize')

    inverse_processed_df = mdpp.preprocess_data_inverse(
        processed_df['Predicted'].iloc[1:], final_scaler=final_scaler_df, last_undiff_value=test_df['Predicted'][0],
        standardizeit='standardize', data_transform='difference')

    processed_df_2, last_undiff_value_df_2, final_scaler_df_2 = mdpp.preprocess_data(
        data_df, remove_outliers=1, correlation_threshold=0.9, data_transform='difference',
        standardizeit='standardize')

    inverse_processed_df_2 = mdpp.preprocess_data_inverse(
        processed_df_2[1:, 0], final_scaler=final_scaler_df_2, last_undiff_value=test_df['Predicted'][0],
        standardizeit='standardize', data_transform='difference')

    correct_preprocessing = np.array([[-0.707107, -0.707107],
                                      [1.414214, 1.414214],
                                      [-0.707107, -0.707107]])

    check_1 = np.allclose(processed_df.values, correct_preprocessing)
    check_2 = np.allclose(processed_df_2, correct_preprocessing)

    correct_inveerse_preprocessing = np.array([22., 23.])

    check_3 = np.allclose(inverse_processed_df, correct_inveerse_preprocessing)
    check_4 = np.allclose(inverse_processed_df_2, correct_inveerse_preprocessing)

    assert check_1 and check_2 and check_3 and check_4


def test_visual():
    visual_test(print_preprocessing=1, print_postprocessing=1)


##################
### Unit tests ###
##################

### Data load ###

def test_test_data():
    assert mdpp.load_data('test').ndim


def test_numpy_and_dataframe():
    assert mdpp.load_data(np.random.randn(100, 3)).ndim and mdpp.load_data(pd.DataFrame(np.random.randn(100, 3))).ndim


def test_numpys_and_pandas():
    assert mdpp.load_data([np.random.randn(20, 3), np.random.randn(25, 3)]).ndim and mdpp.load_data((pd.DataFrame(np.random.randn(20, 3)), pd.DataFrame(np.random.randn(25, 3)))).ndim


def test_dict():
    assert mdpp.load_data({'col_1': [3, 2, 1, 0], 'col_2': [3, 2, 1, 0]}, data_orientation='index').ndim


def test_list_of_dicts():
    assert mdpp.load_data(
        [{'col_1': [3, 2, 1, 0], 'col_2': [3, 2, 1, 0]},
         {'col_1': [4, 2, 1, 0], 'col_2': [3, 2, 1, 0]},
         {'col_1': [5, 2, 1, 0], 'col_2': [3, 2, 1, 0]}],
        data_orientation='columns').ndim


def test_list():
    assert mdpp.load_data([['Jon', 'Smith', 21], ['Mark', 'Brown', 38], ['Maria', 'Lee', 42]]).ndim


def test_tuple():
    assert mdpp.load_data((('Jon', 'Smith', 21), ('Mark', 'Brown', 38), ('Maria', 'Lee', 42))).ndim


def test_more_files():

    data_loaded = mdpp.load_data("https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv", request_datatype_suffix=".json", predicted_table='txs', data_orientation="index")
    data_loaded2 = mdpp.load_data(["https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv", "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"], request_datatype_suffix=".json", predicted_table='txs', data_orientation="index")

    assert len(data_loaded2) == 2 * len(data_loaded)


### Consolidation ###
def test_remove_nans():
    data = np.random.randn(50, 10)
    data[data < 0] = np.nan

    mdpp.data_consolidation(data, remove_nans_or_replace='mean', remove_nans_threshold=0.5)
    mdpp.data_consolidation(data, remove_nans_or_replace='neighbor', remove_nans_threshold=0.5)
    mdpp.data_consolidation(data, remove_nans_or_replace='remove', remove_nans_threshold=0.5)
    mdpp.data_consolidation(data, remove_nans_or_replace=0, remove_nans_threshold=0.5)


### Preprocessing ###
def test_local_files():

    df_imported = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv')
    df_part = df_imported.iloc[:10, :]

    df_imported.to_csv('tested.csv')
    df_part.to_csv('tested2.csv')
    df_imported.to_parquet('tested.parquet', compression='gzip')
    df_imported.to_hdf('tested.h5', key='df')

    loaded_data = requests.get('https://blockchain.info/unconfirmed-transactions?format=json').content
    with open('tested.json', 'w') as outfile:
        json.dump(json.loads(loaded_data), outfile)

    try:
        df_csv = mdpp.load_data('tested.csv')
        df_csv_joined = mdpp.load_data(['tested.csv', 'tested2.csv'])
        df_json = mdpp.data_consolidation(mdpp.load_data('tested.json', request_datatype_suffix=".json", predicted_table='txs'))
        df_parquet = mdpp.load_data('tested.parquet')
        df_hdf = mdpp.load_data('tested.h5')

    except Exception:
        pass

    finally:
        os.remove('tested.csv')
        os.remove('tested2.csv')
        os.remove('tested.json')
        os.remove('tested.parquet')
        os.remove('tested.h5')

    assert df_csv.ndim and df_json.ndim and df_parquet.ndim and df_hdf.ndim and len(df_csv_joined) == len(df_csv) + 10


def test_resample():
    data = mdpp.load_data('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv')
    resampled = mdpp.data_consolidation(data, datetime_column='Date', freq='M')
    assert len(resampled) < len(data) and len(resampled) > 1


def test_add_frequency_columns():
    data = np.array([[0, 2] * 64, [0, 0, 0, 5] * 32]).T
    mdpp.add_frequency_columns(data, window=8)


def test_add_derived_columns():
    data = pd.DataFrame([range(30), range(30, 60)]).T
    mdpp.add_derived_columns(data, differences=True, second_differences=True, multiplications=True, rolling_means=True, rolling_stds=True, mean_distances=True, window=10)


def test_split():
    data = pd.DataFrame([range(30), range(30, 60)]).T
    mdpp.split(data)


def test_embedding():
    data = pd.DataFrame([[1, 'e', 'e'], [2, 'e', 'l'], [3, 'r', 'v'], [4, 'e', 'r'], [5, 'r', 'o']])
    embedded_one_hot = mdpp.categorical_embedding(data, embedding='one-hot', unique_threshlold=0.6)
    embedded_label = mdpp.categorical_embedding(data, embedding='label', unique_threshlold=0.6)

    label_supposed_result = np.array([[1, 0], [2, 0], [3, 1], [4, 0], [5, 1]])
    one_hot_supposed_result = np.array([[1, 1, 0], [2, 1, 0], [3, 0, 1], [4, 1, 0], [5, 0, 1]])

    assert(np.array_equal(embedded_label.values, label_supposed_result) and np.array_equal(embedded_one_hot.values, one_hot_supposed_result))


def test_make_sequences():
    data = np.array([[1, 3, 5, 2, 3, 4, 5, 66, 3]]).T
    seqs, Y, x_input, test_inputs = mdp.inputs.make_sequences(data, predicts=7, repeatit=3, n_steps_in=6, n_steps_out=1, constant=1)
    assert seqs.ndim and Y.ndim and test_inputs.ndim

# if __name__ == "__main__":

#     pass
