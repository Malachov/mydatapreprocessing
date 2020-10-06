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
import mydatapreprocessing.preprocessing as mdp

mylogging._COLORIZE = 0

np.random.seed(2)


#########################
### Integration tests ###
#########################

def test_integration():

    data = "https://blockchain.info/unconfirmed-transactions?format=json"

    # Load data from file or URL
    data_loaded = mdp.load_data(data, request_datatype_suffix=".json", predicted_table='txs')

    # Transform various data into defined format - pandas dataframe - convert to numeric if possible, keep
    # only numeric data and resample ifg configured. It return array, dataframe
    data_consolidated = mdp.data_consolidation(
        data_loaded, predicted_column="weight", data_orientation="index", remove_nans_threshold=0.9, remove_nans_or_replace='interpolate')

    # Preprocess data. It return preprocessed data, but also last undifferenced value and scaler for inverse
    # transformation, so unpack it with _
    data_preprocessed_df, _, _ = mdp.preprocess_data(data_consolidated, remove_outliers=True, smoothit=(11, 2),
                                                     correlation_threshold=True, data_transform=True, standardizeit='standardize')

    data_preprocessed, _, _ = mdp.preprocess_data(data_consolidated.values, remove_outliers=True, smoothit=(11, 2),
                                                  correlation_threshold=0.9, data_transform='difference', standardizeit='01')

    assert not np.isnan(np.min(data_preprocessed_df.values)) and not np.isnan(np.min(data_preprocessed))


def test_preprocessing():

    ### Column with nan should be removed, row with outlier big value should be removed.
    ### Preprocessing and inverse will be made and than just  compare with good results

    test_df = pd.DataFrame(np.array([range(5), range(20, 25), range(25, 30), np.random.randn(5)]).T, columns=["First", "Predicted", "Ignored", "Ignored 2"])
    test_df.iloc[2, 1] = 500
    test_df.iloc[2, 2] = np.nan

    df_df = mdp.data_consolidation(test_df, predicted_column=1, other_columns=1, datetime_column=False, remove_nans_threshold=0.9)
    data_df = df_df.values.copy()

    # Predicted column moved to index 0, but for test reason test, use different one
    processed_df, last_undiff_value_df, final_scaler_df = mdp.preprocess_data(
        df_df, remove_outliers=1, correlation_threshold=0.9, data_transform='difference', standardizeit='standardize')

    inverse_processed_df = mdp.preprocess_data_inverse(
        processed_df['Predicted'].iloc[1:], final_scaler=final_scaler_df, last_undiff_value=test_df['Predicted'][0],
        standardizeit='standardize', data_transform='difference')

    processed_df_2, last_undiff_value_df_2, final_scaler_df_2 = mdp.preprocess_data(
        data_df, remove_outliers=1, correlation_threshold=0.9, data_transform='difference',
        standardizeit='standardize')

    inverse_processed_df_2 = mdp.preprocess_data_inverse(
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


### Preprocessing

def test_local_file():

    df_imported = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv')
    df_imported.to_csv('tested.csv')
    df_imported.to_parquet('tested.parquet', compression='gzip')
    df_imported.to_hdf('tested.h5', key='df')

    loaded_data = requests.get('https://blockchain.info/unconfirmed-transactions?format=json').content
    with open('tested.json', 'w') as outfile:
        json.dump(json.loads(loaded_data), outfile)

    try:
        df_csv = mdp.load_data('tested.csv')
        df_json = mdp.data_consolidation(mdp.load_data('tested.json', request_datatype_suffix=".json", predicted_table='txs'))
        df_parquet = mdp.load_data('tested.parquet')
        df_hdf = mdp.load_data('tested.h5')

    except Exception:
        pass

    finally:
        os.remove('tested.csv')
        os.remove('tested.json')
        os.remove('tested.parquet')
        os.remove('tested.h5')

    assert df_csv.ndim and df_json.ndim and df_parquet.ndim and df_hdf.ndim


def test_test_data():
    assert mdp.load_data('test').ndim


### Consolidation

def test_numpy():
    assert mdp.data_consolidation(np.random.randn(100, 3)).ndim


def test_dict():
    assert mdp.data_consolidation({'col_1': [3, 2, 1, 0], 'col_2': [3, 2, 1, 0]}, data_orientation='index').ndim


def test_resample():
    data = mdp.load_data('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv')
    resampled = mdp.data_consolidation(data, datetime_column='Date', freq='M')
    assert len(resampled) < len(data) and len(resampled) > 1


def test_remove_nans():
    data = np.random.randn(50, 10)
    data[data < 0] = np.nan

    mdp.data_consolidation(data, remove_nans_or_replace='mean', remove_nans_threshold=0.5)
    mdp.data_consolidation(data, remove_nans_or_replace='neighbor', remove_nans_threshold=0.5)
    mdp.data_consolidation(data, remove_nans_or_replace='remove', remove_nans_threshold=0.5)
    mdp.data_consolidation(data, remove_nans_or_replace=0, remove_nans_threshold=0.5)


def test_add_frequency_columns():
    data = np.array([[0, 2] * 64, [0, 0, 0, 5] * 32]).T
    mdp.add_frequency_columns(data, window=8)


def test_add_derived_columns():
    data = pd.DataFrame([range(30), range(30, 60)]).T
    mdp.add_derived_columns(data, differences=True, second_differences=True, multiplications=True, rolling_means=True, rolling_stds=True, mean_distances=True, window=10)


def test_split():
    data = pd.DataFrame([range(30), range(30, 60)]).T
    mdp.split(data)


# For deeper debug, uncomment problematic test
# if __name__ == "__main__":

#     pass

df = pd.DataFrame(np.random.randn(50, 2))
df.columns = ['sd', 'vv']
df.to_parquet("test.parquet", compression='gzip')
dff = pd.read_parquet("test.parquet")
