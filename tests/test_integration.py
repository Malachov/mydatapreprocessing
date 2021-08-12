import sys
from pathlib import Path
import inspect
import os
import subprocess
import mypythontools

import numpy as np

from mypythontools.utils import run_tests
import mylogging

# Find paths and add to sys.path to be able to import local modules
test_dir_path = Path(os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename)).parent
root_path = test_dir_path.parent

if test_dir_path.as_posix() not in sys.path:
    sys.path.insert(0, test_dir_path.as_posix())

if root_path.as_posix() not in sys.path:
    sys.path.insert(0, root_path.as_posix())

mylogging.config.COLOR = 0
np.random.seed(2)

import mydatapreprocessing as mdp


def test_integration():
    # Load data from file or URL
    data_loaded = mdp.load_data.load_data(
        "https://blockchain.info/unconfirmed-transactions?format=json",
        request_datatype_suffix=".json",
        predicted_table="txs",
        data_orientation="index",
    )

    # Transform various data into defined format - pandas dataframe - convert to numeric if possible, keep
    # only numeric data and resample ifg configured. It return array, dataframe
    data_consolidated = mdp.preprocessing.data_consolidation(
        data_loaded,
        predicted_column="weight",
        remove_nans_threshold=0.9,
        remove_nans_or_replace="interpolate",
    )

    # Preprocess data. It return preprocessed data, but also last undifferenced value and scaler for inverse
    # transformation, so unpack it with _
    data_preprocessed_df, _, _ = mdp.preprocessing.preprocess_data(
        data_consolidated,
        remove_outliers=True,
        smoothit=(11, 2),
        correlation_threshold=False,
        data_transform=True,
        standardizeit="standardize",
    )

    data_preprocessed, _, _ = mdp.preprocessing.preprocess_data(
        data_consolidated.values,
        remove_outliers=True,
        smoothit=(11, 2),
        correlation_threshold=0.9,
        data_transform="difference",
        standardizeit="01",
    )

    assert not np.isnan(np.min(data_preprocessed_df.values)) and not np.isnan(np.min(data_preprocessed))
