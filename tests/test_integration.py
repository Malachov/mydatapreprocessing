import numpy as np

import mypythontools

# Find paths and add to sys.path to be able to import local modules
mypythontools.tests.setup_tests()

import mydatapreprocessing as mdp

np.random.seed(2)


def test_integration():
    # Load data from file or URL
    data_loaded = mdp.load_data.load_data(
        "https://www.ncdc.noaa.gov/cag/global/time-series/globe/land_ocean/ytd/12/1880-2016.json",
        request_datatype_suffix=".json",
        predicted_table="data",
        data_orientation="index",
    )

    # Transform various data into defined format - pandas dataframe - convert to numeric if possible, keep
    # only numeric data and resample ifg configured. It return array, dataframe
    data_consolidated = mdp.preprocessing.data_consolidation(
        data_loaded,
        predicted_column=0,
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
