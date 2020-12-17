# -*- coding: utf-8 -*-

"""
Load data from web link or local file (json, csv, excel file, parquet, h5...), consolidate it
and do preprocessing like resampling, standardization, string embedding, new columns derivation,
feature extraction etc. based on configuration.

Library contain 3 modules.

First - `preprocessing` load data and preprocess it. It contains functions like load_data,
data_consolidation, preprocess_data, preprocess_data_inverse, add_frequency_columns, rolling_windows,
add_derived_columns etc.

Examples:

    >>> import mydatapreprocessing.preprocessing as mdpp

    >>> data = "https://blockchain.info/unconfirmed-transactions?format=json"

    >>> # Load data from file or URL
    >>> data_loaded = mdpp.load_data(data, request_datatype_suffix=".json", predicted_table='txs')

    >>> # Transform various data into defined format - pandas dataframe - convert to numeric if possible, keep
    >>> # only numeric data and resample ifg configured. It return array, dataframe
    >>> data_consolidated = mdpp.data_consolidation(
    >>>     data_loaded, predicted_column="weight", data_orientation="index", remove_nans_threshold=0.9, remove_nans_or_replace='interpolate')

    >>> # Preprocess data. It return preprocessed data, but also last undifferenced value and scaler for inverse
    >>> # transformation, so unpack it with _
    >>> data_preprocessed, _, _ = mdpp.preprocess_data(data_consolidated, remove_outliers=True, smoothit=False,
    >>>                                               correlation_threshold=False, data_transform=False, standardizeit='standardize')


    >>> # Allowed data formats for load_data are examples

    >>> # myarray_or_dataframe # Numpy array or Pandas.DataFrame
    >>> # r"/home/user/my.json" # Local file. The same with .parquet, .h5, .json or .xlsx. On windows it's necessary to use raw string - 'r' in front of string because of escape symbols \
    >>> # "https://yoururl/your.csv" # Web url (with suffix). Same with json.
    >>> # "https://blockchain.info/unconfirmed-transactions?format=json" # In this case you have to specify also 'request_datatype_suffix': "json", 'data_orientation': "index", 'predicted_table': 'txs',
    >>> # [{'col_1': 3, 'col_2': 'a'}, {'col_1': 0, 'col_2': 'd'}] # List of records
    >>> # {'col_1': [3, 2, 1, 0], 'col_2': ['a', 'b', 'c', 'd']} # Dict with colums or rows (index) - necessary to setup data_orientation!

    >>> # You can use more files in list and data will be concatenated. It can be list of paths or list of python objects. Example:

    >>> # [{'col_1': 3, 'col_2': 'a'}, {'col_1': 0, 'col_2': 'd'}]  # List of records
    >>> # [np.random.randn(20, 3), np.random.randn(25, 3)]  # Dataframe same way
    >>> # ["https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv", "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"]  # List of URLs
    >>> # ["path/to/my1.csv", "path/to/my1.csv"]

Second module is `inputs`. It take tabular time series data and put it into format that can be inserted
into machine learning models for example on sklearn or tensorflow. It contain functions make_sequences,
create_inputs and create_tests_outputs


Examples:

    >>> import mydatapreprocessing.inputs as mdpi

    >>> data = np.array([[1, 3, 5, 2, 3, 4, 5, 66, 3]]).T
    >>> seqs, Y, x_input, test_inputs = mdpi.inputs.make_sequences(data, predicts=7, repeatit=3, n_steps_in=6, n_steps_out=1, constant=1)

    >>> # Now you can insert X, Y, and x_input into scikit model.

Third module is `generatedata`. It generate some basic data like sin, ramp random. In the future,
it will also import some real datasets for models KPI.

"""

from . import preprocessing
from . import inputs
from . import generatedata

__version__ = "1.1.13"
__author__ = "Daniel Malachov"
__license__ = "MIT"
__email__ = "malachovd@seznam.cz"

__all__ = ['preprocessing', 'inputs', 'generatedata']
