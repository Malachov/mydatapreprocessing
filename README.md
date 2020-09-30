# mydatapreprocessing

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/mydatapreprocessing.svg)](https://pypi.python.org/pypi/mydatapreprocessing/) [![PyPI version](https://badge.fury.io/py/mydatapreprocessing.svg)](https://badge.fury.io/py/mydatapreprocessing) [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/Malachov/mydatapreprocessing.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/Malachov/mydatapreprocessing/context:python) [![Build Status](https://travis-ci.com/Malachov/mydatapreprocessing.svg?branch=master)](https://travis-ci.com/Malachov/mydatapreprocessing) [![Documentation Status](https://readthedocs.org/projects/mydatapreprocessing/badge/?version=master)](https://mydatapreprocessing.readthedocs.io/en/master/?badge=master) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![codecov](https://codecov.io/gh/Malachov/mydatapreprocessing/branch/master/graph/badge.svg)](https://codecov.io/gh/Malachov/mydatapreprocessing)

Load data web link or local file(json, csv, excel file, parquet, parquet...), consolidate it, do preprocessing like resampling standardization, string embedding, generating new columns, feature extraction etc. based on configuration.

Library contain 3 modules.

First - `preprocessing` load data and preprocess it. It contains functions like load_data, data_consolidation, preprocess_data, preprocess_data_inverse, add_frequency_columns, rolling_windows, add_derived_columns etc.

## Example

```python
data = "https://blockchain.info/unconfirmed-transactions?format=json"

# Load data from file or URL
data_loaded = mdp.load_data(data, request_datatype_suffix=".json", predicted_table='txs')

# Transform various data into defined format - pandas dataframe - convert to numeric if possible, keep
# only numeric data and resample ifg configured. It return array, dataframe
data_consolidated = mdp.data_consolidation(
    data_loaded, predicted_column="weight", data_orientation="index", remove_nans_threshold=0.9, remove_nans_or_replace='interpolate')

# Preprocess data. It return preprocessed data, but also last undifferenced value and scaler for inverse
# transformation, so unpack it with _
data_preprocessed, _, _ = mdp.preprocess_data(data_consolidated, remove_outliers=True, smoothit=False,
                                              correlation_threshold=False, data_transform=False, standardizeit='standardize')

```

Allowed data formats for load_data are examples

```python
# myarray_or_dataframe # Numpy array or Pandas.DataFrame
# r"/home/user/my.json" # Local file. The same with .parquet, .h5, .json or .xlsx. On windows it's necessary to use raw string - 'r' in front of string because of escape symbols \
# "https://yoururl/your.csv" # Web url (with suffix). Same with json.
# "https://blockchain.info/unconfirmed-transactions?format=json" # In this case you have to specify also 'request_datatype_suffix': "json", 'data_orientation': "index", 'predicted_table': 'txs',
# [{'col_1': 3, 'col_2': 'a'}, {'col_1': 0, 'col_2': 'd'}] # List of records
# {'col_1': [3, 2, 1, 0], 'col_2': ['a', 'b', 'c', 'd']} # Dict with colums or rows (index) - necessary to setup data_orientation!
```

Second module is `inputs`. It take tabular time series data and put it into format that can be inserted into machine learning models for example on sklearn or tensorflow. It contain functions make_sequences, create_inputs and create_tests_outputs

Third module is `generatedata`. It generate some basic data like sin, ramp random. In the future, it will also import some real datasets for models KPI.
