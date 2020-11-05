# mydatapreprocessing

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/mydatapreprocessing.svg)](https://pypi.python.org/pypi/mydatapreprocessing/) [![PyPI version](https://badge.fury.io/py/mydatapreprocessing.svg)](https://badge.fury.io/py/mydatapreprocessing) [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/Malachov/mydatapreprocessing.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/Malachov/mydatapreprocessing/context:python) [![Build Status](https://travis-ci.com/Malachov/mydatapreprocessing.svg?branch=master)](https://travis-ci.com/Malachov/mydatapreprocessing) [![Documentation Status](https://readthedocs.org/projects/mydatapreprocessing/badge/?version=latest)](https://mydatapreprocessing.readthedocs.io/en/latest/?badge=latest) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![codecov](https://codecov.io/gh/Malachov/mydatapreprocessing/branch/master/graph/badge.svg)](https://codecov.io/gh/Malachov/mydatapreprocessing)

Load data from web link or local file (json, csv, excel file, parquet, h5...), consolidate it and do preprocessing like resampling, standardization, string embedding, new columns derivation, feature extraction etc. based on configuration.

Library contain 3 modules.

## Preprocessing

First - `preprocessing` load data, consolidate it and do the preprocessing. It contains functions like load_data, data_consolidation, preprocess_data, preprocess_data_inverse, add_frequency_columns, rolling_windows, add_derived_columns etc.

### Example

```python
### Preprocessing module

import mydatapreprocessing.preprocessing as mdpp

data = "https://blockchain.info/unconfirmed-transactions?format=json"

# Load data from file or URL
data_loaded = mdpp.load_data(data, request_datatype_suffix=".json", predicted_table='txs')


#Some examples of other inputs to data_load function

# myarray_or_dataframe # Numpy array or Pandas.DataFrame
# r"/home/user/my.json" # Local file. The same with .parquet, .h5, .json or .xlsx. On windows it's necessary to use raw string - 'r' in front of string because of escape symbols \
# "https://yoururl/your.csv" # Web url (with suffix). Same with json.
# "https://blockchain.info/unconfirmed-transactions?format=json" # In this case you have to specify also 'request_datatype_suffix': "json", 'data_orientation': "index", 'predicted_table': 'txs',
# {'col_1': [3, 2, 1, 0], 'col_2': ['a', 'b', 'c', 'd']} # Dict with colums or rows (index) - necessary to setup data_orientation!


# You can use more files in list and data will be concatenated. It can be list of paths or list of python objects. Example:

# [{'col_1': 3, 'col_2': 'a'}, {'col_1': 0, 'col_2': 'd'}]  # List of records
# [np.random.randn(20, 3), np.random.randn(25, 3)]  # Dataframe same way
# ["https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv", "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"]  # List of URLs
# ["path/to/my1.csv", "path/to/my1.csv"]


# Transform various data into defined format - pandas dataframe - convert to numeric if possible, keep
# only numeric data and resample ifg configured. It return array, dataframe
data_consolidated = mdpp.data_consolidation(
    data_loaded, predicted_column="weight", data_orientation="index", remove_nans_threshold=0.9, remove_nans_or_replace='interpolate')

# You can add some extra informations to the data that can help (beware it can slow down the machine learning model)
to_be_extended = np.array([[0, 2] * 64, [0, 0, 0, 5] * 32]).T
extended = mdpp.add_frequency_columns(to_be_extended, window=8)


to_be_extended2 = pd.DataFrame([range(30), range(30, 60)]).T
extended2 = mdpp.add_derived_columns(to_be_extended2, differences=True, second_differences=True, multiplications=True,
                                    rolling_means=True, rolling_stds=True, mean_distances=True, window=10)

# Feature extraction is under development  :[

# Preprocess data. It return preprocessed data, but also last undifferenced value and scaler for inverse
# transformation, so unpack it with _
data_preprocessed, _, _ = mdpp.preprocess_data(data_consolidated, remove_outliers=True, smoothit=False,
                                              correlation_threshold=False, data_transform=False, standardizeit='standardize')
```

## Inputs

Second module is `inputs`. It take tabular time series data and put it into format (input vector X, output vector y and input for predicted value x_input) that can be inserted into machine learning models for example on sklearn or tensorflow. It contain functions make_sequences, create_inputs and create_tests_outputs

### Example

```python
import mydatapreprocessing.inputs as mdpi

data = np.array([[1, 3, 5, 2, 3, 4, 5, 66, 3]]).T
seqs, Y, x_input, test_inputs = mdpi.inputs.make_sequences(data, predicts=7, repeatit=3, n_steps_in=6, n_steps_out=1, constant=1)

```

Third module is `generatedata`. It generate some basic data like sin, ramp random. In the future, it will also import some real datasets for models KPI.
