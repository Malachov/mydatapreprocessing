# mydatapreprocessing

[![Python versions](https://img.shields.io/pypi/pyversions/mydatapreprocessing.svg)](https://pypi.python.org/pypi/mydatapreprocessing/) [![PyPI version](https://badge.fury.io/py/mydatapreprocessing.svg)](https://badge.fury.io/py/mydatapreprocessing) [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/Malachov/mydatapreprocessing.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/Malachov/mydatapreprocessing/context:python) [![Build Status](https://travis-ci.com/Malachov/mydatapreprocessing.svg?branch=master)](https://travis-ci.com/Malachov/mydatapreprocessing) [![Documentation Status](https://readthedocs.org/projects/mydatapreprocessing/badge/?version=latest)](https://mydatapreprocessing.readthedocs.io/en/latest/?badge=latest) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![codecov](https://codecov.io/gh/Malachov/mydatapreprocessing/branch/master/graph/badge.svg)](https://codecov.io/gh/Malachov/mydatapreprocessing)

Library contain 3 modules - preprocessing, inputs and generatedata.

## Installation

Python >=3.6 (Python 2 is not supported).

Install just with

```console
pip install mydatapreprocessing
```

There are some libraries that not every user will be using (for some data inputs).
If you want to be sure to have all libraries, you can download `requirements_advanced.txt` and then install
advanced requirements with `pip install -r requirements_advanced.txt`.

## Preprocessing

Load data from web link or local file (json, csv, excel file, parquet, h5...), consolidate it (to pandas dataframe)
and do preprocessing like resampling, standardization, string embedding, new columns derivation.
If you want to see how functions work - working examples with printed results are in tests - visual.py.

There are many small functions, but there they are called automatically with main preprocess functions.

    - load_data
    - data_consolidation
    - preprocess_data
    - preprocess_data_inverse

Note:
    In data consolidation, predicted column is moved on index 0 !!!

### Example

```python
import mydatapreprocessing.preprocessing as mdpp
```


You can use local files as well as web urls

```python

data1 = mdpp.load_data(PATH_TO_FILE.csv)

data2 = mdpp.load_data(
    "https://blockchain.info/unconfirmed-transactions?format=json",
    request_datatype_suffix=".json",
    data_orientation="index",
    predicted_table="txs",
)  
```

Transform various data into defined format - pandas dataframe - convert to numeric if possible, keep
only numeric data and resample ifg configured. It return array, dataframe

```python
data_consolidated = mdpp.data_consolidation(
    data_loaded, predicted_column="weight", remove_nans_threshold=0.9, remove_nans_or_replace="interpolate"
)
```

`preprocess_data` returns preprocessed data, but also last undifferenced value and scaler for inverse transformation, so unpack it with _

data_preprocessed, _, _ = mdpp.preprocess_data(
    data_consolidated,
    remove_outliers=True,
    smoothit=False,
    correlation_threshold=False,
    data_transform=False,
    standardizeit="standardize",
)


Allowed data formats for load_data are examples

    myarray_or_dataframe # Numpy array or Pandas.DataFrame
    r"/home/user/my.json" # Local file. The same with .parquet, .h5, .json or .xlsx.
    "https://yoururl/your.csv" # Web url (with suffix). Same with json.
    "https://blockchain.info/unconfirmed-transactions?format=json" # In this case you have to specify
        also 'request_datatype_suffix': "json", 'data_orientation': "index", 'predicted_table': 'txs',
    [{'col_1': 3, 'col_2': 'a'}, {'col_1': 0, 'col_2': 'd'}] # List of records
    {'col_1': [3, 2, 1, 0], 'col_2': ['a', 'b', 'c', 'd']} # Dict with colums or rows (index) - necessary
        to setup data_orientation!

You can use more files in list and data will be concatenated. It can be list of paths or list of python objects. For example::

    [{'col_1': 3, 'col_2': 'a'}, {'col_1': 0, 'col_2': 'd'}]  # List of records
    [np.random.randn(20, 3), np.random.randn(25, 3)]  # Dataframe same way
    ["https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"]  # List of URLs
    ["path/to/my1.csv", "path/to/my1.csv"]

On windows it's necessary to use raw string - 'r' in front of string because of escape symbols \

## Inputs

It take tabular time series data and put it into format (input vector X, output vector y and input for predicted value x_input) that can be inserted into machine learning models for example on sklearn or tensorflow. It contain functions make_sequences, create_inputs and create_tests_outputs

### Example

```python
import mydatapreprocessing.inputs as mdpi

data = np.array([[1, 3, 5, 2, 3, 4, 5, 66, 3]]).T
seqs, Y, x_input, test_inputs = mdpi.inputs.make_sequences(data, predicts=7, repeatit=3, n_steps_in=6, n_steps_out=1, constant=1)
```

## generatedata

This module generate data that can be used for example for validating machine learning time series prediction results. It can define data like sig, sign, ramp signal or download ECG heart signal.

### Example

```python
import mydatapreprocessing as mdp

data = mdp.generatedata.gen_sin(1000)
```
