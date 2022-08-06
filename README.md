# mydatapreprocessing

[![Python versions](https://img.shields.io/pypi/pyversions/mydatapreprocessing.svg)](https://pypi.python.org/pypi/mydatapreprocessing/) [![PyPI version](https://badge.fury.io/py/mydatapreprocessing.svg)](https://badge.fury.io/py/mydatapreprocessing) [![Downloads](https://pepy.tech/badge/mydatapreprocessing)](https://pepy.tech/project/mydatapreprocessing) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Malachov/mydatapreprocessing/HEAD?filepath=demo.ipynb) [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/Malachov/mydatapreprocessing.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/Malachov/mydatapreprocessing/context:python) [![Documentation Status](https://readthedocs.org/projects/mydatapreprocessing/badge/?version=latest)](https://mydatapreprocessing.readthedocs.io/?badge=latest) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![codecov](https://codecov.io/gh/Malachov/mydatapreprocessing/branch/master/graph/badge.svg)](https://codecov.io/gh/Malachov/mydatapreprocessing)

Load data from web link or local file (json, csv, Excel file, parquet, h5...), consolidate it (resample data, clean NaN values, do string embedding) derive new features via columns derivation and do preprocessing like
standardization or smoothing. If you want to see how functions works, check it's docstrings - working examples with printed results are also in tests - visual.py.

## Links

[Repo on GitHub](https://github.com/Malachov/mydatapreprocessing)

[Official readthedocs documentation](https://mydatapreprocessing.readthedocs.io)


## Installation

Python >=3.6 (Python 2 is not supported).

Install just with

```console
pip install mydatapreprocessing
```

There are some libraries that not every user will be using (for some specific data inputs for example). If you want to be sure to have all libraries, you can provide extras requirements like.

```console
pip install mydatapreprocessing[datatypes]
```

Available extras are ["all", "datasets", "datatypes"]


## Examples

You can use live [jupyter demo on binder](https://mybinder.org/v2/gh/Malachov/mydatapreprocessing/HEAD?filepath=demo.ipynb)

<!--phmdoctest-setup-->
```python
import mydatapreprocessing as mdp
import pandas as pd
import numpy as np
```

### Load data

You can use:

- python formats (numpy.ndarray, pd.DataFrame, list, tuple, dict)
- local files
- web urls

Supported path formats are:

- csv
- xlsx and xls
- json
- parquet
- h5

You can load more data at once in list.

Syntax is always the same.

<!--phmdoctest-label test_load_data-->
<!--phmdoctest-share-names-->
```python
data = mdp.load_data.load_data(
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
)
# data2 = mdp.load_data.load_data([PATH_TO_FILE.csv, PATH_TO_FILE2.csv])
```

### Consolidation
If you want to use data for some machine learning models, you will probably want to remove Nan values, convert string columns to numeric if possible, do encoding or keep only numeric data and resample.

Consolidation is working with pandas DataFrame as column names matters here.

There are many functions, but there is main function pipelining other functions `consolidate_data`


<!--phmdoctest-label test_consolidation-->
<!--phmdoctest-share-names-->
```python
consolidation_config = mdp.consolidation.consolidation_config.default_consolidation_config.do.copy()
consolidation_config.datetime.datetime_column = 'Date'
consolidation_config.resample.resample = 'M'
consolidation_config.resample.resample_function = "mean"
consolidation_config.dtype = 'float32'

consolidated = mdp.consolidation.consolidate_data(data, consolidation_config)
print(consolidated.head())
```

### Feature engineering
Functions in `feature_engineering` and `preprocessing` expects that data are in form (*n_samples*, *n_features*).
*n_samples* are usually much bigger and therefore transformed in `consolidate_data` if necessary.

In config, you can use shorter update dict syntax as all values names are unique.

### Feature engineering

Create new columns that can be for example used as another machine learning model input.

```python
import mydatapreprocessing.feature_engineering as mdpf
import mydatapreprocessing as mdp

data = pd.DataFrame(
    [mdp.datasets.sin(n=30), mdp.datasets.ramp(n=30)]
).T

extended = mdpf.add_derived_columns(data, differences=True, rolling_means=10)
print(extended.columns)
print(f"\nit has less rows then on input {len(extended)}")
```

Functions in `feature_engineering` and `preprocessing` expects that data are in form (n_samples, n_features). n_samples are usually much bigger and therefore transformed in `consolidate_data`
if necessary.

### Preprocessing

Preprocessing can be used on pandas DataFrame as well as on numpy array. Column names are not important as it's just matrix with defined dtype.

There is many functions, but there is main function pipelining other functions `preprocess_data` Preprocessed data can be converted back with `preprocess_data_inverse`


<!--phmdoctest-label test_preprocess_data-->
<!--phmdoctest-share-names-->
```python

from mydatapreprocessing import preprocessing as mdpp

df = pd.DataFrame(np.array([range(5), range(20, 25), np.random.randn(5)]).astype("float32").T)
df.iloc[2, 0] = 500

config = mdpp.preprocessing_config.default_preprocessing_config.do.copy()
config.do.update({"remove_outliers": None, "difference_transform": True, "standardize": "standardize"})
data_preprocessed, inverse_config = mdpp.preprocess_data(df.values, config)
inverse_config.difference_transform = df.iloc[0, 0]
data_preprocessed_inverse = mdpp.preprocess_data_inverse(
    data_preprocessed[:, 0], inverse_config
)
```
