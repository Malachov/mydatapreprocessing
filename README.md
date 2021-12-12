# mydatapreprocessing

[![Python versions](https://img.shields.io/pypi/pyversions/mydatapreprocessing.svg)](https://pypi.python.org/pypi/mydatapreprocessing/) [![PyPI version](https://badge.fury.io/py/mydatapreprocessing.svg)](https://badge.fury.io/py/mydatapreprocessing) [![Downloads](https://pepy.tech/badge/mydatapreprocessing)](https://pepy.tech/project/mydatapreprocessing) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Malachov/mydatapreprocessing/HEAD?filepath=demo.ipynb) [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/Malachov/mydatapreprocessing.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/Malachov/mydatapreprocessing/context:python) [![Documentation Status](https://readthedocs.org/projects/mydatapreprocessing/badge/?version=latest)](https://mydatapreprocessing.readthedocs.io/?badge=latest) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![codecov](https://codecov.io/gh/Malachov/mydatapreprocessing/branch/master/graph/badge.svg)](https://codecov.io/gh/Malachov/mydatapreprocessing)

Load data from web link or local file (json, csv, Excel file, parquet, h5...), consolidate it (resample data, clean NaN values, do string embedding) derive new featurs via columns derivation and do preprocessing like
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

There are some libraries that not every user will be using (for some data inputs).
If you want to be sure to have all libraries, you can download `requirements_advanced.txt` and then install
advanced requirements with `pip install -r requirements_advanced.txt`.


## Examples

You can use live [jupyter demo on binder](https://mybinder.org/v2/gh/Malachov/mydatapreprocessing/HEAD?filepath=demo.ipynb)

<!--phmdoctest-setup-->
```python
import mydatapreprocessing as mdp
```

### Load data
You can use
- python formats (numpy.ndarray, pd.DataFrame, list, tuple, dict)
- local files
- web urls

You can load more data at once in list.

Syntax is always the same.

<!--phmdoctest-label test_load_data-->
<!--phmdoctest-share-names-->
```python
data = mdp.load_data.load_data(
    "https://www.ncdc.noaa.gov/cag/global/time-series/globe/land_ocean/ytd/12/1880-2016.json",
    request_datatype_suffix=".json",
    data_orientation="index",
    predicted_table="data",
)
# data2 = mdp.load_data.load_data([PATH_TO_FILE.csv, PATH_TO_FILE2.csv])
```

### Consolidation
If you want to use data for some machine learning models, you will probably want to remove Nan values, convert string columns to numeric if possible, do encoding or keep only numeric data and resample.

<!--phmdoctest-label test_consolidation-->
<!--phmdoctest-share-names-->
```python
data_consolidated = mdp.preprocessing.data_consolidation(
    data, predicted_column=0, remove_nans_threshold=0.9, remove_nans_or_replace="interpolate"
)
```

### Feature engineering
Functions in `feature_engineering` and `preprocessing` expects that data are in form (*n_samples*, *n_features*).
*n_samples* are ususally much bigger and therefore transformed in `data_consolidation` if necessary.

Extend original data with

<!--phmdoctest-label test_feature_engineering-->
<!--phmdoctest-share-names-->
```python
data_extended = mdp.feature_engineering.add_derived_columns(data_consolidated, differences=True, rolling_means=32)
```

### Preprocessing
`preprocess_data` returns preprocessed data, but also last undifferenced value and scaler for inverse
transformation, so unpack it with _

<!--phmdoctest-label test_preprocess_data-->
<!--phmdoctest-share-names-->
```python
data_preprocessed, _, _ = mdp.preprocessing.preprocess_data(
    data_extended,
    remove_outliers=3,
    smoothit=None,
    correlation_threshold=False,
    data_transform=False,
    standardizeit="standardize",
)
```

### Creating inputs
Create models inputs with

<!--phmdoctest-label test_create_inputs-->
<!--phmdoctest-share-names-->
```python
seqs, Y, x_input, test_inputs = mdp.create_model_inputs.make_sequences(
    data_extended.values, predicts=7, repeatit=3, n_steps_in=6, n_steps_out=1, constant=1
)
```
