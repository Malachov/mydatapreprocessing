import numpy as np
import pandas as pd
import sys
from pathlib import Path
import inspect
import os

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

import mydatapreprocessing.feature_engineering as mdpf


def test_add_frequency_columns():
    data = np.array([[0, 2] * 64, [0, 0, 0, 5] * 32]).T
    mdpf.add_frequency_columns(data, window=8)


def test_add_derived_columns():
    data = pd.DataFrame([range(30), range(30, 60)]).T
    mdpf.add_derived_columns(
        data,
        differences=True,
        second_differences=True,
        multiplications=True,
        rolling_means=10,
        rolling_stds=10,
        mean_distances=True,
    )
