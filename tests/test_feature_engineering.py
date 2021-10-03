import numpy as np
import pandas as pd

import mypythontools

mypythontools.tests.setup_tests()


import mydatapreprocessing.feature_engineering as mdpf

np.random.seed(2)


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
