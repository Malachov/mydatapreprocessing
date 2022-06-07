from __future__ import annotations

import pandas as pd
import numpy as np


def compare_values(values1, values2):
    """Ignore dtype."""
    values_str_1 = pd.DataFrame(values1).select_dtypes("O")
    values_str_2 = pd.DataFrame(values2).select_dtypes("O")

    values_numeric_1 = pd.DataFrame(values1).select_dtypes("number")
    values_numeric_2 = pd.DataFrame(values2).select_dtypes("number")

    return np.all(values_str_1.values == values_str_2.values) and np.allclose(
        values_numeric_1.values, values_numeric_2.values
    )
