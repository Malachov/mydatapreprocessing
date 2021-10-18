from __future__ import annotations
from typing import TypeVar

import numpy as np
import pandas as pd

DataFrameOrArrayGeneric = TypeVar("DataFrameOrArrayGeneric", pd.DataFrame, np.ndarray)
