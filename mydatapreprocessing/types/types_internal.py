"""Module for types subpackage."""

from __future__ import annotations
from typing import TypeVar, Any, Sequence, Union, Dict

from typing_extensions import Literal
import numpy as np
import pandas as pd

from mypythontools.paths import PathLike

DataFrameOrArrayGeneric = TypeVar("DataFrameOrArrayGeneric", pd.DataFrame, np.ndarray)
Numeric = Union[float, int, np.number]
PandasIndex = Union[str, int, pd.Index, np.integer]
DataFormat = Union[
    PathLike,
    pd.DataFrame,
    np.ndarray,
    Sequence[Dict[str, Any]],
    Sequence[Sequence[Any]],
    Dict[str, Sequence[Any]],
    Sequence[Union[PathLike, pd.DataFrame, np.ndarray, Sequence[Dict[str, Any]], Dict[str, Sequence[Any]]]],
    Literal["test_ramp", "test_sin", "test_random", "test_ecg"],
]
