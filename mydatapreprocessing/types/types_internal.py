"""Module for types subpackage."""

from __future__ import annotations
from typing import TypeVar, Any, Sequence

from typing_extensions import Literal
import numpy as np
import pandas as pd

from mypythontools.paths import PathLike

DataFrameOrArrayGeneric = TypeVar("DataFrameOrArrayGeneric", pd.DataFrame, np.ndarray)
Numeric = float | int | np.number
PandasIndex = str | int | pd.Index | np.integer
DataFormat = (
    PathLike
    | pd.DataFrame
    | np.ndarray
    | Sequence[dict[str, Any]]
    | Sequence[Sequence[Any]]
    | dict[str, Sequence[Any]]
    | Sequence[PathLike | pd.DataFrame | np.ndarray | Sequence[dict[str, Any]] | dict[str, Sequence[Any]]]
    | Literal["test_ramp", "test_sin", "test_random", "test_ecg"]
)
