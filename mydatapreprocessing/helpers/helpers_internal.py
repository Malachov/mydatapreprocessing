"""Module for helpers subpackage."""

from __future__ import annotations
from typing import overload

import numpy as np
import pandas as pd

from ..types import DataFrameOrArrayGeneric, PandasIndex


def check_column_in_df(df: pd.DataFrame, name_or_index: PandasIndex, source: None | str = None) -> None:
    """If defined column is not in DataFrame, it raise Error.

    Args:
        df (pd.DataFrame): Input data.
        name_or_index (PandasIndex): Integer index, name or pandas.Index.
        source (str, optional): In raised message wanted column can be referenced. Defaults to None.

    Raises:
        KeyError: If column not found in DataFrame.

    Example:
        >>> df = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
        >>> check_column_in_df(df, "a")
        >>> check_column_in_df(df, "z")
        Traceback (most recent call last):
        KeyError...
    """
    if isinstance(name_or_index, str):
        if not name_or_index in df.columns:
            caption = "Column not found error." if not source else f"Error in '{source}.'"
            raise KeyError(
                f"{caption} Column '{name_or_index}' not found in data. Available columns: {list(df.columns)}"
            )

    elif isinstance(name_or_index, (int, np.integer)):
        if name_or_index >= len(df.columns):
            raise ValueError(
                f"Column '{name_or_index}' not available as df has only {df.shape[1]} columns.",
            )

    elif isinstance(name_or_index, pd.Index):
        if name_or_index.size == 1:
            column_name = name_or_index.values[0]
            check_column_in_df(df, column_name)
        else:
            raise ValueError(
                "If checking column in DataFrame with pandas.Index, it has to have size 1.",
            )

    else:
        raise TypeError("Checking column in DataFrame is possible only with PandasIndex types.")


@overload
def get_column_name(df: pd.DataFrame, index: str | int | np.integer) -> str:
    ...


@overload
def get_column_name(df: pd.DataFrame, index: pd.Index) -> pd.Index:
    ...


def get_column_name(df: pd.DataFrame, index: PandasIndex) -> str | pd.Index:
    """Return index that can be used to access column directly.

    In user input the column can be defined by name or by it's index. Then selecting the column has
    different syntax. It's verified whether column is available. If it's integer index, it's converted
    to string so the syntax is always the same.

    Args:
        df (pd.DataFrame): Input data
        index (PandasIndex): Also integer index.

    Example:
        >>> df = pd.DataFrame([[1, 2, 3]], columns=["a", "b", "c"])
        >>> get_column_name(df, "b")
        'b'
        >>> get_column_name(df, 2)
        'c'
        >>> get_column_name(df, "z")
        Traceback (most recent call last):
        KeyError...
    """
    check_column_in_df(df, index)

    if isinstance(index, str):
        return index

    elif isinstance(index, (np.integer, int)):
        return df.columns.values[index]

    elif isinstance(index, pd.Index):
        return index

    else:
        raise TypeError("Column type must be one of str | int | np.integer | pd.Index.")


def get_copy_or_view(data: DataFrameOrArrayGeneric, inplace: bool) -> DataFrameOrArrayGeneric:
    """As DataFrame copy function needs to be casted for correct type hints this helps to solve it.

    Args:
        data (DataFrameOrArrayGeneric): Input data
        inplace (bool): Whether to return copy or not.

    Returns:
        DataFrameOrArrayGeneric: Copy or original data.

    Example:
        >>> a = np.array([1, 2, 3])
        >>> b = get_copy_or_view(a, inplace=True)
        >>> id(a) == id(b)
        True
        >>> b = get_copy_or_view(a, inplace=False)
        >>> id(a) == id(b)
        False
    """
    if inplace:
        return data

    if isinstance(data, pd.DataFrame):
        df_copy = data.copy()
        return df_copy

    if isinstance(data, np.ndarray):
        return data.copy()


def check_not_empty(data: DataFrameOrArrayGeneric):
    """Check whether there are data. It can happen that in some functions empty data would result error.

    Args:
        data (DataFrameOrArrayGeneric): Data

    Raises:
        TypeError: If `data.size == 0`
    """
    if not data.size:
        if isinstance(data, np.ndarray):
            used_type = "numpy ndarray"
        else:
            used_type = "pandas DataFrame"
        raise TypeError(f"Used {used_type} is empty which would cause further error.")
