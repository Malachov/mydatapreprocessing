"""Module for misc subpackage."""

from __future__ import annotations
import json
import textwrap
from typing import overload, cast, Union

import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import numpy.lib.stride_tricks

from ..preprocessing.preprocessing_functions import remove_outliers
from ..helpers import check_not_empty
from ..types import Numeric


def rolling_windows(data: np.ndarray, window: int) -> np.ndarray:
    """Generate matrix of rolling windows.

    It uses numpy slide tricks so it returns a view. Benefit is that
    it is much more memory efficient, but you must beware that if you change new array, changes will
    occur also on original data.

    Example:
        >>> rolling_windows(np.array([1, 2, 3, 4, 5]), window=2)
        array([[1, 2],
               [2, 3],
               [3, 4],
               [4, 5]])

        If you dimension bigger than 2 you can use it as well

        >>> rolling_windows(np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]), window=2)
        array([[[ 1,  2],
                [ 2,  3],
                [ 3,  4],
                [ 4,  5]],
               [[ 6,  7],
                [ 7,  8],
                [ 8,  9],
                [ 9, 10]]])

    Args:
        data (np.ndarray): Array data input.
        window (int): Number of values in created window.

    Returns:
        np.ndarray: Array of defined windows
    """
    check_not_empty(data)
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)
    return numpy.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


@overload
def split(data: pd.DataFrame, predicts: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    ...


@overload
def split(data: np.ndarray, predicts: int = 7) -> tuple[np.ndarray, np.ndarray]:
    ...


def split(data, predicts=7):
    """Divide data set on train and test set.

    Predicted column is supposed to be 0. This is mostly for time series predictions,
    so in test set there is only predicted column that can be directly used for error criterion evaluation. So this function is
    different than usual train / test split.

    Args:
        data (pd.DataFrame | np.ndarray): Time series data. ndim has to be 2, reshape if necessary.
        predicts (int, optional): Number of predicted values. Defaults to 7.

    Returns:
        tuple[pd.DataFrame | np.ndarray, pd.Series | np.ndarray]: Train set and test set. If input in numpy array, then also output in array,
        if DataFrame input, then DataFrame output.

    Example:
        >>> data = np.array([[1], [2], [3], [4]])
        >>> train, test = split(data, predicts=2)
        >>> train
        array([[1],
               [2]])
        >>> test
        array([3, 4])
    """
    if isinstance(data, pd.DataFrame):
        train = data.iloc[:-predicts, :]
        test = data.iloc[-predicts:, 0]
    else:
        train = data[:-predicts, :]
        test = data[-predicts:, 0]

    return train, test


def add_none_to_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """If empty windows in sampled signal, it will add None values (one row) to the empty window start.

    Reason is to correct plotting. Points are connected, but not between two gaps.

    Args:
        df (pd.DataFrame): DataFrame with time index.

    Returns:
        pd.DataFrame: DataFrame with None row inserted in time gaps.

    Raises:
        NotImplementedError: String values are not supported, use only numeric columns.

    Note:
        Df will be converted to float64 dtypes, to be able to use np.nan.

    Example:
        >>> data = pd.DataFrame([[0, 1]] * 7, index=[0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 2.0])
        >>> data
             0  1
        0.1  0  1
        0.2  0  1
        0.3  0  1
        1.0  0  1
        1.1  0  1
        1.2  0  1
        2.0  0  1
        >>> df_gaps = add_none_to_gaps(data)
        >>> df_gaps
               0    1
        0.1  0.0  1.0
        0.2  0.0  1.0
        0.3  0.0  1.0
        0.4  NaN  NaN
        1.0  0.0  1.0
        1.1  0.0  1.0
        1.2  0.0  1.0
        1.3  NaN  NaN
        2.0  0.0  1.0
    """
    sampling = remove_outliers(np.diff(df.index[:50]).reshape(-1, 1), threshold=1).mean()
    sampling_threshold = sampling * 3
    nons = []
    memory = None

    for i in df.index:
        if memory and i - memory > sampling_threshold:
            nons.append(
                pd.DataFrame(
                    [[np.nan] * df.shape[1]],
                    index=[memory + sampling],
                    columns=df.columns,
                )
            )
        memory = i

    try:
        result = pd.concat([df, *nons]).sort_index()
        return pd.DataFrame(result)

    except NotImplementedError as err:
        raise NotImplementedError(
            "If object dtype in DataFrame, it will fail. Use only numeric dtypes."
        ) from err


def edit_table_to_printable(
    df: pd.DataFrame,
    line_length_limit: int = 16,
    round_decimals: int = 3,
    number_length_limit: Numeric = 10e8,
) -> pd.DataFrame:
    """Edit DataFrame to be able to use in tabulate (or somewhere else).

    Args:
        df (pd.DataFrame): Input data with numeric or text columns.
        line_length_limit (int, optional): Add line breaks if line too long. Defaults to 16.
        round_decimals (int, optional): Round numeric columns to defined decimals. Defaults to 3.
        number_length_limit (Numeric, optional): If there is some very big or very small number,
            convert format to scientific notation. Defaults to 10e8.

    Returns:
        pd.DataFrame: DataFrame with shorter and more readable to be printed (usually in table).

    Note:
        DataFrame column names can be changed (``'\\n'`` is added).

    Example:
        >>> df = pd.DataFrame([[151646516516, 1.5648646, "Lorem ipsum something else"], [1, 2, "3"]])
        >>> for_table = edit_table_to_printable(df).values[0]
        >>> for_table[0]
        '1.516e+11'
        >>> for_table[1]
        1.565
        >>> for_table[2]
        'Lorem ipsum\\nsomething else'
    """
    df.columns = [
        (textwrap.fill(i, line_length_limit) if (isinstance(i, str) and len(i)) > line_length_limit else i)
        for i in df.columns
    ]

    for _, df_i in df.iteritems():

        if is_numeric_dtype(df_i):
            # Replace very big numbers with scientific notation
            if df_i.max() > number_length_limit or df_i.min() < -number_length_limit:
                for k, l in df_i.iteritems():
                    if l > number_length_limit or l < -number_length_limit:
                        df_i[k] = f"{l:.3e}"

        else:
            for k, l in df_i.iteritems():
                k = cast(Union[str, int], k)
                # Add line breaks to long strings
                if isinstance(l, str) and len(l) > line_length_limit:
                    df_i[k] = textwrap.fill(df_i[k], line_length_limit)
                # Convert dictionaries to formated strings
                if isinstance(l, dict):
                    for i, j in df_i[k].items():
                        if isinstance(j, (int, float)):
                            if j > 10**-round_decimals:
                                df_i[k][i] = round(j, round_decimals)
                            if j > number_length_limit or j < -number_length_limit:
                                df_i[k][i] = f"{j:.3e}"

                    df_i[k] = json.dumps(l, indent=2)

    df = df.round(round_decimals)

    return df
