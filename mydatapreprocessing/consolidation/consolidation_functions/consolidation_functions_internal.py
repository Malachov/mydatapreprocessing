"""Module for consolidation_functions subpackage."""

from __future__ import annotations
from typing import Any

from typing_extensions import Literal
import pandas as pd

import mylogging

from ...types import DataFrameOrArrayGeneric, PandasIndex, Numeric
from ...helpers import get_copy_or_view, get_column_name

# Lazy loaded
# from pandas.tseries.frequencies import to_offset

# TODO implement inplace parameter as in preprocessing


def check_shape_and_transform(data: DataFrameOrArrayGeneric, inplace=False) -> DataFrameOrArrayGeneric:
    """Check whether input data has expected shape.

    Some functions work with defined shape of data - (n_samples, n_features). If this is not the case,
    it will transpose the data and log that it happened.

    Args:
        data (DataFrameOrArrayGeneric): Input data.
        inplace (bool, optional): If True, then original data are edited. If False, copy is created.
            Defaults to False.

    Returns:
        DataFrameOrArrayGeneric: Data with verified shape.

    Example:
        >>> import numpy as np
        >>> data = np.array([range(10), range(10)])
        >>> data.shape
        (2, 10)
        >>> data = check_shape_and_transform(data)
        >>> data.shape
        (10, 2)
    """
    if data.shape[0] < data.shape[1]:
        data = get_copy_or_view(data, inplace)
        mylogging.info(
            "Input data must be in shape (n_samples, n_features) that means (rows, columns) Your shape is "
            f" {data.shape}. It's unusual to have more features than samples. Probably wrong shape.",
            caption="Data transposed warning!!!",
        )
        data = data.T

    return data


def categorical_embedding(
    data: pd.DataFrame,
    embedding: Literal["label", "one-hot"] = "label",
    unique_threshold: Numeric = 0.6,
    inplace=False,
) -> pd.DataFrame:
    """Transform string categories such as 'US', 'FR' into numeric values.

    This is necessary for example in machine learnings models.

    Args:
        data (pd.DataFrame): Data with string (pandas Object dtype) columns.
        embedding("label", "one-hot", optional): 'label' or 'one-hot'. Categorical encoding. Create numbers
            from strings. 'label' give each category (unique string) concrete number. Result will have same
            number of columns. 'one-hot' create for every category new column. Only columns, where are strings
            repeating (unique_threshold) will be used. Defaults to "label".
        unique_threshold(Numeric, optional): Remove string columns, that have to many categories (ids, hashes
            etc.). E.g 0.9 defines that in column of length 100, max number of categories to not to be deleted
            is 10 (90% non unique repeating values). Defaults to 0.6. Min is 0, max is 1. Defaults is 0.6.
        inplace (bool, optional): If True, then original data are edited. If False, copy is created.
            Defaults to False.

    Returns:
        pd.DataFrame: DataFrame where string columns transformed to numeric.

    Raises:
        TypeError: If there is unhashable object in values for example.

    Example:
        >>> df = pd.DataFrame(["One", "Two", "One", "Three", "One"])
        >>> categorical_embedding(df, embedding="label", unique_threshold=0.1)
           0
        0  0
        1  2
        2  0
        3  1
        4  0
        >>> categorical_embedding(df, embedding="one-hot", unique_threshold=0.1)
           One  Three  Two
        0    1      0    0
        1    0      0    1
        2    1      0    0
        3    0      1    0
        4    1      0    0
    """
    data = get_copy_or_view(data, inplace)

    to_drop = []

    for i in data.select_dtypes(
        exclude=["number"],
    ):

        try:
            if (data[i].nunique() / len(data[i])) >= (1 - unique_threshold):
                to_drop.append(i)
                continue
        except TypeError:
            to_drop.append(i)
            continue

        data[i] = data[i].astype("category", copy=False)

        if embedding == "label":
            data[i] = data[i].cat.codes

        if embedding == "one-hot":
            data = data.join(pd.get_dummies(data[i]))
            to_drop.append(i)

    # Drop columns with too few categories - drop all columns at once to better performance
    data.drop(to_drop, axis=1, inplace=True)

    return data


def set_datetime_index(
    df: pd.DataFrame,
    name_or_index: PandasIndex,
    on_error: Literal["ignore", "raise"] = "ignore",
    inplace: bool = False,
) -> pd.DataFrame:
    """Set defined column as index and convert it to datetime.

    Args:
        df (pd.DataFrame): Input data.
        name_or_index (PandasIndex): Name or index of datetime column that will be set as index.
            Defaults to None.
        on_error (Literal["ignore", "raise"]): What happens if converting to datetime fails.
            Defaults to "ignore".
        inplace (bool, optional): If True, then original data are edited. If False, copy is created.
            Defaults to False.

    Raises:
        ValueError: If defined column failed to convert to datetime.

    Returns:
        pd.DataFrame: Data with datetime index.

    Example:
        >>> from datetime import datetime
        ...
        >>> df = pd.DataFrame(
        ...     {
        ...         "col_1": [1] * 3,
        ...         "col_2": [2] * 3,
        ...         "date": [
        ...             datetime(2022, 1, 1),
        ...             datetime(2022, 2, 1),
        ...             datetime(2022, 3, 1),
        ...         ],
        ...     }
        ... )
        >>> df = set_datetime_index(df, 'date', inplace=True)
        >>> isinstance(df.index, pd.DatetimeIndex)
        True
    """
    df = get_copy_or_view(df, inplace)

    index_name = get_column_name(df, name_or_index)

    df.set_index(index_name, drop=True, inplace=True)

    try:
        df.index = pd.to_datetime(df.index)  # type: ignore

    except ValueError as err:
        if on_error == "raise":
            raise ValueError(
                "Error in 'mydatapreprocessing' package in 'set_datetime_index' function. Setting of "
                f"datetime index from column '{index_name}' failed.",
            ) from err

    return df


def infer_frequency(
    df: pd.DataFrame, on_error: Literal[None, "warn", "raise"] = "warn", inplace=False
) -> pd.DataFrame:
    """When DataFrame has datetime index, it will try to infer it's frequency.

    Args:
        df (pd.DataFrame): Input data.
        on_error (Literal[None, "warn", "raise"]): Define what to do when index is not inferred.
            Defaults to "warn.
        inplace (bool, optional): If True, then original data are edited. If False, copy is created.
            Defaults to False.

    Raises:
        ValueError: If defined column failed to convert to datetime.

    Returns:
        pd.DataFrame: Data with datetime index.

    Example:
        >>> df = pd.DataFrame([[1], [2], [3]], index=["08/04/2022", "09/04/2022", "10/04/2022"])
        >>> df.index = pd.to_datetime(df.index)
        >>> df = infer_frequency(df)
        >>> df.index.freq
    """
    df = get_copy_or_view(df, inplace)

    if isinstance(df.index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
        if df.index.freq is None:
            freq = pd.infer_freq(df.index)
            if freq:
                from pandas.tseries.frequencies import to_offset

                df.index.freq = to_offset(freq)

        if df.index.freq is None:
            message = (
                "Error in 'mydatapreprocessing' package in 'infer_frequency' function. Frequency inferring "
                "failed. Check the datetime index and try 'set_datetime_index' first."
            )

            if on_error == "warn":
                mylogging.warn(message, caption="Error in 'infer_frequency'")

            elif on_error == "raise":
                raise TypeError(message)
    else:
        raise TypeError(
            "Error in 'mydatapreprocessing' package in 'infer_frequency' function. Index is not"
            "pd.DatetimeIndex | pd.TimedeltaIndex type. You can use 'set_datetime_index' function to convert"
            "it from string."
        )

    return df


def resample(
    df: pd.DataFrame,
    freq: Literal["S", "min", "H", "M", "Y"] | str,
    resample_function: Literal["sum", "mean"],
):
    """Change the sampling frequency.

    Args:
        df (pd.DataFrame): Input data.
        freq (Literal["S", "min", "H", "M", "Y"] | str): Frequency of resampled data. For possible options
            check pandas 'Offset aliases'.
        resample_function (Literal['sum', 'mean'], optional): 'sum' or 'mean'. Whether sum resampled
            columns, or use average. Defaults to 'sum'.

    Returns:
        pd.DataFrame: Resampled data.

    Example:
        >>> from datetime import datetime, timedelta
        ...
        >>> df = pd.DataFrame(
        ...     {
        ...         "date": [
        ...             datetime(2022, 1, 1),
        ...             datetime(2022, 1, 2),
        ...             datetime(2022, 2, 1),
        ...             datetime(2022, 4, 1)
        ...         ],
        ...         "col_1": [1] * 4,
        ...         "col_2": [2] * 4,
        ...     }
        ... )
        >>> df
                date  col_1  col_2
        0 2022-01-01      1      2
        1 2022-01-02      1      2
        2 2022-02-01      1      2
        3 2022-04-01      1      2
        >>> df = df.set_index("date")
        >>> df = resample(df, "M", "sum")
        >>> df
                    col_1  col_2
        date
        2022-01-31      2      4
        2022-02-28      1      2
        2022-03-31      0      0
        2022-04-30      1      2
        >>> df.index.freq
        <MonthEnd>
    """
    df.sort_index(inplace=True)
    if resample_function == "mean":
        df = pd.DataFrame(df.resample(freq).mean())
    elif resample_function == "sum":
        df = pd.DataFrame(df.resample(freq).sum())

    return df


def move_on_first_column(df: pd.DataFrame, name_or_index: PandasIndex) -> pd.DataFrame:
    """Move defined column on index 0.

    Use case for that can be for example to be good visible in generated table.

    Args:
        df (pd.DataFrame): Input data.
        name_or_index (PandasIndex): Index or name of the column that will be moved.

    Raises:
        KeyError: Defined column not found in data.

    Returns:
        pd.DataFrame: DataFrame with defined column at index 0.

    Example:
        >>> move_on_first_column(pd.DataFrame([[1, 2, 3]], columns=["One", "Two", "Three"]), "Two").columns
        Index(['Two', 'One', 'Three']...
    """
    index = get_column_name(df, name_or_index)

    df.insert(0, index, df.pop(index))  # type: ignore - It's validated in get_column_name

    return df


def remove_nans(
    data: DataFrameOrArrayGeneric,
    remove_all_column_with_nans_threshold: None | Numeric = None,
    remove_nans_type: None | Literal["interpolate", "mean", "neighbor", "remove"] | Any = None,
    inplace: bool = False,
) -> DataFrameOrArrayGeneric:
    """Remove NotANumber values.

    Columns where are too many NaN values are dropped. Then in rest of columns rows with NaNs are
    removed or Nans are interpolated.

    Args:
        data (DataFrameOrArrayGeneric): Data in shape (n_samples, n_features).
        remove_all_column_with_nans_threshold (None | Numeric, optional): From 0 to 1. Require that many non-nan
            numeric values in column to not be deleted. E.G if value is 0.9 with column with 10 values, 90%
            must be numeric that implies max 1 np.nan can be presented, otherwise column will be deleted.
            Defaults to 0.85.
        remove_nans_type (None | Literal["interpolate", "mean", "neighbor", "remove"] | Any, optional): Remove or
            replace rest nan values. If you want to use concrete value, just use value directly.
            Defaults to 'interpolate'.
        inplace (bool, optional): If True, then original data are edited. If False, copy is created.
            Defaults to False.

    Example:
        >>> import numpy as np
        ...
        >>> array = np.array([[1, 2, np.nan], [2, np.nan, np.nan], [3, 4, np.nan]])
        >>> array
        array([[ 1.,  2., nan],
               [ 2., nan, nan],
               [ 3.,  4., nan]])
        >>> cleaned_df = remove_nans(
        ...     array,
        ...     remove_all_column_with_nans_threshold=0.5,
        ...     remove_nans_type="interpolate"
        ... )
        >>> cleaned_df
        array([[1., 2.],
               [2., 3.],
               [3., 4.]])
    """
    data = get_copy_or_view(data, inplace)
    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data

    # Remove columns that have to much nan values
    if remove_all_column_with_nans_threshold:
        df = df.dropna(axis=1, thresh=int(len(df) * (remove_all_column_with_nans_threshold)))

    if remove_nans_type is not None:
        # Replace rest of nan values
        if remove_nans_type == "interpolate":
            df.interpolate(inplace=True)

        elif remove_nans_type == "remove":
            df.dropna(axis=0, inplace=True)

        elif remove_nans_type == "neighbor":
            # Need to use both directions if first or last value is nan
            df.fillna(method="ffill", inplace=True)

        elif remove_nans_type == "mean":
            for col in df.columns:
                df[col] = df[col].fillna(df[col].mean())

        else:
            df.fillna(remove_nans_type, inplace=True)

        # Forward fill and interpolate can miss som nans if on first row
        if remove_nans_type in ["interpolate", "neighbor"]:
            df.fillna(method="bfill", inplace=True)

    if isinstance(data, pd.DataFrame):
        return df
    else:
        return df.values


def cast_str_to_numeric(df: pd.DataFrame, on_error: Literal["ignore", "raise"] = "ignore") -> pd.DataFrame:
    """Convert string values in DataFrame.

    Args:
        df (pd.DataFrame): Data
        on_error (Literal["ignore", "raise"]): What to do if meet error. Defaults to 'ignore'.

    Returns:
        pd.DataFrame: Data with possibly converted types.
    """
    df = df.apply(pd.to_numeric, errors=on_error)  # type: ignore
    return df
