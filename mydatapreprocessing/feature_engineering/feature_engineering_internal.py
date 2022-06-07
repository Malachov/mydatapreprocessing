"""Module for consolidation_pipeline subpackage."""

from __future__ import annotations
import itertools
import warnings

import numpy as np
import pandas as pd

import mylogging

from ..misc import rolling_windows
from ..types import DataFrameOrArrayGeneric


def keep_correlated_data(data: DataFrameOrArrayGeneric, threshold: float = 0.5) -> DataFrameOrArrayGeneric:
    """Remove columns that are not correlated enough to predicted columns.

    Predicted column is supposed to be 0.

    Args:
        data (DataFrameOrArrayGeneric): Time series data.
        threshold (float, optional): After correlation matrix is evaluated, all columns that are correlated
            less than threshold are deleted. Defaults to 0.5.

    Returns:
        DataFrameOrArrayGeneric: Data with no columns that are not correlated with predicted column.
        If input in numpy array, then also output in array, if DataFrame input, then DataFrame output.
    """
    # TODO inplace param
    if data.ndim == 1 or data.shape[1] == 1:
        return data

    if isinstance(data, np.ndarray):
        # If some row have no variance - RuntimeWarning warning in correlation matrix computing
        # and then in comparing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            corr = np.corrcoef(data.T)
            corr = np.nan_to_num(corr, False, 0.0)

            range_array = np.array(range(corr.shape[0]))
            columns_to_del = range_array[abs(corr[0]) <= threshold]

            data = np.delete(data, columns_to_del, axis=1)

    elif isinstance(data, pd.DataFrame):
        corr = data.corr().iloc[0, :]
        corr = corr[~corr.isnull()]
        names_to_del = list(corr[abs(corr) <= threshold].index)
        data.drop(columns=names_to_del)  # inplace=True

    return data


def add_derived_columns(
    data: pd.DataFrame,
    differences: bool = True,
    second_differences: bool = True,
    multiplications: bool = True,
    rolling_means: int | None = 10,
    rolling_stds: int | None = 10,
    mean_distances: bool = True,
) -> pd.DataFrame:
    """Create many columns with new information about dataset.

    Add data like difference, rolling mean or distance from average. Computed columns will be appended to
    original data. It will process all the columns, so a lot of redundant data will be created. It is
    necessary do some feature extraction afterwards to remove non-correlated columns.

    Note:
        Length on output is different as rolling windows needs to be prepend before first values.

    Args:
        data (pd.DataFrame): Data that we want to extract more information from.
        differences (bool, optional): Compute difference between n and n-1 sample. Defaults to True.
        second_differences (bool, optional): Compute second difference. Defaults to True.
        multiplications (bool, optional): Column multiplicated with other column. Defaults to True.
        rolling_means (int | None), optional): Rolling mean with defined window. Defaults to 10.
        rolling_stds (int | None): Rolling std with defined window. Defaults to 10.
        mean_distances (bool, optional): Distance from average. Defaults to True.

    Returns:
        pd.DataFrame: Data with more columns, that can have more information,
        than original data. Number of rows can be little bit smaller. Data has the same type as input.

    Example:
        >>> import mydatapreprocessing as mdp
        >>> data = pd.DataFrame(
        ...     [mdp.datasets.sin(n=30), mdp.datasets.ramp(n=30)]
        ... ).T
        ...
        >>> extended = add_derived_columns(data, differences=True, rolling_means=10)
        >>> extended.columns
        Index([                      0,                       1,
                      '0 - Difference',        '1 - Difference',
               '0 - Second difference', '1 - Second difference',
                'Multiplicated (0, 1)',      '0 - Rolling mean',
                    '1 - Rolling mean',       '0 - Rolling std',
                     '1 - Rolling std',     '0 - Mean distance',
                   '1 - Mean distance'],
              dtype='object')
        >>> len(extended)
        21
    """
    results = [data]

    if differences:
        results.append(
            pd.DataFrame(
                np.diff(data.values, axis=0),
                columns=[f"{i} - Difference" for i in data.columns],
            )
        )

    if second_differences:
        results.append(
            pd.DataFrame(
                np.diff(data.values, axis=0, n=2),
                columns=[f"{i} - Second difference" for i in data.columns],
            )
        )

    if multiplications:

        combinations = list(itertools.combinations(data.columns, 2))
        combinations_names = [f"Multiplicated {i}" for i in combinations]
        multiplicated = np.zeros((len(data), len(combinations)))

        for i, j in enumerate(combinations):
            multiplicated[:, i] = data[j[0]] * data[j[1]]

        results.append(pd.DataFrame(multiplicated, columns=combinations_names))

    if rolling_means:
        results.append(
            pd.DataFrame(
                np.mean(rolling_windows(data.values.T, rolling_means), axis=2).T,
                columns=[f"{i} - Rolling mean" for i in data.columns],
            )
        )

    if rolling_stds:
        results.append(
            pd.DataFrame(
                np.std(rolling_windows(data.values.T, rolling_stds), axis=2).T,
                columns=[f"{i} - Rolling std" for i in data.columns],
            )
        )

    if mean_distances:
        mean_distanced = np.zeros(data.T.shape)

        for i in range(data.shape[1]):
            mean_distanced[i] = data.values.T[i] - data.values.T[i].mean()
        results.append(pd.DataFrame(mean_distanced.T, columns=[f"{i} - Mean distance" for i in data.columns]))

    min_length = min(len(i) for i in results)

    return pd.concat(
        [i.iloc[-min_length:, :].set_index(data.index[-min_length:], drop=True) for i in results], axis=1
    )


def add_frequency_columns(data: pd.DataFrame | np.ndarray, window: int) -> pd.DataFrame:
    """Use fourier transform on running window and add it's maximum and std as new data column.

    Args:
        data (pd.DataFrame | np.ndarray): Data we want to use.
        window (int): length of running window.

    Returns:
        pd.DataFrame: Data with new columns, that contain information of running frequency analysis.

    Example:
        >>> import mydatapreprocessing as mdp
        >>> data = pd.DataFrame(
        ...     [mdp.datasets.sin(n=100), mdp.datasets.ramp(n=100)]
        ... ).T
        >>> extended = add_frequency_columns(data, window=32)
    """
    data = pd.DataFrame(data).copy()
    if window > len(data.values):
        mylogging.warn(
            "Length of data much be much bigger than window used for generating new data columns",
            caption="Adding frequency columns failed",
        )

    windows = rolling_windows(data.values.T, window)

    ffted = np.fft.fft(windows, axis=2) / window

    absolute = np.abs(ffted)[:, :, 1:]
    angle = np.angle(ffted)[:, :, 1:]  # type: ignore

    data = data.iloc[-ffted.shape[1] :, :]

    for i, j in enumerate(data):
        data[f"{j} - FFT windowed abs max index"] = np.nanargmax(absolute, axis=2)[i]
        data[f"{j} - FFT windowed angle max index"] = np.nanargmax(angle, axis=2)[i]
        data[f"{j} - FFT windowed abs max"] = np.nanmax(absolute, axis=2)[i]
        data[f"{j} - FFT windowed abs std"] = np.nanstd(absolute, axis=2)[i]
        data[f"{j} - FFT windowed angle max"] = np.nanmax(angle, axis=2)[i]
        data[f"{j} - FFT windowed angle std"] = np.nanstd(angle, axis=2)[i]

    return data
