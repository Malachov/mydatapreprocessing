"""
You can add new derived columns. This new generated data can help to machine learning models to better results.

In `add_derived_columns` you add first and second derivations, multiplication of columns, rolling means and rolling standard deviation.

In `add_frequency_columns` you can add fast fourier transform results maximums on running window.
"""

import itertools

import numpy as np
import pandas as pd

import mylogging

from .misc import rolling_windows


def add_derived_columns(
    data,
    differences=True,
    second_differences=True,
    multiplications=True,
    rolling_means=10,
    rolling_stds=10,
    mean_distances=True,
):
    """This will create many columns that can be valuable for making predictions like difference, or
    rolling mean or distance from average. Computed columns will be appened to original data. It will process all the columns,
    so a lot of redundant data will be created. It is necessary do some feature extraction afterwards to remove noncorrelated columns.

    Args:
        data (pd.DataFrame): Data that we want to extract more information from.
        differences (bool, optional): Compute difference between n and n-1 sample. Defaults to True.
        second_difference (bool, optional): Compute second difference. Defaults to True.
        multiplications (bool, optional): Column multiplicated with other column. Defaults to True.
        rolling_means ((int, None), optional): Rolling mean with defined window. Defaults to 10.
        rolling_stds ((int, None), optional): Rolling std with defined window. Defaults to 10.
        window (int, optional): Window used for rolling_stds and rolling_means.
        mean_distances (bool, optional): Distance from average. Defaults to True.

    Returns:
        pd.DataFrame: Data with more columns, that can have more informations,
        than original data. Number of rows can be little bit smaller. Data has the same type as input.

    Example:
        >>> import mydatapreprocessing as mdp
        >>> data = pd.DataFrame(
        ...     [mdp.generate_data.sin(n=100), mdp.generate_data.ramp(n=100)]
        ... ).T
        ...
        >>> extended = add_derived_columns(data, differences=True, rolling_means=32)
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

    return pd.concat([i.iloc[-min_length:].reset_index(drop=True) for i in results], axis=1)


def add_frequency_columns(data, window):
    """Use fourier transform on running window and add it's maximum and std as new data column.

    Args:
        data (pd.DataFrame): Data we want to use.
        window (int): length of running window.

    Returns:
        pd.Dataframe: Data with new columns, that contain informations of running frequency analysis.

    Example:

        >>> import mydatapreprocessing as mdp
        >>> data = pd.DataFrame(
        ...     [mdp.generate_data.sin(n=100), mdp.generate_data.ramp(n=100)]
        ... ).T
        >>> extended = add_frequency_columns(data, window=32)
    """
    data = pd.DataFrame(data)

    if window > len(data.values):
        mylogging.warn(
            "Length of data much be much bigger than window used for generating new data columns",
            caption="Adding frequency columns failed",
        )

    windows = rolling_windows(data.values.T, window)

    ffted = np.fft.fft(windows, axis=2) / window

    absolute = np.abs(ffted)[:, :, 1:]
    angle = np.angle(ffted)[:, :, 1:]

    data = data[-ffted.shape[1] :]

    for i, j in enumerate(data):
        data[f"{j} - FFT windowed abs max index"] = np.nanargmax(absolute, axis=2)[i]
        data[f"{j} - FFT windowed angle max index"] = np.nanargmax(angle, axis=2)[i]
        data[f"{j} - FFT windowed abs max"] = np.nanmax(absolute, axis=2)[i]
        data[f"{j} - FFT windowed abs std"] = np.nanstd(absolute, axis=2)[i]
        data[f"{j} - FFT windowed angle max"] = np.nanmax(angle, axis=2)[i]
        data[f"{j} - FFT windowed angle std"] = np.nanstd(angle, axis=2)[i]

    return data
