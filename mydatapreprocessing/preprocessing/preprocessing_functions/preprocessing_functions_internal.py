"""Module for preprocessing_functions subpackage."""

from __future__ import annotations
from typing import TYPE_CHECKING, cast, Union

from typing_extensions import Literal
import numpy as np
import pandas as pd

from mypythontools.system import check_library_is_available

from ...types import DataFrameOrArrayGeneric, Numeric
from ...helpers import get_copy_or_view


if TYPE_CHECKING:
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    ScalerType = Union[MinMaxScaler, RobustScaler, StandardScaler]

# Lazy load
# import scipy.signal
# import scipy.stats
# from sklearn import preprocessing


def remove_the_outliers(
    data: DataFrameOrArrayGeneric,
    threshold: Numeric = 3,
) -> DataFrameOrArrayGeneric:
    """Deprecated function. Historically, remove_outliers was parameter in pipeline and in the same module,
    function needed different name. Use `remove_outliers` if possible. This will be removed in new major."""
    return remove_outliers(data, threshold)


def remove_outliers(
    data: DataFrameOrArrayGeneric,
    threshold: Numeric = 3,
) -> DataFrameOrArrayGeneric:
    """Remove values far from mean - probably errors.

    If more columns, then only rows that have outlier on predicted column will be deleted. Predicted column
    (column where we are searching for outliers) is supposed to be 0.

    Args:
        data (DataFrameOrArrayGeneric): Time series data. Must have ndim = 2, so if univariate, reshape.
        threshold (Numeric, optional): How many times must be standard deviation from mean to be ignored.
            Defaults to 3.

    Returns:
        DataFrameOrArrayGeneric: Cleaned data.

    Examples:
        >>> data = np.array(
        ...     [
        ...         [1, 7],
        ...         [66, 3],
        ...         [5, 5],
        ...         [2, 3],
        ...         [2, 3],
        ...         [3, 9],
        ...     ]
        ... )
        >>> remove_outliers(data, threshold=2)
        array([[1, 7],
               [5, 5],
               [2, 3],
               [2, 3],
               [3, 9]])
    """
    if isinstance(data, np.ndarray):
        data_mean = data[:, 0].mean()
        data_std = data[:, 0].std()

        range_array = np.array(range(data.shape[0]))
        names_to_del = range_array[abs(data[:, 0] - data_mean) > threshold * data_std]
        data = np.delete(data, names_to_del, axis=0)

    elif isinstance(data, pd.DataFrame):
        main_column = data.columns[0]

        data_mean = data[main_column].mean()
        data_std = data[main_column].std()

        data = data[abs(data[main_column] - data_mean) < threshold * data_std]

    return data


def do_difference(data: DataFrameOrArrayGeneric) -> DataFrameOrArrayGeneric:
    """Transform data into neighbor difference.

    Args:
        data (DataFrameOrArrayGeneric): Data.

    Returns:
        DataFrameOrArrayGeneric: Differenced data in same format as inserted.

    Examples:
        >>> data = np.array([1, 3, 5, 2])
        >>> print(do_difference(data))
        [ 2  2 -3]
    """
    if isinstance(data, np.ndarray):
        return np.diff(data, axis=0)

    elif isinstance(data, (pd.DataFrame, pd.Series)):
        return data.diff().iloc[1:]

    else:
        raise TypeError("Only DataFrame, Series or numpy array supported.")


def inverse_difference(data: np.ndarray, last_undiff_value: Numeric) -> np.ndarray:
    """Transform do_difference transform back.

    Args:
        data (np.ndarray): One dimensional differenced data from do_difference function.
        last_undiff_value (Numeric): First value to computer the rest.

    Returns:
        np.ndarray: Normal data, not the additive series.

    Examples:
        >>> data = np.array([1, 1, 1, 1])
        >>> print(inverse_difference(data, 1))
        [2 3 4 5]
    """
    assert data.ndim == 1, "Data input must be one-dimensional."

    return np.insert(data, 0, last_undiff_value).cumsum()[1:]


def standardize(
    data: DataFrameOrArrayGeneric, used_scaler: Literal["standardize", "01", "-11", "robust"] = "standardize"
) -> tuple[DataFrameOrArrayGeneric, "ScalerType"]:
    """Standardize or normalize data.

    More standardize methods available. Predicted column is supposed to be 0.

    Args:
        data (DataFrameOrArrayGeneric): Time series data.
        used_scaler (Literal['standardize', '01', '-11', 'robust'], optional): '01' and '-11' means scope
            from to for normalization. 'robust' use RobustScaler and 'standardize' use StandardScaler - mean
            is 0 and std is 1. Defaults to 'standardize'.

    Returns:
        tuple[DataFrameOrArrayGeneric, ScalerType]: Standardized data and scaler for inverse transformation.
    """
    check_library_is_available("sklearn")

    from sklearn import preprocessing

    if used_scaler == "01":
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    elif used_scaler == "-11":
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    elif used_scaler == "robust":
        scaler = preprocessing.RobustScaler()
    elif used_scaler == "standardize":
        scaler = preprocessing.StandardScaler()

    else:
        raise TypeError(
            f"Your scaler {used_scaler} not in options. Use one of ['01', '-11', 'robust', 'standardize']"
        )

    # First normalized values are calculated, then scaler just for predicted value is computed again so no
    # full matrix is necessary for inverse
    if isinstance(data, pd.DataFrame):
        normalized = data.copy()
        normalized.iloc[:, :] = scaler.fit_transform(data.copy().values)
        final_scaler = scaler.fit(data.values[:, 0].reshape(-1, 1))

    else:
        normalized = scaler.fit_transform(data)
        final_scaler = scaler.fit(data[:, 0].reshape(-1, 1))

    return normalized, final_scaler  # type: ignore


def standardize_one_way(
    data: DataFrameOrArrayGeneric,
    minimum: float,
    maximum: float,
    axis: Literal[0, 1] = 0,
    inplace: bool = False,
) -> DataFrameOrArrayGeneric:
    """Own implementation of standardization. No inverse transformation available.

    Reason is for builded applications to do not carry sklearn with build.

    Args:
        data (DataFrameOrArrayGeneric): Data.
        minimum (float): Minimum in transformed axis.
        maximum (float): Max in transformed axis.
        axis (Literal[0, 1], optional): 0 to columns, 1 to rows. Defaults to 0.
        inplace (bool, optional): If True, then original data are edited. If False, copy is created.
            Defaults to False.

    Returns:
        DataFrameOrArrayGeneric: Standardized data. If numpy inserted, numpy returned, same for DataFrame.
        If input in numpy array, then also output in array, if DataFrame input, then DataFrame output.
    """
    data = get_copy_or_view(data, inplace)

    values = data.values if isinstance(data, pd.DataFrame) else data

    if axis == 0:
        values[:, :] = (values - np.nanmin(values, axis=0)) / (
            np.nanmax(values, axis=0) - np.nanmin(values, axis=0)
        ) * (maximum - minimum) + minimum

    elif axis == 1:
        values[:, :] = (
            (values.T - np.nanmin(values.T, axis=0))
            / (np.nanmax(values.T, axis=0) - np.nanmin(values.T, axis=0))
            * (maximum - minimum)
            + minimum
        ).T

    return data


def binning(
    data: DataFrameOrArrayGeneric, bins: int, binning_type: Literal["cut", "qcut"] = "cut"
) -> DataFrameOrArrayGeneric:
    """Discretize value on defined number of bins.

    It will return the same shape of data, where middle (average) values of bins interval returned.

    Args:
        data (DataFrameOrArrayGeneric): Data for preprocessing. ndim = 2 (n_samples, n_features).
        bins (int): Number of bins - unique values.
        binning_type (Literal["cut", "qcut"], optional): "cut" for equal size of bins intervals (different
            number of members in bins) or "qcut" for equal number of members in bins and various size of bins.
            It uses pandas cut or qcut function. Defaults to "cut".

    Returns:
        DataFrameOrArrayGeneric: Discretized data of same type as input. If input in numpy
        array, then also output in array, if DataFrame input, then DataFrame output.

    Example:
        >>> binning(np.array(range(5)), bins=3, binning_type="cut")
        array([[0.6645],
               [0.6645],
               [2.    ],
               [3.3335],
               [3.3335]])

    """
    df = pd.DataFrame(data)

    if binning_type == "qcut":
        func = pd.qcut
    elif binning_type == "cut":
        func = pd.cut
    else:
        raise TypeError("`binning_type` has to be one of ['cut', 'qcut'].")

    for i in df:
        df[i] = func(df[i].values, bins)
        df[i] = df[i].map(lambda x: x.mid)

    if isinstance(data, np.ndarray):
        return df.values
    else:
        return df


def smooth(
    data: DataFrameOrArrayGeneric,
    window=101,
    polynomial_order=2,
    inplace: bool = False,
) -> DataFrameOrArrayGeneric:
    """Smooth data (reduce noise) with Savitzky-Golay filter. For more info on filter check scipy docs.

    Args:
        data (DataFrameOrArrayGeneric): Input data.
        window (int, optional): Length of sliding window. Must be odd. Defaults to 101.
        polynomial_order (int, optional) - Must be smaller than window. Defaults to 2.
        inplace (bool, optional): If True, then original data are edited. If False, copy is created.
            Defaults to False.

    Returns:
        DataFrameOrArrayGeneric: Cleaned data with less noise.
    """
    check_library_is_available("scipy")

    import scipy.signal

    data = get_copy_or_view(data, inplace)

    if isinstance(data, pd.DataFrame):
        for i in range(data.shape[1]):
            data.iloc[:, i] = scipy.signal.savgol_filter(data.values[:, i], window, polynomial_order)

    elif isinstance(data, np.ndarray):
        for i in range(data.shape[1]):
            data[:, i] = scipy.signal.savgol_filter(data[:, i], window, polynomial_order)

    return data


def fitted_power_transform(
    data: np.ndarray,
    fitted_stdev: float,
    mean: float | None = None,
    fragments: int = 10,
    iterations: int = 5,
) -> np.ndarray:
    """Function transforms data, so it will have similar standard deviation and mean.

    It use Box-Cox power transform in SciPy lib.

    Args:
        data (np.ndarray): Array of data that should be transformed (one column => ndim = 1).
        fitted_stdev (float): Standard deviation that we want to have.
        mean (float | None, optional): Mean of transformed data. Defaults to None.
        fragments (int, optional): How many lambdas will be used in one iteration. Defaults to 10.
        iterations (int, optional): How many iterations will be used to find best transform. Defaults to 5.

    Returns:
        np.ndarray: Transformed data with demanded standard deviation and mean.
    """
    check_library_is_available("scipy")

    import scipy.stats

    if data.ndim == 2 and 1 not in data.shape:
        raise ValueError("Only one column can be power transformed. Use ravel if have shape (n, 1)")

    lmbda_low = 0
    lmbda_high = 3
    lmbda_arr = np.linspace(lmbda_low, lmbda_high, fragments)
    lmbda_best_stdev_error = np.inf
    lmbda_best_id = 0
    lmbda_best = 0

    for i in range(iterations):
        for j, lmbda in enumerate(lmbda_arr):
            power_transformed = scipy.stats.yeojohnson(data, lmbda=lmbda)
            transformed_stdev = np.std(power_transformed)
            if abs(transformed_stdev - fitted_stdev) < lmbda_best_stdev_error:
                lmbda_best_stdev_error = abs(transformed_stdev - fitted_stdev)
                lmbda_best_id = j
                lmbda_best = lmbda

        if i != iterations:
            if lmbda_best_id > 0:
                lmbda_low = lmbda_arr[lmbda_best_id - 1]
            if lmbda_best_id < len(lmbda_arr) - 1:
                lmbda_high = lmbda_arr[lmbda_best_id + 1]
            lmbda_arr = np.linspace(lmbda_low, lmbda_high, fragments)

    transformed_results = scipy.stats.yeojohnson(data, lmbda=lmbda_best)

    transformed_results = cast(np.ndarray, transformed_results)

    if mean is not None:
        mean_difference = np.mean(transformed_results) - mean
        transformed_results = transformed_results - mean_difference

    return transformed_results
