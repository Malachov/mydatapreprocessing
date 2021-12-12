"""Module for data preprocessing.

You can consolidate data with `data_consolidation` and optimize it for example for machine learning models.

Then you can preprocess the data to be able to achieve even better results.

There are many small functions that you can use separately, but there is main function `preprocess_data` that
call all the functions based on input params for you. For inverse preprocessing use `preprocess_data_inverse`
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Generic, Any, cast, Union
from dataclasses import dataclass, astuple
import warnings
import importlib.util

from typing_extensions import Literal
import numpy as np
import pandas as pd

import mylogging

from .custom_types import DataFrameOrArrayGeneric


if TYPE_CHECKING:
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    ScalerType = Union[MinMaxScaler, RobustScaler, StandardScaler]

# Lazy load

# import scipy.signal
# import scipy.stats
# from sklearn import preprocessing


def data_consolidation(
    data: pd.DataFrame | np.ndarray,
    predicted_column: int | str = None,
    other_columns: int = 1,
    datalength: int = 0,
    datetime_column: str | int | None = "",
    resample_freq: str | None = None,
    resample_function: Literal[None, "sum", "mean"] = "sum",
    embedding: Literal[None, "label", "one-hot"] = "label",
    unique_threshold: float = 0.6,
    remove_nans_threshold: float = 0.85,
    remove_nans_or_replace: str | float = "interpolate",
    dtype: str | np.dtype | pd.DataFrame | list = "float32",
) -> pd.DataFrame:
    """Transform input data in various formats and shapes into data in defined shape optimal for machine learning models, that other functions rely on.
    If you have data in other format than dataframe, use `load_data` first.

    Note:
        This function return only numeric data. All string columns will be removed (use embedding if you need)
        Predicted column is moved on index 0 !!!

    Args:
        data (pd.DataFrame | np.ndarray): Input data in well standardized format.
        predicted_column (int | str, optional): Predicted column name or index. Move on first column and test if number.
            If None, it's ignored. Defaults to None.
        other_columns (int, optional): Whether use other columns or only predicted one. Defaults to 1.
        datalength (int, optional): Data length after resampling. Defaults to 0.
        datetime_column (str | int | None, optional): Name or index of datetime column. Defaults to None.
        resample_freq (str | None, optional): Frequency of resampled data. Defaults to None.
        resample_function (Literal[None, 'sum', 'mean'], optional): 'sum' or 'mean'. Whether sum resampled columns, or use average. Defaults to 'sum'.
        embedding(Literal[None, "label", "one-hot"], optional): 'label' or 'one-hot'. Categorical encoding. Create numbers from strings. 'label' give each
            category (unique string) concrete number. Result will have same number of columns. 'one-hot' create for every
            category new column. Only columns, where are strings repeating (unique_threshold) will be used. Defaults to 'label'.
        unique_threshold(float, optional): Remove string columns, that have to many categories. E.g 0.9 define, that if
            column contain more that 90% of NOT unique values it's deleted. Min is 0, max is 1. It will remove ids,
            hashes etc. Defaults to 0.6.
        remove_nans_threshold (float, optional): From 0 to 1. Require that many non-nan numeric values to not be deleted.
            E.G if value is 0.9 with column with 10 values, 90% must be numeric that implies max 1 np.nan can be presented,
            otherwise column will be deleted. Defaults to 0.85.
        remove_nans_or_replace (str | float, optional): 'interpolate', 'remove', 'neighbor', 'mean' or value. Remove or replace
            rest nan values. If you want to keep nan, setup value to np.nan. If you want to use concrete value, use float or
            int type. Defaults to 'interpolate'.
        dtype (str | np.dtype | pd.DataFrame | list, optional): Output dtype. For possible inputs check pandas function `astype`. Defaults to 'float32'.

    Raises:
        KeyError, TypeError: May happen if wrong params. E.g. if predicted column name not found in dataframe.


    Returns:
        np.ndarray, pd.DataFrame, str: Data in standardized form. Data array for prediction - predicted column on index 0,
        and column for ploting as pandas dataframe. Data has the same type as input.

    """

    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(data)
        except Exception as err:
            raise (
                RuntimeError(
                    mylogging.return_str(
                        "Check configuration file for supported formats. It can be path of file (csv, json, parquet...) or it "
                        "can be data in python format (numpy array, pandas dataframe or series, dict or list, ). It can also be other "
                        "format, but then it have to work with pd.DataFrame(your_data)."
                        f"\n\n Detailed error: \n\n {err}",
                        caption="Data load failed",
                    )
                )
            )

    data_for_predictions_df = data.copy()

    if data_for_predictions_df.shape[0] < data_for_predictions_df.shape[1]:
        mylogging.info(
            "Input data must be in shape (n_samples, n_features) that means (rows, columns) Your shape is "
            f" {data.shape}. It's unusual to have more features than samples. Probably wrong shape.",
            caption="Data transposed warning!!!",
        )
        data_for_predictions_df = data_for_predictions_df.T

    if predicted_column or predicted_column == 0:
        if isinstance(predicted_column, str):

            predicted_column_name = predicted_column

            if predicted_column_name not in data_for_predictions_df.columns:

                raise KeyError(
                    mylogging.return_str(
                        f"Predicted column name - '{predicted_column}' not found in data. Change 'predicted_column' in config"
                        f". Available columns: {list(data_for_predictions_df.columns)}",
                        caption="Column not found error",
                    )
                )

        elif isinstance(predicted_column, int) and isinstance(
            data_for_predictions_df.columns[predicted_column], str
        ):
            predicted_column_name = data_for_predictions_df.columns[predicted_column]

        else:
            predicted_column_name = "Predicted column"
            data_for_predictions_df.rename(
                columns={data_for_predictions_df.columns[predicted_column]: predicted_column_name},
                inplace=True,
            )

        # Make predicted column index 0
        data_for_predictions_df.insert(
            0, predicted_column_name, data_for_predictions_df.pop(predicted_column_name)
        )

    else:
        predicted_column_name = None

    reset_index = False

    if datetime_column not in [None, False, ""]:

        try:
            if isinstance(datetime_column, str):
                data_for_predictions_df.set_index(datetime_column, drop=True, inplace=True)

            else:
                data_for_predictions_df.set_index(
                    data_for_predictions_df.columns[datetime_column], drop=True, inplace=True,
                )

            data_for_predictions_df.index = pd.to_datetime(data_for_predictions_df.index)

        except Exception:
            raise KeyError(
                mylogging.return_str(
                    f"Datetime name / index from config - '{datetime_column}' not found in data or not datetime format. "
                    f"Change in config - 'datetime_column'. Available columns: {list(data_for_predictions_df.columns)}"
                )
            )

    # Convert strings numbers (e.g. '6') to numbers
    data_for_predictions_df = data_for_predictions_df.apply(pd.to_numeric, errors="ignore")

    if embedding:
        data_for_predictions_df = categorical_embedding(
            data_for_predictions_df, embedding=embedding, unique_threshold=unique_threshold,
        )

    # Keep only numeric columns
    data_for_predictions_df = data_for_predictions_df.select_dtypes(include="number")

    if predicted_column_name:
        # TODO [predictit] setup other columns in define input so every model can choose and simplier config input types
        if not other_columns:
            data_for_predictions_df = pd.DataFrame(data_for_predictions_df[predicted_column_name])

        if predicted_column_name not in data_for_predictions_df.columns:
            raise KeyError(
                mylogging.return_str(
                    "Predicted column is not number datatype. Setup correct 'predicted_column' in py. "
                    f"Available columns with number datatype: {list(data_for_predictions_df.columns)}",
                    caption="Prediction available only on number datatype column.",
                )
            )

    if datetime_column not in [None, False, ""]:
        if resample_freq:
            data_for_predictions_df.sort_index(inplace=True)
            if resample_function == "mean":
                data_for_predictions_df = data_for_predictions_df.resample(resample_freq).mean()
            elif resample_function == "sum":
                data_for_predictions_df = data_for_predictions_df.resample(resample_freq).sum()
            data_for_predictions_df = data_for_predictions_df.asfreq(resample_freq, fill_value=0)

        else:
            data_for_predictions_df.index.freq = pd.infer_freq(data_for_predictions_df.index)

            if data_for_predictions_df.index.freq is None:
                reset_index = True
                mylogging.warn(
                    "Datetime index was provided from config, but frequency guess failed. "
                    "Specify 'resample_freq' in config to resample and have equal sampling if you want "
                    "to have date in plot or if you want to have equal sampling. Otherwise index will "
                    "be reset because cannot generate date indexes of predicted values.",
                    caption="Datetime frequency not inferred",
                )

    # If frequency is not configured nor infered or index is not datetime, it's reset to be able to generate next results
    if reset_index or not isinstance(
        data_for_predictions_df.index,
        (pd.core.indexes.datetimes.DatetimeIndex, pd._libs.tslibs.timestamps.Timestamp),
    ):
        data_for_predictions_df.reset_index(inplace=True, drop=True)

    # Define concrete dtypes in number columns
    if dtype:
        data_for_predictions_df = data_for_predictions_df.astype(dtype, copy=False)

    # Trim the data on defined length
    data_for_predictions_df = data_for_predictions_df.iloc[-datalength:, :]

    data_for_predictions_df = pd.DataFrame(data_for_predictions_df)

    # Remove columns that have to much nan values
    if remove_nans_threshold:
        data_for_predictions_df = data_for_predictions_df.iloc[:, 0:1].join(
            data_for_predictions_df.iloc[:, 1:].dropna(
                axis=1, thresh=len(data_for_predictions_df) * (remove_nans_threshold)
            )
        )

    # Replace rest of nan values
    if remove_nans_or_replace == "interpolate":
        data_for_predictions_df.interpolate(inplace=True)

    elif remove_nans_or_replace == "remove":
        data_for_predictions_df.dropna(axis=0, inplace=True)

    elif remove_nans_or_replace == "neighbor":
        # Need to use both directions if first or last value is nan
        data_for_predictions_df.fillna(method="ffill", inplace=True)

    elif remove_nans_or_replace == "mean":
        for col in data_for_predictions_df.columns:
            data_for_predictions_df[col] = data_for_predictions_df[col].fillna(
                data_for_predictions_df[col].mean()
            )

    if isinstance(remove_nans_or_replace, (int, float) or np.isnan(remove_nans_or_replace)):
        data_for_predictions_df.fillna(remove_nans_or_replace, inplace=True)

    # Forward fill and interpolate can miss som nans if on first row
    else:
        data_for_predictions_df.fillna(method="bfill", inplace=True)

    return data_for_predictions_df


@dataclass
class PreprocessedData(Generic[DataFrameOrArrayGeneric]):
    """To be able to do inverse preprocessing (function preprocess_data_inverse), there are some values
    going with preprocessed data.

    Attributes:
        preprocessed (DataFrameOrArrayGeneric): Preprocessed data.
        last_undiff_value (Any): Last value from predicted column. This is necessary for inverse diff transformation.
        final_scaler("ScalerType" | None): For inverse standardization.
    """

    __slots__ = ("preprocessed", "last_undiff_value", "final_scaler")

    preprocessed: DataFrameOrArrayGeneric
    last_undiff_value: Any
    final_scaler: "ScalerType" | None

    def __iter__(self):
        yield from astuple(self)


def preprocess_data(
    data: DataFrameOrArrayGeneric,
    remove_outliers: int | float | None = False,
    smoothit: None | tuple[int, int] = None,
    correlation_threshold: float | int | None = None,
    data_transform: Literal["difference", None] = None,
    standardizeit: Literal[None, "standardize", "01", "-11", "robust"] = "standardize",
    bins: None | int = False,
    binning_type: Literal["cut", "qcut"] = "cut",
) -> PreprocessedData[DataFrameOrArrayGeneric]:
    """Main preprocessing function, that call other functions based on configuration. Mostly for preparing
    data to be optimal as input into machine learning models.

    Args:
        data (DataFrameOrArrayGeneric): Input data that we want to preprocess.
        remove_outliers (int | float | None, optional): Whether remove unusual values far from average. Defaults to False.
        smoothit (None | tuple[int, int], optional): Whether smooth the data with Savitzky-Golay filter.
            Insert tuple with (window, polynom_order) parameters as in `smooth` function e.g (11, 2). Defaults to False.
        correlation_threshold (float | None, optional): Whether remove columns that are corelated less than configured value
            Value must be between 0 and 1. But if 0, than None correlation threshold is applied. Defaults to 0.
        data_transform (Literal["difference", None], optional): Whether transform data. 'difference' transform data into differences between
            neighbor values. Defaults to None.
        standardizeit (Literal[None, "standardize", "-11", "01", "robust"], optional): How to standardize data. '01' and '-11' means scope from to for normalization.
            'robust' use RobustScaler and 'standard' use StandardScaler - mean is 0 and std is 1. If no standardization, use None.
            Defaults to 'standardize'.
        bins (None | int, optional): Whether to discretize values into defined number of bins (their average). None make no discretization,
            int define number of bins. Defaults to False.
        binning_type (str, optional): "cut" for equal size of bins intervals (different number of members in bins)
            or "qcut" for equal number of members in bins and various size of bins. It uses pandas cut
            or qcut function. Defaults to 'cut'.

    Returns:
        PreprocessedData: If input in numpy array, then also output in array, if dataframe input, then dataframe output.
    """

    preprocessed = data.copy()

    if remove_outliers:
        preprocessed = remove_the_outliers(preprocessed, threshold=remove_outliers)

    if smoothit:
        preprocessed = smooth(preprocessed, smoothit[0], smoothit[1])

    if correlation_threshold:
        preprocessed = keep_corelated_data(preprocessed, threshold=correlation_threshold)

    if data_transform == "difference":
        if isinstance(preprocessed, np.ndarray):
            last_undiff_value = preprocessed[-1, 0]
        else:
            last_undiff_value = preprocessed.iloc[-1, 0]
        preprocessed = do_difference(preprocessed)
    else:
        last_undiff_value = None

    if standardizeit:
        preprocessed, final_scaler = standardize(preprocessed, used_scaler=standardizeit)
    else:
        final_scaler = None

    if bins:
        preprocessed = binning(preprocessed, bins, binning_type)

    preprocessed = cast(DataFrameOrArrayGeneric, preprocessed)

    return PreprocessedData[DataFrameOrArrayGeneric](preprocessed, last_undiff_value, final_scaler)


def preprocess_data_inverse(
    data: np.ndarray,
    standardizeit: str | None = None,
    final_scaler: "ScalerType" | None = None,
    data_transform: Literal["difference", None] = None,
    last_undiff_value: None | Any = None,
) -> np.ndarray:
    """Undo all data preprocessing to get real data. Not not inverse all the columns, but only predicted one.
    Only predicted column is also returned. Order is reverse than preprocessing. Output is in numpy array.

    Args:
        data (np.ndarray): One dimension (one column) preprocessed data. Do not use ndim > 1.
        standardizeit (str | None, optional): Whether use inverse standardization and what. Choices [None, 'standardize', '-11', '01', 'robust']. Defaults to False.
        final_scaler ("ScalerType" | None, optional): Scaler used in standardization. Defaults to None.
        data_transform (Literal["difference", None], optional): Use data transformation. Choices [False, 'difference]. Defaults to False.
        last_undiff_value (None | Any, optional): Last used value in difference transform. Defaults to None.

    Returns:
        np.ndarray: Inverse preprocessed data

    Raises:
        TypeError: If variable necessary for particular inverse conversion is not defined.
    """

    if standardizeit:
        if final_scaler:
            if TYPE_CHECKING:
                final_scaler = cast(ScalerType, final_scaler)

            data = final_scaler.inverse_transform(data.reshape(1, -1)).ravel()
        else:
            raise TypeError(
                mylogging.return_str(
                    "If using `standardizeit`, then you need to define `final_scaler` for inverse transform."
                )
            )

    if data_transform == "difference":
        if last_undiff_value:
            last_undiff_value = cast(float, last_undiff_value)
            data = inverse_difference(data, last_undiff_value)
        else:
            raise TypeError(
                mylogging.return_str(
                    "If using `data_transform == 'difference'`, then you need to define `last_undiff_value` for inverse transform."
                )
            )

    return data


def categorical_embedding(
    data: pd.DataFrame, embedding: Literal["label", "one-hot"] = "label", unique_threshold: float = 0.6
) -> pd.DataFrame:
    """Transform string categories such as 'US', 'FR' into numeric values, that can be used in machine learning model.

    Args:
        data (pd.DataFrame): Data with string (pandas Object dtype) columns.
        embedding("label", "one-hot", optional): 'label' or 'one-hot'. Categorical encoding. Create numbers from strings. 'label'
            give each category (unique string) concrete number. Result will have same number of columns.
            'one-hot' create for every category new column. Only columns, where are strings repeating (unique_threshold)
            will be used. Defaults to "label".
        unique_threshold(float, optional): Remove string columns, that have to many categories (ids, hashes etc.).
            E.g 0.9 defines that in column of length 100, max number of categories to not to be deleted is
            10 (90% non unique repeating values). Defaults to 0.6. Min is 0, max is 1. Defaults is 0.6.

    Returns:
        pd.DataFrame: Dataframe where string columns transformed to numeric.
    """
    data_for_embedding = data.copy()
    to_drop = []

    for i in data_for_embedding.select_dtypes(exclude=["number"]):

        try:

            if (data_for_embedding[i].nunique() / len(data_for_embedding[i])) > (1 - unique_threshold):
                to_drop.append(i)
                continue

            data_for_embedding[i] = data_for_embedding[i].astype("category", copy=False)

            if embedding == "label":
                data_for_embedding[i] = data_for_embedding[i].cat.codes

            if embedding == "one-hot":
                data_for_embedding = data_for_embedding.join(pd.get_dummies(data_for_embedding[i]))
                to_drop.append(i)

        except Exception:
            to_drop.append(i)

    # Drop columns with too few categories - drop all columns at once to better performance
    data_for_embedding.drop(to_drop, axis=1, inplace=True)

    return data_for_embedding


### Data preprocessing functions...


def keep_corelated_data(data: DataFrameOrArrayGeneric, threshold: float = 0.5) -> DataFrameOrArrayGeneric:
    """Remove columns that are not corelated enough to predicted columns. Predicted column is supposed to be 0.

    Args:
        data (DataFrameOrArrayGeneric): Time series data.
        threshold (float, optional): After correlation matrix is evaluated, all columns that are correlated less
            than threshold are deleted. Defaults to 0.5.

    Returns:
        DataFrameOrArrayGeneric: Data with no columns that are not corelated with predicted column.
        If input in numpy array, then also output in array, if dataframe input, then dataframe output.
    """
    if data.ndim == 1 or data.shape[1] == 1:
        return data

    if isinstance(data, np.ndarray):
        # If some row have no variance - RuntimeWarning warning in correlation matrix computing and then in comparing
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            corr = np.corrcoef(data.T)
            corr = np.nan_to_num(corr, 0)

            range_array = np.array(range(corr.shape[0]))
            columns_to_del = range_array[abs(corr[0]) <= threshold]

            data = np.delete(data, columns_to_del, axis=1)

    elif isinstance(data, pd.DataFrame):
        corr = data.corr().iloc[0, :]
        corr = corr[~corr.isnull()]
        names_to_del = list(corr[abs(corr) <= threshold].index)
        data.drop(columns=names_to_del, inplace=True)

    return data


def remove_the_outliers(
    data: DataFrameOrArrayGeneric, threshold: int | float = 3, main_column: int | str = 0
) -> DataFrameOrArrayGeneric:
    """Remove values far from mean - probably errors. If more columns, then only rows that have outlier on
    predicted column will be deleted. Predicted column is supposed to be 0.

    Args:
        data (DataFrameOrArrayGeneric): Time series data. Must have ndim = 2, if univariate, reshape...
        threshold (int, optional): How many times must be standard deviation from mean to be ignored. Defaults to 3.
        main_column (int | str, optional): Main column that we relate outliers to. Defaults to 0.

    Returns:
        DataFrameOrArrayGeneric: Cleaned data.

    Examples:

        >>> data = np.array([[1, 3, 5, 2, 3, 4, 5, 66, 3]])
        >>> print(remove_the_outliers(data))
        [[ 1  3  5  2  3  4  5 66  3]]
    """

    if isinstance(data, np.ndarray):
        data_mean = data[:, main_column].mean()
        data_std = data[:, main_column].std()

        range_array = np.array(range(data.shape[0]))
        names_to_del = range_array[abs(data[:, main_column] - data_mean) > threshold * data_std]
        data = np.delete(data, names_to_del, axis=0)

    elif isinstance(data, pd.DataFrame):
        if isinstance(main_column, int):
            main_column = data.columns[main_column]

        data_mean = data[main_column].mean()
        data_std = data[main_column].std()

        data = data[abs(data[main_column] - data_mean) < threshold * data_std]

    return data


def do_difference(data: DataFrameOrArrayGeneric) -> DataFrameOrArrayGeneric:
    """Transform data into neighbor difference. For example from [1, 2, 4] into [1, 2].

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
        raise TypeError(mylogging.return_str("Only DataFrame, Series or numpy array supported."))


def inverse_difference(differenced_predictions: np.ndarray, last_undiff_value: float) -> np.ndarray:
    """Transform do_difference transform back.

    Args:
        differenced_predictions (np.ndarray): One dimensional!! differenced data from do_difference function.
        last_undiff_value (float): First value to computer the rest.

    Returns:
        np.ndarray: Normal data, not the additive series.

    Examples:

        >>> data = np.array([1, 1, 1, 1])
        >>> print(inverse_difference(data, 1))
        [2 3 4 5]
    """

    assert differenced_predictions.ndim == 1, "Data input must be one-dimensional."

    return np.insert(differenced_predictions, 0, last_undiff_value).cumsum()[1:]


def standardize(
    data: DataFrameOrArrayGeneric, used_scaler: Literal["standardize", "01", "-11", "robust"] = "standardize"
) -> tuple[DataFrameOrArrayGeneric, "ScalerType"]:
    """Standardize or normalize data. More standardize methods available. Predicted column is supposed to be 0.

    Args:
        data (DataFrameOrArrayGeneric): Time series data.
        used_scaler (Literal['standardize', '01', '-11', 'robust'], optional): '01' and '-11' means scope from to for normalization.
            'robust' use RobustScaler and 'standardize' use StandardScaler - mean is 0 and std is 1. Defaults to 'standardize'.

    Returns:
        tuple[DataFrameOrArrayGeneric, ScalerType]: Standardized data and scaler for inverse transformation.
    """
    if not importlib.util.find_spec("sklearn"):
        raise ImportError(
            "sklearn library is necessary for standardize function. Install via `pip install sklearn`"
        )

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
            mylogging.return_str(
                f"Your scaler {used_scaler} not in options. Use one of ['01', '-11', 'robust', 'standardize']"
            )
        )

    # First normalized values are calculated, then scler just for predicted value is computed again so no full matrix is necessary for inverse
    if isinstance(data, pd.DataFrame):
        normalized = data.copy()
        normalized.iloc[:, :] = scaler.fit_transform(data.copy().values)
        final_scaler = scaler.fit(data.values[:, 0].reshape(-1, 1))

    else:
        normalized = scaler.fit_transform(data)
        final_scaler = scaler.fit(data[:, 0].reshape(-1, 1))

    return normalized, final_scaler


def standardize_one_way(
    data: DataFrameOrArrayGeneric, min: float, max: float, axis: Literal[0, 1] = 0, inplace: bool = False,
) -> DataFrameOrArrayGeneric:
    """Own implementation of standardization. No inverse transformation available.
    Reason is for builded applications to do not carry sklearn with build.

    Args:
        data (DataFrameOrArrayGeneric): Data.
        min (float): Minimum in transformed axis.
        max (float): Max in transformed axis.
        axis (Literal[0, 1], optional): 0 to columns, 1 to rows. Defaults to 0.
        inplace (bool, optional): If true, no copy will be returned, but original object. Defaults to False.

    Returns:
        DataFrameOrArrayGeneric: Standardized data. If numpy inserted, numpy returned, same for dataframe.
        If input in numpy array, then also output in array, if dataframe input, then dataframe output.
    """
    if not inplace:
        data = data.copy()

    values = data.values if isinstance(data, pd.DataFrame) else data

    if axis == 0:
        values[:, :] = (values - np.nanmin(values, axis=0)) / (
            np.nanmax(values, axis=0) - np.nanmin(values, axis=0)
        ) * (max - min) + min

    elif axis == 1:
        values[:, :] = (
            (values.T - np.nanmin(values.T, axis=0))
            / (np.nanmax(values.T, axis=0) - np.nanmin(values.T, axis=0))
            * (max - min)
            + min
        ).T

    return data


def binning(
    data: DataFrameOrArrayGeneric, bins: int, binning_type: Literal["cut", "qcut"] = "cut"
) -> DataFrameOrArrayGeneric:
    """Discretize value on defined number of bins. It will return the same shape of data, where middle
    (average) values of bins interval returned.

    Args:
        data (DataFrameOrArrayGeneric): Data for preprocessing. ndim = 2 (n_samples, n_features).
        bins (int): Number of bins - unique values.
        binning_type (Literal["cut", "qcut"], optional): "cut" for equal size of bins intervals (different number of members in bins)
            or "qcut" for equal number of members in bins and various size of bins. It uses pandas cut
            or qcut function. Defaults to "cut".

    Returns:
        DataFrameOrArrayGeneric: Discretized data of same type as input. If input in numpy
        array, then also output in array, if dataframe input, then dataframe output.

    Example:

    >>> import mydatapreprocessing.preprocessing as mdpp
    ...
    >>> mdpp.binning(np.array(range(10)), bins=3, binning_type="cut")
    array([[1.4955],
           [1.4955],
           [1.4955],
           [1.4955],
           [4.5   ],
           [4.5   ],
           [4.5   ],
           [7.5   ],
           [7.5   ],
           [7.5   ]])

    """

    convert_to_array = True if isinstance(data, np.ndarray) else False

    data = pd.DataFrame(data)

    if binning_type == "qcut":
        func = pd.qcut
    elif binning_type == "cut":
        func = pd.cut
    else:
        raise TypeError(mylogging.return_str("`binning_type` has to be one of ['cut', 'qcut']."))

    for i in data:
        data[i] = func(data[i].values, bins)
        data[i] = data[i].map(lambda x: x.mid)

    if convert_to_array:
        return data.values
    else:
        return data


def smooth(data: DataFrameOrArrayGeneric, window=101, polynom_order=2) -> DataFrameOrArrayGeneric:
    """Smooth data (reduce noise) with Savitzky-Golay filter. For more info on filter check scipy docs.

    Args:
        data (DataFrameOrArrayGeneric): Input data.
        window (int, optional): Length of sliding window. Must be odd. Defaults to 101.
        polynom_order (int, optional) - Must be smaller than window. Defaults to 2.

    Returns:
        DataFrameOrArrayGeneric: Cleaned data with less noise.
    """
    if not importlib.util.find_spec("scipy"):
        raise ImportError("scipy library is necessary for smooth function. Install via `pip install scipy`")

    import scipy.signal

    if isinstance(data, pd.DataFrame):
        for i in range(data.shape[1]):
            data.iloc[:, i] = scipy.signal.savgol_filter(data.values[:, i], window, polynom_order)

    elif isinstance(data, np.ndarray):
        for i in range(data.shape[1]):
            data[:, i] = scipy.signal.savgol_filter(data[:, i], window, polynom_order)

    return data


def fitted_power_transform(
    data: np.ndarray,
    fitted_stdev: float,
    mean: float | None = None,
    fragments: int = 10,
    iterations: int = 5,
) -> np.ndarray:
    """Function mostly for data postprocessing. Function transforms data, so it will have
    similar standar deviation, similar mean if specified. It use Box-Cox power transform in SciPy lib.

    Args:
        data (np.ndarray): Array of data that should be transformed (one column => ndim = 1).
        fitted_stdev (float): Standard deviation that we want to have.
        mean (float | None, optional): Mean of transformed data. Defaults to None.
        fragments (int, optional): How many lambdas will be used in one iteration. Defaults to 10.
        iterations (int, optional): How many iterations will be used to find best transform. Defaults to 5.

    Returns:
        np.ndarray: Transformed data with demanded standard deviation and mean.
    """

    if not importlib.util.find_spec("scipy"):
        raise ImportError("scipy library is necessary for smooth function. Install via `pip install scipy`")

    import scipy.stats

    if data.ndim == 2 and 1 not in data.shape:
        raise ValueError(
            mylogging.return_str("Only one column can be power transformed. Use ravel if have shape (n, 1)")
        )

    lmbda_low = 0
    lmbda_high = 3
    lmbd_arr = np.linspace(lmbda_low, lmbda_high, fragments)
    lbmda_best_stdv_error = 1000000

    for _ in range(iterations):
        for j in range(len(lmbd_arr)):

            power_transformed = scipy.stats.yeojohnson(data, lmbda=lmbd_arr[j])
            transformed_stdev = np.std(power_transformed)
            if abs(transformed_stdev - fitted_stdev) < lbmda_best_stdv_error:
                lbmda_best_stdv_error = abs(transformed_stdev - fitted_stdev)
                lmbda_best_id = j

        if lmbda_best_id > 0:
            lmbda_low = lmbd_arr[lmbda_best_id - 1]
        if lmbda_best_id < len(lmbd_arr) - 1:
            lmbda_high = lmbd_arr[lmbda_best_id + 1]
        lmbd_arr = np.linspace(lmbda_low, lmbda_high, fragments)

    transformed_results = scipy.stats.yeojohnson(data, lmbda=lmbd_arr[j])

    transformed_results = cast(np.ndarray, transformed_results)

    if mean is not None:
        mean_difference = np.mean(transformed_results) - mean
        transformed_results = transformed_results - mean_difference

    return transformed_results
