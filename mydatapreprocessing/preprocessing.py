"""Module for data preprocessing. It consolidate data from various resources, cleane the data, generate
new derived columns, do feature exxtraction and do data preprocessing like. If you want to see how functions work -
working examples with printed results are in tests - visual.py.

Default output data shape is (n_samples, n_features)!

There are many small functions, but there they are called automatically with main preprocess functions.
    - load_data
    - data_consolidation
    - preprocess_data
    - preprocess_data_inverse

In data consolidation, predicted column is moved on index 0!

All processing funtions after data_consolidation use numpy.ndarray with ndim == 2,
so reshape(1, -1) if necessary...

"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import urllib
import requests
import itertools
from sklearn import preprocessing

from mylogging import user_warning, user_message


def load_data(loaded_data, header=0, csv_style={'separator': ",", 'decimal': "."}, predicted_table='', max_imported_length=0, request_datatype_suffix=''):
    """Load data from path or url. Available formats are csv, excel xlsx, parquet, json or h5.
    It can also be 'test' or 'sql' - you need to setup database name and query then.

    Args:
        data (str, pathlib.Path): Path, url or 'test' or 'sql'. Check configuration file for examples.
        header (int, optional): Row index used as column names. Defaults to 0.
        csv_style (dict, optional): Define CSV separators. En locale usually use {'sep': ",", 'decimal': "."} some Europian country use {'sep': ";", 'decimal': ","}. Defaults to {'separator': ",", 'decimal': "."}.
        predicted_table (str, optional): If using excel (xlsx) - it means what sheet to use, if json, it means what key values, if SQL, then it mean what table. Else it have no impact. Defaults to ''.
        max_imported_length (int, optional): Max length of imported samples (before resampling). If 0, than full length. Defaults to 0.

    Returns:
        pd.DataFrame, dict, list : Loaded data. Usually in pd.DataFrame format, but sometimes as dict or list, if it needs to be processed before conversion (because of orientation).
    """

    if str(loaded_data).lower() == 'test':
        from mydatapreprocessing import generatedata

        data = generatedata.gen_random()
        print(user_message(("Test data was used. Setup config.py 'data'. Check official readme or do help(preprocessing)")))

        return data

    # ############# Load SQL data #############
    # elif str(loaded_data).lower() == 'sql':
    #     try:
    #         data = predictit.database.database_load(server=config.server, database=config.database, freq=config.freq,
    #                                                 data_limit=config.max_imported_length)

    #     except Exception:
    #         raise RuntimeError(user_message("ERROR - Data load from SQL server failed - "
    #                                         "Setup server, database and predicted column name in config"))

    #     return data

    data_path = Path(loaded_data)

    try:
        if data_path.exists():
            loaded_data = Path(loaded_data).as_posix()
            file_path_exist = True
        else:
            raise

    except Exception:

        # Maybe file path is relative and in test_path folder
        data_path = 'test_data' / data_path

        try:
            if data_path.exists():
                loaded_data = Path(loaded_data).as_posix()
                file_path_exist = True

            else:
                raise
        except Exception:
            file_path_exist = False

    # On url, take everything after last dot
    data_type_suffix = data_path.suffix[1:].lower()

    # If not suffix inferred, then maybe url that return as request - than suffix have to be configured
    if not data_type_suffix or (data_type_suffix not in ['csv', 'json', 'xlsx'] and request_datatype_suffix):
        data_type_suffix = request_datatype_suffix.lower()

    if data_type_suffix.startswith('.'):
        data_type_suffix = data_type_suffix[1:]

        # If it's URL with suffix, we usually need url, if its url link with no suffix, we need get request response
        if not file_path_exist:
            loaded_data = requests.get(loaded_data).content

    if not data_type_suffix:
        raise TypeError(user_message("Data has no suffix (e.g. csv) and is not 'test' or 'sql'. "
                                     "If using url with no suffix, setup 'request_datatype_suffix'"
                                     "Or insert data with local path or insert data for example in "
                                     "dataframe or numpy array", caption="Data load error"))

    try:

        if data_type_suffix == 'csv':

            if not header and header != 0:
                header = 'infer'

            data = pd.read_csv(loaded_data, header=header, sep=csv_style['separator'],
                               decimal=csv_style['decimal']).iloc[-max_imported_length:, :]

        elif data_type_suffix == 'xlsx':
            data = pd.read_excel(loaded_data, sheet_name=predicted_table).iloc[-max_imported_length:, :]

        elif data_type_suffix == 'json':

            import json

            if file_path_exist:
                with open(loaded_data) as json_file:
                    data = json.load(json_file)[predicted_table] if predicted_table else json.load(json_file)

            else:
                data = json.loads(loaded_data)[predicted_table] if predicted_table else json.loads(loaded_data)

        # elif data_type_suffix in ('h5'):
        #     data = pd.read_csv(loaded_data).iloc[-max_imported_length:, :]

        else:
            raise TypeError

    except TypeError:
        raise TypeError(user_message(f"Your file format {data_type_suffix} not implemented yet. You can use csv, excel, parquet or txt.", "Wrong (not implemented) format"))

    except urllib.error.URLError:
        raise Exception(user_message(
            "Configured URL not found, check if page is available.",
            caption="URL error"))

    except Exception as err:
        if not file_path_exist:
            raise FileNotFoundError(user_message(
                "File not found on configured path. If you are using relative path, file must have be in CWD "
                "(current working directory) or must be inserted in system paths (sys.path.insert(0, 'your_path')). If url, check if page is available.",
                caption="File not found error"))
        else:
            raise(RuntimeError(user_message("Data load error. File found on path, but not loaded. Check if you use "
                                            "corrent locales - correct value and decimal separators in config (different in US and EU...). "
                                            " If it's web link, URL has to have .csv suffix. If it's link, that will generate csv link after "
                                            f"load, it will not work.\n\n Detailed error: \n\n {err}", caption="Data load failed")))

    return data


def data_consolidation(data, predicted_column=0, other_columns=1, datalength=0, data_orientation='', datetime_index='', unique_threshlold=0.1,
                       embedding='label', freq=0, resample_function='sum', remove_nans_threshold=0.85, remove_nans_or_replace='interpolate', dtype='float32'):
    """Transform input data in various formats and shapes into data in defined shape,
    that other functions rely on.

    Args:
        data (np.ndarray, pd.DataFrame): Input data in well standardized format.
        predicted_column ((int, str), optional): Predicted column name or index. Defaults to 0.
        other_columns (int, optional): Whether use other columns or only predicted one. Defaults to 1.
        datalength (int, optional): Data length after resampling. Defaults to 0.
        data_orientation (str, optional): 'columns' or 'rows'. If dictionary (or json), specify data orientation. Defaults to ''.
        datetime_index (str, optional): Name or index of datetime column. Defaults to ''.
        freq (int, optional): Frequency of resampled data. Defaults to 0.
        resample_function (str, optional): 'sum' or 'mean'. Whether sum resampled columns, or use average. Defaults to 'sum'.
        remove_nans_threshold (float, optional): From 0 to 1. How much not nans (not a number) can be in column to not be deleted.
        remove_nans_or_replace (str, float, optional): 'interpolate', 'remove', 'neighbor', 'mean' or value. Remove or replace rest nan values.
        dtype (str, optional): Output dtype. E.g. 'float32'.

    Raises:
        KeyError, TypeError: If wrong configuration in configuration.py.
            E.g. if predicted column name not found in dataframe.


    Returns:
        np.ndarray, pd.DataFrame, str: Data in standardized form. Data array for prediction - predicted column on index 0,
        and column for ploting as pandas dataframe.

    """

    if isinstance(data, np.ndarray):

        if not isinstance(predicted_column, int):
            raise TypeError(user_message("'predicted_column' in config is a string and data in numpy array format. Numpy does not allow "
                                         "string assignment", caption="Numpy string assignment not allowed"))

        data = pd.DataFrame(data)

    elif isinstance(data, list):
        data = pd.DataFrame.from_records(data)

    elif isinstance(data, dict):
        data = {i: [j] for (i, j) in data.items() if not isinstance(j, list)}
        orientation = 'columns' if not data_orientation else data_orientation
        data = pd.DataFrame.from_dict(data, orient=orientation)

    else:
        if not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame(data)
            except Exception as err:
                raise(RuntimeError(user_message("Check configuration file for supported formats. It can be path of file (csv, json, parquer...) or it "
                                                "can be data in python format (numpy array, pandas dataframe or series, dict or list, ). It can also be other "
                                                "format, but then it have to work with pd.DataFrame(your_data)."
                                                f"\n\n Detailed error: \n\n {err}", caption="Data load failed")))

    if isinstance(data, pd.DataFrame):

        data_for_predictions_df = data.copy()

        if data_for_predictions_df.shape[0] < data_for_predictions_df.shape[1]:
            print(user_message("Input data must be in shape (n_samples, n_features) that means (rows, columns) Your shape is "
                               f" {data.shape}. It's unusual to have more features than samples. Probably wrong shape.",
                               caption="Data transposed warning!!!"))
            data_for_predictions_df = data_for_predictions_df.T

        if isinstance(predicted_column, str):

            predicted_column_name = predicted_column

            if predicted_column_name not in data_for_predictions_df.columns:

                raise KeyError(user_message(
                    f"Predicted column name - '{predicted_column}' not found in data. Change 'predicted_column' in config"
                    f". Available columns: {list(data_for_predictions_df.columns)}", caption="Column not found error"))

        else:
            if isinstance(data_for_predictions_df.columns[predicted_column], str):
                predicted_column_name = data_for_predictions_df.columns[predicted_column]

            else:
                predicted_column_name = 'Predicted column'
                data_for_predictions_df.rename(columns={data_for_predictions_df.columns[predicted_column]: predicted_column_name}, inplace=True)
        reset_index = False

        if datetime_index not in [None, False, '']:

            try:
                if isinstance(datetime_index, str):
                    data_for_predictions_df.set_index(datetime_index, drop=True, inplace=True)

                else:
                    data_for_predictions_df.set_index(
                        data_for_predictions_df.columns[datetime_index], drop=True, inplace=True)

            except Exception:
                raise KeyError(user_message(
                    f"Datetime name / index from config - '{datetime_index}' not found in data or not datetime format. "
                    f"Change in config - 'datetime_index'. Available columns: {list(data_for_predictions_df.columns)}"))

            try:
                data_for_predictions_df.index = pd.to_datetime(data_for_predictions_df.index)

            except Exception:
                raise TypeError(user_message(
                    f"Datetime name / index from config - '{datetime_index}' could not been transformed to datetime format. "
                    "Try some common datetime string or convert column manually. - 'datetime_index'."))


        # Make predicted column index 0
        data_for_predictions_df.insert(0, predicted_column_name, data_for_predictions_df.pop(predicted_column_name))

        # Convert strings numbers (e.g. '6') to numbers
        data_for_predictions_df = data_for_predictions_df.apply(pd.to_numeric, errors='ignore')

        # Categorical embedding - Create numbers from strings (e.g 'US')

        to_drop = []

        for i in data_for_predictions_df.select_dtypes(exclude=['number']):

            try:

                if data_for_predictions_df[i].nunique() > unique_threshlold:
                    to_drop.append(i)
                    break

                data_for_predictions_df[i] = data_for_predictions_df[i].astype('category', copy=False)

                if embedding == 'label':
                    data_for_predictions_df[i] = data_for_predictions_df[i].cat.codes

            except Exception:
                to_drop.append(i)

        if embedding == 'one-hot':
            data_for_predictions_df = data_for_predictions_df.join(pd.get_dummies(data_for_predictions_df.select_dtypes(exclude=['number'])))

        # Drop columns with too few caterogies - drop all columns at once to better performance
        data_for_predictions_df.drop(to_drop, axis=1, inplace=True)

        # Keep only numeric columns
        data_for_predictions_df = data_for_predictions_df.select_dtypes(include='number')

        if datetime_index not in [None, False, '']:
            if freq:
                data_for_predictions_df.sort_index(inplace=True)
                if resample_function == 'mean':
                    data_for_predictions_df = data_for_predictions_df.resample(freq).mean()
                elif resample_function == 'sum':
                    data_for_predictions_df = data_for_predictions_df.resample(freq).sum()
                data_for_predictions_df = data_for_predictions_df.asfreq(freq, fill_value=0)

            else:
                data_for_predictions_df.index.freq = pd.infer_freq(data_for_predictions_df.index)

                if data_for_predictions_df.index.freq is None:
                    reset_index = True
                    user_warning("Datetime index was provided from config, but frequency guess failed. "
                                 "Specify 'freq' in config to resample and have equal sampling if you want "
                                 "to have date in plot or if you want to have equal sampling. Otherwise index will "
                                 "be reset because cannot generate date indexes of predicted values.",
                                 caption="Datetime frequency not inferred")

        # If frequency is not configured nor infered or index is not datetime, it's reset to be able to generate next results
        if reset_index or not isinstance(data_for_predictions_df.index, (pd.core.indexes.datetimes.DatetimeIndex, pd._libs.tslibs.timestamps.Timestamp)):
            data_for_predictions_df.reset_index(inplace=True, drop=True)

        if predicted_column_name not in data_for_predictions_df.columns:
            raise KeyError(user_message(
                "Predicted column is not number datatype. Setup correct 'predicted_column' in py. "
                f"Available columns with number datatype: {list(data_for_predictions_df.columns)}",
                caption="Prediction available only on number datatype column."))

        # Define concrete dtypes in number columns
        if dtype:
            data_for_predictions_df = data_for_predictions_df.astype(dtype, copy=False)

        # Trim the data on defined length
        data_for_predictions_df = data_for_predictions_df.iloc[-datalength:, :]

        if not other_columns or data_for_predictions_df.ndim <= 1:
            data_for_predictions_df = data_for_predictions_df[predicted_column_name]

        data_for_predictions_df = pd.DataFrame(data_for_predictions_df)

        # Remove columns that have to much nan values
        data_for_predictions_df = data_for_predictions_df.iloc[:, 0:1].join(
            data_for_predictions_df.iloc[:, 1:].dropna(axis=1, thresh=len(data_for_predictions_df) * (remove_nans_threshold)))

        # Replace rest of nan values
        if remove_nans_or_replace == 'interpolate':
            data_for_predictions_df.interpolate(inplace=True)

        elif remove_nans_or_replace == 'remove':
            data_for_predictions_df.dropna(axis=0, inplace=True)

        elif remove_nans_or_replace == 'neighbor':
            # Need to use both directions if first or last value is nan
            data_for_predictions_df.fillna(method='ffill', inplace=True)

        elif remove_nans_or_replace == 'mean':
            for col in data_for_predictions_df.columns:
                data_for_predictions_df[col] = data_for_predictions_df[col].fillna(data_for_predictions_df[col].mean())

        elif isinstance(remove_nans_or_replace, (int, float)):
            data_for_predictions_df.fillna(remove_nans_or_replace, inplace=True)

        # Forward fill and interpolate can miss som nans if on first row
        if data_for_predictions_df.isnull().values.any():
            data_for_predictions_df.fillna(method='bfill', inplace=True)

    else:
        raise TypeError(user_message(
            "Input data must be in pd.dataframe, pd.series, numpy array or in a path (str or pathlib) with supported formats"
            " - csv, xlsx, txt or parquet. Check config comments for more informations...", "Data format error"))

    return data_for_predictions_df


def add_frequency_columns(data, window):
    """Use fourier transform on running window and add it's maximum and std as new data column.

    Args:
        data (pd.DataFrame): Data we want to use.
        window (int): length of running window.

    Returns:
        pd.Dataframe: Data with new columns, that contain informations of running frequency analysis.
    """
    data = pd.DataFrame(data)

    if window > len(data.values.T):
        user_warning("Length of data much be much bigger than window used for generating new data columns",
                     caption="Adding frequency columns failed")

    windows = rolling_windows(data.values.T, window)

    ffted = np.fft.fft(windows, axis=2) / windows.shape[0]

    absolute = np.abs(ffted)[:, :, 1:]
    angle = np.angle(ffted)[:, :, 1:]

    data = data[-ffted.shape[1]:]

    for i, j in enumerate(data):
        data[f"{j} - FFT windowed abs max"] = np.nanmax(absolute, axis=2)[i]
        data[f"{j} - FFT windowed abs std"] = np.nanstd(absolute, axis=2)[i]
        data[f"{j} - FFT windowed angle1 max"] = np.nanmax(angle, axis=2)[i]
        data[f"{j} - FFT windowed angle1 std"] = np.nanstd(angle, axis=2)[i]

    return data


def rolling_windows(data, window):
    """Generate matrix of rolling windows. E.g for matrix [1, 2, 3, 4, 5] and window 2
    it will create [[1 2], [2 3], [3 4], [4 5]]. From matrix [[1, 2, 3], [4, 5, 6]] it will create
    [[[1 2], [2 3]], [[4 5], [5 6]]].

    Args:
        data (np.ndarray): Array data input.
        window (int): Number of values in created window.

    Returns:
        np.ndarray: Array of defined windows
    """
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def add_derived_columns(data, differences=True, second_differences=True, multiplications=True, rolling_means=True, rolling_stds=True, mean_distances=True, window=10):
    """This will create many columns that can be valuable for making predictions like difference, or
    rolling mean or distance from average. Computed columns will be appened to original data. It will process all the columns,
    so a lot of redundant data will be created. It is necessary do some feature extraction afterwards to remove noncorrelated columns.

    Examples:
        TODO

    Args:
        data (pd.DataFrame): Data that we want to extract more information from.
        difference (bool): Compute difference between n and n-1 sample
        second_difference (bool): Compute second difference.
        multiplicated (bool): Column multiplicated with other column.
        rolling_mean (bool): Rolling mean with defined window.
        rolling_std (bool): Rolling std with defined window.
        mean_distance (bool): Distance from average.

    Returns:
        pd.DataFrame: Data with more columns, that can have more informations,
        than original data. Number of rows can be little bit smaller.
    """

    results = [data]

    if differences:
        results.append(pd.DataFrame(np.diff(data.values, axis=0), columns=[f"{i} - Difference" for i in data.columns]))

    if second_differences:
        results.append(pd.DataFrame(np.diff(data.values, axis=0, n=2), columns=[f"{i} - Second difference" for i in data.columns]))

    if multiplications:

        combinations = list(itertools.combinations(data.columns, 2))
        combinations_names = [f"Multiplicated {i}" for i in combinations]
        multiplicated = np.zeros((len(data), len(combinations)))

        for i, j in enumerate(combinations):
            multiplicated[:, i] = data[j[0]] * data[j[1]]

        results.append(pd.DataFrame(multiplicated, columns=combinations_names))

    if rolling_means:
        results.append(pd.DataFrame(np.mean(rolling_windows(data.values.T, window), axis=2).T, columns=[f"{i} - Rolling mean" for i in data.columns]))

    if rolling_stds:
        results.append(pd.DataFrame(np.std(rolling_windows(data.values.T, window), axis=2).T, columns=[f"{i} - Rolling std" for i in data.columns]))

    if mean_distances:
        mean_distanced = np.zeros(data.T.shape)

        for i in range(data.shape[1]):
            mean_distanced[i] = data.values.T[i] - data.values.T[i].mean()
        results.append(pd.DataFrame(mean_distanced.T, columns=[f"{i} - Mean distance" for i in data.columns]))

    min_length = min(len(i) for i in results)

    return pd.concat([i.iloc[-min_length:].reset_index(drop=True) for i in results], axis=1)


def preprocess_data(data_for_predictions, remove_outliers=False, smoothit=False,
                    correlation_threshold=False, data_transform=False, standardizeit=False):
    if remove_outliers:
        data_for_predictions = remove_the_outliers(
            data_for_predictions, threshold=remove_outliers)

    if smoothit:
        data_for_predictions = smooth(
            data_for_predictions, smoothit[0], smoothit[1])

    if correlation_threshold:
        data_for_predictions = keep_corelated_data(data_for_predictions, threshold=correlation_threshold)

    if data_transform == 'difference':
        if isinstance(data_for_predictions, np.ndarray):
            last_undiff_value = data_for_predictions[-1, 0]
        else:
            last_undiff_value = data_for_predictions.iloc[-1, 0]
        data_for_predictions = do_difference(data_for_predictions)
    else:
        last_undiff_value = None

    if standardizeit:
        data_for_predictions, final_scaler = standardize(
            data_for_predictions, used_scaler=standardizeit)
    else:
        final_scaler = None

    return data_for_predictions, last_undiff_value, final_scaler


def preprocess_data_inverse(data, standardizeit=False, final_scaler=None, data_transform=False, last_undiff_value=None):
    """Undo all data preprocessing to get real data. Not not inverse all the columns, but only predicted one.
    Only predicted column is also returned. Order is reverse than preprocessing. Output is in numpy array.

    Args:
        data (np.ndarray, pd.DataFrame): Preprocessed data
        standardizeit (bool, optional): Whether use inverse standardization and what. Choices [None, 'standardize', '-11', '01', 'robust']. Defaults to False.
        final_scaler (sklearn.preprocessing.__x__scaler, optional): Scaler used in standardization. Defaults to None.
        data_transform (bool, optional): Use data transformation. Choices [False, 'difference]. Defaults to False.
        last_undiff_value (float, optional): Last used value in difference transform. Defaults to None.

    Returns:
        np.ndarray: Inverse preprocessed data
    """

    if isinstance(data, (pd.Series, pd.DataFrame)):
        data = data.values

    if standardizeit:
        data = final_scaler.inverse_transform(data.reshape(1, -1)).ravel()

    if data_transform == 'difference':
        data = inverse_difference(data, last_undiff_value)

    return data


### Data preprocessing functions...

def keep_corelated_data(data, threshold=0.5):
    """Remove columns that are not corelated enough to predicted columns. Predicted column is supposed to be 0.

    Args:
        data (np.array, pd.DataFrame): Time series data.
        threshold (float, optional): After correlation matrix is evaluated, all columns that are correlated less than threshold are deleted. Defaults to 0.2.

    Returns:
        np.array, pd.DataFrame: Data with no columns that are not corelated with predicted column.
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
            columns_to_del = range_array[abs(
                corr[0]) <= threshold]

            data = np.delete(data, columns_to_del, axis=1)

    elif isinstance(data, pd.DataFrame):
        corr = data.corr()
        names_to_del = list(
            corr[abs(corr[corr.columns[0]]) <= threshold].index)
        data.drop(columns=names_to_del, inplace=True)

    return data


def remove_the_outliers(data, threshold=3):
    """Remove values far from mean - probably errors. If more columns, then only rows that have outlier on
    predicted column will be deleted. Predicted column is supposed to be 0.

    Args:
        data (np.array, pd.DataFrame): Time series data. Must have ndim = 2, if univariate, reshape...
        threshold (int, optional): How many times must be standard deviation from mean to be ignored. Defaults to 3.

    Returns:
        np.array: Cleaned data.

    Examples:

        >>> data = np.array([[1, 3, 5, 2, 3, 4, 5, 66, 3]])
        >>> print(remove_the_outliers(data))
        [1, 3, 5, 2, 3, 4, 5, 3]
    """

    if isinstance(data, np.ndarray):
        data_mean = data[:, 0].mean()
        data_std = data[:, 0].std()

        range_array = np.array(range(data.shape[0]))
        names_to_del = range_array[abs(
            data[:, 0] - data_mean) > threshold * data_std]
        data = np.delete(data, names_to_del, axis=0)

    elif isinstance(data, pd.DataFrame):
        data_mean = data.iloc[:, 0].mean()
        data_std = data.iloc[:, 0].std()

        data = data[abs(data[data.columns[0]] - data_mean) < threshold * data_std]

    return data


def do_difference(data):
    """Transform data into neighbor difference. For example from [1, 2, 4] into [1, 2].

    Args:
        dataset (np.ndarray, pd.DataFrame): Numpy on or multi dimensional array.

    Returns:
        ndarray: Differenced data.

    Examples:

        >>> data = np.array([1, 3, 5, 2])
        >>> print(do_difference(data))
        [2, 2, -3]
    """

    if isinstance(data, np.ndarray):
        return np.diff(data, axis=0)
    else:
        return data.diff().iloc[1:]


def inverse_difference(differenced_predictions, last_undiff_value):
    """Transform do_difference transform back.

    Args:
        differenced_predictions (ndarray): One dimensional!! differenced data from do_difference function.
        last_undiff_value (float): First value to computer the rest.

    Returns:
        np.ndarray: Normal data, not the additive series.

    Examples:

        >>> data = np.array([1, 1, 1, 1])
        >>> print(inverse_difference(data, 1))
        [2, 3, 4, 5]
    """

    assert differenced_predictions.ndim == 1, 'Data input must be one-dimensional.'

    return np.insert(differenced_predictions, 0, last_undiff_value).cumsum()[1:]


def standardize(data, used_scaler='standardize'):
    """Standardize or normalize data. More standardize methods available. Predicted column is supposed to be 0.

    Args:
        data (np.ndarray): Time series data.
        used_scaler (str, optional): '01' and '-11' means scope from to for normalization.
            'robust' use RobustScaler and 'standard' use StandardScaler - mean is 0 and std is 1. Defaults to 'standardize'.

    Returns:
        ndarray: Standardized data.
    """

    if used_scaler == '01':
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    elif used_scaler == '-11':
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    elif used_scaler == 'robust':
        scaler = preprocessing.RobustScaler()
    elif used_scaler == 'standardize':
        scaler = preprocessing.StandardScaler()

    # First normalized values are calculated, then scler just for predicted value is computed again so no full matrix is necessary for inverse
    if isinstance(data, pd.DataFrame):
        normalized = data.copy()
        normalized.iloc[:, :] = scaler.fit_transform(data.copy().values)
        final_scaler = scaler.fit(data.values[:, 0].reshape(-1, 1))

    else:
        normalized = scaler.fit_transform(data)
        final_scaler = scaler.fit(data[:, 0].reshape(-1, 1))

    return normalized, final_scaler


def split(data, predicts=7):
    """Divide data set on train and test set. Predicted column is supposed to be 0.

    Args:
        data (pandas.DataFrame, ndarray): Time series data.
        predicts (int, optional): Number of predicted values. Defaults to 7.

    Returns:
        ndarray, ndarray: Train set and test set.

    Examples:

        >>> data = np.array([1, 2, 3, 4])
        >>> train, test = (split(data, predicts=2))
        >>> print(train, test)
        [2, 3]
        [4, 5]
    """
    if isinstance(data, pd.DataFrame):
        train = data.iloc[:-predicts, :]
        test = data.iloc[-predicts:, 0]
    else:
        train = data[:-predicts, :]
        test = data[-predicts:, 0]

    return train, test


def smooth(data, window=101, polynom_order=2):
    """Smooth data (reduce noise) with Savitzky-Golay filter. For more info on filter check scipy docs.

    Args:
        data (ndarray): Input data.
        window (tuple, optional): Length of sliding window. Must be odd.
        polynom_order - Must be smaller than window.

    Returns:
        ndarray: Cleaned data with less noise.
    """
    import scipy.signal

    for i in range(data.shape[1]):
        data[:, i] = scipy.signal.savgol_filter(
            data[:, i], window, polynom_order)

    return data


def fitted_power_transform(data, fitted_stdev, mean=None, fragments=10, iterations=5):
    """Function mostly for data postprocessing. Function transforms data, so it will have
    similiar standar deviation, similiar mean if specified. It use Box-Cox power transform in SciPy lib.

    Args:
        data (np.array): Array of data that should be transformed.
        fitted_stdev (float): Standard deviation that we want to have.
        mean (float, optional): Mean of transformed data. Defaults to None.
        fragments (int, optional): How many lambdas will be used in one iteration. Defaults to 9.
        iterations (int, optional): How many iterations will be used to find best transform. Defaults to 4.

    Returns:
        np.array: Transformed data with demanded standard deviation and mean.
    """

    import scipy.stats

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

    if mean is not None:
        mean_difference = np.mean(transformed_results) - mean
        transformed_results = transformed_results - mean_difference

    return transformed_results