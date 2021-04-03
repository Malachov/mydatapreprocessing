"""Module for data loading and preprocessing.

Load data from web link or local file (json, csv, excel file, parquet, h5...), consolidate it (to pandas dataframe)
and do preprocessing like resampling, standardization, string embedding, new columns derivation.
If you want to see how functions work - working examples with printed results are in tests - visual.py.

There are many small functions, but there they are called automatically with main preprocess functions.
    - load_data
    - data_consolidation
    - preprocess_data
    - preprocess_data_inverse

Note:
    In data consolidation, predicted column is moved on index 0!!!

Example:

    >>> import mydatapreprocessing.preprocessing as mdpp
    >>> data = "https://blockchain.info/unconfirmed-transactions?format=json"

    Load data from file or URL

    >>> data_loaded = mdpp.load_data(data, request_datatype_suffix=".json", predicted_table='txs')

    Transform various data into defined format - pandas dataframe - convert to numeric if possible, keep
    only numeric data and resample ifg configured. It return array, dataframe

    >>> data_consolidated = mdpp.data_consolidation(
    ...     data_loaded, predicted_column="weight", data_orientation="index", remove_nans_threshold=0.9,
    ...     remove_nans_or_replace='interpolate')

    Preprocess data. It return preprocessed data, but also last undifferenced value and scaler for inverse
    transformation, so unpack it with _

    >>> data_preprocessed, _, _ = mdpp.preprocess_data(
    ...     data_consolidated, remove_outliers=True, smoothit=False, correlation_threshold=False,
    ...     data_transform=False, standardizeit='standardize')


    Allowed data formats for load_data are examples

    >>> # myarray_or_dataframe # Numpy array or Pandas.DataFrame
    >>> # r"/home/user/my.json" # Local file. The same with .parquet, .h5, .json or .xlsx.
    ...     On windows it's necessary to use raw string - 'r' in front of string because of escape symbols \
    >>> # "https://yoururl/your.csv" # Web url (with suffix). Same with json.
    >>> # "https://blockchain.info/unconfirmed-transactions?format=json" # In this case you have to specify
    ...     also 'request_datatype_suffix': "json", 'data_orientation': "index", 'predicted_table': 'txs',
    >>> # [{'col_1': 3, 'col_2': 'a'}, {'col_1': 0, 'col_2': 'd'}] # List of records
    >>> # {'col_1': [3, 2, 1, 0], 'col_2': ['a', 'b', 'c', 'd']} # Dict with colums or rows (index)
    ...     - necessary to setup data_orientation!
    ...
    >>> # You can use more files in list and data will be concatenated. It can be list of paths or list of python objects. Example:
    >>> # [{'col_1': 3, 'col_2': 'a'}, {'col_1': 0, 'col_2': 'd'}]  # List of records
    >>> # [np.random.randn(20, 3), np.random.randn(25, 3)]  # Dataframe same way
    >>> # ["https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
    ...     "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"]  # List of URLs
    >>> # ["path/to/my1.csv", "path/to/my1.csv"]
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import urllib
import requests
import itertools
import importlib

import mylogging


def get_file_paths(
    filetypes=[
        ("csv", ".csv"),
        ("Excel (xlsx, xls)", ".xlsx .xls"),
        ("h5", ".h5"),
        ("parquet", ".parquet"),
        ("json", ".json"),
    ],
    title="Select files",
):
    """Open dialog window where you can choose files you want to use. It will return tuple with string paths.

    Args:
        filetypes (list, optional): Accepted file types / suffixes. List of strings or list of tuples.
            Defaults to [("csv", ".csv")].
        title (str, optional): Just a name of dialog window. Defaults to 'Select file'.

    Returns:
        tuple: Tuple with string paths.
    """
    import tkinter as tk
    from tkinter import filedialog

    # Open dialog window where user can choose which files to use
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)

    return filedialog.askopenfilenames(filetypes=filetypes, title=title)


def load_data(
    data,
    header=None,
    csv_style=None,
    predicted_table="",
    max_imported_length=0,
    request_datatype_suffix="",
    data_orientation="",
):
    """Load data from path or url or other python format (numpy array, list, dict) into dataframe.
    Available formats are csv, excel xlsx, parquet, json or h5. Allow multiple files loading at once - just put
    it in list e.g. [df1, df2, df3] or ['myfile1.csv', 'myfile2.csv']. Structure of files does not have to be the same.
    If you have files in folder and not in list, you can use `get_file_paths` function to open system dialog window,
    select files and get the list of paths.

    Data param can also be 'test' or 'sql' - you need to setup database name and query then.

    Args:
        data (str, pathlib.Path): Path, url or 'test' or 'sql'. Check configuration file for examples.
        header (int, optional): Row index used as column names. Defaults to 0.
        csv_style ((dict, None), optional): Define CSV separator and decimal. En locale usually use {'sep': ",", 'decimal': "."}
            some Europian country use {'sep': ";", 'decimal': ","}. If None - values are infered. Defaults to None.
        predicted_table (str, optional): If using excel (xlsx) - it means what sheet to use, if json,
            it means what key values, if SQL, then it mean what table. Else it have no impact. Defaults to ''.
        max_imported_length (int, optional): Max length of imported samples (before resampling). If 0, than full length. Defaults to 0.
        request_datatype_suffix(str, optional): 'json' for example. If using url with no extension,
            define whichdatatype is on this url with GET request
        data_orientation(str, optional): 'columns' or 'index'. If using json or dictionary, it describe how data are
            oriented. Default is 'columns' if None used. If orientation is records (in pandas terminology), it's detected automatically.

    Returns:
        pd.DataFrame, dict, list : Loaded data. Usually in pd.DataFrame format, but sometimes as dict or list,
            if it needs to be processed before conversion (because of orientation).
    """

    list_of_dataframes = []

    # It can be list of more files or it can be just one path. Put it in list to same way of processing
    if not isinstance(data, (list, tuple)):
        data = [data]

    if isinstance(data[0], (list, tuple)):
        return pd.DataFrame.from_records(data)

    # If data is only path or URL or test or SQL
    elif isinstance(data, (list, tuple)) and isinstance(data[0], (str, Path)):

        if str(data[0]).lower() == "test":
            from mydatapreprocessing import generatedata

            data = generatedata.gen_random()
            mylogging.info(
                "Test data was used. Setup config.py 'data'. Check official readme for help how to configure data."
            )

            return data

        # ############# Load SQL data #############
        # elif str(data).lower() == 'sql':
        #     try:
        #         data = mydatapreprocessing.database.database_load(server=config.server, database=config.database, freq=config.freq,
        #                                                 data_limit=config.max_imported_length)

        #     except Exception:
        #         raise RuntimeError(user_message("ERROR - Data load from SQL server failed - "
        #                                         "Setup server, database and predicted column name in config"))

        #     return data

        for i in data:
            data_path = Path(i)
            iterated_data = None

            try:
                if data_path.exists():
                    iterated_data = data_path.as_posix()
                    is_file = True
                else:
                    iterated_data = i
                    is_file = False
            except Exception:
                iterated_data = i
                is_file = False

            if not is_file:
                try:
                    request = requests.get(iterated_data)
                except Exception:
                    request = None

                if not request or not (
                    request.status_code >= 200 and request.status_code < 300
                ):

                    raise FileNotFoundError(
                        mylogging.return_str(
                            "File not found on configured path. If you are using relative path, file must have be in CWD "
                            "(current working directory) or must be inserted in system path (sys.path.insert(0, 'your_path'))."
                            "If url, check if page is available.",
                            caption="File not found error",
                        )
                    )

            # For example csv or json. On url, take everything after last dot
            data_type_suffix = data_path.suffix[1:].lower()

            # If not suffix inferred, then maybe url that return as request - than suffix have to be configured
            if not data_type_suffix or (
                data_type_suffix not in ["csv", "json", "xlsx", "xls"]
                and request_datatype_suffix
            ):
                data_type_suffix = request_datatype_suffix.lower()

                if data_type_suffix.startswith("."):
                    data_type_suffix = data_type_suffix[1:]

            # If it's URL with suffix, we usually need url, if its url link with no suffix, we need get request response
            if not is_file:
                if data_type_suffix == "json":
                    iterated_data = request.content

                if data_type_suffix == "csv":
                    iterated_data = i

            if not data_type_suffix:
                raise TypeError(
                    mylogging.return_str(
                        "Data has no suffix (e.g. csv) and is not 'test' or 'sql'. "
                        "If using url with no suffix, setup 'request_datatype_suffix'"
                        "Or insert data with local path or insert data for example in "
                        f"dataframe or numpy array. \n\nYour configured data are {data}",
                        caption="Data load error",
                    )
                )

            if data_type_suffix == "csv":

                if not header:
                    header = "infer"

                if not csv_style:
                    sep = pd.read_csv(
                        iterated_data, sep=None, iterator=True
                    )._engine.data.dialect.delimiter

                    if sep not in [",", ";", "\t"]:
                        raise ValueError(
                            mylogging.return_str(
                                "CSV separator not infered. Infering not possible if description with symbols on "
                                "first few lines. Define parameter csv_style - separator and decimal manually and "
                                "skip description with header parameter."
                            )
                        )

                    decimal = "," if sep == ";" else "."

                else:
                    sep = csv_style["sep"]
                    decimal = csv_style["decimal"]

                try:
                    list_of_dataframes.append(
                        pd.read_csv(
                            iterated_data, header=header, sep=sep, decimal=decimal
                        ).iloc[-max_imported_length:, :]
                    )
                except UnicodeDecodeError:
                    list_of_dataframes.append(
                        pd.read_csv(
                            iterated_data,
                            header=header,
                            sep=csv_style["sep"],
                            decimal=csv_style["decimal"],
                            encoding="cp1252",
                        ).iloc[-max_imported_length:, :]
                    )

            elif data_type_suffix == "xls":
                if not importlib.util.find_spec("xlrd"):
                    raise ModuleNotFoundError(
                        mylogging.return_str(
                            "If using excel 'xlsx' file, library xlrd is necessary. Use \n\n\t`pip install xlrd`"
                        )
                    )

                if not predicted_table:
                    predicted_table = 0
                    list_of_dataframes.append(
                        pd.read_excel(iterated_data, sheet_name=predicted_table).iloc[
                            -max_imported_length:, :
                        ]
                    )

            elif data_type_suffix == "xlsx":
                if not importlib.util.find_spec("openpyxl"):
                    raise ModuleNotFoundError(
                        mylogging.return_str(
                            "If using excel 'xls' file, library openpyxl is necessary. Use \n\n\t`pip install openpyxl`"
                        )
                    )

                if not predicted_table:
                    predicted_table = 0
                list_of_dataframes.append(
                    pd.read_excel(
                        iterated_data, sheet_name=predicted_table, engine="openpyxl"
                    ).iloc[-max_imported_length:, :]
                )

            elif data_type_suffix == "json":

                import json

                if is_file:
                    with open(iterated_data) as json_file:
                        list_of_dataframes.append(
                            json.load(json_file)[predicted_table]
                            if predicted_table
                            else json.load(json_file)
                        )

                else:
                    list_of_dataframes.append(
                        json.loads(iterated_data)[predicted_table]
                        if predicted_table
                        else json.loads(iterated_data)
                    )

            elif data_type_suffix in ("h5", "hdf5"):
                list_of_dataframes.append(
                    pd.read_hdf(iterated_data).iloc[-max_imported_length:, :]
                )

            elif data_type_suffix in ("parquet"):
                list_of_dataframes.append(
                    pd.read_parquet(iterated_data).iloc[-max_imported_length:, :]
                )

            else:
                raise TypeError(
                    mylogging.return_str(
                        f"Your file format {data_type_suffix} not implemented yet. You can use csv, excel, parquet, h5 or txt.",
                        "Wrong (not implemented) format",
                    )
                )

    else:
        list_of_dataframes = data

    orientation = "columns" if not data_orientation else data_orientation

    for i, j in enumerate(list_of_dataframes):
        if isinstance(j, dict):
            # If just one column, put in list to have same syntax further
            if not isinstance(next(iter(j.values())), list):
                list_of_dataframes[i] = {k: [l] for (k, l) in j.items()}

            list_of_dataframes[i] = pd.DataFrame.from_dict(j, orient=orientation)

        elif isinstance(j, list):
            list_of_dataframes[i] = pd.DataFrame.from_records(j)

        elif not isinstance(j, pd.DataFrame):
            list_of_dataframes[i] = pd.DataFrame(j)

    data = pd.concat(list_of_dataframes, ignore_index=True)

    if data.empty:
        raise TypeError(
            mylogging.return_str(
                "Input data must be in pd.dataframe, pd.series, numpy array or in a path (str or pathlib) with supported formats"
                " - csv, xlsx, txt or parquet. It also can be a list of paths, files etc. If you want to generate list of file paths, "
                "you can use get_file_paths(). Check config comments for more informations...",
                "Data format error",
            )
        )

    return data


def data_consolidation(
    data,
    predicted_column=None,
    other_columns=1,
    datalength=0,
    datetime_column="",
    freq=0,
    resample_function="sum",
    embedding="label",
    unique_threshlold=0.6,
    remove_nans_threshold=0.85,
    remove_nans_or_replace="interpolate",
    dtype="float32",
):
    """Transform input data in various formats and shapes into data in defined shape, that other functions rely on.
    If you have data in other format than dataframe, use `load_data` first.

    Args:
        data (pd.DataFrame): Input data in well standardized format.
        predicted_column ((int, str), optional): Predicted column name or index. Move on first column and test if number. If None, it's ignored. Defaults to None.
        other_columns (int, optional): Whether use other columns or only predicted one. Defaults to 1.
        datalength (int, optional): Data length after resampling. Defaults to 0.
        datetime_column (str, optional): Name or index of datetime column. Defaults to ''.
        freq (int, optional): Frequency of resampled data. Defaults to 0.
        resample_function (str, optional): 'sum' or 'mean'. Whether sum resampled columns, or use average. Defaults to 'sum'.
        embedding(str, optional): 'label' or 'one-hot'. Categorical encoding. Create numbers from strings. 'label' give each category (unique string) concrete number. Defaults to 'interpolate'.
            Result will have same number of columns. 'one-hot' create for every category new column. Only columns, where are strings repeating (unique_threshlold) will be used.
        unique_threshlold(float, optional): Remove string columns, that have to many categories. E.g 0.9 define, that if column contain more that 90% of NOT unique values it's deleted. Defaults to 0.6.
            Min is 0, max is 1. It will remove ids, hashes etc.
        remove_nans_threshold (float, optional): From 0 to 1. Require that many non-nan numeric values to not be deleted. E.G if value is 0.9 with column
            with 10 values, 90% must be numeric that implies max 1 np.nan can be presented, otherwise column will be deleted.
        remove_nans_or_replace (str, float, optional): 'interpolate', 'remove', 'neighbor', 'mean' or value. Remove or replace rest nan values.
            If you want to keep nan, setup value to np.nan. If you want to use concrete value, use float or int type. Defaults to 'interpolate'.
        dtype (str, optional): Output dtype. E.g. 'float32'.

    Raises:
        KeyError, TypeError: If wrong configuration in configuration.py.
            E.g. if predicted column name not found in dataframe.


    Returns:
        np.ndarray, pd.DataFrame, str: Data in standardized form. Data array for prediction - predicted column on index 0,
        and column for ploting as pandas dataframe.

    """

    if not isinstance(data, pd.DataFrame):
        try:
            data = pd.DataFrame(data)
        except Exception as err:
            raise (
                RuntimeError(
                    mylogging.return_str(
                        "Check configuration file for supported formats. It can be path of file (csv, json, parquer...) or it "
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
                columns={
                    data_for_predictions_df.columns[
                        predicted_column
                    ]: predicted_column_name
                },
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
                data_for_predictions_df.set_index(
                    datetime_column, drop=True, inplace=True
                )

            else:
                data_for_predictions_df.set_index(
                    data_for_predictions_df.columns[datetime_column],
                    drop=True,
                    inplace=True,
                )

            data_for_predictions_df.index = pd.to_datetime(
                data_for_predictions_df.index
            )

        except Exception:
            raise KeyError(
                mylogging.return_str(
                    f"Datetime name / index from config - '{datetime_column}' not found in data or not datetime format. "
                    f"Change in config - 'datetime_column'. Available columns: {list(data_for_predictions_df.columns)}"
                )
            )

    # Convert strings numbers (e.g. '6') to numbers
    data_for_predictions_df = data_for_predictions_df.apply(
        pd.to_numeric, errors="ignore"
    )

    if embedding:
        data_for_predictions_df = categorical_embedding(
            data_for_predictions_df,
            embedding=embedding,
            unique_threshlold=unique_threshlold,
        )

    # Keep only numeric columns
    data_for_predictions_df = data_for_predictions_df.select_dtypes(include="number")

    if predicted_column_name:
        # TODO setup other columns in define input so every model can choose and simplier config input types
        if not other_columns:
            data_for_predictions_df = pd.DataFrame(
                data_for_predictions_df[predicted_column_name]
            )

        if predicted_column_name not in data_for_predictions_df.columns:
            raise KeyError(
                mylogging.return_str(
                    "Predicted column is not number datatype. Setup correct 'predicted_column' in py. "
                    f"Available columns with number datatype: {list(data_for_predictions_df.columns)}",
                    caption="Prediction available only on number datatype column.",
                )
            )

    if datetime_column not in [None, False, ""]:
        if freq:
            data_for_predictions_df.sort_index(inplace=True)
            if resample_function == "mean":
                data_for_predictions_df = data_for_predictions_df.resample(freq).mean()
            elif resample_function == "sum":
                data_for_predictions_df = data_for_predictions_df.resample(freq).sum()
            data_for_predictions_df = data_for_predictions_df.asfreq(freq, fill_value=0)

        else:
            data_for_predictions_df.index.freq = pd.infer_freq(
                data_for_predictions_df.index
            )

            if data_for_predictions_df.index.freq is None:
                reset_index = True
                mylogging.warn(
                    "Datetime index was provided from config, but frequency guess failed. "
                    "Specify 'freq' in config to resample and have equal sampling if you want "
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

    # TODO fix error after option - no predicted value - iter from 0...
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

    if isinstance(
        remove_nans_or_replace, (int, float) or np.isnan(remove_nans_or_replace)
    ):
        data_for_predictions_df.fillna(remove_nans_or_replace, inplace=True)

    # Forward fill and interpolate can miss som nans if on first row
    else:
        data_for_predictions_df.fillna(method="bfill", inplace=True)

    return data_for_predictions_df


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


def add_frequency_columns(data, window):
    """Use fourier transform on running window and add it's maximum and std as new data column.

    Args:
        data (pd.DataFrame): Data we want to use.
        window (int): length of running window.

    Returns:
        pd.Dataframe: Data with new columns, that contain informations of running frequency analysis.
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


def add_derived_columns(
    data,
    differences=True,
    second_differences=True,
    multiplications=True,
    rolling_means=True,
    rolling_stds=True,
    mean_distances=True,
    window=10,
):
    """This will create many columns that can be valuable for making predictions like difference, or
    rolling mean or distance from average. Computed columns will be appened to original data. It will process all the columns,
    so a lot of redundant data will be created. It is necessary do some feature extraction afterwards to remove noncorrelated columns.

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
                np.mean(rolling_windows(data.values.T, window), axis=2).T,
                columns=[f"{i} - Rolling mean" for i in data.columns],
            )
        )

    if rolling_stds:
        results.append(
            pd.DataFrame(
                np.std(rolling_windows(data.values.T, window), axis=2).T,
                columns=[f"{i} - Rolling std" for i in data.columns],
            )
        )

    if mean_distances:
        mean_distanced = np.zeros(data.T.shape)

        for i in range(data.shape[1]):
            mean_distanced[i] = data.values.T[i] - data.values.T[i].mean()
        results.append(
            pd.DataFrame(
                mean_distanced.T, columns=[f"{i} - Mean distance" for i in data.columns]
            )
        )

    min_length = min(len(i) for i in results)

    return pd.concat(
        [i.iloc[-min_length:].reset_index(drop=True) for i in results], axis=1
    )


def preprocess_data(
    data,
    remove_outliers=False,
    smoothit=False,
    correlation_threshold=0,
    data_transform=None,
    standardizeit="standardize",
):
    """Main preprocessing function, that call other functions based on configuration. Mostly for preparing
    data to be optimal as input into machine learning models.

    Args:
        data (np.ndarray, pd.DataFrame): Input data that we want to preprocess.
        remove_outliers (bool, optional): Whether remove unusual values far from average. Defaults to False.
        smoothit (bool, optional): Whether smooth the data. Defaults to False.
        correlation_threshold (float, optional): Whether remove columns that are corelated less than configured value
            Value must be between 0 and 1. But if 0, than None correlation threshold is applied. Defaults to 0.
        data_transform (str, optional): Whether transform data. 'difference' transform data into differences between
            neighbor values. Defaults to None.
        standardizeit (str, optional): How to standardize data. '01' and '-11' means scope from to for normalization.
            'robust' use RobustScaler and 'standard' use StandardScaler - mean is 0 and std is 1. Defaults to 'standardize'.

    Returns:
        np.ndarray, pd.DataFrame: Preprocessed data. If input in numpy array, then also output in array, if dataframe input, then dataframe output.
    """

    preprocessed = data

    if remove_outliers:
        preprocessed = remove_the_outliers(preprocessed, threshold=remove_outliers)

    if smoothit:
        preprocessed = smooth(preprocessed, smoothit[0], smoothit[1])

    if correlation_threshold:
        preprocessed = keep_corelated_data(
            preprocessed, threshold=correlation_threshold
        )

    if data_transform == "difference":
        if isinstance(preprocessed, np.ndarray):
            last_undiff_value = preprocessed[-1, 0]
        else:
            last_undiff_value = preprocessed.iloc[-1, 0]
        preprocessed = do_difference(preprocessed)
    else:
        last_undiff_value = None

    if standardizeit:
        preprocessed, final_scaler = standardize(
            preprocessed, used_scaler=standardizeit
        )
    else:
        final_scaler = None

    return preprocessed, last_undiff_value, final_scaler


def preprocess_data_inverse(
    data,
    standardizeit=False,
    final_scaler=None,
    data_transform=False,
    last_undiff_value=None,
):
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

    if data_transform == "difference":
        data = inverse_difference(data, last_undiff_value)

    return data


### Data consolidation functions...


def categorical_embedding(data, embedding="label", unique_threshlold=0.6):
    """Transform string categories such as 'US', 'FR' into numeric values, that can be used in machile learning model.

    Args:
        data (pd.DataFrame): Data with string (Object) columns.
        embedding(str, optional): 'label' or 'one-hot'. Categorical encoding. Create numbers from strings. 'label' give each category (unique string) concrete number.
            Result will have same number of columns. 'one-hot' create for every category new column. Only columns, where are strings repeating (unique_threshlold) will be used.
        unique_threshlold(float, optional): Remove string columns, that have to many categories (ids, hashes etc.). E.g 0.9 defines that in column of length 100, max number of
            categories to not to be deleted is 10 (90% non unique repeating values). Defaults to 0.6. Min is 0, max is 1.

    Returns:
        pd.DataFrame: Dataframe where string columns transformed to numeric.
    """
    data_for_embedding = data.copy()
    to_drop = []

    for i in data_for_embedding.select_dtypes(exclude=["number"]):

        try:

            if (data_for_embedding[i].nunique() / len(data_for_embedding[i])) > (
                1 - unique_threshlold
            ):
                to_drop.append(i)
                continue

            data_for_embedding[i] = data_for_embedding[i].astype("category", copy=False)

            if embedding == "label":
                data_for_embedding[i] = data_for_embedding[i].cat.codes

            if embedding == "one-hot":
                data_for_embedding = data_for_embedding.join(
                    pd.get_dummies(data_for_embedding[i])
                )
                to_drop.append(i)

        except Exception:
            to_drop.append(i)

    # Drop columns with too few caterogies - drop all columns at once to better performance
    data_for_embedding.drop(to_drop, axis=1, inplace=True)

    return data_for_embedding


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
            columns_to_del = range_array[abs(corr[0]) <= threshold]

            data = np.delete(data, columns_to_del, axis=1)

    elif isinstance(data, pd.DataFrame):
        corr = data.corr().iloc[0, :]
        corr = corr[~corr.isnull()]
        names_to_del = list(corr[abs(corr) <= threshold].index)
        data.drop(columns=names_to_del, inplace=True)

    return data


def remove_the_outliers(data, threshold=3, main_column=0):
    """Remove values far from mean - probably errors. If more columns, then only rows that have outlier on
    predicted column will be deleted. Predicted column is supposed to be 0.

    Args:
        data (np.array, pd.DataFrame): Time series data. Must have ndim = 2, if univariate, reshape...
        threshold (int, optional): How many times must be standard deviation from mean to be ignored. Defaults to 3.
        main_column ((int, index)):
    Returns:
        np.array: Cleaned data.

    Examples:

        >>> data = np.array([[1, 3, 5, 2, 3, 4, 5, 66, 3]])
        >>> print(remove_the_outliers(data))
        [1, 3, 5, 2, 3, 4, 5, 3]
    """

    if isinstance(data, np.ndarray):
        data_mean = data[:, main_column].mean()
        data_std = data[:, main_column].std()

        range_array = np.array(range(data.shape[0]))
        names_to_del = range_array[
            abs(data[:, main_column] - data_mean) > threshold * data_std
        ]
        data = np.delete(data, names_to_del, axis=0)

    elif isinstance(data, pd.DataFrame):
        if isinstance(main_column, int):
            main_column = data.columns[main_column]

        data_mean = data[main_column].mean()
        data_std = data[main_column].std()

        data = data[abs(data[main_column] - data_mean) < threshold * data_std]

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

    assert differenced_predictions.ndim == 1, "Data input must be one-dimensional."

    return np.insert(differenced_predictions, 0, last_undiff_value).cumsum()[1:]


def standardize(data, used_scaler="standardize"):
    """Standardize or normalize data. More standardize methods available. Predicted column is supposed to be 0.

    Args:
        data (np.ndarray): Time series data.
        used_scaler (str, optional): '01' and '-11' means scope from to for normalization.
            'robust' use RobustScaler and 'standardize' use StandardScaler - mean is 0 and std is 1. Defaults to 'standardize'.

    Returns:
        ndarray: Standardized data.
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


def standardize_one_way(data, min, max, axis=0, inplace=False):
    """Own implementation of standardization. No inverse transformation available.
    Reason is for builded applications to do not carry sklearn with build.

    Args:
        data ((np.array, pd.DataFrame)): Data.
        min (float): Minimum in transformed axis.
        max (float): Max in transformed axis.
        axis (int, optional): 0 to columns, 1 to rows. Defaults to 0.
        inplace (bool, optional): If true, no copy will be returned, but original object. Defaults to False.

    Returns:
        (np.array, pd.DataFrame): Standardized data. If numpy inserted, numpy returned, same for dataframe.
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
    if not importlib.util.find_spec("scipy"):
        raise ImportError(
            "scipy library is necessary for smooth function. Install via `pip install scipy`"
        )

    import scipy.signal

    if isinstance(data, pd.DataFrame):
        for i in range(data.shape[1]):
            data.iloc[:, i] = scipy.signal.savgol_filter(
                data.values[:, i], window, polynom_order
            )

    elif isinstance(data, np.ndarray):
        for i in range(data.shape[1]):
            data[:, i] = scipy.signal.savgol_filter(data[:, i], window, polynom_order)

    return data


def fitted_power_transform(data, fitted_stdev, mean=None, fragments=10, iterations=5):
    """Function mostly for data postprocessing. Function transforms data, so it will have
    similiar standar deviation, similiar mean if specified. It use Box-Cox power transform in SciPy lib.

    Args:
        data (np.array): Array of data that should be transformed (one column => ndim = 1).
        fitted_stdev (float): Standard deviation that we want to have.
        mean (float, optional): Mean of transformed data. Defaults to None.
        fragments (int, optional): How many lambdas will be used in one iteration. Defaults to 9.
        iterations (int, optional): How many iterations will be used to find best transform. Defaults to 4.

    Returns:
        np.array: Transformed data with demanded standard deviation and mean.
    """

    if data.ndim == 2 and 1 not in data.shape:
        raise ValueError(
            mylogging.return_str(
                "Only one column can be power transformed. Use ravel if have shape (n, 1)"
            )
        )

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


def add_none_to_gaps(df):
    """If empty windows in sampled signal, it will add None values (one row) to the empty window start.
    Reason is to correct plotting. Points are connected, but not between two gaps.

    Args:
        df (pd.DataFrame): Dataframe with time index.
    Returns:
        df: Df with None row inserted in time gaps.
    """

    sampling_threshold = (
        remove_the_outliers(np.diff(df.index[:50]).reshape(-1, 1), threshold=1).mean()
        * 3
    )
    nons = []
    memory = None

    for i in df.index:
        if memory and i - memory > sampling_threshold:
            nons.append(
                pd.DataFrame(
                    [[np.nan] * df.shape[1]], index=[memory + sampling_threshold]
                )
            )
        memory = i

    return pd.concat([df, *nons]).sort_index()
