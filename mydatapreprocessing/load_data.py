"""This module helps you to load data from path as well as from web url in various formats.

Supported path formats are:

- csv
- xlsx and xls
- json
- parquet
- h5

You can insert more files (urls) at once and your data will be automatically concatenated.

Main function is `load_data` where you can find working examples.

There is also function `get_file_paths` which open an dialog window in your operation system and let you
choose your files in convenient way. This tuple output you can then insert into `load_data`.
"""
import mylogging

import importlib
from pathlib import Path

import pandas as pd

from . import generate_data

# Lazy load
# import json
# import tkinter as tk
# from tkinter import filedialog


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
            Defaults to [("csv", ".csv"), ("Excel (xlsx, xls)", ".xlsx .xls"), ("h5", ".h5"), ("parquet", ".parquet"), ("json", ".json")].
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
    ssl_verification=False,
):

    """Load data from path or url or other python format (numpy array, list, dict) into dataframe.
    Available formats are csv, excel xlsx, parquet, json or h5. Allow multiple files loading at once - just put
    it in list e.g. [df1, df2, df3] or ['myfile1.csv', 'myfile2.csv']. Structure of files does not have to be the same.
    If you have files in folder and not in list, you can use `get_file_paths` function to open system dialog window,
    select files and get the list of paths.


    Args:
        data (Any): Path, url. For examples check examples section.
        header (int, optional): Row index used as column names. Defaults to None.
        csv_style ((dict, None), optional): Define CSV separator and decimal. En locale usually use {'sep': ",", 'decimal': "."}
            some Europian country use {'sep': ";", 'decimal': ","}. If None - values are infered. Defaults to None.
        predicted_table (str, optional): If using excel (xlsx) - it means what sheet to use, if json,
            it means what key values, if SQL, then it mean what table. Else it have no impact. Defaults to ''.
        max_imported_length (int, optional): Max length of imported samples (before resampling). If 0, than full length.
            Defaults to 0.
        request_datatype_suffix(str, optional): 'json' for example. If using url with no extension,
            define whichdatatype is on this url with GET request. Defaults to "".
        data_orientation(str, optional): 'columns' or 'index'. If using json or dictionary, it describe how data are
            oriented. Default is 'columns' if None used. If orientation is records (in pandas terminology), it's detected
            automatically. Defaults to "".
        ssl_verification(bool, optional): If using data from web, it use requests and sometimes, there can be ssl verification
            error, this skip verification, with adding verify param to requests call.

    Raises:
        FileNotFoundError, TypeError, ValueError, ModuleNotFoundError: If not existing file, or url, or if necessary
            dependency library not found.

    Returns:
        pd.DataFrame : Loaded data in pd.DataFrame format.

    Examples:
        You can use local files as well as web urls

        >>> data_loaded = load_data("https://blockchain.info/unconfirmed-transactions?format=json", request_datatype_suffix=".json", data_orientation="index", predicted_table='txs')
        >>> # data2 = load_data(PATH_TO_FILE.csv)

        Allowed data formats for load_data are examples::

            myarray_or_dataframe # Numpy array or Pandas.DataFrame
            r"/home/user/my.json" # Local file. The same with .parquet, .h5, .json or .xlsx.
            "https://yoururl/your.csv" # Web url (with suffix). Same with json.
            "https://blockchain.info/unconfirmed-transactions?format=json" # In this case you have to specify
                also 'request_datatype_suffix': "json", 'data_orientation': "index", 'predicted_table': 'txs',
            [{'col_1': 3, 'col_2': 'a'}, {'col_1': 0, 'col_2': 'd'}] # List of records
            {'col_1': [3, 2, 1, 0], 'col_2': ['a', 'b', 'c', 'd']} # Dict with colums or rows (index) - necessary
                to setup data_orientation!

        You can use more files in list and data will be concatenated. It can be list of paths or list of python objects. For example::

            [{'col_1': 3, 'col_2': 'a'}, {'col_1': 0, 'col_2': 'd'}]  # List of records
            [np.random.randn(20, 3), np.random.randn(25, 3)]  # Dataframe same way
            ["https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
                "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"]  # List of URLs
            ["path/to/my1.csv", "path/to/my1.csv"]

    Note:
        On windows it's necessary to use raw string - 'r' in front of string because of escape symbols \
    """

    list_of_dataframes = []

    # It can be list of more files or it can be just one path. Put it in list to same way of processing
    if not isinstance(data, (list, tuple)):
        data = [data]

    if isinstance(data[0], (list, tuple)):
        return pd.DataFrame.from_records(data)

    # If data is only path or URL or test or SQL
    elif isinstance(data, (list, tuple)) and isinstance(data[0], (str, Path)):

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

                import requests

                try:
                    request = requests.get(iterated_data, verify=ssl_verification)
                except Exception:
                    request = None

                if not request or not (request.status_code >= 200 and request.status_code < 300):

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
                data_type_suffix not in ["csv", "json", "xlsx", "xls"] and request_datatype_suffix
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
                        "Data has no suffix (e.g. csv). If using url with no suffix, setup"
                        "'request_datatype_suffix' or insert data with local path or insert data for example in"
                        f"dataframe or numpy array. \n\nYour configured data are {data}",
                        caption="Data load error",
                    )
                )

            if data_type_suffix == "csv":

                if not header:
                    header = "infer"

                if not csv_style:
                    sep = pd.read_csv(iterated_data, sep=None, iterator=True)._engine.data.dialect.delimiter

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
                        pd.read_csv(iterated_data, header=header, sep=sep, decimal=decimal).iloc[
                            -max_imported_length:, :
                        ]
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
                    pd.read_excel(iterated_data, sheet_name=predicted_table, engine="openpyxl").iloc[
                        -max_imported_length:, :
                    ]
                )

            elif data_type_suffix == "json":

                import json

                if is_file:
                    with open(iterated_data) as json_file:
                        list_of_dataframes.append(
                            json.load(json_file)[predicted_table] if predicted_table else json.load(json_file)
                        )

                else:
                    list_of_dataframes.append(
                        json.loads(iterated_data)[predicted_table]
                        if predicted_table
                        else json.loads(iterated_data)
                    )

            elif data_type_suffix in ("h5", "hdf5"):
                list_of_dataframes.append(pd.read_hdf(iterated_data).iloc[-max_imported_length:, :])

            elif data_type_suffix in ("parquet"):
                list_of_dataframes.append(pd.read_parquet(iterated_data).iloc[-max_imported_length:, :])

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
