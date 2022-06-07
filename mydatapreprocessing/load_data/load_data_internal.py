"""Module for load_data subpackage."""

from __future__ import annotations
from pathlib import Path
from typing import Iterable

import pandas as pd
from typing_extensions import Literal

from mypythontools.system import check_library_is_available

from .load_data_functions import data_parsers, get_file_type, return_test_data, download_data_from_url
from ..types import DataFormat

# Lazy load
# import requests
# import json
# import tkinter as tk
# from tkinter import filedialog


def get_file_paths(
    filetypes: Iterable[tuple[str, str | list[str] | tuple[str, ...]]]
    | None = [
        ("csv", ".csv"),
        ("Excel (xlsx, xls)", ".xlsx .xls"),
        ("h5", ".h5"),
        ("parquet", ".parquet"),
        ("json", ".json"),
    ],
    title: str = "Select files",
) -> tuple[str, ...] | Literal[""]:
    """Open dialog window where you can choose files you want to use. It will return tuple with string paths.

    Args:
        filetypes (Iterable[tuple[str, str | list[str] | tuple[str, ...]]] | None, optional): Accepted file
            types / suffixes. List of strings or list of tuples.
            Defaults to [
                ("csv", ".csv"),
                ("Excel (xlsx, xls)", ".xlsx .xls"),
                ("h5", ".h5"),
                ("parquet", ".parquet"),
                ("json", ".json")].
        title (str, optional): Just a name of dialog window. Defaults to 'Select file'.

    Returns:
        tuple[str, ...] | Literal[""]: Tuple with string paths. If dialog window is closed, it will return empty string.
    """
    import tkinter as tk
    from tkinter import filedialog

    # Open dialog window where user can choose which files to use
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)

    return filedialog.askopenfilenames(filetypes=filetypes, title=title)


def load_data(
    data: DataFormat,
    header: Literal["infer"] | int | None = "infer",
    csv_style: dict | Literal["infer"] = "infer",
    field: str = "",
    sheet: str | int = 0,
    max_imported_length: None | int = None,
    request_datatype_suffix: str | None = "",
    data_orientation: Literal["columns", "index"] = "columns",
    ssl_verification: None | bool | str = None,
) -> pd.DataFrame:
    """Load data from path or url or other python format (numpy array, list, dict) into DataFrame.

    Available formats are csv, excel xlsx, parquet, json or h5. Allow multiple files loading at once - just
    put it in list e.g. [df1, df2, df3] or ['my_file1.csv', 'my_file2.csv']. Structure of files does not have
    to be the same. If you have files in folder and not in list, you can use `get_file_paths` function to open
    system dialog window, select files and get the list of paths.

    Args:
        data (Any): Path to file, url or python data. For examples check examples section.
        header (Literal['infer'] | int | None, optional): Row index used as column names. If 'infer', it will
            be automatically chosen. Defaults to 'infer'.
        csv_style (dict | Literal["infer"], optional): Define CSV separator and decimal. En locale usually
            use ``{'sep': ",", 'decimal': "."}`` some European country use ``{'sep': ";", 'decimal': ","}``.
            If 'infer' one of those two locales is automatically used. Defaults to 'infer'.
        field (str , optional): If using json, it means what field to use. For example "field.subfield.data"
            as if json was dict means ``data[field][subfield][data]`` key values, if SQL, then it mean what table.
            If empty string, root level is used. Defaults to ''.
        sheet (str | int, optional): If using xls or xlsx excel file it define what sheet will be used. If
            using h5, it define what key (chanel group) to use. Defaults to 0.
        max_imported_length (None | int, optional): Max length of imported samples (before resampling).
            If 0, than full length. Defaults to None.
        request_datatype_suffix(str | None, optional): 'json' for example. If using url with no extension,
            define which datatype is on this url with GET request. Defaults to "".
        data_orientation(Literal["columns", "index"], optional): 'columns' or 'index'. If using json or
            dictionary, it describe how data are oriented. Defaults to "columns".
        ssl_verification(None | bool | str, optional): If using data from web, it use requests and sometimes,
            there can be ssl verification error, this skip verification, with adding verify param to requests
            call. It!s param of requests get function. Defaults to None.

    Raises:
        FileNotFoundError, TypeError, ValueError, ModuleNotFoundError: If not existing file, or url, or if
            necessary dependency library not found.

    Returns:
        pd.DataFrame : Loaded data in pd.DataFrame format.

    Examples:
        **Some python formats if data are already in python**

        Numpy array or pandas DataFrame

        >>> import numpy as np
        >>> array_or_DataFrame = np.random.randn(10, 2)

        List of records

        >>> records = [{'col_1': 3, 'col_2': 'a'}, {'col_1': 0, 'col_2': 'd'}] # List of records

        Dict with columns or rows (index) - necessary to setup data_orientation!

        >>> dict_data = {'col_1': [3, 2, 1, 0], 'col_2': ['a', 'b', 'c', 'd']}

        **External files**

        Local file.

        >>> local_file = r"tests/test_files/tested_data.csv"  # The same with .parquet, .h5, .json or .xlsx.

        Web URL. If it has suffix in URL, it's not necessary to define the type, but when it's missing,
        you have to specify also e.g. 'request_datatype_suffix': "json". Sometimes, you need to define also
        further information like 'data_orientation': "index" whether use index or columns, or for json
        'field': 'data' which define in what field data are stored (it can be nested with dots like
        'dataset_1.data.summer').

        >>> url = "https://www.ncdc.noaa.gov/cag/global/time-series/globe/land_ocean/ytd/12/1880-2016.json"

        You can use more files in list and data will be concatenated. It can be list of paths or list of python objects. For example::

        >>> multiple_dict_data = [{'col_1': 3, 'col_2': 'a'}, {'col_1': 0, 'col_2': 'd'}]  # List of records
        >>> multiple_arrays_or_dfs = [np.random.randn(20, 3), np.random.randn(25, 3)]  # DataFrame same way
        >>> multiple_urls = ["https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv", "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"]
        >>> multiple_local_files = ["tests/test_files/tested_data.csv", "tests/test_files/tested_data.csv"]

        You can also use some test data. Use one of 'test_ecg' 'test_ramp' 'test_sin' 'test_random'.

        >>> data_loaded = []
        >>> for i in [
        ...     array_or_DataFrame, local_file, records, dict_data, url,
        ...     multiple_dict_data, multiple_arrays_or_dfs, multiple_urls, multiple_local_files
        ... ]:
        ...     data_loaded.append(load_data(i).size > 0)
        >>> all(data_loaded)
        True

    Note:
        On windows when using paths, it's necessary to use raw string - 'r' in front of string because of
        escape symbols
    """
    if isinstance(data, str) and data in ["test_ramp", "test_sin", "test_random", "test_ecg"]:
        return return_test_data(data)  # type: ignore

    # It can be list of more files or it can be just one path. Put it in list for same way of processing

    datas = data if isinstance(data, (list, tuple)) else [data]

    if all((isinstance(i, (list, tuple)) for i in datas)):
        return pd.DataFrame.from_records(data)

    list_of_DataFrames: list[pd.DataFrame] = []

    for iterated_data in datas:
        if isinstance(iterated_data, pd.DataFrame):
            result_data = iterated_data

        elif isinstance(iterated_data, dict):
            # If just one column, put in list to have same syntax further
            result_data = data_parsers.load_dict(iterated_data, data_orientation)

        elif isinstance(iterated_data, list):
            result_data = pd.DataFrame(iterated_data)

        # If data is only path or URL to data
        elif isinstance(iterated_data, (str, Path)):

            data_path = Path(iterated_data)

            file_type = get_file_type(data_path, request_datatype_suffix)

            supported_formats = ["csv", "json", "xlsx", "xls"]
            
            if file_type not in supported_formats:
                raise NotImplementedError(
                    f"File extension not implemented. Supported file formats are {supported_formats}. Used "
                    f"format is '{file_type}'.")
                
            if data_path.exists():
                data_content = data_path.as_posix()
            else:
                try:
                    # Is URL
                    data_content = download_data_from_url(str(iterated_data), ssl_verification)
                except FileNotFoundError as err:
                    raise FileNotFoundError(
                        "Error in 'mydatapreprocessing' package in 'load_data' function. "
                        "File not found on configured path. If you are using relative path, file must "
                        "have be in CWD (current working directory) or must be inserted in system path "
                        "(sys.path.insert(0, 'your_path')). If url, check if page is available.",
                    ) from err

            if file_type == "csv":
                result_data = data_parsers.csv_load(data_content, csv_style, header, max_imported_length)

            elif file_type == "json":
                result_data = data_parsers.json_load(
                    data_content, field=field, data_orientation=data_orientation
                )

            elif file_type == "xls":
                check_library_is_available("xlrd")
                if header == "infer":
                    header = 0
                result_data = pd.read_excel(data_content, sheet_name=sheet, header=header)  # type: ignore

            elif file_type == "xlsx":
                check_library_is_available("openpyxl")

                result_data = pd.read_excel(
                    data_content, sheet_name=sheet, header=header, engine="openpyxl"  # type: ignore
                )

            elif file_type in ("h5", "hdf5"):
                result_data = pd.read_hdf(data_content, key=sheet)

            elif file_type in ("parquet"):
                result_data = pd.read_parquet(data_content)

            else:
                raise TypeError(
                    "Error in 'mydatapreprocessing' package in 'load_data' function. "
                    f"Your file format {file_type} not implemented yet. You can use csv, xls, xlsx, "
                    "parquet, h5 or txt.",
                )

        else:
            try:
                result_data = pd.DataFrame(iterated_data)

            except Exception as err:
                raise ValueError(
                    "Error in 'mydatapreprocessing' package in 'load_data' function. "
                    "If using python format as input, Input data must be in pd.DataFrame, pd.series, "
                    "numpy array or some format that can be used in pd.DataFrame() constructor."
                ) from err

        list_of_DataFrames.append(result_data)

    data = pd.concat(list_of_DataFrames, ignore_index=True)

    return data
