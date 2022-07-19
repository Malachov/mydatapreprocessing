"""Module for load_data_functions subpackage."""

from __future__ import annotations
from pathlib import Path
import io

import pandas as pd
from typing_extensions import Literal

from ... import datasets


def download_data_from_url(url: str, ssl_verification: None | bool | str = None) -> io.BytesIO:
    """Download data from defined url and returns io.BytesIO.

    Args:
        url (str): Url with defined file.
        ssl_verification (None | bool | str, optional): Same meaning as in requests library.

    Raises:
        FileNotFoundError: If url is not available.

    Returns:
        io.BytesIO:  Converted to io.BytesIO so it can later be used for example in pandas read_x functions.

    Example:
        >>> downloaded = download_data_from_url(
        ...     "https://github.com/Malachov/mydatapreprocessing/blob/master/tests/test_files/csv.csv?raw=true"
        ... )
        >>> downloaded
        <_io.BytesIO object at...
        >>> downloaded.readline()
        b'Column 1, Column 2...
    """
    import requests

    try:
        request = requests.get(url, verify=ssl_verification)
    except requests.exceptions.RequestException as err:
        raise FileNotFoundError(f"Url '{url}' probably not available or no permissions available.") from err

    if not request or not (200 <= request.status_code < 300):
        raise RuntimeError(
            f"Request failed with status {request.status_code}.",
        )

    return io.BytesIO(request.content)


def return_test_data(data: Literal["test_ramp", "test_sin", "test_random", "test_ecg"]) -> pd.DataFrame:
    """If want some test data, define just name and get data.

    Args:
        data (Literal['test_ramp', 'test_sin', 'test_random', 'test_ecg']): Possible test data. Most of it
            is generated, test_ecg is real data.

    Returns:
        pd.DataFrame: Test data.

    Example:
        >>> return_test_data('test_ramp')
               0
        0      0
        1      1
        2      2
        ...

    """
    if data == "test_ramp":
        return pd.DataFrame(datasets.ramp())

    elif data == "test_sin":
        return pd.DataFrame(datasets.sin())

    elif data == "test_random":
        return pd.DataFrame(datasets.random())

    elif data == "test_ecg":
        return pd.DataFrame(datasets.get_ecg())


def get_file_type(data_path: Path, request_datatype_suffix: None | str = None):
    """Give file name or url with extension and return file extension.

    If file extension not at end of url add it extra.

    Args:
        data_path (Path): Defined path. It can also be URL, but it must be pathlib.Path in format.
        request_datatype_suffix (None | str): If there is no extension in name, it can be defined via
            parameter. Defaults to None.

    Raises:
        TypeError: If extension not inferred and not defined with param.

    Returns:
        str: Extension lowered like for example 'csv'.
    """
    if request_datatype_suffix:
        file_type = request_datatype_suffix.lower()

        if file_type.startswith("."):
            file_type = file_type[1:]

    # If not suffix inferred, then maybe url that return as request - than suffix have to be configured
    else:
        # For example csv or json. On url, take everything after last dot
        file_type = data_path.suffix[1:].lower()

    if not file_type:
        raise TypeError(
            "Data has no suffix (e.g. csv). If using url with no suffix, setup"
            "'request_datatype_suffix' or insert data with local path or insert data for example in"
            f"DataFrame or numpy array. \n\nParsed data are '{data_path}'",
        )

    return file_type
