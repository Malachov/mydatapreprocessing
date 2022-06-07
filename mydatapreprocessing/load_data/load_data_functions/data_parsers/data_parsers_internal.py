"""Module for data_parsers subpackage."""

from __future__ import annotations
from typing import Any
import io
from pathlib import Path

from typing_extensions import Literal
import pandas as pd

import mylogging


def csv_load(
    data: io.BytesIO | str | Path,
    csv_style: Literal["infer"] | dict = "infer",
    header: Literal["infer"] | None | int = None,
    max_imported_length: None | int = None,
) -> pd.DataFrame:
    """Load CSV data and infer used separator.

    Args:
        data (io.BytesIO | str | Path): Input data.
        csv_style (Literal["infer"] | dict, optional): If infer, inferred automatically else dictionary
            with `sep` and `decimal`. E.g. {'sep': ';', 'dec': ','}. Defaults to "infer".
        header (Literal['infer'] | None | int, optional): First row used. Usually with column names.
            Defaults to None.
        max_imported_length (int, optional): Last N rows used. Defaults to None.

    Raises:
        RuntimeError: If loading fails.

    Returns:
        pd.DataFrame: Loaded data.
    """
    if csv_style == "infer":

        if isinstance(data, io.BytesIO):
            data_line = _get_further_data_line(data)
            data.seek(0)

        else:
            with open(data, "r") as data_read:
                data_line = _get_further_data_line(data_read)

        if data_line.count(";") and data_line.count(";") >= data_line.count(",") - 1:
            sep = ";"
            decimal = ","
        else:
            sep = ","
            decimal = "."

    else:
        sep = csv_style["sep"]
        decimal = csv_style["decimal"]

    try:
        loaded = pd.read_csv(data, header=header, sep=sep, decimal=decimal)

    except UnicodeDecodeError:
        loaded = pd.read_csv(
            data,
            header=header,
            sep=sep,
            decimal=decimal,
            encoding="cp1252",
        )

    except Exception as err:
        raise RuntimeError(
            "CSV load failed. Try to set correct `header` and `csv_style`",
        ) from err

    if max_imported_length:
        loaded = loaded.iloc[-max_imported_length:, :]

    return loaded


def load_dict(data: dict[str, Any], data_orientation: Literal["index", "columns"] = "columns"):
    """Load dict with values to DataFrame.

    Args:
        data (dict[str, Any]): Data with array like values.
        data_orientation (Literal["index", "columns"], optional): Define dict data orientation.
            Defaults to "columns".

    Returns:
        pd.DataFrame: Loaded data.
    """
    if isinstance(next(iter(data.values())), list):
        dict_of_lists = data
    else:
        dict_of_lists = {k: [l] for (k, l) in data.items()}

    return pd.DataFrame.from_dict(dict_of_lists, orient=data_orientation)


def json_load(
    data: str | Path | io.BytesIO, field: str, data_orientation: Literal["index", "columns"] = "columns"
):
    """Load data from json to DataFrame.

    The reason why pandas read_json is not used is that usually just some subfield with inner json is used.

    Args:
        data (str | Path | io.BytesIO): Input data. Path to file or io.BytesIO created for example from
            request content.
        field (str, optional): If you need to use just a node from data.  You can use dot for entering another
            levels of nested data. For example "key_1.sub_key_1"
        data_orientation (Literal["index", "columns"], optional): Define dict data orientation.
            Defaults to "columns".

    Raises:
        KeyError: If defined key is not available.

    Returns:
        pd.DataFrame: Loaded data.
    """
    # pandas is not used so it's possible to use just one fields values as subset to original data
    import json

    if isinstance(data, io.BytesIO):
        data_dict = json.loads(data.read())

    else:
        with open(data) as json_file:
            data_dict = json.load(json_file)

    if field:
        for i in field.split("."):
            try:
                data_dict = data_dict[i]
            except KeyError as err:
                raise KeyError(
                    mylogging.format_str(f"Data load error. Defined field '{field}' not found in data.")
                ) from err

    return load_dict(data_dict, data_orientation)


def _get_further_data_line(data) -> str:
    data_line = ""
    for _ in range(20):
        new_line = data.readline()
        if new_line:
            data_line = new_line
        else:
            break

    return str(data_line)
