"""Tests for load_data package."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from mypythontools_cicd import tests

tests.setup_tests()

import mydatapreprocessing.load_data as mdpd

from tests.helpers import compare_values

test_files_path = Path(__file__).parent / "test_files"

# pylint: disable=missing-function-docstring


def test_exceptions():

    try:
        mdpd.load_data("testfile")
    except TypeError as err:
        assert isinstance(err, TypeError)

    try:
        mdpd.load_data("testfile.csv")
    except FileNotFoundError as err:
        assert isinstance(err, FileNotFoundError)

    try:
        mdpd.load_data("https://www.ncgdfgddc.noaa.gov/")
    except NotImplementedError as err:
        assert isinstance(err, NotImplementedError)


def test_test_data():
    for i in ["test_ramp", "test_sin", "test_random", "test_ecg"]:
        assert mdpd.load_data(i).ndim


def test_ndarray_and_df():
    assert mdpd.load_data(np.random.randn(100, 3)).shape == (100, 3)
    assert mdpd.load_data(pd.DataFrame(np.random.randn(100, 3))).shape == (100, 3)


def test_csv():
    assert mdpd.load_data(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    ).ndim


def test_dicts():

    assert compare_values(
        mdpd.load_data({"row_1": [3, 2, 1, 0], "row_2": [3, 2, 1, 0]}, data_orientation="index"),
        np.array([[3, 2, 1, 0], [3, 2, 1, 0]]),
    )

    assert compare_values(
        mdpd.load_data(
            [
                {"col_1": [3, 2, 1], "col_2": [3, 2, 1]},
                {"col_1": [4, 2, 1], "col_2": [3, 2, 1]},
            ],
            data_orientation="columns",
        ),
        np.array([[3, 3], [2, 2], [1, 1], [4, 3], [2, 2], [1, 1]]),
    )

    assert compare_values(
        mdpd.load_data(
            [
                {"col_1": 3, "col_2": 3},
                {"col_1": 4, "col_2": 6},
            ],
            data_orientation="columns",
        ),
        np.array([[3, 3], [4, 6]]),
    )

    assert compare_values(
        mdpd.load_data(
            [
                {"row1": [3, 2, 1, 0]},
                {"row2": [4, 2, 1, 0]},
            ],
            data_orientation="index",
        ),
        np.array([[3, 2, 1, 0], [4, 2, 1, 0]]),
    )


def test_list():
    assert compare_values(
        mdpd.load_data([["Jon", "Smith", 21], ["Mark", "Brown", 38], ["Maria", "Lee", 42]]),
        pd.DataFrame([["Jon", "Smith", 21], ["Mark", "Brown", 38], ["Maria", "Lee", 42]]),
    )


def test_tuple():
    assert compare_values(
        mdpd.load_data((("Jon", "Smith", 21), ("Mark", "Brown", 38), ("Maria", "Lee", 42))),
        pd.DataFrame([["Jon", "Smith", 21], ["Mark", "Brown", 38], ["Maria", "Lee", 42]]),
    )


def test_json():
    assert compare_values(
        mdpd.load_data(
            test_files_path / "json_nested.json", field="main_field_1.sub_field_1.sub_sub_field_1"
        ),
        np.array([["value1", "value2"]]),
    )

    assert compare_values(
        mdpd.load_data(test_files_path / "json_flat.json", field=""),
        np.array([["value1", "value3"], ["value2", "value4"]]),
    )

    assert mdpd.load_data(
        "https://raw.githubusercontent.com/Malachov/mydatapreprocessing/master/tests/test_files/list.json",
    ).ndim


def test_more_files():

    assert mdpd.load_data([test_files_path / "csv.csv", test_files_path / "csv.csv"]).shape == (6, 3)

    assert mdpd.load_data([np.random.randn(20, 3), np.random.randn(25, 3)]).shape == (45, 3)
    assert mdpd.load_data(
        (pd.DataFrame(np.random.randn(20, 3)), pd.DataFrame(np.random.randn(25, 3)))
    ).shape == (45, 3)


def test_files():
    expected = pd.DataFrame([[1, 1.0, "One"], [2, 2.0, "Two"], [3, 3.0, "Three"]])

    # json test aside as its works in different way

    file_types = ["xls", "xlsx", "csv", "parquet"]

    for i in file_types:
        loaded_local = mdpd.load_data(test_files_path / f"{i}.{i}")
        assert compare_values(loaded_local, expected), f"Load of {i} failed."

        loaded_web = mdpd.load_data(
            f"https://github.com/Malachov/mydatapreprocessing/blob/master/tests/test_files/{i}.{i}?raw=true",
            request_datatype_suffix=i,
        )
        assert compare_values(loaded_web, expected), f"Load of {i} failed."

    if sys.version_info.minor > 7:
        # H5 not supported from url and need kwargs
        assert compare_values(
            mdpd.load_data(test_files_path / "h5.h5", sheet="h5"), expected
        ), "Load of 'h5.h5' failed."

    # Replace h5 and parquet with new data with
    # loaded["csv"].to_parquet((test_files_path / "parquet.parquet").as_posix(), compression="gzip")
    # loaded["csv"].to_hdf((test_files_path / "h5.h5").as_posix(), key="h5")


if __name__ == "__main__":
    # test_files()
    pass
