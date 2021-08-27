import json
from pathlib import Path

import requests
import numpy as np
import pandas as pd

import mypythontools

# Find paths and add to sys.path to be able to import local modules
mypythontools.tests.setup_tests()


import mydatapreprocessing.load_data as mdpd

np.random.seed(2)


def test_exceptions():

    exceptions = []

    try:
        mdpd.load_data("testfile")
    except Exception as e:
        exceptions.append(isinstance(e, TypeError))

    try:
        mdpd.load_data("testfile.csv")
    except Exception as e:
        exceptions.append(isinstance(e, FileNotFoundError))

    try:
        mdpd.load_data("https://www.ncgdfgddc.noaa.gov/")
    except Exception as e:
        exceptions.append(isinstance(e, FileNotFoundError))

    assert all(exceptions)


def test_numpy_and_dataframe():
    assert (
        mdpd.load_data(np.random.randn(100, 3)).ndim
        and mdpd.load_data(pd.DataFrame(np.random.randn(100, 3))).ndim
    )


def test_numpys_and_pandas():
    assert (
        mdpd.load_data([np.random.randn(20, 3), np.random.randn(25, 3)]).ndim
        and mdpd.load_data((pd.DataFrame(np.random.randn(20, 3)), pd.DataFrame(np.random.randn(25, 3)))).ndim
    )


def test_dict():
    assert mdpd.load_data({"col_1": [3, 2, 1, 0], "col_2": [3, 2, 1, 0]}, data_orientation="index").ndim


def test_list_of_dicts():
    assert mdpd.load_data(
        [
            {"col_1": [3, 2, 1, 0], "col_2": [3, 2, 1, 0]},
            {"col_1": [4, 2, 1, 0], "col_2": [3, 2, 1, 0]},
            {"col_1": [5, 2, 1, 0], "col_2": [3, 2, 1, 0]},
        ],
        data_orientation="columns",
    ).ndim


def test_list():
    assert mdpd.load_data([["Jon", "Smith", 21], ["Mark", "Brown", 38], ["Maria", "Lee", 42]]).ndim


def test_tuple():
    assert mdpd.load_data((("Jon", "Smith", 21), ("Mark", "Brown", 38), ("Maria", "Lee", 42))).ndim


def test_more_files():

    data_loaded = mdpd.load_data(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
        request_datatype_suffix=".json",
        predicted_table="txs",
        data_orientation="index",
    )
    data_loaded2 = mdpd.load_data(
        [
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
        ],
        request_datatype_suffix=".json",
        predicted_table="txs",
        data_orientation="index",
    )

    assert len(data_loaded2) == 2 * len(data_loaded)


def test_local_files():

    test_files = Path(__file__).parent / "test_files"
    xls = mdpd.load_data(test_files / "file_example_xls.xls")
    xlsx = mdpd.load_data(test_files / "file_example_xlsx.xlsx")

    df_imported = pd.read_csv(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
    )
    df_part = df_imported.iloc[:10, :]

    csv_path = Path(__file__).parent / "tested.csv"
    csv_path2 = Path(__file__).parent / "tested2.csv"
    json_path = Path(__file__).parent / "tested.json"
    parquet_path = Path(__file__).parent / "tested.parquet"
    # hdf_path = Path(__file__).parent / "tested.h5"

    df_imported.to_csv(csv_path.as_posix())
    df_part.to_csv(csv_path2.as_posix())
    df_imported.to_parquet(parquet_path.as_posix(), compression="gzip")
    # df_imported.to_hdf(hdf_path.as_posix(), key="df")

    loaded_data = requests.get(
        "https://www.ncdc.noaa.gov/cag/global/time-series/globe/land_ocean/ytd/12/1880-2016.json"
    ).content

    with open(json_path, "w") as outfile:
        json.dump(json.loads(loaded_data), outfile)

    try:
        df_csv = mdpd.load_data(csv_path)
        df_csv_joined = mdpd.load_data([csv_path, csv_path2])
        df_json = mdpd.load_data(
            json_path, request_datatype_suffix=".json", predicted_table="data", data_orientation="index"
        )
        df_parquet = mdpd.load_data(parquet_path)
        # df_hdf = mdpd.load_data(hdf_path)

    except Exception:
        pass

    finally:
        for i in [csv_path, csv_path2, json_path, parquet_path]:  # , hdf_path
            i.unlink()

    assert all(
        [
            xls.ndim,
            xlsx.ndim,
            df_csv.ndim,
            df_json.ndim,
            df_parquet.ndim,
            # df_hdf.ndim,
            len(df_csv_joined) == len(df_csv) + 10,
        ]
    )
