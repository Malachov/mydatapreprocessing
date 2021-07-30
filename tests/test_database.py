""" Test module. Auto pytest that can be started in IDE or with::

    python -m pytest

in terminal in tests folder.
"""
#%%

import pandas as pd
import docker
import time

import mylogging

import sys
from pathlib import Path
import inspect
import os
import numpy as np

# Find paths and add to sys.path to be able to import local modules
test_dir_path = Path(os.path.abspath(inspect.getframeinfo(inspect.currentframe()).filename)).parent
root_path = test_dir_path.parent

if test_dir_path.as_posix() not in sys.path:
    sys.path.insert(0, test_dir_path.as_posix())

if root_path.as_posix() not in sys.path:
    sys.path.insert(0, root_path.as_posix())

mylogging.config.COLOR = 0
np.random.seed(2)

import mydatapreprocessing as mdp


def test_databases():
    client = docker.from_env()
    container = client.containers.run("mssql:latest", ports={1433: 1433}, detach=True)

    time.sleep(20)

    df = pd.DataFrame([range(1, 11), ["Product " + str(i) for i in range(10)]]).T
    df.columns = ["ID", "ProductName"]

    try:
        mdp.database.database_write(
            df,
            server=".",
            database="DemoData",
            table="Products",
            username="sa",
            password="Ahojdatadata123",
            if_exists="replace",
        )

        data = mdp.database.database_load(
            server=".",
            database="DemoData",
            query="""
                SELECT TOP (1000) [ID] ,[ProductName]
                FROM [DemoData].[dbo].[Products]
            """,
            username="sa",
            password="Ahojdatadata123",
        )

    except Exception:
        pass

    finally:
        container.stop()

    assert len(data) == 10
