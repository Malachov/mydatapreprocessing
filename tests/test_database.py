import pandas as pd
import docker
import time

import mylogging

import numpy as np

import mydatapreprocessing as mdp

mylogging.config.COLOR = 0
np.random.seed(2)


def test_databases():
    client = docker.from_env()
    container = client.containers.run("mssql:latest", ports={1433: 1433}, detach=True)

    time.sleep(50)

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
        mylogging.traceback()

    finally:
        container.stop()

    assert len(data) == 10