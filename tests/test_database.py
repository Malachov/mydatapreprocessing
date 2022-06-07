"""Tests for database package."""

import time

import pandas as pd
import docker
import docker.errors

from mypythontools_cicd import tests

tests.setup_tests()

import mydatapreprocessing as mdp

# pylint: disable=missing-function-docstring


def test_databases():
    try:
        client = docker.from_env()
    except docker.errors.DockerException as err:
        raise RuntimeError("Docker error, check if Docker is running.") from err

    try:
        container = client.containers.run(
            "mssql:latest",
            ports={1433: 1433},
            detach=True,
            auto_remove=True,
            environment={"ACCEPT_EULA": "Y"},
        )
    except docker.errors.ImageNotFound as err:
        raise docker.errors.ImageNotFound(
            "mssql docker image not made. First build Dockerfile here in tests."
        ) from err

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
            password="HelloPassword123",
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
            password="HelloPassword123",
        )

    except Exception:  # pylint: disable=try-except-raise
        raise

    finally:
        container.stop()

    assert len(data) == 10
