"""Tests for database package."""

import time
from pathlib import Path
import platform

import pandas as pd
import docker
import docker.errors

import mylogging
from mypythontools_cicd import tests

tests.setup_tests()

import mydatapreprocessing as mdp

# pylint: disable=missing-function-docstring


def run_sql_container(client: docker.DockerClient, tag):
    return client.containers.run(
        tag,
        ports={1433: 1433},
        detach=True,
        auto_remove=True,
    )


def test_databases():
    # Unfortunately not on linux (wsl) issues with pyodbc - solved.
    # Then lazy to install driver. Todo one day
    if platform.system() == "Windows":
        tag = "mdp-mssql:latest"
        try:
            client = docker.from_env()
        except docker.errors.DockerException as err:
            raise RuntimeError("Docker error, check if Docker is running.") from err

        try:
            container = run_sql_container(client, tag)

        except docker.errors.ImageNotFound:
            mylogging.error(
                "docker.errors.ImageNotFound: mssql docker image not made. First build Dockerfile here in tests. "
                "Trying to build it.\n\n"
            )
            client.images.build(path=Path("tests/docker/mssql").as_posix(), tag=tag)

            try:
                container = run_sql_container(client, tag)
            except docker.errors.ImageNotFound as err:
                raise docker.errors.ImageNotFound(
                    "docker.errors.ImageNotFound: mssql docker image not made. First build Dockerfile here in tests. "
                    "Trying to build it."
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


if __name__ == "__main__":
    # test_databases()
    pass
