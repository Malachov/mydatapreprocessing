"""Module for database subpackage."""

from __future__ import annotations
from typing import TYPE_CHECKING

from typing_extensions import Literal
import pandas as pd

from mypythontools.system import check_library_is_available

# Lazy imports
if TYPE_CHECKING:
    import sqlalchemy
    import urllib


def database_load(
    query: str,
    server: str,
    database: str,
    port: str | int | None = None,
    driver: str = "{SQL Server}",
    username: str | None = None,
    password: str | None = None,
    trusted_connection: bool = False,
) -> pd.DataFrame:
    """Load database into DataFrame.

    Args:
        query (str): Used query.
        server (str): Name of server.
        database (str): Name of database.
        port (str | int | None, optional): Used port. May work with None. Defaults to None.
        driver (str, optional): Used driver. One can be downloaded on https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server?view=sql-server-ver15
            Defaults to '{SQL Server}'.
        username (str | None, optional): Username. 'sa' for root on mssql.
        password (str | None, optional): Password.
        trusted_connection (bool, optional): If using windows authentication.

    Returns:
        pd.DataFrame: DataFrame with data from database based on input SQL query.

    Example: ::

        data = mdp.database.database_load(
            server=".",
            database="DemoData",
            query='''
                SELECT TOP (1000) [ID] ,[ProductName]
                FROM [DemoData].[dbo].[Products]
            ''',
            username="sa",
            password="HelloPassword123",
        )
    """
    connection = _create_connection(
        server=server,
        database=database,
        port=port,
        driver=driver,
        username=username,
        password=password,
        trusted_connection=trusted_connection,
    )
    df = pd.read_sql(query, connection)

    return df


def database_write(
    df: pd.DataFrame,
    server: str,
    database: str,
    table: str,
    index: bool = False,
    port: str | int | None = None,
    driver: str = "{SQL Server}",
    username: str | None = None,
    password: str | None = None,
    trusted_connection: bool = False,
    schema: None | str = None,
    if_exists: Literal["fail", "replace", "append"] = "append",
) -> None:
    """Deploy DataFrame to SQL server.

    Args:
        df (pd.DataFrame): DataFrame passed to database.
        server (str): Name of server.
        database (str): Name of database.
        table (str): Used table.
        index (bool, optional): Whether use index as a column Defaults to False.
        port (str | int | None, optional): Used port. May work with None. Defaults to None.
        driver (str, optional): Used driver. One can be downloaded on https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server?view=sql-server-ver15
            Defaults to '{SQL Server}'.
        username (str | None, optional): Username. 'sa' for root on mssql. Defaults to None.
        password (str | None, optional): Password. Defaults to None.
        trusted_connection (bool, optional): If using windows authentication. You don't need username and
            password then. Defaults to False.
        schema (None | str, optional): Used schema. Defaults to None.
        if_exists (Literal['fail', 'replace', 'append'], optional): Define whether append new data on the end,
            remove and replace or fail if table exists. Defaults to 'append'.

    Example: ::

            import pandas as pd

            df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])

            mdp.database.database_write(
                df,
                server=".",
                database="DemoData",
                table="Products",
                username="sa",
                password="HelloPassword123",
                if_exists="replace",
            )
    """
    connection = _create_connection(
        server=server,
        database=database,
        driver=driver,
        port=port,
        username=username,
        password=password,
        trusted_connection=trusted_connection,
    )

    df.to_sql(name=table, con=connection, if_exists=if_exists, index=index, schema=schema)


def _create_connection(
    server: str,
    database: str,
    port: str | int | None = None,
    driver: str = "{SQL Server}",
    username: str | None = None,
    password: str | None = None,
    trusted_connection: bool = False,
) -> "sqlalchemy.engine.base.Engine":
    """Create connection, that can be used in another function to connect the database.

    Args:
        server (str): Name of server.
        database (str): Name of database.
        port (str | int | None, optional): Used port. May work with None. Defaults to None.
        driver (str, optional): Used driver. One can be downloaded on https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server?view=sql-server-ver15
            Defaults to '{SQL Server}'.
        username (str | None, optional): Username. 'sa' for root on mssql. Defaults to None.
        password (str | None, optional): Password. Defaults to None.
        trusted_connection (bool, optional): If using windows authentication. Defaults to False.
    """
    check_library_is_available("sqlalchemy")

    from sqlalchemy import create_engine
    import urllib.parse

    connection_string = f"DRIVER={driver};SERVER={server};DATABASE={database};"
    if port:
        connection_string = connection_string + f"PORT={str(port)};"
    if trusted_connection:
        connection_string = connection_string + "Trusted_Connection=yes;"
    if username:
        connection_string = connection_string + f"UID={username};"
    if password:
        connection_string = connection_string + f"PWD={password};"

    params = urllib.parse.quote_plus(connection_string)

    conn_str = f"mssql+pyodbc:///?odbc_connect={params}"

    return create_engine(conn_str)
