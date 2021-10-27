"""Module include two functions: database_load and database_deploy. First download data from database - it's
necessary to set up connect credentials. The database_deploy than deploy data to the database server.

It is working only for mssql server so far.

Example:

    ::

        data = mdp.database.database_load(
            server=".",
            database="DemoData",
            query='''
                SELECT TOP (1000) [ID] ,[ProductName]
                FROM [DemoData].[dbo].[Products]
            '''
            username="sa",
            password="Ahojdatadata123",
        )
"""

from __future__ import annotations
import importlib.util
from typing import TYPE_CHECKING

import pandas as pd

import mylogging

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
    """Load database into dataframe and create datetime index. !!! This function have to be change for every
    particular database !!!

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
        pd.DataFrame: Dataframe with data from database based on input SQL query.

    Example:

        This is how the query could look like in python::

            query_example = f'''

                SELECT TOP ({data_limit})
                    {col},
                    sum([Number]) SumNumber,
                    sum([Duration]) SumDuration

                FROM [dbo].[Table] F
                    INNER JOIN dbo.DimDateTime D
                    ON F.DimDateTimeId = D.DimDateTimeId

                WHERE      condition = 1
                    and    condition2 = 1
                    and    DimOperationOutId = 69

                GROUP BY
                    {col}

                ORDER BY
                    {col_desc}
            '''
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
    schema: str = None,
    if_exists: str = "append",
) -> None:
    """Deploy dataframe to SQL server.

    Args:
        df (pd.DataFrame): Dataframe passed to database.
        server (str): Name of server.
        database (str): Name of database.
        table (str): Used table.
        index (bool, optional): Whether use index as a column Defaults to False.
        port (str | int | None, optional): Used port. May work with None. Defaults to None.
        driver (str, optional): Used driver. One can be downloaded on https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server?view=sql-server-ver15
            Defaults to '{SQL Server}'.
        username (str | None, optional): Username. 'sa' for root on mssql. Defaults to None.
        password (str | None, optional): Password. Defaults to None.
        trusted_connection (bool): If using windows authentication. You dont need username and password then. Defaults to False.
        schema (str): Used schema. Defaults to None.
        if_exists (str): 'fail', 'replace', 'append'. Define whether append new
            data on the end, remove and replace or fail if table exists. Defaults to 'append'.
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
        trusted_connection (bool, optional): If using windows authentication. Defaults to False."""

    if not importlib.util.find_spec("sqlalchemy"):
        raise ModuleNotFoundError(mylogging.return_str("Using databases. Install with `pip install wfdb`"))

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
