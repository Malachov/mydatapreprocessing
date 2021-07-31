"""Module include two functions: database_load and database_deploy. First download data from database - it's necessary to set up
connect credentials. The database_deploy than deploy data to the database server.

It is working only for mssql server so far.

Example::

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

# Lazy imports
# import pandas as pd
# import pyodbc
# from sqlalchemy import create_engine
# import urllib


def database_load(
    query,
    server,
    database,
    driver="{SQL Server}",
    username=None,
    password=None,
    trusted_connection=None,
):
    """Load database into dataframe and create datetime index. !!! This function have to be change for every particular database !!!

    Args:
        query (str, optional): Used query.
        server (string): Name of server.
        database (str): Name of database.
        driver (str): Used driver. One can be downloaded on https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server?view=sql-server-ver15
            Defaults to '{SQL Server}'.
        username ((str, None)): Username. 'sa' for root on mssql.
        password (str): Password.
        trusted_connection (bool): If using windows authontification.

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

                WHERE      contidion = 1
                    and    condition2 = 1
                    and    DimOperationOutId = 69

                GROUP BY
                    {col}

                ORDER BY
                    {col_desc}
            '''
    """

    import pandas as pd

    connection = create_connection(
        server=server,
        database=database,
        driver=driver,
        username=username,
        password=password,
        trusted_connection=trusted_connection,
    )
    df = pd.read_sql(query, connection)

    return df


def database_write(
    df,
    server,
    database,
    table,
    port=None,
    index=False,
    driver="{SQL Server}",
    username=None,
    password=None,
    trusted_connection=None,
    schema=None,
    if_exists="append",
):
    """Deploy dataframe to SQL server.

    Args:
        df (pd.DataFrame): Dataframe passed to database.
        server (string): Name of server.
        database (str): Name of database.
        table (str): Used table.
        port ((str, int)): Used port.
        index (bool): Whether use index as a column
        driver (str): Used driver. One can be downloaded on https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server?view=sql-server-ver15
            Defaults to '{SQL Server}'.        username
        username ((str, None)): Username. 'sa' for root on mssql.
        password (str): Password.
        trusted_connection (bool): If using windows authontification.
        schema (str): Used schema. Defaults to None.
        if_exists (str): 'fail', 'replace', 'append'. Define whether append new
            data on the end, remove and replace or fail if table exists.
    """

    connection = create_connection(
        server=server,
        database=database,
        driver=driver,
        port=port,
        username=username,
        password=password,
        trusted_connection=trusted_connection,
    )

    df.to_sql(name=table, con=connection, if_exists=if_exists, index=index, schema=schema)


def create_connection(
    server, database, port=None, driver="{SQL Server}", username=None, password=None, trusted_connection=None
):
    """Create connection, that can be used in another function to connect the databse.

    Args:
        server (string): Name of server.
        database (str): Name of database.
        port ((str, int)): Used port. May work with None.
        driver (str): Used driver. One can be downloaded on https://docs.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server?view=sql-server-ver15
            Defaults to '{SQL Server}'.
        username ((str, None)): Username. 'sa' for root on mssql.
        password (str): Password.
        trusted_connection (bool): If using windows authontification."""
    from sqlalchemy import create_engine
    import urllib

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
