"""Process database. Read or write.

It is working only for mssql server so far.
"""
from mydatapreprocessing.database.database_internal import database_load, database_write

__all__ = ["database_load", "database_write"]
