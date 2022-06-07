"""This module helps you to load data from path as well as from web url in various formats.

Supported path formats are:

- csv
- xlsx and xls
- json
- parquet
- h5

You can insert more files (urls) at once and your data will be automatically concatenated.

Main function is `load_data` where you can find working examples.

There is also function `get_file_paths` which open an dialog window in your operation system and let you
choose your files in convenient way. This tuple output you can then insert into `load_data`.
"""

from mydatapreprocessing.load_data import load_data_functions
from mydatapreprocessing.load_data.load_data_internal import load_data

__all__ = ["load_data", "load_data_functions"]
