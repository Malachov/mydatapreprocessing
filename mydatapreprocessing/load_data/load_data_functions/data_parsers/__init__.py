"""If there are some formats, that need extra logic, it's put aside here."""

from mydatapreprocessing.load_data.load_data_functions.data_parsers.data_parsers_internal import (
    csv_load,
    json_load,
    load_dict,
)

__all__ = ["csv_load", "json_load", "load_dict"]
