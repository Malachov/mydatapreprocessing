"""Miscellaneous functions that do not fit into other modules.

You can find here for example functions for
train / test split, function for rolling windows, function that clean the DataFrame for print as table or
function that will add gaps to time series data where are no data so two remote points are not joined in plot.
"""
from mydatapreprocessing.misc.misc_internal import (
    add_none_to_gaps,
    edit_table_to_printable,
    rolling_windows,
    split,
)

__all__ = ["add_none_to_gaps", "edit_table_to_printable", "rolling_windows", "split"]
