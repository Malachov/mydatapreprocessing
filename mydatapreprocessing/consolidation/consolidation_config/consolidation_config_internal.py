"""Module with config  for consolidation pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd

from mypythontools.config import Config, MyProperty

from ...types import PandasIndex
from . import subconfigurations


class ConsolidationConfig(Config):
    """Config class for `consolidate_data` pipeline.

    There is `default_consolidation_config` object already created. You can import it, edit and use. Static
    type check and intellisense should work.
    """

    def __init__(self) -> None:
        """Create subconfigs."""
        self.datetime: subconfigurations.Datetime = subconfigurations.Datetime()
        """Set datetime index and convert it to datetime type."""
        self.resample: subconfigurations.Resample = subconfigurations.Resample()
        """Change sampling frequency on defined frequency if there is a datetime column. You can use sum or
        average."""
        self.remove_missing_values: subconfigurations.RemoveMissingValues = (
            subconfigurations.RemoveMissingValues()
        )
        """Define whether and how to remove NotANumber values."""
        self.strings_to_numeric: subconfigurations.StringsToNumeric = subconfigurations.StringsToNumeric()
        """Remove or replace string values with numbers."""

    @MyProperty
    @staticmethod
    def inplace() -> bool:
        """Define whether work on inserted data itself, or on a copy.

        Type:
            bool

        Default:
            False

        Copy is created just once, then internally all the consolidating functions are used inplace. Syntax is
        a bit different than in for example Pandas. Use assigning to variable e.g. `df = consolidate_data(df)`
        even with inplace. If True your inserted data will be changed.
        """
        return False

    @MyProperty
    @staticmethod
    def check_shape_and_transform() -> bool:
        """Check whether correct shape is used and eventually transpose.

        Type:
            bool

        Default:
            True

        Usually there is much more rows than columns in table. If not, it can mean that dimensions are swapped
        from data load. This will check this, transform if necessary and log it.
        """
        return True

    @MyProperty
    @staticmethod
    def first_column() -> None | PandasIndex:
        """Move defined column on index 0.

        Type:
            None | PandasIndex

        Default:
            None
        """
        return None

    @MyProperty
    @staticmethod
    def data_length() -> int:
        """Limit the data length after resampling.

        Type:
            int

        Default:
            0

        If 0, then all the data is used.
        """
        return 0

    @MyProperty
    @staticmethod
    def dtype() -> str | np.dtype | pd.Series | list[str | np.dtype]:
        """Set output dtype.

        Type:
            str | np.dtype | pd.Series | list[str | np.dtype]

        Default:
            "float32"

        For possible inputs check pandas function `astype`.
        """
        return "float32"


default_consolidation_config = ConsolidationConfig()
