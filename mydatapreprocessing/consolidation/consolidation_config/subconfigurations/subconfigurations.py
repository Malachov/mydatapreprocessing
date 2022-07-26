"""Content for consolidation subconfigs subpackage."""
from __future__ import annotations
from typing import Any

from typing_extensions import Literal
import numpy as np
import pandas as pd

from mypythontools.config import Config, MyProperty

from ....types import PandasIndex, Numeric


class Datetime(Config):
    """Define whether to set datetime index."""

    @MyProperty
    def datetime_column(self) -> PandasIndex | None:
        """Name or index of datetime column that will be set as index and converted to datetime.

        Type:
            PandasIndex | None

        Default:
            None

        If None, then no column will be set as index.
        """
        return None

    @MyProperty
    def on_set_datetime_error(self) -> Literal["ignore", "raise"]:
        """Define what happens if converting to datetime fails.

        Type:
            Literal["ignore", "raise"]

        Default:
            "ignore"
        """
        return "ignore"


class Resample(Config):
    """Change the sampling frequency."""

    @MyProperty
    def resample(self) -> None | Literal["S", "min", "H", "M", "Y"] | str:
        """Frequency of resampled data.

        Type:
            None | Literal["S", "min", "H", "M", "Y"] | str

        Default:
            None

        If None, then data are not resampled.
        """
        return None

    @MyProperty
    def resample_function(self) -> Literal["sum", "mean"]:
        """Define whether resampled values are sum of values or it's average.

        Type:
            Literal["sum", "mean"]

        Default:
            "sum"
        """
        return "sum"


class RemoveMissingValues(Config):
    """Remove NaN values."""

    @MyProperty
    def remove_all_column_with_nans_threshold(self) -> None | Numeric:
        """Delete all the column based on amount of NaN values.

        Type:
            None | Numeric

        Default:
            0.85

        From 0 to 1. Require that many non-nan numeric values to not be deleted. E.G if value is 0.9 with
        column with 10 values, 90% must be numeric that implies max 1 np.nan can be presented, otherwise
        column will be deleted.
        """
        return 0.85

    @MyProperty
    def remove_nans_type(self) -> None | Literal["interpolate", "mean", "neighbor", "remove"] | Any:
        """Remove rows where NaN or replace rest nan values.

        Type:
            None | Literal["interpolate", "mean", "neighbor", "remove"] | Any

        Default:
            "interpolate"

        If None, NaN are not removed. If you want to replace with concrete value, use float or int type.
        """
        return "interpolate"


class StringsToNumeric(Config):
    """Remove or replace string values with numeric."""

    @MyProperty
    def embedding(self) -> None | Literal["label", "one-hot"]:
        """Implement categorical encoding.

        Type:
            None | Literal["label", "one-hot"]

        Default:
            "label"

        Create numbers from strings. 'label' give each category (unique string) concrete number. Result will
        have the same number of columns. 'one-hot' create for every category new column. Only columns, where
        are strings repeating (unique_threshold) will be used.
        """
        return "label"

    @MyProperty
    def cast_str_to_numeric(self) -> bool:
        """Try to convert strings to numeric.

        Type:
            bool

        Default:
            True

        Errors will be ignored, so if column cannot be converted to numeric, it's untouched.
        """
        return True

    @MyProperty
    def only_numeric(self) -> bool:
        """Remove all non numeric values.

        Type:
            bool

        Default:
            True

        If True, all the non numeric columns will be dropped. 'cast_str_to_numeric' and 'embedding' are used
        before dropping columns.
        """
        return True

    @MyProperty
    def unique_threshold(self) -> Numeric:
        """Remove string columns, that have to many categories.

        Type:
            Numeric

        Default:
            0.6

        E.g 0.9 define, that if column contain more that 90% of NOT unique values it's deleted. Min is 0, max
        is 1. It will remove ids, hashes etc.
        """
        return 0.6
