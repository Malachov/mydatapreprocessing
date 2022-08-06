"""Module for consolidation_pipeline subpackage."""

from __future__ import annotations

import pandas as pd

from . import consolidation_functions
from .consolidation_functions import (
    categorical_embedding,
    check_shape_and_transform,
    move_on_first_column,
    remove_nans,
    resample,
    set_datetime_index,
)
from ..types import DataFrameOrArrayGeneric
from ..helpers import get_copy_or_view
from .consolidation_config import default_consolidation_config, ConsolidationConfig


def consolidate_data(
    data: DataFrameOrArrayGeneric, config: ConsolidationConfig = default_consolidation_config
) -> pd.DataFrame:
    """Transform input data in various formats and shapes into data in defined shape.

    This can be beneficial for example in machine learning. If you have data in other format than DataFrame,
    use `load_data` first.

    Note:
        This function returns only numeric data with default config. All string columns will be removed (use
        embedding if you need)

    Args:
        data (DataFrameOrArrayGeneric): Input data.
        config (ConsolidationConfig): Configure data consolidation. It's documented in `consolidation_config`
            subpackage. You can import and edit `default_consolidation_config`. Intellisense and static type
            analysis should work. Defaults to `default_consolidation_config`

    Raises:
        KeyError, TypeError: May happen if wrong params. E.g. if predicted column name not found in DataFrame.

    Returns:
        pd.DataFrame: Data in standardized form.

    Example:
        >>> import mydatapreprocessing.consolidation as mdpc
        >>> import pandas as pd
        >>> import numpy as np
        >>> np.random.seed(2)
        ...
        >>> df = pd.DataFrame(
        ...    np.array([range(4), range(20, 24), np.random.randn(4)]).T,
        ...    columns=["Column", "First", "Random"],
        ... )
        ...
        >>> df.iloc[2, 0] = np.nan
        >>> df
           Column  First    Random
        0     0.0   20.0 -0.416758
        1     1.0   21.0 -0.056267
        2     NaN   22.0 -2.136196
        3     3.0   23.0  1.640271

        >>> config = mdpc.consolidation_config.default_consolidation_config.do.copy()
        >>> config.first_column = "First"
        >>> config.datetime.datetime_column = None
        >>> config.remove_missing_values.remove_all_column_with_nans_threshold = 0.6
        ...
        >>> consolidate_data(df, config)
           First  Column    Random
        0   20.0     0.0 -0.416758
        1   21.0     1.0 -0.056267
        2   22.0     2.0 -2.136196
        3   23.0     3.0  1.640271
    """
    data = get_copy_or_view(data, config.inplace)

    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data

    if config.check_shape_and_transform:
        df = check_shape_and_transform(df, inplace=True)

    # Trim the data on defined length
    df = df.iloc[-config.data_length :, :]

    if config.first_column:
        df = move_on_first_column(df, config.first_column)

    if config.datetime.datetime_column:
        df = set_datetime_index(
            df, config.datetime.datetime_column, on_error=config.datetime.on_set_datetime_error, inplace=True
        )

    if config.strings_to_numeric.cast_str_to_numeric:
        # Convert strings numbers (e.g. '6') to numbers
        df = consolidation_functions.cast_str_to_numeric(df)

    if config.strings_to_numeric.embedding:
        df = categorical_embedding(
            df,
            embedding=config.strings_to_numeric.embedding,
            unique_threshold=config.strings_to_numeric.unique_threshold,
            inplace=True,
        )

    if config.strings_to_numeric.only_numeric:
        df = df.select_dtypes(include="number")

    if (
        config.remove_missing_values.remove_all_column_with_nans_threshold
        or config.remove_missing_values.remove_nans_type
    ):
        df = remove_nans(
            df,
            config.remove_missing_values.remove_all_column_with_nans_threshold,
            config.remove_missing_values.remove_nans_type,
        )

    if config.resample.resample:
        df = resample(df, config.resample.resample, config.resample.resample_function)

    # Define concrete dtypes in number columns
    if config.dtype:
        df = df.astype(config.dtype, copy=False)  # type: ignore

    return df
