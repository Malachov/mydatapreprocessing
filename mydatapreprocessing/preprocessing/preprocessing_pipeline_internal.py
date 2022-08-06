"""Module for preprocessing subpackage."""

from __future__ import annotations
from typing import TYPE_CHECKING, Generic, cast, Union

import numpy as np
import pandas as pd

from ..types import DataFrameOrArrayGeneric, Numeric
from . import preprocessing_functions as pf
from .preprocessing_config import default_preprocessing_config, PreprocessingConfig
from .inverse_preprocessing_config import InversePreprocessingConfig


if TYPE_CHECKING:
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    ScalerType = Union[MinMaxScaler, RobustScaler, StandardScaler]

# Lazy load

# import scipy.signal
# import scipy.stats
# from sklearn import preprocessing


def preprocess_data(
    data: DataFrameOrArrayGeneric, config: PreprocessingConfig = default_preprocessing_config
) -> tuple[DataFrameOrArrayGeneric, InversePreprocessingConfig]:
    """Main preprocessing function, that call other functions based on configuration.

    Mostly for preparing data to be optimal as input into machine learning models.

    Args:
        data (DataFrameOrArrayGeneric): Input data that we want to preprocess.
        config (PreprocessingConfig): Configure data preprocessing. It's documented in `preprocessing_config`
            subpackage. You can import and edit `default_preprocessing_config`. Intellisense and static type
            analysis should work. Defaults to `default_preprocessing_config`

    Returns:
        PreprocessedData: If input in numpy array, then also output in array, if DataFrame input, then
        DataFrame output.

    Example:
        >>> import pandas as pd
        >>> import mydatapreprocessing.preprocessing as mdpp

        >>> df = pd.DataFrame(
        ...     np.array([range(5), range(20, 25), np.random.randn(5)]).astype("float32").T,
        ... )
        >>> df.iloc[2, 0] = 500
        ...
        >>> config = mdpp.preprocessing_config.default_preprocessing_config.do.copy()
        >>> config.do.update({"remove_outliers": 1, "difference_transform": True, "standardize": "standardize"})

        Predicted column moved to index 0, but for test reason test, use different one

        >>> processed_df, inverse_preprocessing_config_df = mdpp.preprocess_data(df, config)
        >>> processed_df
                  0         1         2
        1 -0.707107 -0.707107  0.377062
        3  1.414214  1.414214  0.991879
        4 -0.707107 -0.707107 -1.368941

        Inverse preprocessing is done for just one column - the one on index 0
        You can use `first_column` in consolidation to move the column to that index.

        >>> inverse_preprocessing_config_df.difference_transform = df.iloc[0, 0]
        >>> inverse_processed_df = mdpp.preprocess_data_inverse(
        ...     processed_df.iloc[:, 0].values, inverse_preprocessing_config_df
        ... )
        >>> np.allclose(inverse_processed_df, np.array([1, 3, 4]))
        True
    """
    preprocessed = data.copy()

    if config.remove_outliers:
        preprocessed = pf.remove_outliers(preprocessed, threshold=config.remove_outliers)

    if config.smooth:
        preprocessed = pf.smooth(preprocessed, config.smooth[0], config.smooth[1])

    if config.difference_transform:
        if isinstance(preprocessed, np.ndarray):
            last_undiff_value = preprocessed[-1, 0]
        else:
            last_undiff_value = preprocessed.iloc[-1, 0]
        last_undiff_value = cast(Numeric, last_undiff_value)

        preprocessed = pf.do_difference(preprocessed)
    else:
        last_undiff_value = None

    if config.standardize:
        preprocessed, final_scaler = pf.standardize(preprocessed, used_scaler=config.standardize)
    else:
        final_scaler = None

    if config.discretization.discretize:
        preprocessed = pf.binning(
            preprocessed, config.discretization.discretize, config.discretization.binning_type
        )

    preprocessed = cast(DataFrameOrArrayGeneric, preprocessed)

    inverse_preprocessing_config = InversePreprocessingConfig()
    inverse_preprocessing_config.difference_transform = last_undiff_value
    inverse_preprocessing_config.standardize = final_scaler

    return preprocessed, inverse_preprocessing_config


def preprocess_data_inverse(
    data: pd.DataFrame | np.ndarray, config: InversePreprocessingConfig
) -> np.ndarray:
    """Undo all data preprocessing to get real data.

    Does not inverse all the columns, but only defined one. Only predicted column is also returned. Order is
    reverse than preprocessing. Output is in numpy array.

    Args:
        data (pd.DataFrame | np.ndarray): One dimension (one column) preprocessed data. Do not use ndim > 1.
        config (InversePreprocessingConfig): Data necessary for inverse transformation. It does not need to be
            configured, but it is returned from `preprocess_data`.

    Returns:
        np.ndarray: Inverse preprocessed data

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> import mydatapreprocessing.preprocessing as mdpp
        ...
        >>> df = pd.DataFrame(
        ...     np.array([range(5), range(20, 25), np.random.randn(5)]).astype("float32").T,
        ... )
        >>> preprocessed, inverse_config = mdpp.preprocess_data(df.values)
        >>> preprocessed
        array([[-1.4142135 , -1.4142135 ,  0.1004863 ],
               [-0.70710677, -0.70710677,  0.36739323],
               [ 0.        ,  0.        , -1.1725829 ],
               [ 0.70710677,  0.70710677,  1.6235067 ],
               [ 1.4142135 ,  1.4142135 , -0.9188035 ]], dtype=float32)
        >>> preprocess_data_inverse(preprocessed[:, 0], inverse_config)
        array([0., 1., 2., 3., 4.], dtype=float32)
    """
    inverse = data.values if isinstance(data, pd.DataFrame) else data

    if config.standardize:
        inverse = config.standardize.inverse_transform(inverse.reshape(1, -1)).ravel()

    if config.difference_transform is not None:
        inverse = pf.inverse_difference(inverse, config.difference_transform)

    return inverse
