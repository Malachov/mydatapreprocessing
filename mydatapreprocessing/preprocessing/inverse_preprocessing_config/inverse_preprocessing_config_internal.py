"""Module with config  for consolidation pipeline."""
from __future__ import annotations
from typing import TYPE_CHECKING, Union, TypeAlias
from mypythontools.config import Config, MyProperty

from ...types import Numeric

ScalerTypeVar = None

if TYPE_CHECKING:
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    ScalerTypeVar = Union[MinMaxScaler, RobustScaler, StandardScaler]

ScalerType: TypeAlias = ScalerTypeVar


class InversePreprocessingConfig(Config):
    """Data necessary for inverse preprocessing.

    It's not created manually, but returned from 'preprocess_data' function. Still is sometimes necessary to
    edit the values.
    """

    @MyProperty
    @staticmethod
    def standardize() -> None | ScalerType:
        """Define whether use inverse standardization and what type.

        Type:
            None | "ScalerType"

        Default:
            None

        Scaler is necessary for inverse standardization.
        """
        return None

    @MyProperty
    @staticmethod
    def difference_transform() -> None | Numeric:
        """Define whether use inverse difference transform and first starting value.

        Type:
            None | Numeric

        Default:
            None

        By default, last value is used.
        """
        return None
