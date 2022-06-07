"""Module with config  for consolidation pipeline."""
from __future__ import annotations
from typing import TYPE_CHECKING, Union

from typing_extensions import Literal

from mypythontools.config import Config, MyProperty

from .subconfigurations import Discretization
from ...types import Numeric

if TYPE_CHECKING:
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    ScalerType = Union[MinMaxScaler, RobustScaler, StandardScaler]


class PreprocessingConfig(Config):
    """Config class for `preprocess_data` pipeline.

    There is `default_preprocessing_config` object already created. You can import it, edit and use. Static
    type check and intellisense should work.
    """

    def __init__(self) -> None:
        """Create subconfigs."""
        self.discretization: Discretization = Discretization()

    @MyProperty
    @staticmethod
    def remove_outliers() -> None | Numeric:
        """Remove unusual values far from average.

        Type:
            None | Numeric

        Default:
            None
        """
        return None

    @MyProperty
    @staticmethod
    def smooth() -> None | tuple[int, int]:
        """Smooth the data with Savitzky-Golay filter.

        Type:
            None | tuple[int, int]

        Default:
            None

        Setup with tuple (window, polynomial_order) as in `smooth` function e.g (11, 2).
        """
        return None

    @MyProperty
    @staticmethod
    def difference_transform() -> bool:
        """Transform the data.

        Type:
            bool

        Default:
            False

        'difference' transform data into differences between neighbor values.
        """
        return False

    @MyProperty
    @staticmethod
    def standardize() -> None | Literal["standardize", "-11", "01", "robust"]:
        """Standardize the data.

        Type:
            None | Literal["standardize", "-11", "01", "robust"]

        Default:
            'standardize'

        '01' and '-11' means scope from to for normalization. 'robust' use RobustScaler and 'standard' use
        StandardScaler - mean is 0 and std is 1. If no standardization, use None.
        """
        return "standardize"


default_preprocessing_config = PreprocessingConfig()
"""Default config you can use."""
