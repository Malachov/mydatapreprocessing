"""Content for preprocessing subconfigs subpackage."""
from __future__ import annotations

from typing_extensions import Literal


from mypythontools.config import Config, MyProperty


class Discretization(Config):
    """Define whether to discretize values."""

    @MyProperty
    def discretize(self) -> None | int:
        """Define whether discretize values into defined number of bins (their average).

        Type:
            None | int

        Default:
            None
        """
        return None

    @MyProperty
    def binning_type(self) -> Literal["cut", "qcut"]:
        """Define how the bins will be defined.

        Type:
            Literal['cut', 'qcut']

        Default:
            'cut'

        'cut' for equal size of bins intervals (different number of members in bins) or 'qcut' for equal
        number of members in bins and various size of bins. It uses pandas 'cut' or 'qcut' function
        """
        return "cut"
