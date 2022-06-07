"""Place for storing the types used across the library. Type for data input format for example.

Attributes:
    DataFrameOrArrayGeneric (typing.TypeVar): Many functions works for numpy arrays as well as for pandas
        DataFrame. The same type as on input is returned on output usually.
    Numeric (typing.TypeAlias): Define basic numeric type usually used in computation. Union of float, int and
        `numpy.number`.
    PandasIndex (typing.TypeAlias): Index that can be used in this library in function parameter. It can be
        str, but also int index. It's usually narrowed to str | pd.Index afterwards so it can be used to
        access column with the same syntax as with columns name.
    DataFormat (typing.TypeAlias)
"""
from __future__ import annotations
from mydatapreprocessing.types.types_internal import DataFrameOrArrayGeneric, PandasIndex, DataFormat, Numeric

__all__ = ["DataFrameOrArrayGeneric", "PandasIndex", "DataFormat", "Numeric"]
