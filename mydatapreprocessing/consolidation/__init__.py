"""Module for data consolidation.

Consolidation means that output is somehow standardized and you know that it will be working in your
algorithms even when data are not known beforehand. It includes for example shape
verification, string embedding, setting datetime index, resampling or NaN cleaning.

You can consolidate data with `consolidate_data` and prepare it for example for machine learning models.

There are many small functions that you can use separately in `consolidation_functions`, but there is main
pipeline function `consolidate_data` that calls all the functions based on config for you.

Functions usually use DataFrame as consolidation is first phase of data preparation and columns names are
still important here.

There is an 'inplace' parameter on many places. This means, that it change your original data, but syntax is
bit different as it will return anyway, so use for example ``df = consolidation_function(df, inplace=True)``
"""
from mydatapreprocessing.consolidation.consolidation_pipeline_internal import (
    consolidate_data,
)
from . import consolidation_functions
from . import consolidation_config
from .consolidation_config import default_consolidation_config

__all__ = [
    "consolidate_data",
    "consolidation_config",
    "default_consolidation_config",
    "consolidation_functions",
]
