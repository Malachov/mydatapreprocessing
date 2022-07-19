"""Preprocessing config and subconfig classes.

Attributes:
    default_preprocessing_config (mydatapreprocessing.preprocessing.preprocessing_config.PreprocessingConfig):
        Default config, that you can use. You can use intellisense with help tooltip to see what you can setup
        there or you can use `update` method for bulk configuration.
"""

from mydatapreprocessing.preprocessing.preprocessing_config.preprocessing_config_internal import (
    PreprocessingConfig,
    default_preprocessing_config,
)

from mydatapreprocessing.preprocessing.preprocessing_config import subconfigurations

__all__ = ["PreprocessingConfig", "default_preprocessing_config", "subconfigurations"]
