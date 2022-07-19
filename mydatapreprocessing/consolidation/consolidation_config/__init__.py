"""Consolidation config and subconfig classes.

Attributes:
    default_consolidation_config (mydatapreprocessing.consolidation.consolidation_config.ConsolidationConfig):
        Default config, that you can use. You can use intellisense with help tooltip to see what you can setup
        there or you can use `update` method for bulk configuration.
"""

from mydatapreprocessing.consolidation.consolidation_config.consolidation_config_internal import (
    ConsolidationConfig,
    default_consolidation_config,
)

from mydatapreprocessing.consolidation.consolidation_config import subconfigurations

__all__ = ["ConsolidationConfig", "default_consolidation_config", "subconfigurations"]
