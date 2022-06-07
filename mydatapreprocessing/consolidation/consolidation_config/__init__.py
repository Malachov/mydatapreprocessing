"""Consolidation config and subconfig classes."""
from mydatapreprocessing.consolidation.consolidation_config.consolidation_config_internal import (
    ConsolidationConfig,
    default_consolidation_config,
)

from mydatapreprocessing.consolidation.consolidation_config import subconfigurations

__all__ = ["ConsolidationConfig", "default_consolidation_config", "subconfigurations"]
