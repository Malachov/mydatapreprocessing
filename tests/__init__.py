""" Test module. Auto pytest that can be started in IDE or with::

    python -m pytest

in terminal in tests folder.
"""

from . import (
    test_create_model_inputs,
    test_database,
    test_feature_engineering,
    test_integration,
    test_load_data,
    test_preprocessing,
    test_visual,
)

__all__ = [
    "test_create_model_inputs",
    "test_database",
    "test_feature_engineering",
    "test_integration",
    "test_load_data",
    "test_preprocessing",
    "test_visual",
]
