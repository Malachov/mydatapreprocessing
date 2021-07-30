""" Test module. Auto pytest that can be started in IDE or with::

    python -m pytest

in terminal in tests folder.
"""

from . import test_database, test_inputs, test_preprocessing, test_visual

__all__ = ["test_database", "test_inputs", "test_preprocessing", "test_visual"]
