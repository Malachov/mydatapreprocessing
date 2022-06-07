"""Tests for helpers package."""

import pandas as pd

from mypythontools_cicd import tests

tests.setup_tests()

import mydatapreprocessing.helpers as mdph

# pylint: disable=missing-function-docstring


def test_get_copy_or_view():
    a = pd.DataFrame([1, 2, 3])
    b = mdph.get_copy_or_view(a, inplace=True)
    assert id(a) == id(b), "It should be identical"

    b = mdph.get_copy_or_view(a, inplace=False)
    assert not (id(a) == id(b)), "It should not be identical"
