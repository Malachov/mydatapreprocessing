"""Runs before every pytest test. Used automatically (at least at VS Code)."""
import mypythontools_cicd.tests as tests

import pytest


@pytest.fixture(autouse=True)
def setup_tests():
    tests.setup_tests()
