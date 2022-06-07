"""Runs before every pytest test. Used automatically (at least at VS Code)."""

from mypythontools_cicd import tests

tests.setup_tests()
