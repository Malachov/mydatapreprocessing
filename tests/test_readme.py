import mypythontools

# Find paths and add to sys.path to be able to import local modules
mypythontools.tests.setup_tests()


def test_readme():
    mypythontools.tests.test_readme()
