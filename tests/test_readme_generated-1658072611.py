"""pytest file built from d:/Github/mydatapreprocessing/README.md"""
import pytest

from phmdoctest.fixture import managenamespace


@pytest.fixture(scope="module")
def _phm_setup_teardown(managenamespace):
    # setup code line 35.
    import mydatapreprocessing as mdp
    import pandas as pd
    import numpy as np

    managenamespace(operation="update", additions=locals())
    yield
    # <teardown code here>

    managenamespace(operation="clear")


pytestmark = pytest.mark.usefixtures("_phm_setup_teardown")


def test_load_data(managenamespace):
    data = mdp.load_data.load_data(
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
    )
    # data2 = mdp.load_data.load_data([PATH_TO_FILE.csv, PATH_TO_FILE2.csv])

    # Caution- no assertions.
    managenamespace(operation="update", additions=locals())


def test_consolidation(managenamespace):
    consolidation_config = mdp.consolidation.consolidation_config.default_consolidation_config.copy()
    consolidation_config.datetime.datetime_column = "Date"
    consolidation_config.resample.resample = "M"
    consolidation_config.resample.resample_function = "mean"
    consolidation_config.dtype = "float32"

    consolidated = mdp.consolidation.consolidate_data(data, consolidation_config)
    print(consolidated.head())

    # Caution- no assertions.
    managenamespace(operation="update", additions=locals())


def test_code_101():
    import mydatapreprocessing.feature_engineering as mdpf
    import mydatapreprocessing as mdp

    data = pd.DataFrame([mdp.datasets.sin(n=30), mdp.datasets.ramp(n=30)]).T

    extended = mdpf.add_derived_columns(data, differences=True, rolling_means=10)
    print(extended.columns)
    print(f"\nit has less rows then on input {len(extended)}")

    # Caution- no assertions.


def test_preprocess_data(managenamespace):

    from mydatapreprocessing import preprocessing as mdpp

    df = pd.DataFrame(np.array([range(5), range(20, 25), np.random.randn(5)]).astype("float32").T)
    df.iloc[2, 0] = 500

    config = mdpp.preprocessing_config.default_preprocessing_config.copy()
    config.update({"remove_outliers": None, "difference_transform": True, "standardize": "standardize"})
    data_preprocessed, inverse_config = mdpp.preprocess_data(df.values, config)
    inverse_config.difference_transform = df.iloc[0, 0]
    data_preprocessed_inverse = mdpp.preprocess_data_inverse(data_preprocessed[:, 0], inverse_config)

    # Caution- no assertions.
    managenamespace(operation="update", additions=locals())
