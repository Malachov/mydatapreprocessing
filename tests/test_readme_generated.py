"""pytest file built from C:/Users/Malac/ownCloud/Github/mydatapreprocessing/README.md"""
import pytest

from phmdoctest.fixture import managenamespace


@pytest.fixture(scope="module")
def _phm_setup_teardown(managenamespace):
    # setup code line 34.
    import mydatapreprocessing as mdp

    managenamespace(operation="update", additions=locals())
    yield
    # <teardown code here>

    managenamespace(operation="clear")


pytestmark = pytest.mark.usefixtures("_phm_setup_teardown")


def test_load_data(managenamespace):
    data = mdp.load_data.load_data(
        "https://www.ncdc.noaa.gov/cag/global/time-series/globe/land_ocean/ytd/12/1880-2016.json",
        request_datatype_suffix=".json",
        data_orientation="index",
        predicted_table="data",
    )
    # data2 = mdp.load_data.load_data([PATH_TO_FILE.csv, PATH_TO_FILE2.csv])

    # Caution- no assertions.
    managenamespace(operation="update", additions=locals())


def test_consolidation(managenamespace):
    data_consolidated = mdp.preprocessing.data_consolidation(
        data, predicted_column=0, remove_nans_threshold=0.9, remove_nans_or_replace="interpolate"
    )

    # Caution- no assertions.
    managenamespace(operation="update", additions=locals())


def test_feature_engineering(managenamespace):
    data_extended = mdp.feature_engineering.add_derived_columns(data_consolidated, differences=True, rolling_means=32)

    # Caution- no assertions.
    managenamespace(operation="update", additions=locals())


def test_preprocess_data(managenamespace):
    data_preprocessed, _, _ = mdp.preprocessing.preprocess_data(
        data_extended,
        remove_outliers=True,
        smoothit=False,
        correlation_threshold=False,
        data_transform=False,
        standardizeit="standardize",
    )

    # Caution- no assertions.
    managenamespace(operation="update", additions=locals())


def test_create_inputs(managenamespace):
    seqs, Y, x_input, test_inputs = mdp.create_model_inputs.make_sequences(
        data_extended.values, predicts=7, repeatit=3, n_steps_in=6, n_steps_out=1, constant=1
    )

    # Caution- no assertions.
    managenamespace(operation="update", additions=locals())
