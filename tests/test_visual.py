#%%
""" Visual test on various components. Mostly for data preparation and input creating functions.
Just run and check results to know what functions do and how.
"""
import numpy as np

from mypythontools_cicd import tests

tests.setup_tests()


import mydatapreprocessing.preprocessing as mdpp
import mydatapreprocessing as mdp

# pylint: disable=line-too-long


def test_visual(print_preprocessing=1, print_postprocessing=1):
    """Call function and particular parts of used functions with results will be printed."""
    np.set_printoptions(suppress=True, precision=1)

    # Data must have 2 dimensions. If you have only one column, reshape(-1, 1)!!!
    data = np.array([[1, 3, 5, 2, 3, 4, 5, 66, 3, 2, 4, 5, 6, 0, 0, 0, 0, 7, 3, 4, 55, 3, 2]]).T

    data_multi_col = np.array(
        [
            [1, 22, 3, 3, 5, 8, 3, 3, 5, 8],
            [5, 6, 7, 6, 7, 8, 3, 9, 5, 8],
            [8, 9, 10, 6, 8, 8, 3, 3, 7, 8],
        ]
    ).T

    normalized, scaler = mdpp.preprocessing_functions.standardize(data)
    normalized_multi, _ = mdpp.preprocessing_functions.standardize(data_multi_col)

    if print_preprocessing:

        print(
            f"""

                ##########################
                ### preprocessing ###
                ##########################

            ##################
            ### One column ###
            ################## \n

        ### Remove outliers ### \n

        With outliers: \n {data} \n\nWith no outliers: \n{mdpp.preprocessing_functions.remove_the_outliers(data, threshold = 3)} \n

        ### Difference transform ### \n
        Original data: \n {data} \n\nDifferenced data: \n{mdpp.preprocessing_functions.do_difference(data[:, 0])} \n\n
        Backward difference: \n{mdpp.preprocessing_functions.do_difference(data[:, 0])}\n

        ### Standardize ### \n
        Original: \n {data} \n\nStandardized: \n{normalized} \n

        Inverse standardization: \n{scaler.inverse_transform(normalized)} \n

        ### Split ### \n
        Original: \n {data} \n\nsplitted train: \n{mdp.misc.split(data)[0]} \n \n\nsplitted test: \n{mdp.misc.split(data)[1]} \n

            ####################
            ### More columns ###
            ####################\n

        ### Remove outliers ### \n
        With outliers: \n {data_multi_col} \n\nWith no outliers: \n{mdpp.preprocessing_functions.remove_the_outliers(data_multi_col, threshold = 1)} \n

        ### Standardize ### \n
        Original: \n {data_multi_col} \n\nStandardized: \n{normalized_multi} \n

        ### Split ### \n
        Original: \n {data_multi_col} \n\nsplitted train: \n{mdp.misc.split(data_multi_col, predicts=2)[0]} \n \n\nsplitted test: \n{mdp.misc.split(data_multi_col, predicts=2)[1]} \n
        """
        )

    if print_postprocessing:
        print(
            f"""

                ###########################
                ### Data_postprocessing ###
                ###########################\n
        ### Fitted power transform ### \n
        Original: \n {data}, original std = {data.std()}, original mean = {data.mean()} \n\ntransformed: \n{mdpp.preprocessing_functions.fitted_power_transform(data, 10, 10)} \n\ntransformed std = {mdpp.preprocessing_functions.fitted_power_transform(data, 10, 10).std()},
        transformed mean = {mdpp.preprocessing_functions.fitted_power_transform(data, 10, 10).mean()} (should be 10 and 10)\n

        """
        )


if __name__ == "__main__":
    test_visual()
