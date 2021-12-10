#%%
""" Visual test on various components. Mostly for data preparation and input creating functions.
Just run and check results to know what functions do and how.
"""
import numpy as np

import mypythontools

mypythontools.tests.setup_tests()


import mydatapreprocessing.create_model_inputs as mdi
import mydatapreprocessing.preprocessing as mdpp
import mydatapreprocessing as mdp

np.random.seed(2)


def test_visual(print_preprocessing=1, print_postprocessing=1):

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

    # Some calculations, that are to long to do in f-strings - Just ignore...

    seqs, Y, x_input, test_inputs = mdi.make_sequences(
        data, predicts=7, repeatit=3, n_steps_in=6, n_steps_out=1, constant=True
    )
    seqs_2, Y_2, x_input2, test_inputs2 = mdi.make_sequences(
        data, predicts=7, repeatit=3, n_steps_in=4, n_steps_out=2, constant=False
    )
    seqs_m, Y_m, x_input_m, test_inputs_m = mdi.make_sequences(
        data_multi_col,
        predicts=7,
        repeatit=3,
        n_steps_in=4,
        n_steps_out=1,
        default_other_columns_length=None,
        constant=1,
    )
    seqs_2_m, Y_2_m, x_input2_m, test_inputs2_m = mdi.make_sequences(
        data_multi_col,
        predicts=7,
        repeatit=3,
        n_steps_in=3,
        n_steps_out=2,
        default_other_columns_length=1,
        constant=0,
    )

    normalized, scaler = mdpp.standardize(data)
    normalized_multi, scaler_multi = mdpp.standardize(data_multi_col)

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

        With outliers: \n {data} \n\nWith no outliers: \n{mdpp.remove_the_outliers(data, threshold = 3)} \n

        ### Difference transform ### \n
        Original data: \n {data} \n\nDifferenced data: \n{mdpp.do_difference(data[:, 0])} \n\n
        Backward difference: \n{mdpp.inverse_difference(mdpp.do_difference(data[:, 0]), data[0, 0])}\n

        ### Standardize ### \n
        Original: \n {data} \n\nStandardized: \n{normalized} \n

        Inverse standardization: \n{scaler.inverse_transform(normalized)} \n

        ### Split ### \n
        Original: \n {data} \n\nsplited train: \n{mdp.misc.split(data)[0]} \n \n\nsplited test: \n{mdp.misc.split(data)[1]} \n

        ### Make sequences - n_steps_in = 6, n_steps_out = 1, constant = 1 ### \n
        Original: \n {data} \n\nsequences: \n{seqs} \n\nY: \n{Y} \n\nx_input:{x_input} \n\n Tests inputs:{test_inputs}\n

        ### Make batch sequences - n_steps_in = 4, n_steps_out = 2, constant = 0 ### \n
        Original: \n {data} \n\nsequences: \n{seqs_2} \n\nY: \n{Y_2} \n \nx_input: \n{x_input2} \n\n Tests inputs:{test_inputs2} \n

            ####################
            ### More columns ###
            ####################\n

        ### Remove outliers ### \n
        With outliers: \n {data_multi_col} \n\nWith no outliers: \n{mdpp.remove_the_outliers(data_multi_col, threshold = 1)} \n

        ### Standardize ### \n
        Original: \n {data_multi_col} \n\nStandardized: \n{normalized_multi} \n

        ### Split ### \n
        Original: \n {data_multi_col} \n\nsplited train: \n{mdp.misc.split(data_multi_col, predicts=2)[0]} \n \n\nsplited test: \n{mdp.misc.split(data_multi_col, predicts=2)[1]} \n

        ### Make sequences - n_steps_in=4, n_steps_out=1, default_other_columns_length=None, constant=1 ### \n
        Original: \n {data_multi_col} \n\nsequences: \n{seqs_m} \n\nY: \n{Y_m} \nx_input: \n\n{x_input_m} \n\n Tests inputs:{test_inputs_m} \n

        ### Make batch sequences - n_steps_in=3, n_steps_out=2, default_other_columns_length=1, constant=0 ### \n
        Original: \n {data_multi_col} \n\nsequences: \n{seqs_2_m} \n\nY: \n{Y_2_m} \nx_input: \n\n{x_input2_m} \n\n Tests inputs:{test_inputs2_m} \n

        """
        )

    if print_postprocessing:
        print(
            f"""

                ###########################
                ### Data_postprocessing ###
                ###########################\n
        ### Fitt power transform ### \n
        Original: \n {data}, original std = {data.std()}, original mean = {data.mean()} \n\ntransformed: \n{mdpp.fitted_power_transform(data, 10, 10)} \n\ntransformed std = {mdpp.fitted_power_transform(data, 10, 10).std()},
        transformed mean = {mdpp.fitted_power_transform(data, 10, 10).mean()} (shoud be 10 and 10)\n

        """
        )


if __name__ == "__main__":
    test_visual()
