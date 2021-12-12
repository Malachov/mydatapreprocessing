import numpy as np

import mypythontools

mypythontools.tests.setup_tests()


import mydatapreprocessing.create_model_inputs as mdpi


def test_make_sequences():
    data = np.array(
        [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16], [17, 18, 19, 20, 21, 22, 23, 24],]
    ).T
    X, y, x_input, _ = mdpi.make_sequences(data, n_steps_in=2, n_steps_out=3, constant=1)

    X_res = np.array(
        [
            [1.0, 1.0, 2.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0, 17.0, 18.0, 19.0, 20.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 13.0, 18.0, 19.0, 20.0, 21.0],
        ]
    )
    y_res = np.array([[5, 6, 7], [6, 7, 8]])
    x_inpu_res = np.array([[1.0, 5.0, 6.0, 7.0, 8.0, 13.0, 14.0, 15.0, 16.0, 21.0, 22.0, 23.0, 24.0]])

    data2 = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]).T
    X2, y2, x_input2, test_inputs2 = mdpi.make_sequences(
        data2, n_steps_in=2, n_steps_out=1, constant=0, predicts=3, repeatit=2
    )

    X2_res = np.array(
        np.array(
            [
                [1, 2, 11, 12],
                [2, 3, 12, 13],
                [3, 4, 13, 14],
                [4, 5, 14, 15],
                [5, 6, 15, 16],
                [6, 7, 16, 17],
                [7, 8, 17, 18],
                [8, 9, 18, 19],
            ]
        )
    )
    y2_res = np.array(([[3], [4], [5], [6], [7], [8], [9], [10]]))
    x_input2_res = np.array(([[9, 10, 19, 20]]))
    test_inputs2_res = np.array([[[5, 6, 15, 16]], [[6, 7, 16, 17]]])

    assert all(
        [
            np.allclose(X, X_res),
            np.allclose(y, y_res),
            np.allclose(x_input, x_inpu_res),
            np.allclose(X2, X2_res),
            np.allclose(y2, y2_res),
            np.allclose(x_input2, x_input2_res),
            np.allclose(test_inputs2, test_inputs2_res),
        ]
    )


if __name__ == "__main__":
    pass
