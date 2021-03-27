"""This is module that from time series data create inputs for machine learning models like scikit learn or tensorflow.
Usual data inputs types are (X, y, x_input). X stands for vector of inputs, y for vector of outputs and
x_input is input for new predictions we want to create.

There are three functions. `make_sequences` that create seqences from time samples, `create_inputs`
that tell the first function what sequences create for what models and `create_tests_outputs`
that for defined inputs create outputs that we can compute error criterion like rmse with.

Functions are documented in it's docstrings.

Input data are in shape (x_samples, x_features).

Example:
    >>> import mydatapreprocessing as mdp
    ...
    >>> data = np.array(
    ...     [
    ...         [1, 2, 3, 4, 5, 6, 7, 8],
    ...         [9, 10, 11, 12, 13, 14, 15, 16],
    ...         [17, 18, 19, 20, 21, 22, 23, 24],
    ...     ]
    ... ).T
    >>> X, y, x_input, _ = mdp.inputs.make_sequences(data, n_steps_in= 2, n_steps_out=3)
    >>> data
    data = array([[1, 9, 17],
                  [2, 10, 18],
                  [3, 11, 19],
                  [4, 12, 20],
                  [5, 13, 21],
                  [6, 14, 22],
                  [7, 15, 23],
                  [8, 16, 24]])
    >>> X
    array([[1, 2, 3, 9, 10, 11, 17, 18, 19],
           [2, 3, 4, 10, 11, 12, 18, 19, 20],
           [3, 4, 5, 11, 12, 13, 19, 20, 21],
           [4, 5, 6, 12, 13, 14, 20, 21, 22]])
    >>> y
    array([[4, 5],
           [5, 6],
           [6, 7],
           [7, 8]]
    >>> x_input
    array([[ 6, 7, 8, 14, 15, 16, 22, 23, 24]])

    If constant param is True, then bias 1 is added to every sample on index 0.
"""

import pandas as pd

pd.DataFrame


import numpy as np

from mydatapreprocessing.preprocessing import rolling_windows
import mylogging


def make_sequences(
    data,
    n_steps_in,
    n_steps_out=1,
    constant=None,
    predicts=7,
    repeatit=10,
    predicted_column_index=0,
    serialize_columns=1,
    default_other_columns_length=None,
):
    """Function that create inputs and outputs to models.

        Example for n_steps_in = 3 and n_steps_out = 1

        From [[1], [2], [3], [4], [5], [6]]

        Creates [1, 2, 3]  [4]
                [2, 3, 4]  [5]
                [3, 4, 5]  [6]

        Args:
            data (np.ndarray): Time series data. Shape is (n_samples, n_feature)
            n_steps_in (int): Number of input members.
            n_steps_out (int, optional): Number of output members. For one-step models use 1. Defaults to 1.
            constant (bool, optional): If use bias (add 1 to first place to every member). Defaults to None.
            predicts (int, optional): How many values are predicted. Define output length for batch models.
            repeatit (int, optional): How many inputs will be tested.
            predicted_column_index (int, optional): If multiavriate data, index of predicted column. Defaults to 0.
            serialize_columns(bool, optional): If multivariate data, serialize columns sequentions into one row.
            default_other_columns_length (int, optional): Length of non-predicted columns that are evaluated in inputs. If None, than same length as predicted column. Defaults to None.

        Returns:
            np.array, np.array: X and y. Inputs and outputs (that can be used for example in sklearn models).

    Example:
        >>> import mydatapreprocessing as mdp
        ...
        >>> data = np.array(
        ...     [
        ...         [1, 2, 3, 4, 5, 6, 7, 8],
        ...         [9, 10, 11, 12, 13, 14, 15, 16],
        ...         [17, 18, 19, 20, 21, 22, 23, 24],
        ...     ]
        ... ).T
        >>> X, y, x_input, _ = mdp.inputs.make_sequences(data, n_steps_in= 2, n_steps_out=3)
        >>> data
        data = array([[1, 9, 17],
                      [2, 10, 18],
                      [3, 11, 19],
                      [4, 12, 20],
                      [5, 13, 21],
                      [6, 14, 22],
                      [7, 15, 23],
                      [8, 16, 24]])
        >>> X
        array([[1, 2, 3, 9, 10, 11, 17, 18, 19],
               [2, 3, 4, 10, 11, 12, 18, 19, 20],
               [3, 4, 5, 11, 12, 13, 19, 20, 21],
               [4, 5, 6, 12, 13, 14, 20, 21, 22]])
        >>> y
        array([[4, 5],
               [5, 6],
               [6, 7],
               [7, 8]]
        >>> x_input
        array([[ 6, 7, 8, 14, 15, 16, 22, 23, 24]])

        If parameter constant is one (mean bias for neural nets), result can be something like

        X = array([[ 1, 1, 2, 3],
                   [ 1, 4, 5, 6],
                   [ 1, 7, 8, 9])
    """

    if n_steps_out > n_steps_in:
        n_steps_in = n_steps_out + 1
        mylogging.warn(
            "n_steps_out was bigger than n_steps_in - n_steps_in changed during prediction!"
        )

    if default_other_columns_length == 0:
        data = data[:, 0].reshape(1, -1)

    X = rolling_windows(data.T, n_steps_in)

    if predicted_column_index != 0:
        X[[0, predicted_column_index], :] = X[[predicted_column_index, 0], :]

    y = X[0][n_steps_out:, -n_steps_out:]

    if serialize_columns:
        if default_other_columns_length:
            X = np.hstack(
                [X[0, :]]
                + [X[i, :, -default_other_columns_length:] for i in range(1, len(X))]
            )
        else:
            X = X.transpose(1, 0, 2).reshape(1, X.shape[1], -1)[0]

    else:
        X = X.transpose(1, 2, 0)

    if constant:
        X = np.hstack([np.ones((len(X), 1)), X])

    if X.ndim == 3:
        x_input = X[-1].reshape(1, X.shape[1], X.shape[2])
        x_test_inputs = X[-predicts - repeatit : -predicts, :, :]
        x_test_inputs = x_test_inputs.reshape(
            x_test_inputs.shape[0], 1, x_test_inputs.shape[1], x_test_inputs.shape[2]
        )

    else:
        x_input = X[-1].reshape(1, -1)

        x_test_inputs = X[-predicts - repeatit : -predicts, :]
        x_test_inputs = x_test_inputs.reshape(
            x_test_inputs.shape[0], 1, x_test_inputs.shape[1]
        )

    X = X[:-n_steps_out]

    return X, y, x_input, x_test_inputs


def create_inputs(
    data,
    input_type_name,
    input_type_params,
    mode="validate",
    predicts=7,
    repeatit=10,
    predicted_column_index=0,
):
    """Define configured inputs for various models. For some models use `make_sequences` function => check it's
    documentation how it works. For `data` input type name, just return data, if data_one_column, other columns
    are deleted, if something else, it create inputs called X and y - same convention as in sklearn plus x_input
    - input for predicted values. If constant in used name, it will insert bias 1 to every sample input.

    Args:
        data (np.ndarray): Time series data.
        input_type_name (str): Name of input. Choices are ['data', 'data_one_column', 'one_in_one_out_constant','one_in_one_out',
            'one_in_batch_out', 'something_else']. If something else, than input type params define produces inputs.
        input_type_params (dict): Dict of params used in make_sequences. E.g. {'n_steps_in': cls.default_n_steps_in,
            'n_steps_out': cls.predicts, 'default_other_columns_length': cls.default_other_columns_length, 'constant': 0}
        mode (str, optional): 'validate' or 'predictNumber of predicted valuesvalidate'.
        predicts (int, optional): Number of predicted values. Defaults to 7.
        repeatit (int, optional): Number of tested sequentions. Defaults to 10.
        predicted_column_index (int, optional): Predicted column index. Defaults to 0.

    Returns:
        tuple: (model_train_input, model_predict_input, model_test_inputs)

            `model_train_input` means data inserting model - it can be timeseries like [1, 2, 3, 4, 5...] or it can be
            the tuple of X = [[1, 2, 3], [2, 3, 4]...] and y = [[4], [5]...]

            `model_predict_input` means x_input - data inserting model to create prediction.

            `model_test_inputs` if mode is not validate, it will choose some inputs and it's results for model error evaluation.
            Thing is, that these inputs was in training set, so it's not accurate (but as much as possible used for train).
            If mode is validate, than only one testing input wit result is generated (real error, but small sample - need to validate
            in more datapoints - retrain everytime).
    """

    # Take one input type, make all derivated inputs (save memory, because only slices) and create dictionary of inputs for one iteration
    used_sequentions = {}

    if input_type_name == "data":
        used_sequentions = data

    elif input_type_name == "data_one_column":
        used_sequentions = data[:, predicted_column_index]

    else:
        if input_type_name in [
            "one_in_one_out_constant",
            "one_in_one_out",
            "one_in_batch_out",
        ]:
            used_sequentions = data[
                :, predicted_column_index : predicted_column_index + 1
            ]
        else:
            used_sequentions = data

        used_sequentions = make_sequences(
            used_sequentions, predicts=predicts, repeatit=repeatit, **input_type_params
        )

    if isinstance(used_sequentions, tuple):
        model_train_input = (used_sequentions[0], used_sequentions[1])
        model_predict_input = used_sequentions[2]
        if mode == "validate":
            model_test_inputs = [model_predict_input]
        else:
            model_test_inputs = used_sequentions[3]

    else:
        model_train_input = model_predict_input = used_sequentions
        if mode == "validate":
            model_test_inputs = [model_predict_input]
        else:
            model_test_inputs = []
            if used_sequentions.ndim == 1:
                for i in range(repeatit):
                    model_test_inputs.append(
                        used_sequentions[: -predicts - repeatit + i + 1]
                    )
            else:
                for i in range(repeatit):
                    model_test_inputs.append(
                        used_sequentions[:, : -predicts - repeatit + i + 1]
                    )

    return model_train_input, model_predict_input, model_test_inputs


def create_tests_outputs(data, predicts=7, repeatit=10):
    """Generate list of expected test results to have values to compare the predictions to.

    Args:
        data (np.ndarray): Predicted column.
        predicts (int, optional): Number of predicted values. Defaults to 7.
        repeatit (int, optional): Number of created predictions that will be compared. Defaults to 10.

    Returns:
        list: List of result arrays.
    """
    models_test_outputs = np.zeros((repeatit, predicts))

    for i in range(repeatit):
        models_test_outputs[i] = (
            data[-predicts - i : -i] if i > 0 else data[-predicts - i :]
        )

    models_test_outputs = models_test_outputs[::-1]

    return models_test_outputs
