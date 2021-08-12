import numpy as np
import pandas as pd

from mydatapreprocessing.preprocessing import remove_the_outliers


def rolling_windows(data, window):
    """Generate matrix of rolling windows.

    Example:

        >>> rolling_windows(np.array([1, 2, 3, 4, 5]), window=2)
        array([[1, 2],
               [2, 3],
               [3, 4],
               [4, 5]])

    Args:
        data (np.ndarray): Array data input.
        window (int): Number of values in created window.

    Returns:
        np.ndarray: Array of defined windows
    """
    shape = data.shape[:-1] + (data.shape[-1] - window + 1, window)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def split(data, predicts=7):
    """Divide data set on train and test set. Predicted column is supposed to be 0.

    Args:
        data (pd.DataFrame, np.ndarray): Time series data. ndim has to be 2, reshape if necessary.
        predicts (int, optional): Number of predicted values. Defaults to 7.

    Returns:
        pd.DataFrame, np.ndarray: Train set and test set. If input in numpy array, then also output in array, if dataframe input, then dataframe output.

    Example:

        >>> data = np.array([[1], [2], [3], [4]])
        >>> train, test = split(data, predicts=2)
        >>> train
        array([[1],
               [2]])
        >>> test
        array([3, 4])
    """
    if isinstance(data, pd.DataFrame):
        train = data.iloc[:-predicts, :]
        test = data.iloc[-predicts:, 0]
    else:
        train = data[:-predicts, :]
        test = data[-predicts:, 0]

    return train, test


def add_none_to_gaps(df):
    """If empty windows in sampled signal, it will add None values (one row) to the empty window start.
    Reason is to correct plotting. Points are connected, but not between two gaps.

    Args:
        df (pd.DataFrame): Dataframe with time index.

    Returns:
        pd.DataFrame: Dataframe with None row inserted in time gaps.

    Example:
        >>> data = pd.DataFrame([[0, 1]] * 7, index=[0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 2.0])
        >>> data
             0  1
        0.1  0  1
        0.2  0  1
        0.3  0  1
        1.0  0  1
        1.1  0  1
        1.2  0  1
        2.0  0  1
        >>> df_gaps = add_none_to_gaps(data)
        >>> df_gaps
               0    1
        0.1  0.0  1.0
        0.2  0.0  1.0
        0.3  0.0  1.0
        0.4  NaN  NaN
        1.0  0.0  1.0
        1.1  0.0  1.0
        1.2  0.0  1.0
        1.3  NaN  NaN
        2.0  0.0  1.0
    """
    sampling = remove_the_outliers(np.diff(df.index[:50]).reshape(-1, 1), threshold=1).mean()
    sampling_threshold = sampling * 3
    nons = []
    memory = None

    for i in df.index:
        if memory and i - memory > sampling_threshold:
            nons.append(pd.DataFrame([[np.nan] * df.shape[1]], index=[memory + sampling]))
        memory = i

    return pd.concat([df, *nons]).sort_index()
