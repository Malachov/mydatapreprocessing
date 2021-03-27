""" Test data definition. Data are to be used for validating machine learning time series prediction results.

Example:

    >>> import mydatapreprocessing as mdp
    ...
    >>> data = mdp.generatedata.gen_sin(1000)
    >>> data
    array([ 0.00000000e+00, 3.92598158e-02, 7.84590957e-02, 1.17537397e-01,
            1.56434465e-01, 1.95090322e-01, 2.33445364e-01, 2.71440450e-01,
            ...
            ...
    ])
"""

import numpy as np
import mylogging
import importlib


def gen_sin(n=1000):
    """Generate test data of length n in sinus shape.

    Args:
        n (int): Length of data.

    Returns:
        np.ndarray: Sinus shaped data.
    """

    fs = 8000  # Sample rate
    f = 50
    x = np.arange(n)

    return np.sin(2 * np.pi * f * x / fs)


def gen_sign(n=1000, periods=220):
    """Generate test data of length n in signum shape.

    Args:
        n (int): Length of data.

    Returns:
        np.ndarray: Signum shaped data.
    """

    sin = gen_sin(n=n)

    return np.sign(sin)


# Random
def gen_random(n=1000):
    """Generate random test data of length.

    Args:
        n (int): Length of data.

    Returns:
        np.ndarray: Random test data.
    """

    return np.random.randn(n) * 5 + 10


# Range
def gen_slope(n=1000):
    """Generate random test data of length.

    Args:
        n (int): Length of data.

    Returns:
        np.ndarray: Slope test data.
    """

    return np.array(range(n))


def get_eeg(n=1000):
    """Download eeg data.

    Args:
        n (int): Length of data.

    Returns:
        np.ndarray: Slope test data.
    """

    if not importlib.util.find_spec("xlrd"):
        raise ModuleNotFoundError(
            mylogging.return_str(
                "For parsing eeg signal, wfdb library is necessary. Install with `pip install wfdb`"
            )
        )

    import wfdb

    return wfdb.rdrecord(
        "a103l", pn_dir="challenge-2015/training/", channels=[1], sampto=n
    ).p_signal
