""" Test data definition. Data can be used for example for validating machine learning time series prediction results.

Only 'real' data are ECG heart signal returned with function get_ecg().
"""

import importlib.util
from typing import cast

import numpy as np

import mylogging

# Lazy imports
# import wfdb


def sin(n: int = 1000) -> np.ndarray:
    """Generate test data of length n in sinus shape.

    Args:
        n (int, optional): Length of data. Defaults to 1000.

    Returns:
        np.ndarray: Sinus shaped data.

    Example:
        >>> sin(50)
        array([0.        , 0.03925982, 0.0784591 , 0.1175374 , 0.15643447,...
    """

    fs = 8000  # Sample rate
    f = 50
    x = np.arange(n)

    return np.sin(2 * np.pi * f * x / fs)


def sign(n: int = 1000) -> np.ndarray:
    """Generate test data of length n in signum shape.

    Args:
        n (int, optional): Length of data. Defaults to 1000.

    Returns:
        np.ndarray: Signum shaped data.

    Example:
        >>> sign(50)
        array([0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,...
    """

    return np.sign(sin(n=n))


# Random
def random(n: int = 1000) -> np.ndarray:
    """Generate random test data of defined length.

    Args:
        n (int, optional): Length of data. Defaults to 1000.

    Returns:
        np.ndarray: Random test data.

    Example:
        >>> data = random(50)
        >>> data.shape
        (50,)
    """

    return np.random.randn(n)


# Range
def ramp(n: int = 1000) -> np.ndarray:
    """Generate ramp data (linear slope) of defined length.

    Args:
        n (int, optional): Length of data. Defaults to 1000.

    Returns:
        np.ndarray: Ramp test data.

    Example:
        >>> ramp(50)
        array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,...
    """

    return np.array(range(n))


def get_ecg(n: int = 1000) -> np.ndarray:
    """Download real ECG data.

    Args:
        n (int, optional): Length of data. Defaults to 1000.

    Returns:
        np.ndarray: Slope test data.

    Example:
        >>> data = get_ecg(50)
        >>> data.shape
        (50, 1)
    """

    if not importlib.util.find_spec("wfdb"):
        raise ModuleNotFoundError(
            mylogging.return_str(
                "For parsing ECG signal, wfdb library is necessary. Install with `pip install wfdb`"
            )
        )

    import wfdb

    try:
        data = wfdb.rdrecord("a103l", pn_dir="challenge-2015/training/", channels=[1], sampto=n).p_signal
        data = cast(np.ndarray, data)

    except Exception:
        raise RuntimeError(mylogging.return_str("Data load failed."))

    return data
