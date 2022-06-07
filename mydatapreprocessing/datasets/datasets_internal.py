"""Module for consolidation_pipeline subpackage."""

from __future__ import annotations
from typing import cast

import numpy as np

from mypythontools.system import check_library_is_available

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
    sample_rate = 8000  # Sample rate
    freq = 50
    arr = np.arange(n)

    return np.sin(2 * np.pi * freq * arr / sample_rate)


def sign(n: int = 1000) -> np.ndarray:
    """Generate test data of length n in signum shape.

    Args:
        n (int, optional): Length of data. Defaults to 1000.

    Returns:
        np.ndarray: Signum shaped data.

    Example:
        >>> sign(50)
        array([0., 1., 1., 1., 1., 1., 1., 1., 1., ...
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
        array([ 0,  1,  2,  3,  4,  5,  6,  7, ...
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
    check_library_is_available("wfdb")

    import wfdb

    try:
        data = wfdb.rdrecord(
            "a103l", pn_dir="challenge-2015/training/", channels=[1], sampto=n
        ).p_signal  # type: ignore - Has no type annotations
        data = cast(np.ndarray, data)
        return data

    except Exception as err:
        raise RuntimeError(
            "Error in 'mydatapreprocessing' package in 'get_ecg' function. EVG test data load failed. "
            "Internet connection is necessary to get the data. If it's SSLError on anaconda it may be "
            "necessary to to install openSSL and restart computer.",
        ) from err
