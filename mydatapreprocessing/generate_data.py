""" Test data definition. Data can be used for example for validating machine learning time series prediction results.

Only 'real' data are ECG heart signal returned with function get_eeg().
"""

import numpy as np
import importlib

import mylogging

# Lazy imports
# import wfdb


def sin(n=1000):
    """Generate test data of length n in sinus shape.

    Args:
        n (int, optional): Length of data. Defaults to 1000.

    Returns:
        np.ndarray: Sinus shaped data.


    Example:
        >>> import mydatapreprocessing as mdp
        ...
        >>> data = mdp.generate_data.sin(1000)
        >>> data
        array([ 0.00000000e+00,  3.92598158e-02,  7.84590957e-02,  1.17537397e-01,
                1.56434465e-01,  1.95090322e-01,  2.33445364e-01,  2.71440450e-01,
                ...
    """

    fs = 8000  # Sample rate
    f = 50
    x = np.arange(n)

    return np.sin(2 * np.pi * f * x / fs)


def sign(n=1000):
    """Generate test data of length n in signum shape.

    Args:
        n (int, optional): Length of data. Defaults to 1000.

    Returns:
        np.ndarray: Signum shaped data.
    """

    return np.sign(sin(n=n))


# Random
def random(n=1000):
    """Generate random test data of defined length.

    Args:
        n (int, optional): Length of data. Defaults to 1000.

    Returns:
        np.ndarray: Random test data.
    """

    return np.random.randn(n) * 5 + 10


# Range
def ramp(n=1000):
    """Generate ramp data (linear slope) of defined length.

    Args:
        n (int, optional): Length of data. Defaults to 1000.

    Returns:
        np.ndarray: Ramp test data.
    """

    return np.array(range(n))


def get_eeg(n=1000):
    """Download real EEG data.

    Args:
        n (int, optional): Length of data. Defaults to 1000.

    Returns:
        np.ndarray: Slope test data.
    """

    if not importlib.util.find_spec("wfdb"):
        raise ModuleNotFoundError(
            mylogging.return_str(
                "For parsing EEG signal, wfdb library is necessary. Install with `pip install wfdb`"
            )
        )

    import wfdb

    return wfdb.rdrecord("a103l", pn_dir="challenge-2015/training/", channels=[1], sampto=n).p_signal
