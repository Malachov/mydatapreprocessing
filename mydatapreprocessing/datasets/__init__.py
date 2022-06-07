"""Test data definition.

Data can be used for example for validating machine learning time series prediction results.

Only 'real' data are ECG heart signal returned with function `get_ecg()`.
"""
from mydatapreprocessing.datasets.datasets_internal import get_ecg, ramp, random, sin, sign

__all__ = ["get_ecg", "ramp", "random", "sin", "sign"]
