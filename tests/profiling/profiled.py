# from __future__ import annotations
# from typing import NamedTuple, Any, cast
# from typing_extensions import Literal

# import line_profiler
# import memory_profiler
# import numpy as np
# import pandas as pd

# from mypythontools_cicd import tests

# import mydatapreprocessing

# tests.setup_tests()

# # import predictit

# ##############################

# profile = line_profiler.LineProfiler()  # memory
# # profile = memory_profiler.profile  # time

# ##############################


# # This profiled function is to be profiled from jupyter
# def profiled():
#     array = np.random.randn(100000, 6) + 8 * 6
#     df = pd.DataFrame(array)

#     # Profiled code here


# # This code will be
# if __name__ == "__main__":

#     def copy(df):
#         return df.copy()

#     def plain(df):
#         return df

#     def inplace(df):
#         return

#     # (precision=8)  # Just memory profiler
#     @profile
#     def profile_here():
#         df = pd.DataFrame(np.random.randn(100000, 6) + 8 * 6)

#         df2 = copy(df)
#         inplace(df2)
#         df3 = plain(df)

#     profile_here()

#     if hasattr(profile, "print_stats"):
#         profile.print_stats()
