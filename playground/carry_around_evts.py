import pandas as pd
import numpy as np


def to_spdf(func):
    """Transform generic output of `func` to SPDF.

    Returns
    -------
    wrapper : callable
    """
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        return SPDF(res)

    return wrapper


class SPDF:
    """Special-purpose dataframe.

    Parameters
    ----------
    df : pandas.DataFrame

    """

    def __init__(self, df):
        self.df = df

    def __repr__(self):
        return repr(self.df)

    def __getattr__(self, item):
        res = getattr(self.df, item)

        if callable(res):
            res = to_spdf(res)

        return res


if __name__ == "__main__":

    # construct a generic SPDF
    df = pd.DataFrame(np.eye(4), index=pd.period_range("2000-01", periods=4,
                                                       freq='M'))
    an_spdf = SPDF(df)

    # call .diff() to obtain another SPDF
    print(an_spdf.resample('Q').mean())
    print(pd.__version__)