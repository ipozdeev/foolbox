import pandas as pd
import numpy as np
from foolbox.data_mgmt.set_credentials import set_path
from foolbox.finance import realized_variance
from foolbox.linear_models import PureOls


path_to_fx = set_path("research_data/fx_and_events/")


def deseasonalize(data, use_log=True):
    """

    Parameters
    ----------
    data

    Returns
    -------

    """
    days = pd.Series(data=data.index.weekday, index=data.index)
    dum = pd.get_dummies(days)

    deseased = dict()

    for c, c_col in data.iteritems():
        if use_log:
            y = np.log(c_col)
        else:
            y = c_col.copy()

        mod = PureOls(y, dum, add_constant=False)
        deseased[c] = mod.get_residuals(original=True)

        if use_log:
            deseased[c] = np.exp(deseased[c])

    res = pd.concat(deseased, axis=1)

    return res


def main():
    """

    Returns
    -------

    """
    # data from bloomberg ---------------------------------------------------
    data_bloom = pd.read_pickle(path_to_fx + "fx_hf_ba_2017_2018.p")

    # resample at 15min
    data_bloom = data_bloom["mid"]
    data_bloom.index = data_bloom.index.tz_convert("US/Eastern")
    data_bloom = data_bloom.resample("15T").last()

    # returns
    ret_bloom = np.log(data_bloom).diff()

    # rv
    rv_bloom = realized_variance(ret_bloom, freq='B', n_in_day=None,
                                 r_vola=True)

    # data from fxcm --------------------------------------------------------
    data_fxcm = pd.read_pickle(path_to_fx + "fxcm_counter_usd_m15.p")

    # returns
    data_fxcm = (data_fxcm["bid_close"] + data_fxcm["ask_close"])/2
    data_fxcm.index = data_fxcm.index.tz_convert("US/Eastern")

    # resample at 15min
    data_fxcm = data_fxcm.resample("15T").last()

    ret_fxcm = np.log(data_fxcm).diff()

    # rv
    rv_fxcm = realized_variance(ret_fxcm, freq='B', n_in_day=None,
                                r_vola=True)

    rv_bloom, rv_fxcm = rv_bloom.align(rv_fxcm, axis=0, join="inner")

    # ret_bloom, ret_fxcm = ret_bloom.align(ret_fxcm, axis=0, join="inner")

    # rv_bloom.corrwith(rv_fxcm)

    print(rv_bloom.corrwith(rv_fxcm))


if __name__ == "__main__":
    main()
