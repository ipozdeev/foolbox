import pandas as pd
import numpy as np
from foolbox.data_mgmt.set_credentials import set_path
from foolbox.linear_models import PureOls
from foolbox.finance import EventStudy


path_to_fx = set_path("research_data/fx_and_events/")


def calculate_rv(data, hour_start=None, hour_end=None, n_in_day=1):
    """

    Parameters
    ----------
    data
    hour_start
    hour_end

    Returns
    -------

    """
    if hour_end is None:
        hour_end = 24
    if hour_start is None:
        hour_start = 0

    idx = (data.index.hour > hour_start) & (data.index.hour < hour_end) & \
        (~data.index.weekday.isin([5, 6]))
    ret_trim = data.loc[idx, :]

    rv = ret_trim.pow(2).resample('B').mean() * n_in_day * 252

    rv = np.sqrt(rv) * 100

    return rv


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


if __name__ == "__main__":
    # 15min
    data = pd.read_pickle(path_to_fx + "fxcm_counter_usd_m15.p")
    ret = np.log(((data["bid_close"] + data["ask_close"])/2)).diff()

    # rv
    rv = calculate_rv(ret, 8, 21, n_in_day=24*4)

    # seasonal
    rv_deseas = deseasonalize(rv, use_log=True)

    # rv_deseas.plot()

    events_data = pd.read_pickle(path_to_fx + "events.p")
    evt = events_data["joint_cbs"]
    rv, evt = rv.align(evt, axis=1, join="inner")
    rv_deseas.index = rv_deseas.index.date
    evt.index = evt.index.date
    es = EventStudy(data=rv_deseas, events=evt, window=(-10, -1, 0, 3),
                    mean_type="count_weighted")
    es.the_mean