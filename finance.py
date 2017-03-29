import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset, MonthBegin, MonthEnd, \
    relativedelta

def expect_policy(instrument, meetings, tau, rate=None):
    """
    Parameters
    ----------
    tau : int
        maturity, in months
    """
    assert isinstance(instrument, pd.Series)
    assert isinstance(meetings, pd.Series)

    # do not let
    instrument = instrument.loc[meetings.index[0]:]

    # if no rate provided, meetings must contain a rate
    if rate is None:
        rate = meetings.reindex(index=instrument.index, method="ffill")

    # allocate space for forward rates
    fwd_rate = instrument.copy()*np.nan

    # loop over dates in instrument
    for t in fwd_rate.index:
        # t = fwd_rate.index[2808]

        # break if overshoot
        if t > meetings.last_valid_index():
            break

        # find two meeting dates: the previous, whose rate will serve as
        #   reference rate, and the next, which will possibly set a new rate
        # previous
        prev_meet = meetings.index[meetings.index.get_loc(t, method="ffill")]

        # next closest meeting date
        nx_meet = meetings.index[meetings.index.get_loc(t, method="bfill")]

        # maturity date of ois: t + 1 month (actual/360 convention)
        setl_date = t + DateOffset(months=tau)

        # number of days between them
        ndays = (setl_date - t).days

        # number of days until next meeting
        ndays_until = (nx_meet - t).days

        # if next meeting is earlier than maturity, skip
        if setl_date <=  nx_meet:
            continue

        # previously set rate, to be effective until next meeting
        prev_rate = meetings.loc[prev_meet]
        # prev_rate = sonia.loc[t]

        # implied rate
        r_ois = instrument.loc[t]
        # impl_rate = ((1+/360)**ndays /
        #     (1+prev_rate/360)**ndays_until - 1)*(360/(ndays-ndays_until))
        impl_rate = (
            (
                (r_ois/360 + 1)**ndays/
                (1+prev_rate/360)**(ndays_until)
            )**(1/(ndays-ndays_until))-1)*360

        # store
        fwd_rate.loc[t] = impl_rate

    return fwd_rate
