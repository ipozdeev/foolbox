import pandas as pd
import numpy as np
import pickle

def remove_outliers(data, stds):
    """
    """
    res = data.where(np.abs(data) < data.std()*25)

    return res

def fetch_the_data(path_to_data, drop_curs=[], align=False, add_usd=False):
    """
    """
    # Import the FX data
    with open(path_to_data+"fx_by_tz_aligned_d.p", mode="rb") as fname:
        data_merged_tz = pickle.load(fname)

    # Import the all fixing times for the dollar index
    with open(path_to_data+"fx_by_tz_sp_fixed.p", mode="rb") as fname:
        data_all_tz = pickle.load(fname)

    # Get the individual currencies, spot rates:
    spot_bid = data_merged_tz["spot_bid"]
    spot_ask = data_merged_tz["spot_ask"]
    swap_ask = data_merged_tz["tnswap_ask"]
    swap_bid = data_merged_tz["tnswap_bid"]

    # outliers in swap points
    swap_ask = remove_outliers(swap_ask, 50)
    swap_bid = remove_outliers(swap_bid, 50)

    # Align and ffill the data, first for tz-aligned countries
    (spot_bid, spot_ask, swap_bid, swap_ask) = align_and_fillna(
        (spot_bid, spot_ask, swap_bid, swap_ask),
        "B", method="ffill")

    if add_usd:

        spot_ask_us = data_all_tz["spot_ask"].loc[:,:,"NYC"]
        spot_bid_us = data_all_tz["spot_bid"].loc[:,:,"NYC"]
        swap_ask_us = data_all_tz["tnswap_ask"].loc[:,:,"NYC"]
        swap_bid_us = data_all_tz["tnswap_bid"].loc[:,:,"NYC"]

        swap_ask_us = \
            swap_ask_us.where(np.abs(swap_ask_us) < swap_ask_us.std()*25)\
            .dropna(how="all")
        swap_bid_us = \
            swap_bid_us.where(np.abs(swap_bid_us) < swap_bid_us.std()*25)\
            .dropna(how="all")

        # Now for the dollar index
        (spot_bid_us, spot_ask_us, swap_bid_us, swap_ask_us) =\
            align_and_fillna((spot_bid_us, spot_ask_us,
                              swap_bid_us, swap_ask_us),
                             "B", method="ffill")

        spot_bid.loc[:,"usd"] = spot_bid_us.drop(drop_curs,axis=1).mean(axis=1)
        spot_ask.loc[:,"usd"] = spot_ask_us.drop(drop_curs,axis=1).mean(axis=1)
        swap_bid.loc[:,"usd"] = swap_bid_us.drop(drop_curs,axis=1).mean(axis=1)
        swap_ask.loc[:,"usd"] = swap_ask_us.drop(drop_curs,axis=1).mean(axis=1)

    prices = pd.Panel.from_dict(
        {"bid": spot_bid, "ask": spot_ask},
        orient="items").drop(drop_curs, axis="minor_axis")
    swap_points = pd.Panel.from_dict(
        {"bid": swap_bid, "ask": swap_ask},
        orient="items").drop(drop_curs, axis="minor_axis")

    return prices, swap_points


def align_and_fillna(data, reindex_freq=None, **kwargs):
    """
    Parameters
    ----------
    data : list-like
        of pandas.DataFrames or pandas.Series
    reindex_freq : str
        pandas frequency string, e.g. 'B' for business day
    kwargs : dict
        arguments to .fillna()
    """
    common_idx = pd.concat(data, axis=1, join="outer").index

    if reindex_freq is not None:
        common_idx = pd.date_range(common_idx[0], common_idx[-1],
            freq=reindex_freq)

    new_data = None

    if isinstance(data, dict):
        new_data = {}
        for k,v in data.items():
            new_data.update({k: v.reindex(index=common_idx).fillna(**kwargs)})
    elif isinstance(data, tuple):
        new_data = tuple(
            [p.reindex(index=common_idx).fillna(**kwargs) for p in data])
    elif isinstance(data, list):
        new_data = [p.reindex(index=common_idx).fillna(**kwargs) for p in data]

    return new_data

def add_fake_signal(ret, sig):
    """ Add fake series to signal: the median in each row.
    """
    r, s = ret.copy(), sig.copy()

    # calculate median across rows
    fake_sig = sig.apply(np.nanmedian, axis=1)

    # reinstall
    s.loc[:,"fake"] = fake_sig
    r.loc[:,"fake"] = np.nan

    return r, s

def interevent_quantiles(events, df=None):
    """ Split inter-event intervals in two parts.
    Parameters
    ----------
    events : pandas.Series or pandas.DataFrame
        of events; any non-na value counts as separate event
    df : (optional) pandas.DataFrame
        optional dataframe to concatenate the indicator column to

    Returns
    -------
    res : (if df is None) pandas.DataFrame with new columns: event number and
        quantiles for each column of `events` (if more than one); (if df is
        not None) df.copy with new columns added

    Example
    -------
    pd.set_option("display.max_rows", 20)
    import random
    evts = pd.DataFrame(index=range(100), columns=["one","two"])
    idx = random.sample(list(evts.index), 20)
    evts.loc[idx] = 1
    events = evts.copy()
    interevent_quantiles(events, events)
    """
    # recursion
    if isinstance(events, pd.DataFrame):
        res = interevent_quantiles(
            events=events.iloc[:,1:].squeeze(),
            df=interevent_quantiles(events.iloc[:,0], df))
        return res

    # name
    evt_name = "evt" if events.name is None else events.name

    # leave only 0.0 and nan's
    evts_q = events.notnull().where(events.notnull())*0

    # remove trailing and leading nans: no info where these periods strat(end)
    evts_q = evts_q.loc[evts_q.first_valid_index():evts_q.last_valid_index()]

    # index events
    evts_idx = evts_q.dropna()
    evts_idx = evts_idx.add(np.arange(1,len(evts_idx)+1))
    evts_idx = evts_idx.reindex(index=evts_q.index, method="ffill")

    # helper: this will be modified each iteration
    evts_help = evts_q.copy().astype(float)

    # while there are nan's, forward-fill then backward-fill one cell
    while evts_q.isnull().sum() > 0:
        temp_evts = evts_help.replace(0.0, 1.0)
        evts_q.fillna(temp_evts.fillna(method="ffill", limit=1), inplace=True)
        temp_evts = evts_help.replace(0.0, 2.0)
        evts_q.fillna(temp_evts.fillna(method="bfill", limit=1), inplace=True)
        # make sure helper changes
        evts_help = evts_q.copy()

    # concatenate
    res = pd.concat((evts_idx, evts_q), axis=1)
    res.columns = ["_evt_" + p + evt_name for p in ["num_", "q_"]]

    # patch with two additional columns
    if df is not None:
        df = df.reindex(index=evts_q.index)
        df = pd.concat((df, res), axis=1)

    return (res if df is None else df)
