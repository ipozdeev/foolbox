import pandas as pd
import numpy as np
import pickle

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
    swap_ask = swap_ask.where(np.abs(swap_ask) < swap_ask.std()*25)\
        .dropna(how="all")
    swap_bid = swap_bid.where(np.abs(swap_bid) < swap_bid.std()*25)\
        .dropna(how="all")

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
