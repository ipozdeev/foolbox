# spot, forward and tom/next swap points by time zone
# data_by_tz : dict of Panels with "NYC", "LON", "TOK" on the minor axis
# data_by_tz_aligned : dict of DataFrames, with time dimension in Panels above
#   "collapsed" according to currencies' time zones; swap points fixed

import pandas as pd
# import numpy as np
import pickle

from foolbox.data_mgmt import set_credentials as set_cred
path = set_cred.gdrive_path("research_data/fx_and_events/")

# pickle name ---------------------------------------------------------------
name_data_by_tz = path + "fx_by_tz_d.p"
name_data_by_tz_aligned = path + "fx_by_tz_aligned_d.p"

# xlsx names
fname_spot_mid = "fx_spot_mid_diff_tz_1994_2017_d.xlsx"
fname_spot_bid = "fx_spot_bid_diff_tz_1994_2017_d.xlsx"
fname_spot_ask = "fx_spot_ask_diff_tz_1994_2017_d.xlsx"
fname_tnswap_ask = "fx_tnswap_ask_diff_tz_1994_2017_d.xlsx"
fname_tnswap_bid = "fx_tnswap_bid_diff_tz_1994_2017_d.xlsx"
fnames = [fname_spot_mid, fname_spot_ask, fname_spot_bid, fname_tnswap_ask,
    fname_tnswap_bid]

def parse_bloomberg_xlsx_temp(filename):
    """
    Returns
    -------
    this_data : dict
        with "NYC", "LON", "TOK" as keys, pandas.DataFrames as values
    """
    currencies = pd.read_excel(path+filename, sheetname="iso", header=0)
    currencies = currencies.columns

    N = len(currencies)
    converters = {}
    for p in range(N):
        converters[p*3] = lambda x: pd.to_datetime(x)

    # sheets
    sheetnames = ["NYC","LON","TOK"]

    this_data = dict()

    for s in sheetnames:
        # s = "NYC"
        data = pd.read_excel(
            io=path+filename,
            sheetname=s,
            skiprows=2,
            header=None,
            converters=converters)

        # take every third third column: these are the values
        data = [data.ix[:,(p*3):(p*3+1)].dropna() for p in range(N)]

        # pop dates as index
        for p in range(N):
            data[p].index = data[p].pop(p*3)

        # `data` is a list -> concat to a df
        data = pd.concat(data, axis=1, ignore_index=False)

        # columns are iso letters
        data.columns = currencies

        # store
        this_data[s] = data

    return this_data

# loop over files, collect data ---------------------------------------------
data_by_tz = dict()
for f in fnames:
    # f = fname_tnswap_ask
    this_data = parse_bloomberg_xlsx_temp(f)

    # keys will be "spot_mid", "tnswap_bid" etc.
    data_by_tz['_'.join(f.split('_')[1:3])] = pd.Panel.from_dict(
        this_data, orient="minor")

# tom/next swap points: from pips to units ----------------------------------
data_by_tz["tnswap_ask"] /= 10000
data_by_tz["tnswap_bid"] /= 10000
# yen has 100 as pips
data_by_tz["tnswap_ask"].loc["jpy",:,:] *= 100
data_by_tz["tnswap_bid"].loc["jpy",:,:] *= 100

# organize according to the time zone a currency belongs to -----------------
currencies = data_by_tz["spot_mid"].items
cur_by_tz = dict(zip(
    ['aud', 'cad', 'chf', 'dkk', 'eur', 'gbp', 'jpy', 'nok', 'nzd', 'sek'],
    ["TOK","NYC","LON","LON","LON","LON","TOK","LON","TOK","LON"]))

# split by time zone
data_by_tz_aligned = dict()
for k,v in data_by_tz.items():
    # take spot mid, ask etc.
    this_quote_type = pd.concat((
        v.loc[p,:,cur_by_tz[p]] for p in currencies), axis=1)
    this_quote_type.columns = currencies

    # save
    data_by_tz_aligned[k] = this_quote_type

# "fix" sek, dkk, jpy, nok and chf ------------------------------------------
# since these are in usdxxx form
usdxxx = ["sek","dkk","chf","nok","jpy"]
ask_sp_x = \
    1/(1/data_by_tz_aligned["spot_ask"].loc[:,usdxxx] +
        data_by_tz_aligned["tnswap_bid"].loc[:,usdxxx]
        )-\
        data_by_tz_aligned["spot_ask"].loc[:,usdxxx]
bid_sp_x = \
    1/(1/data_by_tz_aligned["spot_bid"].loc[:,usdxxx] +
        data_by_tz_aligned["tnswap_ask"].loc[:,usdxxx]
        )-\
        data_by_tz_aligned["spot_bid"].loc[:,usdxxx]

data_by_tz_aligned["tnswap_ask"].loc[:,usdxxx] = ask_sp_x
data_by_tz_aligned["tnswap_bid"].loc[:,usdxxx] = bid_sp_x

# adjust for negative ba spreads --------------------------------------------
idx = (data_by_tz_aligned["tnswap_ask"] - data_by_tz_aligned["tnswap_bid"]) < 0
data_by_tz_aligned["tnswap_bid"] = data_by_tz_aligned["tnswap_bid"].mask(
    idx, data_by_tz_aligned["tnswap_ask"])

# pickle (for the greater justment) -----------------------------------------
with open(name_data_by_tz, "wb") as fname:
    pickle.dump(data_by_tz, fname)
with open(name_data_by_tz_aligned, "wb") as fname:
    pickle.dump(data_by_tz_aligned, fname)
