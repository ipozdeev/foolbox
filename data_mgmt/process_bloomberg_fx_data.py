# spot, forward and tom/next swap points by time zone
# data_by_tz : dict of Panels with "NYC", "LON", "TOK" on the minor axis
# data_by_tz_aligned : dict of DataFrames, with time dimension in Panels above
#   "collapsed" according to currencies' time zones; swap points fixed

import pandas as pd
import pickle

from foolbox.data_mgmt import set_credentials as set_cred
path = set_cred.gdrive_path("research_data/fx_and_events/")

# pickle name ---------------------------------------------------------------
name_data_by_tz = path + "fx_by_tz_d.p"
name_data_by_tz_aligned = path + "fx_by_tz_aligned_d.p"
name_data_by_tz_sp_fixed = path + "fx_by_tz_sp_fixed.p"

# xlsx names
fname_spot_mid = "fx_spot_mid_diff_tz_1994_2017_d.xlsx"
fname_spot_bid = "fx_spot_bid_diff_tz_1994_2017_d.xlsx"
fname_spot_ask = "fx_spot_ask_diff_tz_1994_2017_d.xlsx"
fname_tnswap_ask = "fx_tnswap_ask_diff_tz_1994_2017_d.xlsx"
fname_tnswap_bid = "fx_tnswap_bid_diff_tz_1994_2017_d.xlsx"
# fname_1w_ask = "fx_1w_ask_diff_tz_1994_2017_d.xlsx"
# fname_1w_bid = "fx_1w_bid_diff_tz_1994_2017_d.xlsx"
# fname_1w_mid = "fx_1w_mid_diff_tz_1994_2017_d.xlsx"
# fname_1w_settl = "fx_1w_settl_diff_tz_1994_2017_d.xlsx"


# fnames = [fname_spot_mid, fname_spot_ask, fname_spot_bid, fname_tnswap_ask,
#     fname_tnswap_bid, fname_1w_ask, fname_1w_bid, fname_1w_mid,
#     fname_1w_settl]
fnames = [fname_spot_ask, fname_spot_bid, fname_tnswap_ask, fname_tnswap_bid,
    fname_spot_mid]

def parse_bloomberg_xlsx_temp(filename):
    """
    filename = fname_1w_settl
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
    # f = "fx_spot_mid_diff_tz_1994_2017_d.xlsx"
    this_data = parse_bloomberg_xlsx_temp(f)

    # keys will be "spot_mid", "tnswap_bid" etc.
    data_by_tz['_'.join(f.split('_')[1:3])] = pd.Panel.from_dict(
        this_data, orient="minor")

# pickle this raw file full of panels
with open(name_data_by_tz, "wb") as fname:
    pickle.dump(data_by_tz, fname)

data_by_tz["tnswap_ask"] /= 10000
data_by_tz["tnswap_bid"] /= 10000
# yen has 100 as pips
data_by_tz["tnswap_ask"].loc["jpy",:,:] *= 100
data_by_tz["tnswap_bid"].loc["jpy",:,:] *= 100

# swap points: bid and ask --------------------------------------------------
usdxxx = ["cad","sek","dkk","chf","nok","jpy"]
bids = dict()
asks = dict()
for tz in data_by_tz["spot_bid"].minor_axis:
    # tz = "LON"
    ask_sp_y = data_by_tz["tnswap_ask"].loc[usdxxx,:,tz]
    bid_sp_y = data_by_tz["tnswap_bid"].loc[usdxxx,:,tz]
    ask_x_s = data_by_tz["spot_ask"].loc[usdxxx,:,tz]
    bid_x_s = data_by_tz["spot_bid"].loc[usdxxx,:,tz]

    # not cross
    this_ask_direct = 1/(bid_x_s+bid_sp_y) - 1/bid_x_s
    this_bid_direct = 1/(ask_x_s+ask_sp_y) - 1/ask_x_s

    this_tnswap_ask_full = pd.concat((
        data_by_tz["tnswap_ask"].loc[:,:,tz].drop(usdxxx, axis=1),
        this_ask_direct), axis=1)
    this_tnswap_bid_full = pd.concat((
        data_by_tz["tnswap_bid"].loc[:,:,tz].drop(usdxxx, axis=1),
        this_bid_direct), axis=1)

    asks[tz] = this_tnswap_ask_full
    bids[tz] = this_tnswap_bid_full

    # some spots now need to be reversed
    new_spot_asks = 1/data_by_tz["spot_bid"].loc[usdxxx,:,tz]
    new_spot_bids = 1/data_by_tz["spot_ask"].loc[usdxxx,:,tz]
    new_spot_mids = 1/data_by_tz["spot_mid"].loc[usdxxx,:,tz]

    data_by_tz["spot_ask"].loc[usdxxx,:,tz] = new_spot_asks
    data_by_tz["spot_bid"].loc[usdxxx,:,tz] = new_spot_bids
    data_by_tz["spot_mid"].loc[usdxxx,:,tz] = new_spot_mids


data_by_tz["tnswap_ask"] = pd.Panel.from_dict(asks, orient="minor")
data_by_tz["tnswap_bid"] = pd.Panel.from_dict(bids, orient="minor")

# save this better version
with open(name_data_by_tz_sp_fixed, "wb") as fname:
    pickle.dump(data_by_tz, fname)

# organize according to the time zone a currency belongs to -----------------
currencies = data_by_tz["spot_bid"].items
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

idx = (data_by_tz_aligned["tnswap_ask"] - data_by_tz_aligned["tnswap_bid"]) < 0
data_by_tz_aligned["tnswap_bid"] = data_by_tz_aligned["tnswap_bid"].mask(
    idx, data_by_tz_aligned["tnswap_ask"])

# pickle (for the greater justment) -----------------------------------------
with open(name_data_by_tz_aligned, "wb") as fname:
    pickle.dump(data_by_tz_aligned, fname)

# with open(data_path+"overnight_rates.p", "rb") as fname:
#     rf = pickle.load(fname)
