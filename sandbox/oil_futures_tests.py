from foolbox.api import *


filename = data_path + "wti_futures_raw.p"
with open(filename, mode="rb") as fname:
    futures_raw = pickle.load(fname)


column_to_parse = "Settle"  # parses settlement prices

# Set range of futures to parse
start_date = "1993-01"
end_date = "2018-01"

# Dictionary providing a map between futures' names and months
naming_map = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6, "N": 7, "Q": 8, "U": 9,
    "V": 10, "X": 11, "Z": 12,
}

# Loop over the futures, appending the list to be further concatenated
out_list = []
for key in futures_raw.keys():
    # Get the expiry date as a string
    month = str(naming_map[key[0]])
    year = key[1:]
    # Offset to month end, corresponding to the actual expiry date
    exp_date = pd.Timestamp(year + "-" + month) + pd.offsets.MonthEnd()

    # Append the output to list
    tmp_df = futures_raw[key][[column_to_parse]]
    tmp_df.columns = [exp_date]
    out_list.append(tmp_df)

# Concatenate futures of different expiries, sort in chronological order
out = pd.concat(out_list, join="outer", axis=1)
out = out[sorted(out.columns)]


basis = list()
# Loop over months-years
for (y, m), df in out.groupby([out.index.year, out.index.month]):
    # Make a stamp as of the end of month
    tmp_stamp = pd.Timestamp(str(y) + "-" + str(m)) + pd.offsets.MonthEnd()
    # Skip the dates, preceding the expiry date of the first futures available
    if tmp_stamp < out.columns[0]:
        continue

    # Locate the month-year in the columns, forward it by TWO months:
    # last trading day of a December contract is usually 20-23 of November,
    # that means December is continuously covered by the February contract.
    # Sample the contacts for the next 12 months of maturity
    tmp_df = df.loc[:,
             tmp_stamp + pd.offsets.MonthEnd(2):tmp_stamp +
                                                pd.offsets.MonthEnd(12)]

    # Rename columns and append the output
    tmp_df.columns = ["M"+ str(2 + k) for k in range(len(tmp_df.columns))]
    basis.append(tmp_df)

# Concatenate, and get basis in percent
basis = pd.concat(basis, join="outer", axis=0)

# Drop the base month
basis = (basis.div(basis["M2"], axis=0) - 1).drop(["M2"], axis=1)*100




# window
wa,wb,wc,wd = -10,-1,1,5
window = (wa,wb,wc,wd)

s_dt = settings["sample_start"]
e_dt = settings["sample_end"]

s_dt = "1994-01-01"
e_dt = "2017-10-31"

out_path = set_credentials.gdrive_path("opec_meetings/tex/figs/")

# events + drop currencies ----------------------------------------------
with open(data_path + settings["events_data"], mode='rb') as fname:
    events_data = pickle.load(fname)

events = events_data["joint_cbs"]["usd"]
events = events_data["fomc"]["change"]
events = events.loc[s_dt:e_dt]
events = pd.concat([events for k in basis.columns], axis=1)
events.columns = basis.columns


data = np.abs(basis).copy().diff().loc[s_dt:e_dt]
#data = np.abs(basis).copy().loc[s_dt:e_dt]
# events = events_perf["nzd"].dropna()
test_events = events.copy().where(events < 0)

normal_data = 0.0#data.rolling(261).mean().shift(1)
es = EventStudy(data, test_events, window, mean_type="count_weighted",
    normal_data=normal_data, x_overlaps=True)

#es.get_ci(ps=(0.025, 0.975))
ci_boot_c = es.get_ci(ps=(0.025, 0.975), method="boot", n_blocks=10,M=13)
es.plot()


# roll contracts forward 10 days before the last trading day
roll_dates = out.apply(lambda x: x.dropna().index[-10])
rets = out.pct_change()
lol = [continuos_futures(rets, roll_dates, k) for k in range(7)]
lol = pd.concat(lol, axis=1).dropna()*100
#lol = lol.sub(lol["cont_0"], axis=0).drop(["cont_0"], axis=1)

with open(data_path + settings["events_data"], mode='rb') as fname:
    events_data = pickle.load(fname)


s_dt = "1995-01"
e_dt = "2017-06"

events = events_data["joint_cbs"]["usd"]
events = events_data["rba"]["change"]
events = events.loc[s_dt:e_dt]
events = pd.concat([events for k in lol.columns], axis=1)
events.columns = lol.columns


data = lol.loc[s_dt:e_dt]
test_events = events.copy().where(events > 0)

normal_data = 0.0#data.rolling(132).mean().shift(1)
es = EventStudy(data, test_events, window, mean_type="count_weighted",
    normal_data=normal_data, x_overlaps=True)

#es.get_ci(ps=(0.025, 0.975))
ci_boot_c = es.get_ci(ps=(0.025, 0.975), method="boot", n_blocks=10,M=10)
es.plot()

def continuos_futures(rets, roll_dates, depth=0):
    """Constructs a continuous futures returns, given a dataframe 'data' with
    returns and 'roll dates' specifying when the contract is to be rolled

    Parameters
    ----------
    rets: pd.DataFrame
        with each columns containing prices of individual futures contracts
        ordered chronologically by expiry date
    roll_dates: pd.Series
        of 'pd.Timestamp's which are a subset of the index of 'data' and
        ordered chronologically. Indexed by individual contracts
    depth: int


    Returns
    -------
    cont_futures

    """
    roll_df = \
        pd.concat([roll_dates.shift(1), roll_dates], axis=1).dropna()
    roll_df.columns = ["from", "to"]
    roll_df["contract"] = roll_df.index
    roll_df["contract"] = roll_df["contract"].shift(-depth)
    roll_df = roll_df.where(roll_df["from"] != roll_df["to"]).dropna()

    out_name = "cont_"+str(depth)
    cont_futures = \
        pd.DataFrame(index=rets.index, columns=[out_name], dtype="float")

    for index, roll_info in roll_df.iterrows():
        this_contract = roll_info["contract"]
        hold_from = roll_info["from"]
        hold_to = roll_info["to"]

        if index == roll_df.index[-1]:
            cont_futures.loc[hold_from:, out_name] = \
                rets.loc[hold_from:, this_contract]
        else:
            cont_futures.loc[hold_from:hold_to, out_name] = \
                rets.loc[hold_from:hold_to, this_contract]

    return cont_futures
