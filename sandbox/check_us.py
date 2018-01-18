"""Here dwell tests of the announcement-driven FX strategy for the US
"""
__author__ = "fluffykitten"

from foolbox.api import *

# Set the desired ois series, and signal cutoff threshold
ois_data = "icap_1m"
cutoff = 0.125

# Load datasets: exchange rates, events, and impled policy rates measures
with open(data_path+"data_wmr_dev_d.p", mode="rb") as fname:
    fx_data = pickle.load(fname)
with open(data_path+"events.p", mode="rb") as fname:
    events = pickle.load(fname)
with open(data_path+"fed_funds.p", mode="rb") as fname:
    fed_funds_data = pickle.load(fname)
with open(data_path + "policy_rates.p", mode="rb") as fname:
    p_rates = pickle.load(fname)

# Returns are 5-day excess returns on dollar factor. Get em'!
rx = fx_data["rx_5d"].mean(axis=1) * (-1)  # -1 because USD is bought at hikes
rx.name = "dol"

# Get the implied rates, and the corresponding signals
impl_fut = fed_funds_data["implied"]
impl_ois = p_rates[ois_data]["usd"]

# Get the events and the target rate
fomc = events["joint_cbs"]["usd"].dropna()
fomc_target = events["joint_cbs_lvl"]["usd"]\
    .reindex(rx.index).fillna(method="ffill")


# Get signals
signal_fomc = fomc.where(fomc != 0).dropna()
signal_fut = (impl_fut - fomc_target).shift(5).reindex(fomc.index)
signal_fut = signal_fut.where(abs(signal_fut) >= cutoff).dropna()
signal_ois = (impl_ois - fomc_target).shift(5).reindex(fomc.index)
signal_ois = signal_ois.where(abs(signal_ois) >= cutoff).dropna()

# Perfect foresight
perfect = poco.multiple_timing(rx, signal_fomc, xs_avg=False)

# Fed Funds Futures
fff = poco.multiple_timing(rx, signal_fut, xs_avg=False)

# OIS
ois = poco.multiple_timing(rx, signal_ois, xs_avg=False)


# Pooled
pooled = pd.concat([perfect, fff, ois], axis=1).dropna(how="all")
pooled.columns = ["perfect", "futures", "ois"]

print(taf.descriptives(pooled["2001-08-01":], scale=52))

pooled.replace(np.nan, 0)["2001-08-01":].cumsum().plot()
