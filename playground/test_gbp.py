from foolbox.api import *
import pickle


with open(data_path+"data_dev_d.p", mode="rb") as fname:
    data_d = pickle.load(fname)
with open(data_path+"data_dev_m.p", mode="rb") as fname:
    data_m = pickle.load(fname)
hfq_returns = data_d["spot_ret"]
portfolios = poco.rank_sort(data_m["spot_ret"], data_m["fwd_disc"].shift(1))

out = poco.upsample_portfolios(portfolios, hfq_returns)

lf = poco.get_factor_portfolios(portfolios)
hf = poco.get_factor_portfolios(out)

# Import the data
with open(data_path+"data_dev_d.p", mode="rb") as fname:
    data = pickle.load(fname)

with open(data_path+"events.p", mode='rb') as fname:
    events = pickle.load(fname)

s_d = data["spot_ret"]

fomc = events["fomc"].squeeze()
boe = events["boe"].squeeze()

# Generate daily values for rates set by BoE and the Fed
boe_span = boe.resample("D").ffill().to_frame()
fomc_span = fomc.resample("D").ffill().to_frame()

# Align the rates with spot returns
boe_span = boe_span.reindex(s_d.index).fillna(method="ffill")
fomc_span = fomc_span.reindex(s_d.index).fillna(method="ffill")

# Get the daily interest rate differential
id = boe_span - fomc_span

# Get the changes in interest rates across the two countries
policy_shifts = id.diff().where(id.diff() < 0).dropna()
policy_shifts = id.diff().where(id.diff() > 0).dropna()
policy_shifts = id.diff().where(np.abs(id.diff()) > 0).dropna()

# Brew the aspic of the indigeneous people of Eastern Sibeiria
evenk = EventStudy(s_d.mean(axis=1)*100, policy_shifts, [-10, -1, 0, 10])
evenk.get_ci(ps=0.9, method="simple")
evenk.plot()


with open(data_path+"events.p", mode="rb") as fname:
    evts = pickle.load(fname)





# Import the data
with open(data_path+"data_dev_d.p", mode="rb") as fname:
    data = pickle.load(fname)

with open(data_path+"data_dev_m.p", mode="rb") as fname:
    data_m = pickle.load(fname)

with open(data_path+"events.p", mode='rb') as fname:
    events = pickle.load(fname)

#
s_d = data["spot_ret"]
f_d = data["fwd_disc"]

rx = data_m["rx"]
s_d = data["spot_ret"]
fomc = events["fomc"].squeeze()
boe = events["boe"].squeeze()

boe_span = boe.resample("D").ffill().to_frame()
fomc_span = fomc.resample("D").ffill().to_frame()

boe_span = boe_span.reindex(s_d.index).fillna(method="ffill")
fomc_span = fomc_span.reindex(s_d.index).fillna(method="ffill")
id = boe_span - fomc_span
id.columns = ["gbp"]

shifts = id.diff().where(np.abs(id.diff())>0).dropna()

evenk = EventStudy(s_d.gbp, shifts, [-10, -1, 1, 10])
evenk.get_ci(ps=0.9, method="simple")
evenk.plot()

signals = (f_d.rolling(22).mean()-f_d.rolling(66).mean()).rolling(22).max().shift(1)
#signals = (np.abs(f_d.diff()).rolling(130).mean())
signal_m = signals.resample("M").last().shift(1)

signals = (f_d.rolling(22).mean()-f_d.rolling(66).mean()).rolling(22).max().shift(1)
#signals = (f_d).shift(1)


test = poco.rank_sort_adv(s_d["1998":], signals["1998":], 5, 22)

pp = poco.get_factor_portfolios(test, hml=True)
pp.hml.cumsum().plot()


