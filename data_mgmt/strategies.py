# strategies
import pandas as pd
from foolbox.api import *
import pandas_datareader.data as web
import re

all_strategies = dict()
n_portf=2

# data ----------------------------------------------------------------------
with open(path_to_spot+"data_wmr_dev_m.p", mode='rb') as fname:
    data_m = pickle.load(fname)
with open(path_to_spot+"data_wmr_dev_d.p", mode='rb') as fname:
    data_d = pickle.load(fname)

s_m = data_m["spot_ret"]
rx_m = data_m["rx"]
fdisc_m = data_m["fwd_disc"]

s_d = data_d["spot_ret"]
fdisc_d = data_d["fwd_disc"]
fdisc_dm = fdisc_d.resample('M').mean()

# dollar index
dol_s = s_m.mean(axis=1)
dol_rx_m = rx_m.mean(axis=1)

# libor
usd_libor = web.DataReader("USD1MTD156N", "fred", fdisc_m.index[0])
usd_libor = usd_libor.reindex(index=fdisc_m.index, method="ffill")
usd_libor /= (100*12)
usd_libor = usd_libor.squeeze()

# carry ---------------------------------------------------------------------
sig_carry_1 = fdisc_m.shift(1)
sig_carry_2 = fdisc_dm.shift(1)
carry_1 = poco.get_hml(rx_m, sig_carry_1, n_portf=n_portf)
carry_2 = poco.get_hml(rx_m, sig_carry_2, n_portf=n_portf)

carry_1.loc["2008-08-31"] = 0.0
carry_2.loc["2008-08-31"] = 0.0

carry_1.loc["2008-08-31":].cumsum().plot()
carry_2.loc["2008-08-31":].cumsum().plot()

all_strategies["CARRY"] = {
    "last": carry_1,
    "mean": carry_2}

# dollar carry --------------------------------------------------------------
avg_fdisc_m = fdisc_m.mean(axis=1)
avg_fdisc_dm = fdisc_dm.mean(axis=1)

sig_dol_carry_1 = (avg_fdisc_m.ge(usd_libor)*2-1).shift(1)
sig_dol_carry_2 = (avg_fdisc_dm.ge(usd_libor)*2-1).shift(1)

dol_carry_1 = dol_rx_m.multiply(sig_dol_carry_1)
dol_carry_2 = dol_rx_m.multiply(sig_dol_carry_2)

dol_carry_1.loc["2008-08-31"] = 0.0
dol_carry_2.loc["2008-08-31"] = 0.0

dol_carry_1.loc["2008-08-31":].cumsum().plot()
dol_carry_2.loc["2008-08-31":].cumsum().plot()

all_strategies["DOLCARRY"] = {
    "_nolabel": dol_carry_1}

# momentum ------------------------------------------------------------------
sig_mom_1 = s_m.shift(1)
sig_mom_2 = s_m.rolling(3).sum().shift(1)
sig_mom_3 = s_m.rolling(11).sum().shift(2)

mom_1 = poco.get_hml(rx_m, sig_mom_1, n_portf=n_portf)
mom_2 = poco.get_hml(rx_m, sig_mom_2, n_portf=n_portf)
mom_3 = poco.get_hml(rx_m, sig_mom_3, n_portf=n_portf)

mom_1.loc["2008-08-31"] = 0.0
mom_2.loc["2008-08-31"] = 0.0
mom_3.loc["2008-08-31"] = 0.0

mom_1.loc["2008-08-31":].cumsum().plot()
mom_2.loc["2008-08-31":].cumsum().plot()
mom_3.loc["2008-08-31":].cumsum().plot()

all_strategies["MOM"] = {
    "1-1": mom_1,
    "3-1": mom_2,
    "11-2": mom_3}

# variance rp ---------------------------------------------------------------
path = set_credentials.gdrive_path("option_implied_betas_project/")
path_to_spot = set_credentials.gdrive_path("research_data/fx_and_events/")

path_to_raw = path+"data/raw/longer/"
path_to_data = path+"data/estimates/"
tau_str = "1m"
opt_meth = "mfiv"

BImpl = ImpliedBetaEnvironment(
    path_to_raw=path_to_raw,
    path_to_data=path_to_data,
    tau_str=tau_str,
    opt_meth=opt_meth)

v = BImpl._fetch_from_hdf("variances")
v = v.loc[:,[p for p in v.columns if "usd" in p]].resample('M').last()
v.columns = [re.sub("usd",'',p) for p in v.columns]
iv = np.sqrt(v.loc["2008-07":,:])

rv = np.sqrt((s_m**2).groupby(pd.TimeGrouper(freq='M')).sum())
vrp = (rv.shift(1) - iv)*100

var_strat = poco.get_hml(rx_m, vrp.shift(1), n_portf=n_portf)
var_strat.loc["2008-08-31"] = 0.0

var_strat.loc["2008-08":].cumsum().plot()

all_strategies["VRP"] = {
    "_nolabel": var_strat}


#
with open(data_path+"strategies.p", mode="wb") as hangar:
    pickle.dump(all_strategies, hangar)
