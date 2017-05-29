import pandas as pd
from foolbox.api import *

# data --------------------------------------------------------------------
data_path = set_credentials.gdrive_path("research_data/fx_and_events/")

# ois
with open(data_path + "ois.p", mode='rb') as fname:
    ois_data = pickle.load(fname)

# events
with open(data_path + "events.p", mode='rb') as fname:
    meetings = pickle.load(fname)

meetings_level = meetings["joint_cbs_lvl"]
meetings_change = meetings["joint_cbs"]

# reference rates
with open(data_path + "overnight_rates.p", mode='rb') as fname:
    overnight_rates = pickle.load(fname)

# merge two ois datasets
ois_icap = ois_data["icap_1m"]
ois_tr = ois_data["tr_1m"]

ois_icap, ois_tr = ois_icap.align(ois_tr, join="outer")
ois_mrg = ois_icap.fillna(ois_tr)
ois = ois_mrg.dropna(how="all")

# parameters ----------------------------------------------------------------
drop_curs = ["jpy", "dkk"]

instr = ois.drop(drop_curs, axis=1, errors="ignore")
refrce = overnight_rates.drop(drop_curs, axis=1, errors="ignore")

# space for output
impl_rates = instr*np.nan

# loop over currencies ------------------------------------------------------
for c in impl_rates.columns:
    # c = "aud"

    # concat level and change -----------------------------------------------
    this_evt_lvl = meetings_level.loc[:,c]
    this_evt_chg = meetings_change.loc[:,c]
    this_evt = pd.concat((this_evt_lvl, this_evt_chg), axis=1).dropna(
        how="all")
    this_evt.columns = ["rate_level", "rate_change"]

    # estimate
    pe = PolicyExpectation.from_ois(
        meetings=this_evt,
        instrument=instr.loc[:,c],
        reference_rate=refrce.loc[:,c],
        tau=1)

    # store
    impl_rates.loc[:,c] = pe.rate_expectation

    # for usd, also calculate the rate implied in fed funds futures ---------
    if c == "usd":
        with open(data_path+"fed_funds_futures_settle.p", mode='rb') as fname:
            ffut = pickle.load(fname)

        pe = PolicyExpectation.from_funds_futures(this_evt, ffut)

        # implied rates
        r = pe.rate_expectation.dropna().to_frame()
        r.columns = ["usd"]
        with open(data_path + "implied_rates_ffut.p", mode='wb') as fname:
            pickle.dump(r, fname)

# drop na and save ----------------------------------------------------------
impl_rates = impl_rates.dropna(how="all")

with open(data_path + "implied_rates.p", mode='wb') as fname:
    pickle.dump(impl_rates, fname)


# ---------------------------------------------------------------------------
# with open(data_path + "fed_funds_futures_settle.p", mode='rb') as fname:
#     ffut = pickle.load(fname)
#
# pe = PolicyExpectation.from_funds_futures(this_evt, ffut)
# pe.rate_expectation.dropna()
# %matplotlib
# pe.roc_curve(lag=3, avg_impl_over=1)
# pe.assess_forecast_quality(lag=3, threshold=0.14)
# pe.meetings

#
# with open(data_path + "implied_rates.p", mode='rb') as fname:
#     impl_rates = pickle.load(fname)
# impl_rates.loc[:,"usd"] = pe.policy_exp.dropna()
# impl_rates, lol = impl_rates.align(pe.policy_exp.dropna(), axis=0,
#     join="outer")
#
# %matplotlib
# pe = PolicyExpectation.from_pickles(data_path, "usd", s_dt="2001")
# pe.rate_expectation.dropna()
# pe.forecast_policy_change()
# pe = PolicyExpectation.from_pickles(data_path, "usd", s_dt="2001-08",
#     use_ffut=True)
# pe.assess_forecast_quality(lag=3, threshold=0.125)
# lol = pe.rate_expectation.copy()
# pe.rate_expectation = lol.loc["2001":]
# pe.roc_curve(lag=3, avg_impl_over=1)
# %matplotlib
#
# pe.ts_plot(lag=5)
# pe.reference_rate.plot()
# pe.reference_rate.rolling(lag, min_periods=1).mean().loc["2001-11":]
#
# pe = PolicyExpectation.from_pickles(data_path, "sek", s_dt="2001")
# pe.assess_forecast_quality(lag=3, threshold=0.125, avg_impl_over=2,
#     avg_refrce_over=5)
# pe.roc_curve(avg_impl_over=None, avg_refrce_over=5, lag=3)
# events["joint_cbs"].count(axis=1).hist()
