import pandas as pd
from foolbox.api import *

def calculate_implied_rates(ois_data, on_data, events_data):
    """
    """
    # space for implied rates
    implied_rates = dict()

    for c in ois_data.keys():
        # c = "gbp"
        print(c)

        # OIS
        ois = OIS.from_iso(c, maturity=DateOffset(months=1))

        # this event, to frame + rename
        this_evt = events_data.loc[:, c].dropna().to_frame("rate_change")

        # this ois
        this_ois = ois_data.loc[:, c].astype(float)

        # this overnight
        this_on = on_data.loc[:, c].astype(float)

        # align
        this_ois, this_on = this_ois.loc[
            this_ois.first_valid_index():this_ois.last_valid_index()].align(
                this_on, axis=0, join="left")

        # rates expected to prevale before meetings
        rates_until = ois.get_rates_until(this_on, this_evt,
            method="average")

        pe = PolicyExpectation.from_ois_new(ois=ois,
            meetings=this_evt,
            ois_rates=this_ois,
            on_rates=this_on)

        # store
        this_ir = pe.rate_expectation.copy()
        this_ir.name = c
        implied_rates[c] = pe.rate_expectation

    implied_rates = pd.DataFrame.from_dict(implied_rates)

    with open(data_path + "implied_rates_from_1m.p", mode="wb") as hangar:
        pickle.dump(implied_rates, hangar)


if __name__ == "__main__":
    # fetch data ------------------------------------------------------------
    # ois data
    with open(data_path + "ois_merged_1m.p", mode="rb") as hangar:
        ois_data = pickle.load(hangar)

    # overnight rates data
    with open(data_path + "ois_bloomberg.p", mode="rb") as hangar:
        on_data = pickle.load(hangar)

    on_data = pd.concat(
        [p.loc[:, "ON"].to_frame(c) for c, p in on_data.items()],
        axis=1)

    # events
    with open(data_path + "events.p", mode="rb") as hangar:
        events_data = pickle.load(hangar)

    events_data = events_data["joint_cbs"]

    calculate_implied_rates(ois_data, on_data, events_data)

    with open(data_path + "implied_rates_from_1m.p", mode="rb") as hangar:
        implied_rates = pickle.load(hangar)


# # data --------------------------------------------------------------------
# data_path = set_credentials.gdrive_path("research_data/fx_and_events/")
#
# # ois
# with open(data_path + "ois.p", mode='rb') as fname:
#     ois_data = pickle.load(fname)
#
# # events
# with open(data_path + "events.p", mode='rb') as fname:
#     meetings = pickle.load(fname)
#
# meetings_level = meetings["joint_cbs_lvl"]
# meetings_change = meetings["joint_cbs"]
#
# # reference rates
# with open(data_path + "overnight_rates.p", mode='rb') as fname:
#     overnight_rates = pickle.load(fname)
#
# # merge two ois datasets
# ois_icap = ois_data["icap_1m"]
# ois_tr = ois_data["tr_1m"]
#
# ois_icap, ois_tr = ois_icap.align(ois_tr, join="outer")
# ois_mrg = ois_icap.fillna(ois_tr)
# ois = ois_mrg.dropna(how="all")
#
# # parameters ----------------------------------------------------------------
# drop_curs = ["jpy", "dkk"]
#
# instr = ois.drop(drop_curs, axis=1, errors="ignore")
# refrce = overnight_rates.drop(drop_curs, axis=1, errors="ignore")
#
# # space for output
# impl_rates = instr*np.nan
#
# # loop over currencies ------------------------------------------------------
# for c in impl_rates.columns:
#     # c = "aud"
#
#     # concat level and change -----------------------------------------------
#     this_evt_lvl = meetings_level.loc[:,c]
#     this_evt_chg = meetings_change.loc[:,c]
#     this_evt = pd.concat((this_evt_lvl, this_evt_chg), axis=1).dropna(
#         how="all")
#     this_evt.columns = ["rate_level", "rate_change"]
#
#     # estimate
#     pe = PolicyExpectation.from_ois(
#         meetings=this_evt,
#         instrument=instr.loc[:,c],
#         reference_rate=refrce.loc[:,c],
#         tau=1)
#
#     # store
#     impl_rates.loc[:,c] = pe.rate_expectation
#
#     # for usd, also calculate the rate implied in fed funds futures ---------
#     if c == "usd":
#         with open(data_path+"fed_funds_futures_settle.p", mode='rb') as fname:
#             ffut = pickle.load(fname)
#
#         pe = PolicyExpectation.from_funds_futures(this_evt, ffut)
#
#         # implied rates
#         r = pe.rate_expectation.dropna().to_frame()
#         r.columns = ["usd"]
#         with open(data_path + "implied_rates_ffut.p", mode='wb') as fname:
#             pickle.dump(r, fname)
#
# # drop na and save ----------------------------------------------------------
# impl_rates = impl_rates.dropna(how="all")
#
# with open(data_path + "implied_rates.p", mode='wb') as fname:
#     pickle.dump(impl_rates, fname)


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
