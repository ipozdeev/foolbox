import pandas as pd
import numpy as np
from foolbox.api import *

from foolbox.wp_ois.wp_settings import central_banks_start_dates, end_date,\
    cb_fx_map

fx_cb_map = dict((fx, cb) for cb,fx in cb_fx_map.items())

cur = "usd"
cb = "fomc"

lags = np.arange(1, 11)

if __name__ == "__main__":
    # data ------------------------------------------------------------------
    # events
    with open(data_path + "ois_project_events.p", mode="rb") as hangar:
        events_data = pickle.load(hangar)

    events = events_data["joint_cbs"].astype(float)

    # overnight rate
    with open(data_path + "ois_bloomberg.p", mode="rb") as hangar:
        ois_data = pickle.load(hangar)

    on_rates = pd.concat(
        [p.loc[:, "ON"].to_frame(c) for c, p in ois_data.items()],
        axis=1).astype(float)

    vuss = dict()

    for c in events.columns:
        if c == "nok":
            continue

        s_dt = central_banks_start_dates[fx_cb_map[c]]

        this_evt = events.loc[s_dt:end_date, c].dropna()
        this_on = on_rates.loc[s_dt:end_date, c]

        # PolicyExpectation instance
        pe_ois = PolicyExpectation.from_pickles(data_path, c,
            impl_rates_pickle="implied_rates_from_1m.p")

        pe_ois.rate_expectation = pe_ois.rate_expectation.loc[s_dt:end_date]

        r_until_ois = OIS.from_iso(c,
            maturity=DateOffset(months=1)).get_rates_until(this_on, this_evt,
                method="g_average")

        # loop over lags, calculate VUS
        this_vus = pd.Series(index=lags)

        for p in lags:
            this_vus.loc[p] = pe_ois.get_vus(lag=p, ref_rate=r_until_ois)

        vuss[c] = this_vus

    # plus fff --------------------------------------------------------------
    s_dt_ois = pe_ois.rate_expectation.dropna().index[0]
    pe_fff = PolicyExpectation.from_pickles(data_path, c,
        impl_rates_pickle="implied_rates_ffut.p")
    pe_fff.rate_expectation = pe_fff.rate_expectation.loc[s_dt_ois:end_date]
    r_until_fff = OIS.from_iso(c,
        maturity=DateOffset(months=1)).get_rates_until(this_on, this_evt,
            method="a_average")

    this_vus = pd.Series(index=lags)

    for p in lags:
        this_vus.loc[p] = pe_fff.get_vus(lag=p, ref_rate=r_until_fff)

    vuss.update({"usd (ois)": vuss["usd"]})
    vuss.pop("usd")

    vuss_out = pd.DataFrame.from_dict(vuss)

    vuss_out.loc[:, "usd (fff)"] = this_vus

    vuss_out.columns = [p.upper() for p in vuss_out.columns]

    out_path = set_credentials.set_path("../projects/ois/tex/tabs/",
        which="local")
    vuss_out.to_latex(buf=out_path+"tab_vus.tex",
        float_format="{:5.4f}".format,
        column_format="l"+"W"*len(vuss_out.columns))

    # with open(data_path + "overnight_rates.p", mode='wb') as hangar:
    #     pickle.dump(on_rates, hangar)
