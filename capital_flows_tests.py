import pandas as pd
import numpy as np
from foolbox import tables_and_figures as taf
from foolbox.data_mgmt import set_credentials as set_cred
from foolbox.wp_tabs_figs.wp_settings import *


if __name__ == "__main__":
    # Settings ----------------------------------------------------------------
    path_to_data = set_cred.set_path("research_data/fx_and_events/",
                                     which="gdrive")

    s_dt = pd.to_datetime(settings["sample_start"])
    e_dt = pd.to_datetime(settings["sample_end"])

    # Regress returns on capital flows for these currenices
    currs = ["aud", "cad", "chf", "eur", "gbp", "nzd", "sek", "usd"]
    drop_curs = ["jpy", "dkk", "nok"]  # ignore these

    # Flows data  -------------------------------------------------------------
    tic = pd.read_pickle(path_to_data+"tic_flows_bonds_stocks.p")

    # Compute total in-/outflow of assets in the US convert to billion USD
    tic_bonds = (tic["in_total_bonds"] - tic["out_total_bonds"]) / 1e3
    tic_stocks = (tic["in_total_stocks"] - tic["out_total_stocks"]) / 1e3

    # Events data -------------------------------------------------------------
    events_data = pd.read_pickle(path_to_data + settings["events_data"])
    events = events_data["joint_cbs"].drop(drop_curs + ["usd"], axis=1,
                                           errors="ignore")
    events = events.loc[s_dt:e_dt]

    # Saga returns at individual currency level--------------------------------
    saga_indi = pd.read_pickle(path_to_data+"tmp_saga_individual.p")

    # Invert pre-fomc usd strategy to comply with in/outflows for other currs
    saga_indi["usd"] = 1/saga_indi["usd"]

    # Run pooled OLS regressions ----------------------------------------------
    these_bonds = tic_bonds[currs] #+ tic_stocks[curr]
    these_stocks = tic_stocks[currs]
    this_ret = np.log(saga_indi[currs]).diff() \
        .resample("M").sum().replace(0, np.nan) * 1e4  # bps

    y = this_ret
    x = these_bonds
    x_lag = x.shift(1)

    # US vs world is shared by every currency
    x_us = pd.concat([tic_bonds["usd"] for col in these_bonds.columns], 1)
    x_us.columns = these_bonds.columns
    x_us_lag = x_us.shift(1)

    # Stack
    y = y.stack().to_frame("y")
    x = x.stack().to_frame("x")
    x_lag = x_lag.stack().to_frame("x_lag")
    x_us = x_us.stack().to_frame("x_usd")
    x_us_lag = x_us_lag.stack().to_frame("x_usd")

    # Align
    x = x.reindex(y.index)
    x_lag = x_lag.reindex(y.index)
    x_us = x_us.reindex(y.index)
    x_us_lag = x_us_lag.reindex(y.index)

    # Uniquie values in index
    y.index = range(len(y))
    x.index = range(len(y))
    x_lag.index = range(len(y))
    x_us.index = range(len(y))
    x_us_lag.index = range(len(y))

    X = pd.concat([x, x_lag, x_us, x_us_lag], axis=1)
    X.columns = ["x", "x_lag", "x_us", "x_us_lag"]

    X = pd.concat([x], axis=1)
    X.columns = ["x"]

    res = taf.ts_ap_tests(y, X, 1)
    print(res)

    print("kek")
