from foolbox.api import *
import pandas as pd
from foolbox.RegressionModel import DynamicOLS
import ipdb

if __name__ == "__main__":

    # data ------------------------------------------------------------------
    # overniht rates
    with open(data_path + "ois_bloomberg.p", mode="rb") as hangar:
        ois_data = pickle.load(hangar)

    on_rates = pd.concat(
        [p.loc[:, "ON"].to_frame(c) for c, p in ois_data.items()],
        axis=1)

    # policy rates
    with open(data_path + "events.p", mode="rb") as hangar:
        events_data = pickle.load(hangar)

    tgt_rate_changes = events_data["joint_cbs"]

    # lag settings
    all_settings = {
        "aud": 1,
        "cad": 0,
        "chf": 0,
        "eur": 1,
        "gbp": 1,
        "nzd": 0,
        "sek": 1,
        "usd": 1
        }

    # rolling regression ----------------------------------------------------
    roll_betas = dict()

    for c, lag in all_settings.items():
        # c = "usd"
        # lag = 1
        this_on_rate_change = on_rates.loc[:, c].diff()
        this_tgt_rate_change = tgt_rate_changes.loc[:, c].dropna()

        y0 = this_on_rate_change.copy()
        # y0.name = "on"
        x0 = this_tgt_rate_change.shift(-lag)
        # x0.name = "tgt"
        ipdb.set_trace()
        dynamic_mod = DynamicOLS("expanding",
            y0=y0,
            x0=x0,
            min_periods=10)
        this_b = dynamic_mod.fit()

        lol = pd.concat((y0,x0), axis=1).expanding(min_periods=10).cov()
        lol.loc[:,"on","tgt"]
