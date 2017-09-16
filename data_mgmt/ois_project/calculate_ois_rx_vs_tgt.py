import pandas as pd
from foolbox.api import *
from foolbox.wp_tabs_figs.wp_settings import *


if __name__ == "__main__":

    maturity_to_offset = {
        "1W": DateOffset(weeks=1),
        "2W": DateOffset(weeks=2),
        "1M": DateOffset(months=1),
        "3M": DateOffset(months=3),
        "6M": DateOffset(months=6),
        "9M": DateOffset(months=9),
        "1Y": DateOffset(years=1)
        }

    order = ["1W", "2W", "1M", "3M", "6M", "9M", "1Y"]

    # data ------------------------------------------------------------------
    # OIS data
    with open(data_path + "ois_bloomberg.p", mode="rb") as hangar:
        ois_data = pickle.load(hangar)

    # Target rates data
    with open(data_path + "ois_project_events.p", mode="rb") as hangar:
        events_data = pickle.load(hangar)

    tgt_rates = events_data["joint_cbs_plus_unscheduled_lvl_eff"]

    # Get the overnight rates
    on_rates = pd.concat(
        [p.loc[:, "ON"].to_frame(c) for c, p in ois_data.items()],
        axis=1).astype(float)

    # Reindex the target rates and substitute them for overnight rates
    tgt_rates = \
        tgt_rates.reindex(on_rates.index).ffill().bfill().drop(["nok"], axis=1)
    on_rates = tgt_rates

    risk_premium_all = dict()

    # loop over currencies
    for c in on_rates.columns:
        print(c)
        # if c in ["usd", "nzd", "cad", "chf", "aud"]:
        #     continue
        this_ois_data = ois_data[c].astype(float)
        this_on_rate = on_rates[c]

        # space for implied rates
        this_rp = dict()

        for m, m_offset in maturity_to_offset.items():
            # m = "3M"
            print(m)

            this_ois_quotes = this_ois_data[m]

            # OIS
            ois = OIS.from_iso(c, maturity=m_offset)

            # space for data
            rx = this_ois_quotes.dropna() * np.nan

            for t in rx.index:
                ois.quote_dt = t

                # calculate return
                fwd_ret = ois.get_return_of_floating(this_on_rate)
                fwd_ret *= (ois.day_cnt_fix / ois.lifetime * 100)

                rx.loc[t] = this_ois_quotes.loc[t] - fwd_ret

            this_rp[m] = rx

        risk_premium_all[c] = pd.DataFrame.from_dict(this_rp).loc[:, order]

        # taf.descriptives(risk_premium_all[c].loc["2009-06":]*100, 1)

    with open(data_path + "ois_rx_tgt_day_count.p", mode="wb") as fname:
        pickle.dump(risk_premium_all, fname)
