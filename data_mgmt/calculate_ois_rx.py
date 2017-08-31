import pandas as pd
from foolbox.api import *
from foolbox.wp_tabs_figs.wp_settings import *


if __name__ == "__main__":

    maturity_to_offset = {
        "1W" : DateOffset(weeks=1),
        "2W" : DateOffset(weeks=2),
        "1M" : DateOffset(months=1),
        "3M" : DateOffset(months=3),
        "6M" : DateOffset(months=6),
        "9M" : DateOffset(months=9),
        "1Y" : DateOffset(years=1)
        }

    order = ["1W","2W","1M","3M","6M","9M","1Y"]

    # data ------------------------------------------------------------------
    # ois data
    with open(data_path + "ois_bloomberg.p", mode="rb") as hangar:
        ois_data = pickle.load(hangar)

    # overnight rates
    on_rates = pd.concat(
        [p.loc[:, "ON"].to_frame(c) for c, p in ois_data.items()],
        axis=1).astype(float)

    risk_premium_all = dict()

    # loop over currencies
    for c in on_rates.columns:
        if c == "usd":
            continue
        # c = "usd"
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

    with open(data_path + "ois_rx_w_day_count.p", mode="wb") as fname:
        pickle.dump(risk_premium_all, fname)


    # old -------------------------------------------------------------------
    # with open(data_path + "ois_bloomberg.p", mode="rb") as hangar:
    #     ois_data = pickle.load(hangar)
    #
    # # Value date, day count convention and fixing lag settings. Note that US
    # # fixing lag is set to 0, since the overnight data (effr) is already lagged
    # # by the New York Fed
    # all_settings = {
    #     "aud": {"start_date": 1,
    #             "fixing_lag": 0,
    #             "day_count_float": 365,
    #             "date_rolling": "modified following"},
    #
    #     "cad": {"start_date": 0,
    #             "fixing_lag": 1,
    #             "day_count_float": 365,
    #             "date_rolling": "modified following"},
    #
    #     "chf": {"start_date": 2,
    #             "fixing_lag": -1,
    #             "day_count_float": 360,
    #             "date_rolling": "modified following"},
    #
    #     "eur": {"start_date": 2,
    #             "fixing_lag": 0,
    #             "day_count_float": 360,
    #             "date_rolling": "modified following"},
    #
    #     "gbp": {"start_date": 0,
    #             "fixing_lag": 0,
    #             "day_count_float": 365,
    #             "date_rolling": "modified following"},
    #
    #     "jpy": {"start_date": 2,
    #             "fixing_lag": 1,
    #             "day_count_float": 365,
    #             "date_rolling": "modified following"},
    #
    #     "nzd": {"start_date": 2,
    #             "fixing_lag": 0,
    #             "day_count_float": 365,
    #             "date_rolling": "modified following"},
    #
    #     "sek": {"start_date": 2,
    #             "fixing_lag": -1,
    #             "day_count_float": 360,
    #             "date_rolling": "modified following"},
    #
    #     "usd": {"start_date": 2,
    #             "fixing_lag": 0,
    #             "day_count_float": 360,
    #             "date_rolling": "modified following"}
    #     }
    #
    # maturity_to_offset = {
    #     "1W" : DateOffset(weeks=1),
    #     "2W" : DateOffset(weeks=2),
    #     "1M" : DateOffset(months=1),
    #     "3M" : DateOffset(months=3),
    #     "6M" : DateOffset(months=6),
    #     "9M" : DateOffset(months=9),
    #     "1Y" : DateOffset(years=1)
    #     }
    #
    # order = ["1W","2W","1M","3M","6M","9M","1Y"]
    #
    # # test_currencies = ["aud", "cad", "chf", "eur", "gbp", "nzd", "sek", "usd"]
    #
    # risk_premium_all = dict()
    #
    # # loop over currencies
    # for c, c_settings in all_settings.items():
    #     if c in (list(risk_premium_all.keys()) + ["jpy"]):
    #         continue
    #
    #     # space for risk premium of the following maturity
    #     risk_premium_mat = dict()
    #
    #     # Select the data
    #     test_data = ois_data[c]
    #
    #     # Return of the underlying floating rate
    #     returns = test_data.pop("ON").dropna()
    #
    #     for m, m_offset in maturity_to_offset.items():
    #
    #         # Get the inputs for return estimation: trade dates first
    #         trade_dates = test_data[m].dropna().index
    #
    #         # Compute returns and periods over which the returns realized
    #         realized_ret_data = compute_floating_leg_return(
    #             trade_dates, returns, m_offset, c_settings)
    #
    #         # Concatenate the results: fixed, start and end dates, realized
    #         res = pd.concat([test_data[m], realized_ret_data["dates"],
    #                          realized_ret_data["ret"]], join="inner", axis=1)
    #         res.columns = ["fixed", "start", "end", "realized"]
    #
    #         # Plot fixed vs realized float
    #         risk_premium_mat[m] = res["realized"] - res["fixed"]
    #
    #     risk_premium_all[c] = pd.DataFrame.from_dict(risk_premium_mat).dropna(
    #         how="all")
    #
    # with open(data_path + "ois_risk_premia.p", mode="wb") as hangar:
    #     pickle.dump(risk_premium_all, hangar)
