import pandas as pd

from foolbox.api import *
from foolbox.utils import *

out_path = set_credentials.set_path("../projects/ois/", which="local")

if __name__ == "__main__":

    with open(data_path + "ois_bloomberg.p", mode="rb") as hangar:
        ois_data = pickle.load(hangar)

    # Value date, day count convention and fixing lag settings. Note that US
    # fixing lag is set to 0, since the overnight data (effr) is already lagged
    # by the New York Fed
    all_settings = {
        "aud": {"start_date": 1,
                "fixing_lag": 0,
                "day_count_float": 365,
                "date_rolling": "modified following"},

        "cad": {"start_date": 0,
                "fixing_lag": 1,
                "day_count_float": 365,
                "date_rolling": "modified following"},

        "chf": {"start_date": 2,
                "fixing_lag": -1,
                "day_count_float": 360,
                "date_rolling": "modified following"},

        "eur": {"start_date": 2,
                "fixing_lag": 0,
                "day_count_float": 360,
                "date_rolling": "modified following"},

        "gbp": {"start_date": 0,
                "fixing_lag": 0,
                "day_count_float": 365,
                "date_rolling": "modified following"},

        "jpy": {"start_date": 2,
                "fixing_lag": 1,
                "day_count_float": 365,
                "date_rolling": "modified following"},

        "nzd": {"start_date": 2,
                "fixing_lag": 0,
                "day_count_float": 365,
                "date_rolling": "modified following"},

        "sek": {"start_date": 2,
                "fixing_lag": -1,
                "day_count_float": 360,
                "date_rolling": "modified following"},

        "usd": {"start_date": 2,
                "fixing_lag": 0,
                "day_count_float": 360,
                "date_rolling": "modified following"}
        }

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

    # test_currencies = ["aud", "cad", "chf", "eur", "gbp", "nzd", "sek", "usd"]

    risk_premium_all = dict()

    # loop over currencies
    for c, c_settings in all_settings.items():
        if c in (list(risk_premium_all.keys()) + ["jpy"]):
            continue

        # space for risk premium of the following maturity
        risk_premium_mat = dict()

        # Select the data
        test_data = ois_data[c]

        # Return of the underlying floating rate
        returns = test_data.pop("ON").dropna()

        for m, m_offset in maturity_to_offset.items():

            # Get the inputs for return estimation: trade dates first
            trade_dates = test_data[m].dropna().index

            # Compute returns and periods over which the returns realized
            realized_ret_data = compute_floating_leg_return(
                trade_dates, returns, m_offset, c_settings)

            # Concatenate the results: fixed, start and end dates, realized
            res = pd.concat([test_data[m], realized_ret_data["dates"],
                             realized_ret_data["ret"]], join="inner", axis=1)
            res.columns = ["fixed", "start", "end", "realized"]

            # Plot fixed vs realized float
            risk_premium_mat[m] = res["realized"] - res["fixed"]

        risk_premium_all[c] = pd.DataFrame.from_dict(risk_premium_mat).dropna(
            how="all")

    with open(data_path + "ois_risk_premia.p", mode="wb") as hangar:
        pickle.dump(risk_premium_all, hangar)

    with open(data_path + "ois_risk_premia.p", mode="rb") as hangar:
        risk_premium_all = pickle.load(hangar)

    # analyze
    rp_descr = {k: taf.descriptives(
        v.astype(float), 1) for k, v in risk_premium_all.items()}

    rp_mu = {k: v.loc["mean"] for k, v in rp_descr.items()}
    rp_se = {k: v.loc["se_mean"] for k, v in rp_descr.items()}

    rp_mu = pd.DataFrame.from_dict(rp_mu)*100
    rp_se = pd.DataFrame.from_dict(rp_se)*100

    to_better_latex(
        df_coef=rp_mu.loc[order,:],
        df_tstat=rp_mu.loc[order,:]/rp_se.loc[order,:],
        fmt_coef="{:3.2f}", fmt_tstat="{:3.2f}",
        buf=out_path+"tex/tabs/tab_ois_risk_premium.tex",
        column_format="l"+"W"*len(rp_se.columns))

    ois_data = pd.Panel.from_dict(ois_data, orient="minor")
    ois_data = ois_data.loc[order,:,:]
    ois_data.dropna(axis="major_axis", how="all")

    taus = []
    for p in ois_data.minor_axis:
        for q in ois_data.items:
            taus += [np.datetime64(ois_data.loc[q,:,p].first_valid_index())]

    np.median(np.array(taus))

    [np.datetime64(p) for p in taus]
