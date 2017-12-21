import pandas as pd
from foolbox.api import *
from foolbox.utils import *
from wp_ois.wp_settings import ois_start_dates, end_date

out_path = set_credentials.set_path("ois/", which="local")
# out_path = set_credentials.set_path("", which="local")

if __name__ == "__main__":

    order = ["1w", "2w", "1m", "3m", "6m", "9m", "1y"]

    with open(data_path + "ois_rx_w_day_count.p", mode="rb") as hangar:
        risk_premium_all = pd.read_pickle(data_path + "ois_rx_w_day_count.p")

    # analyze
    rp_descr = \
        {k: taf.descriptives(
            100 * v.loc[ois_start_dates[k]:end_date, order].astype(float), 1)
         for k, v in risk_premium_all.items()}

    rp_mu = {k: v.loc["mean", order] for k, v in rp_descr.items()}
    rp_se = {k: v.loc["se_mean", order] for k, v in rp_descr.items()}
    rp_std = {k: v.loc["std", order] for k, v in rp_descr.items()}

    # Median and quantiles
    rp_50 = {
        k: (100 * v.loc[ois_start_dates[k]:end_date, order].astype(float)) \
            .median() for k, v in risk_premium_all.items()
        }

    rp_95 = {
        k: (100 * v.loc[ois_start_dates[k]:end_date, order].astype(float)) \
            .quantile(0.95) for k, v in risk_premium_all.items()
        }

    rp_05 = {
        k: (100 * v.loc[ois_start_dates[k]:end_date, order].astype(float)) \
            .quantile(0.05) for k, v in risk_premium_all.items()
        }

    rp_mu = pd.DataFrame.from_dict(rp_mu)
    rp_se = pd.DataFrame.from_dict(rp_se)
    rp_std = pd.DataFrame.from_dict(rp_std)
    rp_50 = pd.DataFrame.from_dict(rp_50)
    rp_95 = pd.DataFrame.from_dict(rp_95)
    rp_05 = pd.DataFrame.from_dict(rp_05)

    fmt_coef = "{:3.2f}"
    fmt_tstat = "{:3.2f}"

    mu_fmt = rp_mu.applymap(fmt_coef.format)
    std_fmt = rp_std.applymap(fmt_coef.format)
    tstat_fmt = (rp_mu/rp_se).applymap(('('+fmt_tstat+')').format)
    rp_50_fmt = rp_50.applymap(fmt_coef.format)
    rp_05_fmt = rp_05.applymap(fmt_coef.format)
    rp_95_fmt = rp_95.applymap(fmt_coef.format)

    df_coef = pd.DataFrame(columns=rp_mu.columns)
    df_coef_idx = list()
    for row, values in rp_mu.iterrows():
        df_coef = df_coef.append(mu_fmt.loc[row])
        df_coef = df_coef.append(tstat_fmt.loc[row])
        df_coef = df_coef.append(std_fmt.loc[row])
        df_coef = df_coef.append(rp_05_fmt.loc[row])
        df_coef = df_coef.append(rp_50_fmt.loc[row])
        df_coef = df_coef.append(rp_95_fmt.loc[row])
        df_coef_idx.extend([row, " ", " ", " ", " ", " "])

    df_coef.index = df_coef_idx

    out = pd.concat([pd.Series(["mean", "(t)", "std.", "5%", "50%",
                                "95%"] * len(rp_mu),
                               index=df_coef.index), df_coef], axis=1)
    out.columns = ["{}"] + df_coef.columns.tolist()

    out.index = [k.upper() for k in out.index]

    out.to_latex(buf=out_path + "tex/tabs/tab_ois_risk_premium_upd.tex",
                 column_format="l" + "W" * len(rp_se.columns))


    # to_better_latex(
    #     df_coef=df_coef,
    #     df_tstat=rp_mu/rp_se,
    #     fmt_coef="{:3.2f}", fmt_tstat="{:3.2f}",
    #     buf=out_path+"tex/tabs/tab_ois_risk_premium_upd.tex",
    #     column_format="l"+"W"*len(rp_se.columns))
