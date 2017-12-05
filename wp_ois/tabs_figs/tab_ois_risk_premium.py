import pandas as pd
from foolbox.api import *
from foolbox.utils import *
from wp_ois.wp_settings import ois_start_dates, end_date

out_path = set_credentials.set_path("../ois/", which="local")
# out_path = set_credentials.set_path("", which="local")

if __name__ == "__main__":

    order = ["1W", "2W", "1M", "3M", "6M", "9M", "1Y"]

    with open(data_path + "ois_rx_w_day_count.p", mode="rb") as hangar:
        risk_premium_all = pd.read_pickle(data_path + "ois_rx_w_day_count.p")

    # analyze
    rp_descr = \
        {k: taf.descriptives(v[ois_start_dates[k]:end_date].astype(float), 1)
         for k, v in risk_premium_all.items()}

    rp_mu = {k: v.loc["mean", order] for k, v in rp_descr.items()}
    rp_se = {k: v.loc["se_mean", order] for k, v in rp_descr.items()}

    rp_mu = pd.DataFrame.from_dict(rp_mu)*100
    rp_se = pd.DataFrame.from_dict(rp_se)*100
    np.abs(rp_mu/rp_se)

    to_better_latex(
        df_coef=rp_mu,
        df_tstat=rp_mu/rp_se,
        fmt_coef="{:3.2f}", fmt_tstat="{:3.2f}",
        buf=out_path+"tex/tabs/tab_ois_risk_premium.tex",
        column_format="l"+"W"*len(rp_se.columns))
