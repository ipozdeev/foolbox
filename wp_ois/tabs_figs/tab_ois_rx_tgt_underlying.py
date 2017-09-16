import pandas as pd
from foolbox.api import *
from foolbox.utils import *
from wp_ois.wp_settings import ois_start_dates, end_date, \
    central_banks_start_dates, cb_fx_map
from foolbox.linear_models import PureOls

#out_path = set_credentials.set_path("../ois/", which="local")
out_path = set_credentials.set_path("research_data/fx_and_events"
                                    "/wp_figures_limbo/", which="gdrive")

if __name__ == "__main__":

    order = ["1M", "3M", "6M", "9M", "1Y"]

    # Import RX:
    # 1.) Float indexed by underlyings
    with open(data_path + "ois_rx_w_day_count.p", mode="rb") as hangar:
        risk_premium_all = pickle.load(hangar)

    # No meetings for Japan
    risk_premium_all.pop("jpy")

    # 2.) Float indexed by target rates
    with open(data_path + "ois_rx_tgt_day_count.p", mode="rb") as hangar:
        risk_premium_tgt = pickle.load(hangar)

# =============================================================================
# Panel A: Target as the floating rate
# =============================================================================
    # Get descriptives for target-rate indexed returns
    rp_descr = \
        {k: taf.descriptives(v[ois_start_dates[k]:end_date].astype(float), 1)
         for k, v in risk_premium_tgt.items()}

    rp_mu_tgt = {k: v.loc["mean", order] for k, v in rp_descr.items()}
    rp_se_tgt = {k: v.loc["se_mean", order] for k, v in rp_descr.items()}

    # Prepare statistics to report
    rp_mu_tgt = pd.DataFrame.from_dict(rp_mu_tgt)*100
    rp_se_tgt = pd.DataFrame.from_dict(rp_se_tgt)*100
    tst_tgt = rp_mu_tgt / rp_se_tgt


# =============================================================================
# Panel B: Difference in excess returns
# =============================================================================
    # Compute statistics for the differences in cumulative float rates:
    # Get the said differences
    rx_on_vs_tgt = dict()
    for key in risk_premium_tgt.keys():
        # skip nzd, the difference is zero with zero variance
        if key != "nzd":
            rx_on_vs_tgt[key] = risk_premium_all[key] - risk_premium_tgt[key]

    # Similarly, get the descriptives and statistics to report
    rp_descr = \
        {k: taf.descriptives(v[ois_start_dates[k]:end_date].astype(float), 1)
         for k, v in rx_on_vs_tgt.items()}

    rp_mu_diff = {k: v.loc["mean", order] for k, v in rp_descr.items()}
    rp_se_diff = {k: v.loc["se_mean", order] for k, v in rp_descr.items()}

    rp_mu_diff = pd.DataFrame.from_dict(rp_mu_diff)*100
    rp_se_diff = pd.DataFrame.from_dict(rp_se_diff)*100
    tst_diff = rp_mu_diff / rp_se_diff

# =============================================================================
# Panel C: Regression of rates in levels
# =============================================================================
    # Import the underlying data
    with open(data_path + "ois_bloomberg.p", mode="rb") as hangar:
        ois_data = pickle.load(hangar)

    # Get the overnight rates
    on_rates = pd.concat(
        [p.loc[:, "ON"].to_frame(c) for c, p in ois_data.items()],
        axis=1).astype(float)

    # Import target rates
    with open(data_path + "ois_project_events.p", mode="rb") as hangar:
        events_data = pickle.load(hangar)
    tgt_rates = events_data["joint_cbs_plus_unscheduled_lvl_eff"]

    # Get rid of Norway and Japan
    tgt_rates = tgt_rates.reindex(on_rates.index).ffill().drop(["nok"], axis=1)
    on_rates = on_rates.drop(["jpy"], axis=1)

    # Inverted cb_fx_map {"currency": "corresponding_cb"}
    fx_cb_map = dict((fx, cb) for cb, fx in cb_fx_map.items())

    # Preallocate output structure for the estimates
    coef = dict()
    se = dict()

    # Linear restrictions for the Wald test. Joint hypothesis alpha=0, beta=1
    R = pd.DataFrame([[1, 0], [0, 1]], columns=["const", "tgt"])
    r = pd.Series([0, 1], index=R.index)
    wald = dict()

    # Run regression of underlying on target rate for each currency
    for curr in on_rates.columns:
        # Get the start dates from settings
        start_date = central_banks_start_dates[fx_cb_map[curr]]

        # Run the regression
        tmp_on = on_rates[curr]
        tmp_tgt = tgt_rates[curr]
        y0 = tmp_on.rename("on")[start_date:end_date]
        x0 = tmp_tgt.rename("tgt")[start_date:end_date]

        mod = PureOls(y0*100, x0*100, add_constant=True)
        diagnostics = mod.get_diagnostics(HAC=True)

        # Get the estimates and statistics
        coef[curr] = diagnostics.loc["coef"]
        se[curr] = diagnostics.loc["se"]
        # Compute Wald test for the joint hypothesis: alpha=0, beta=1
        wald[curr] = mod.linear_restrictions_test(R, r)

    coef = pd.DataFrame.from_dict(coef)
    coef.index = ["alpha", "beta"]
    se = pd.DataFrame.from_dict(se)
    se.index = ["alpha", "beta"]

    wald = pd.DataFrame.from_dict(wald)
    # Append coef and se dfs for making the table
    coef.loc["chi_sq", :] = wald.loc["chi_sq", :]
    se.loc["chi_sq", :] = wald.loc["p_val", :]

# =============================================================================
    # Stack the results, replace NZD nans with zeros
    estimates = pd.concat([rp_mu_tgt, rp_mu_diff, coef]).replace(np.nan, 0)
    inference = pd.concat([tst_tgt, tst_diff, se]).replace(np.nan, 0)
    to_better_latex(
        df_coef=estimates,
        df_tstat=inference,
        fmt_coef="{:3.2f}", fmt_tstat="{:3.2f}",
        buf=out_path+"tab_rx_on_vs_tgt.tex",
        column_format="l"+"W"*len(inference.columns))
