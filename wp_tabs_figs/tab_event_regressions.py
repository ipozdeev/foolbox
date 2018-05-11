from foolbox.api import *
from events_intraday_sampler import *
from wp_tabs_figs.wp_settings import settings
from pandas.tseries.offsets import BDay
from foolbox.linear_models import PureOls
from foolbox.utils import to_better_latex
from linearmodels.panel import PooledOLS, PanelOLS
import statsmodels.api as sm
from linearmodels.panel import compare


path_to_out = data_path + "wp_figures_limbo/"


if __name__ == "__main__":
    # Settigns ----------------------------------------------------------------
    # Regression variable settings
    specifications = {
        "(1)": ["ois_spread"],
        "(2)": ["event_hike", "event_cut", "event_no change"],
        "(3)": ["ois_spread", "ois_x_hike", "ois_x_cut", "ois_x_no_change"],
        "(4)": ["ois_spread", "event_hike", "event_cut", "event_no change",
                "ois_x_hike", "ois_x_cut", "ois_x_no_change"],
        # "(5)": ["ois_spread", "event_hike", "event_cut", "event_no change",
        #         "ois_x_hike", "ois_x_cut", "ois_x_no_change", "fwd_disc",
        #         "libor_spread"]
        }
    specification_order = ["(1)", "(2)", "(3)", "(4)"] #+ ["(5)"]
    var_order = specifications["(4)"]
    # Fixed effects, and standard error estimation options
    xs_fe = True
    ts_fe = False

    # 'clustered' - clustered around currencies; 'kernel' - Driscoll-Crayfish
    cov_type = "clustered"

    # Event window settings
    pre = (BDay(-10), BDay(-1))
    post = (BDay(-1), BDay(-1))

    # Currency settings
    fixing_time = "LON"
    currs = ["aud", "cad", "chf", "eur", "gbp", "sek", "nzd"]

    # OIS/ON rates settigns
    ois_data_name = "ois_bloomi_1w_30y.p"
    on_data_name = "overnight_rates.p"
    libor_data_name = "libor_spliced_2000_2007_d.p"
    ois_mat = "1m"

    # Lag difference between ois and on rates by this number of days
    ois_lag = 10
    ois_smooth = 1

    libor_lag = 1
    libor_smooth = 1

    fwd_disc_lag = 1
    fwd_disc_smooth = 1

    # Sample settings
    s_dt = settings["sample_start"]
    e_dt = settings["sample_end"]

    # Load data ---------------------------------------------------------------
    events_data = pd.read_pickle(
        data_path+settings["events_data"])["joint_cbs"].loc[
        s_dt:e_dt, currs]

    ois_data = pd.read_pickle(data_path+ois_data_name)[ois_mat].loc[
        s_dt:e_dt, currs]

    on_data = pd.read_pickle(data_path+on_data_name).loc[s_dt:e_dt, currs]

    libor_data = pd.read_pickle(data_path + libor_data_name)
    libor_data = libor_data["1m"].loc[s_dt:e_dt, currs]

    fx_data = pd.read_pickle(data_path+settings["fx_data_fixed"])

    # Compute returns in bps
    spot = fx_data["spot_mid"].loc[currs, s_dt:e_dt, fixing_time]
    spot_ret = np.log(spot).diff() * 1e4

    # Get the forward discount data
    fx_data_wmr = pd.read_pickle(data_path+"data_wmr_dev_d.p")
    fwd_disc = fx_data_wmr["fwd_disc"].loc[s_dt:e_dt, currs] * 1e4  # in bps
    fwd_disc = fwd_disc.rolling(fwd_disc_smooth).mean().shift(fwd_disc_lag)

    # Compute difference between OIS and overnight rates in bps
    ois_diff = (ois_data.rolling(ois_smooth).mean() -
                on_data.rolling(ois_smooth).mean()).shift(ois_lag) * 1e2
    libor_ois = (- ois_data.rolling(libor_smooth).mean() +
                 libor_data.rolling(libor_smooth).mean()).shift(libor_lag) * \
                1e2

    # ois_diff.mask(np.abs(ois_diff) < 0.1, 0, inplace=True)

    # Process the data --------------------------------------------------------
    # Get the stacked returns, and yield curve slopes; mark events classes
    eda = EventDataAggregator(events_data, spot_ret, pre, post)
    stacked_ret = eda.stack_data()

    eda_ois = EventDataAggregator(events_data, ois_diff, pre,
                                  post)
    stacked_ois = eda_ois.stack_data()

    eda_fwd_disc = EventDataAggregator(events_data, fwd_disc, pre, post)
    stacked_fwd_disc = eda_fwd_disc.stack_data()

    eda_libor = EventDataAggregator(events_data, libor_ois, pre, post)
    stacked_libor = eda_libor.stack_data()

    # Fetch dummies for currency-specific fixed effects and event class
    dummies = pd.get_dummies(stacked_ret.drop(["data", "index"], axis=1))
    dummies = dummies.astype(float)

    curr_fe = dummies[["asset_" + curr for curr in currs]]

    event_dummies = dummies[["event_" + evt_type for evt_type in
                             ["hike", "cut", "no change"]]]

    # OIS spread x event dummy, same for libor
    interaction_terms = event_dummies.multiply(stacked_ois["data"], axis=0)
    interaction_terms.columns = ["ois_x_hike", "ois_x_cut", "ois_x_no_change"]

    interaction_terms_libor = event_dummies.multiply(stacked_libor["data"],
                                                     axis=0)
    interaction_terms_libor.columns = ["lib_x_hike", "lib_x_cut",
                                       "lib_x_no_change"]

    ois_spread = stacked_ois["data"].rename("ois_spread")
    fwd_disc = stacked_fwd_disc["data"].rename("fwd_disc")
    libor_spread = stacked_libor["data"].rename("libor_spread")

    # Aggregate everything into a multiindex dataframe
    data = pd.concat(
        [stacked_ret, curr_fe, ois_spread, libor_spread, fwd_disc,
         event_dummies, interaction_terms, interaction_terms_libor], axis=1)
    data = data.set_index(["asset", "index"])
    data.sort_index(level=[0, 1], inplace=True)
    data.dropna(inplace=True)

    # Drop duplicates for overlapping events, fuck eurozone
    data = data[~data.index.duplicated(keep='first')]
    data.drop(["event"], axis=1, inplace=True)
    data = data.rename({"data": "spot_ret"}, axis="columns")
    data.index.names = ["asset", "date"]
    y = data[["spot_ret"]]
    X = data.drop(["spot_ret"], axis=1)

    # Estimate stuff ----------------------------------------------------------
    summary = {}
    for spec in specification_order:
        mod = PanelOLS(y, X[specifications[spec]], entity_effects=xs_fe,
                       time_effects=ts_fe)
        if cov_type == "clustered":
            res_fe = mod.fit(cov_type="clustered", cluster_entity=True,
                             cluster_time=False)

        if cov_type == "kernel":
            res_fe = mod.fit(cov_type="kernel")

        summary[spec] = res_fe

    summary = compare(summary)
    print(summary)

    tstats = summary.tstats.reindex(var_order).T
    coefs = summary.params.reindex(var_order).T

    to_better_latex(coefs, tstats, None, "{:3.2f}", "{:3.2f}",
                    buf=path_to_out + "panel_ols_events.tex")

    # Aggregate the variables
    # y = stacked_ret["data"]
    # X1 = pd.concat([curr_fe, ois_spread], axis=1)
    # X2 = pd.concat([curr_fe, event_dummies], axis=1)
    # X3 = pd.concat([curr_fe, ois_spread, interaction_terms], axis=1)
    # X4 = pd.concat([curr_fe, ois_spread,
    #                 event_dummies, interaction_terms],
    #                axis=1)
    #
    # # Run regressions ---------------------------------------------------------
    # results = dict()
    # ols1 = PureOls(y, X1, add_constant=False)
    # ols1.fit()
    # results[1] = ols1.get_diagnostics(HAC=False)
    #
    # ols2 = PureOls(y, X2, add_constant=False)
    # ols2.fit()
    # results[2] = ols2.get_diagnostics(HAC=False)
    #
    # ols3 = PureOls(y, X3, add_constant=False)
    # ols3.fit()
    # results[3] = ols3.get_diagnostics(HAC=False)
    #
    # ols4 = PureOls(y, X4, add_constant=False)
    # ols4.fit()
    # results[4] = ols4.get_diagnostics(HAC=False)
    #
    # tstats = []
    # coefs = []
    # for k in range(4):
    #     coefs.append(results[k+1].loc[["coef"], :])
    #     tstats.append(results[k+1].loc[["tstat"], :])
    #
    # tstats = pd.concat(tstats)
    # tstats.index = range(len(tstats))
    # coefs = pd.concat(coefs)
    # coefs.index = range(len(tstats))
    #
    # # Report these variables
    # vars_to_report = ["ois_spread", "event_hike", "event_cut",
    #                   "event_no change", "ois_x_hike", "ois_x_cut",
    #                   "ois_x_no_change"]
    # coefs = coefs[vars_to_report]
    # tstats = tstats[vars_to_report]
    #
    # print(coefs)
    # print(tstats)

    # to_better_latex(coefs, tstats, None, "{:3.2f}", "{:3.2f}",
    #                 buf=path_to_out + "panel_ols_events.tex")