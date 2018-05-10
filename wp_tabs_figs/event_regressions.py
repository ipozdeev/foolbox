from foolbox.api import *
from events_intraday_sampler import *
from wp_tabs_figs.wp_settings import settings
from pandas.tseries.offsets import BDay
from foolbox.linear_models import PureOls
import statsmodels.api as sm


if __name__ == "__main__":
    # Settigns ----------------------------------------------------------------
    # Event window settings
    pre = (BDay(-10), BDay(-1))
    post = (BDay(-1), BDay(-1))

    # Currency settings
    fixing_time = "LON"
    currs = ["aud", "cad", "chf", "eur", "gbp", "sek", "nzd"]

    # OIS/ON rates settigns
    ois_data_name = "ois_bloomi_1w_30y.p"
    on_data_name = "overnight_rates.p"
    ois_mat = "1m"

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

    fx_data = pd.read_pickle(data_path+settings["fx_data_fixed"])

    # Compute returns and spreads
    spot = fx_data["spot_mid"].loc[currs, s_dt:e_dt, fixing_time]
    spot_ret = np.log(spot).diff()

    # Compute difference between OIS and overnight rates
    ois_diff = (ois_data.rolling(5).mean() - on_data.rolling(5).mean())*1e2

    # Estimation --------------------------------------------------------------
    # Sample returns outside the event windows
    eda = EventDataAggregator(events_data, spot_ret, pre, post)
    stacked_ret = eda.stack_data()

    eda_ois = EventDataAggregator(events_data, ois_diff.shift(10), pre, post)
    stacked_ois = eda_ois.stack_data()


    dummies = pd.get_dummies(stacked_ret.drop(["data", "index"], axis=1))
    curr_fe = dummies[["asset_" + curr for curr in currs]]
    event_dummies = dummies[["event_" + evt_type for evt_type in
                             ["hike", "cut"]]]

    interaction_terms = event_dummies.multiply(stacked_ois["data"], axis=0)
    interaction_terms.columns = ["ois_x_hike", "ois_x_cut"]

    X = pd.concat([curr_fe, stacked_ois["data"].rename("ois"), event_dummies,
                   interaction_terms],
                   axis=1)

    ols = PureOls(y0=stacked_ret["data"]*1e4, X0=X, add_constant=False)
    ols.fit()
    ols.get_diagnostics(HAC=False)

    # ret_data = stacked_ret.loc[stacked_ret["event"] == "none"].pivot(
    #     index="index", columns="asset", values="data")
    #
    # # lol = taf.ts_ap_tests(ret_data*1e4,
    # #                       ois_diff[["aud"]].shift(5)*1e2, 1)
    #
    # df = pd.get_dummies(stacked_ret)
    # X = df.drop(["index", "data"], axis=1)
    # X = df[["event_cut", "event_hike"]]
    # y = df[["data"]]
    # ols = sm.OLS(y, X)