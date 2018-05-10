from foolbox.api import *
from events_intraday_sampler import *
from wp_tabs_figs.wp_settings import settings
from pandas.tseries.offsets import BDay
from foolbox.linear_models import PureOls

if __name__ == "__main__":
    # Settigns ----------------------------------------------------------------
    # Event window settings
    pre = (BDay(-10), BDay(-1))
    post = (BDay(-1), BDay(-1))

    # Currency settings
    fixing_time = "LON"
    currs = ["aud", "cad", "chf", "eur", "gbp", "sek", "nzd"]

    # Whether to include currency fixed effects or estimate panel ols with a
    # single intercept for all currencies
    include_currency_fe = True

    # OIS/ON rates settigns
    ois_data_name = "ois_bloomi_1w_30y.p"
    on_data_name = "overnight_rates.p"
    ois_mat = "1m"

    # Lag difference between ois and on rates by this number of days
    ois_lag = 10
    ois_smooth = 1

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

    # Compute returns in bps
    if "usd" in currs:
        currs_x_usd = set(currs) - set(["usd"])
        spot = fx_data["spot_mid"].loc[currs_x_usd, s_dt:e_dt, fixing_time]
        spot_ret = np.log(spot).diff()
        spot_ret["usd"] = - 1 * spot_ret.mean(axis=1) * 1e4

    else:

        spot = fx_data["spot_mid"].loc[currs, s_dt:e_dt, fixing_time]
        spot_ret = np.log(spot).diff() * 1e4

    # Compute difference between OIS and overnight rates in bps
    ois_diff = (ois_data.rolling(ois_smooth).mean() -
                on_data.rolling(ois_smooth).mean()).shift(ois_lag) * 1e2

    # Estimation --------------------------------------------------------------
    # Get the stacked returns, and yield curve slopes; mark events classes
    eda = EventDataAggregator(events_data, spot_ret, pre, post)
    stacked_ret = eda.stack_data()

    eda_ois = EventDataAggregator(events_data, ois_diff, pre,
                                  post)
    stacked_ois = eda_ois.stack_data()

    # Fetch dummies for currency-specific fixed effects and event class
    dummies = pd.get_dummies(stacked_ret.drop(["data", "index"], axis=1))

    curr_fe = dummies[["asset_" + curr for curr in currs]]

    event_dummies = dummies[["event_" + evt_type for evt_type in
                             ["hike", "cut", "no change"]]]

    # OIS spread x event dummy
    interaction_terms = event_dummies.multiply(stacked_ois["data"], axis=0)
    interaction_terms.columns = ["ois_x_hike", "ois_x_cut", "ois_x_no_change"]

    # Construct matrix of regressors
    if include_currency_fe:
        X = pd.concat([curr_fe, stacked_ois["data"].rename("ois"),
                       event_dummies, interaction_terms], axis=1)
    else:
        X = pd.concat([stacked_ois["data"].rename("ois_spread"), event_dummies,
                       interaction_terms], axis=1)

    ols = PureOls(y0=stacked_ret["data"], X0=X,
                  add_constant=not include_currency_fe)
    ols.fit()

    res = ols.get_diagnostics(HAC=False)
    print(res)



