from foolbox.fxtrading import *
from foolbox.data_mgmt import set_credentials as set_cred
from foolbox.wp_tabs_figs.wp_settings import settings
from foolbox.finance import get_pe_signals
import pickle
import pandas as pd

path_to_data = set_cred.set_path("research_data/fx_and_events/")

# Set the output path, input data and sample
s_dt = pd.to_datetime(settings["sample_start"])
e_dt = pd.to_datetime(settings["sample_end"])
avg_impl_over = settings["avg_impl_over"]
avg_refrce_over = settings["avg_refrce_over"]

proxy_rate_pkl = "overnight_rates.p"
implied_rate_pkl = "implied_rates_from_1m_ois.p"
fx_pkl = "fx_by_tz_aligned_d.p"
fx_pkl_fomc = "fx_by_tz_sp_fixed.p"

no_good_curs = ["dkk", "jpy", "nok", "aud", "cad", "chf", "eur", "nzd", "sek"]
no_good_curs_fomc = ["dkk"]
# no_ois_curs = ["jpy", "nok"]

fx_data = pd.read_pickle(path_to_data + fx_pkl)
fx_data_fomc = pd.read_pickle(path_to_data + fx_pkl_fomc)

# prepare environment: local announcements--------------------------------
tr_env = FXTradingEnvironment.from_scratch(
    spot_prices={
        "bid": fx_data["spot_bid"].loc[s_dt:e_dt, :],
        "ask": fx_data["spot_ask"].loc[s_dt:e_dt, :]},
    swap_points={
        "bid": fx_data["tnswap_bid"].loc[s_dt:e_dt, :],
        "ask": fx_data["tnswap_ask"].loc[s_dt:e_dt, :]}
    )

# clean-ups -------------------------------------------------------------
tr_env.drop(labels=no_good_curs, axis="minor_axis", errors="ignore")
tr_env.remove_swap_outliers()
tr_env.reindex_with_freq('B')
tr_env.align_spot_and_swap()
tr_env.fillna(which="both", method="ffill")

# prepare environment: FOMC announcements--------------------------------
# Extract FOMC fixing time df's from panels
for key, panel in fx_data_fomc.items():
    fx_data_fomc[key] = panel.loc[:, :, settings["usd_fixing_time"]]
tr_env_fomc = FXTradingEnvironment.from_scratch(
    spot_prices={
        "bid": fx_data_fomc["spot_bid"].loc[s_dt:e_dt, :],
        "ask": fx_data_fomc["spot_ask"].loc[s_dt:e_dt, :]},
    swap_points={
        "bid": fx_data_fomc["tnswap_bid"].loc[s_dt:e_dt, :],
        "ask": fx_data_fomc["tnswap_ask"].loc[s_dt:e_dt, :]}
    )

# clean-ups -------------------------------------------------------------
tr_env_fomc.drop(labels=no_good_curs_fomc, axis="minor_axis", errors="ignore")
tr_env_fomc.remove_swap_outliers()
tr_env_fomc.reindex_with_freq('B')
tr_env_fomc.align_spot_and_swap()
tr_env_fomc.fillna(which="both", method="ffill")


def saga_strategy(trading_env, holding_period, threshold, fomc=False):
    """

    Parameters
    ----------
    trading_env : FXTradingEnvironment
    holding_period : int
    threshold : float

    Returns
    -------

    """
    # forecast direction ----------------------------------------------------
    curs = trading_env.currencies
    fcast_lag = holding_period + 2
    thresh = threshold*100

    signals_fcast = get_pe_signals(curs, fcast_lag, thresh, path_to_data,
                                   fomc=False,
                                   avg_impl_over=avg_impl_over,
                                   avg_refrce_over=avg_refrce_over,
                                   bday_reindex=True)

    if fomc:
        signals_fomc = get_pe_signals(curs, fcast_lag, thresh, path_to_data,
                                      fomc=True,
                                      avg_impl_over=avg_impl_over,
                                      avg_refrce_over=avg_refrce_over,
                                      bday_reindex=True)

        # add nok and jpy
        signals_fcast = signals_fcast.reindex(
            columns=list(signals_fcast.columns))

        # combine signals
        signals_fcast, signals_fomc = signals_fcast.align(
            signals_fomc, axis=0, join="outer")
        signals_fcast = signals_fcast.fillna(signals_fomc)

    # reindex to be sure that no events are out of sample
    signals_fcast = signals_fcast.reindex(
        index=pd.date_range(s_dt, e_dt, freq='B'))

    # trading strategy ------------------------------------------------------
    strategy_fcast = FXTradingStrategy.from_events(signals_fcast,
        blackout=1, hold_period=holding_period, leverage="unlimited")

    # trading ---------------------------------------------------------------
    trading = FXTrading(environment=trading_env, strategy=strategy_fcast)

    # backtest --------------------------------------------------------------
    res = trading.backtest("balance")

    return res

def saga_strategy2(trading_env, trading_env_fomc, holding_period, threshold):
    """

    Parameters
    ----------
    trading_env: 'FXTradingEnvironment' instance with local fixing time data
    trading_env_fomc: 'FXTradingEnvironment' instance with US fixing time data
    holding_period: int
    threshold: float
    fomc: bool

    Returns
    -------

    """
    # forecast direction ----------------------------------------------------
    curs = trading_env.currencies
    curs_fomc = trading_env_fomc.currencies
    fcast_lag = holding_period + 2
    thresh = threshold*100

    signals_fcast = get_pe_signals(curs, fcast_lag, thresh, path_to_data,
                                   fomc=False,
                                   avg_impl_over=avg_impl_over,
                                   avg_refrce_over=avg_refrce_over,
                                   bday_reindex=True)

    signals_fcast_fomc = get_pe_signals(curs_fomc, fcast_lag, thresh,
                                        path_to_data,
                                        fomc=True,
                                        avg_impl_over=avg_impl_over,
                                        avg_refrce_over=avg_refrce_over,
                                        bday_reindex=True)

    # reindex to be sure that no events are out of sample
    signals_fcast = signals_fcast.reindex(
        index=pd.date_range(s_dt, e_dt, freq='B'))
    signals_fcast_fomc = signals_fcast_fomc.reindex(
        index=pd.date_range(s_dt, e_dt, freq='B'))
    # trading strategy ------------------------------------------------------
    strategy_fcast = FXTradingStrategy.from_events(signals_fcast,
        blackout=1, hold_period=holding_period, leverage="unlimited")

    # trading ---------------------------------------------------------------
    trading = FXTrading(environment=trading_env, strategy=strategy_fcast)

    # backtest --------------------------------------------------------------
    res = trading.backtest("balance")


    # trading strategy FOMC--------------------------------------------------
    strategy_fcast_fomc = FXTradingStrategy.from_events(signals_fcast_fomc,
        blackout=1, hold_period=holding_period, leverage="net")

    # trading ---------------------------------------------------------------
    trading_fomc = FXTrading(environment=trading_env_fomc,
                             strategy=strategy_fcast_fomc)

    # backtest --------------------------------------------------------------
    res_fomc = trading_fomc.backtest("balance")

    return res, res_fomc


def saga_wrapper(x):
    print(x)
    res = saga_strategy(tr_env, x[0], x[1] / 100.0, fomc=False)
    res.name = x
    return res


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import multiprocessing
    from itertools import product

    # with multiprocessing.Pool(4) as pool:
    #     res = pool.map(saga_wrapper, product(range(1, 6), range(1, 6)))

    # strats = dict()
    # for h in range(2, 12):
    #     for threshold in range(10, 12):
    #         res = saga_strategy(tr_env, h, threshold/100.0, fomc=False)
    #         strats[(h, threshold)] = res
    #
    # res = pd.DataFrame(strats)

    strats = dict()
    for h in range(10, 11):
        for threshold in range(10, 12):
            res = saga_strategy2(tr_env, tr_env_fomc, h,
                                 threshold/100.0)
            strats[(h, threshold)] = res

    res = pd.DataFrame(strats)

    with open(path_to_data + "temp_all_strats.p", mode="wb") as hangar:
        pickle.dump(res, hangar)