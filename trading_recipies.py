import pandas as pd
from foolbox.fxtrading import *
from foolbox.data_mgmt import set_credentials as set_cred
from foolbox.wp_tabs_figs.wp_settings import settings
from foolbox.finance import get_pe_signals
import matplotlib.pyplot as plt

path_to_data = set_cred.set_path("research_data/fx_and_events/")

# Set the output path, input data and sample
s_dt = pd.to_datetime(settings["sample_start"])
e_dt = pd.to_datetime(settings["sample_end"])
avg_impl_over = settings["avg_impl_over"]
avg_refrce_over = settings["avg_refrce_over"]

proxy_rate_pkl = "overnight_rates.p"
implied_rate_pkl = "implied_rates_from_1m_ois.p"
fx_pkl = "fx_by_tz_aligned_d.p"

no_good_curs = ["dkk"]
no_ois_curs = ["jpy", "nok"]

def saga_strategy(trading_env, holding_period, threshold):
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
    curs = [p for p in trading_env.currencies \
            if p not in no_ois_curs + no_good_curs]
    fcast_lag = holding_period + 2
    thresh = threshold*100

    signals_fcast = get_pe_signals(curs, fcast_lag, thresh, path_to_data,
                                   fomc=False,
                                   avg_impl_over=avg_impl_over,
                                   avg_refrce_over=avg_refrce_over,
                                   bday_reindex=True)

    signals_fomc = get_pe_signals(trading_env.currencies, fcast_lag, thresh,
                                  path_to_data, fomc=True,
                                  avg_impl_over=avg_impl_over,
                                  avg_refrce_over=avg_refrce_over,
                                  bday_reindex=True)

    # add nok and jpy
    signals_fcast = signals_fcast.reindex(
        columns=list(signals_fcast.columns) + no_ois_curs)

    # combine signals
    signals_fcast, signals_fomc = signals_fcast.align(signals_fomc, axis=0,
                                                      join="outer")
    signals_fcast = signals_fcast.fillna(signals_fomc)
    signals_fcast = signals_fcast.reindex(
        index=pd.date_range(s_dt, e_dt, freq='B'))

    # trading strategy ------------------------------------------------------
    strategy_fcast = FXTradingStrategy.from_events(signals_fcast,
        blackout=1, hold_period=holding_period, leverage="net")

    # trading ---------------------------------------------------------------
    trading = FXTrading(environment=trading_env, strategy=strategy_fcast)

    # backtest --------------------------------------------------------------
    res = trading.backtest("unrealized")

    return res

if __name__ == "__main__":
    fx_data = pd.read_pickle(path_to_data + fx_pkl)

    # prepare environment ---------------------------------------------------
    tr_env = FXTradingEnvironment.from_scratch(
        spot_prices={
            "bid": fx_data["spot_bid"].loc[s_dt:, :],
            "ask": fx_data["spot_ask"].loc[s_dt:, :]},
        swap_points={
            "bid": fx_data["tnswap_bid"].loc[s_dt:, :],
            "ask": fx_data["tnswap_ask"].loc[s_dt:, :]}
    )

    # clean-ups -------------------------------------------------------------
    tr_env.drop(labels=no_good_curs,
                axis="minor_axis", errors="ignore")
    tr_env.remove_swap_outliers()
    tr_env.reindex_with_freq('B')
    tr_env.align_spot_and_swap()
    tr_env.fillna(which="both", method="ffill")

    # run!
    res = saga_strategy(tr_env, 1, 0.10)

    res.plot()

    plt.show()