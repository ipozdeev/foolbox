import pandas as pd
import multiprocessing
from itertools import product

from foolbox.fxtrading import *
from foolbox.data_mgmt import set_credentials as set_cred
from foolbox.finance import get_pe_signals

from foolbox.wp_tabs_figs.wp_settings import settings

path_to_data = set_cred.set_path("research_data/fx_and_events/")


def saga_strategy(trading_env, holding_period, threshold, fomc=False,
                  leverage="net", **kwargs):
    """

    Parameters
    ----------
    trading_env : FXTradingEnvironment
    holding_period : int
    threshold : float
        in basis points, e.g. 25 for the mode of fomc decisions
    fomc : bool
    kwargs : dict
        arguments to get_pe_signals (-> to PolicyExpectation.from_pickles()):
            - proxy_rate_pickle
            - e_proxy_rate_pickle
            - meetings_pickle
            - avg_impl_over
            - avg_refrce_over
            - ffill

    Returns
    -------

    """
    # forecast direction ----------------------------------------------------
    curs = trading_env.currencies
    fcast_lag = holding_period + 2

    signals_fcast = get_pe_signals(curs, fcast_lag, threshold, path_to_data,
                                   fomc=False, **kwargs)

    if fomc:
        # get fomc signals
        signals_fomc = get_pe_signals(curs, fcast_lag, threshold, path_to_data,
                                      fomc=True, **kwargs)

        # combine signals
        signals_fcast, _ = signals_fcast.align(signals_fomc, axis=0,
                                               join="outer")
        signals_fcast, _ = signals_fcast.align(signals_fomc, axis=1,
                                               join="outer")
        signals_fcast = signals_fcast.fillna(signals_fomc)

    # reindex to be sure that no events are out of sample
    signals_fcast = signals_fcast.reindex(
        index=pd.date_range(trading_env.spot_prices["bid"].first_valid_index(),
                            trading_env.spot_prices["bid"].last_valid_index(),
                            freq='B'))

    # trading strategy ------------------------------------------------------
    strategy_fcast = FXTradingStrategy.from_events(signals_fcast,
        blackout=1, hold_period=holding_period, leverage=leverage)

    # trading ---------------------------------------------------------------
    trading = FXTrading(environment=trading_env, strategy=strategy_fcast)

    # backtest --------------------------------------------------------------
    res = trading.backtest("unrealized")

    return res


def wrapper_prepare_environment(settings, bid_ask=True, spot_only=False):
    """

    Parameters
    ----------
    settings

    Returns
    -------

    """
    # settings
    s_dt = pd.to_datetime(settings["sample_start"])
    e_dt = pd.to_datetime(settings["sample_end"])
    fx_pkl = settings["fx_data"]
    no_good_curs = settings["drop_currencies"]

    # prepare environment ---------------------------------------------------
    # load data
    fx_data = pd.read_pickle(path_to_data + fx_pkl)

    # construct environment
    if bid_ask:
        spot_prices = {
            "bid": fx_data["spot_bid"].loc[s_dt:e_dt, :],
            "ask": fx_data["spot_ask"].loc[s_dt:e_dt, :]}
        swap_points = {
            "bid": fx_data["tnswap_bid"].loc[s_dt:e_dt, :],
            "ask": fx_data["tnswap_ask"].loc[s_dt:e_dt, :]}
    else:
        spot_prices = (fx_data["spot_bid"].loc[s_dt:e_dt, :] +
                       fx_data["spot_ask"].loc[s_dt:e_dt, :]) / 2
        swap_points = (fx_data["spot_bid"].loc[s_dt:e_dt, :] +
                       fx_data["spot_ask"].loc[s_dt:e_dt, :]) / 2

    if spot_only:
        swap_points *= 0.0

    tr_env = FXTradingEnvironment.from_scratch(spot_prices=spot_prices,
                                               swap_points=swap_points)

    # clean-ups -------------------------------------------------------------
    tr_env.drop(labels=no_good_curs, axis="minor_axis", errors="ignore")
    tr_env.remove_swap_outliers()
    tr_env.reindex_with_freq('B')
    tr_env.align_spot_and_swap()
    tr_env.fillna(which="both", method="ffill")

    return tr_env


def wrapper_saga_strategy(settings, bid_ask=True, spot_only=False, fomc=False,
                          leverage="net", parallelize=False):
    """

    Parameters
    ----------
    settings
    bid_ask
    spot_only
    fomc
    leverage
    parallelize

    Returns
    -------

    """
    # # ranges
    # h_range = range(1, 2)
    # th_range = range(1, 5)
    #
    # # trading environment: spot prices and swap points
    # trading_env = wrapper_prepare_environment(settings, bid_ask, spot_only)
    #
    # if parallelize:
    #
    #
    #
    # else:
    #     pass
    #
    # return res
    pass


def retail_carry(trading_env, fwd_disc=None, map_fwd_disc=None,
                 leverage="net", **kwargs):
    """Construct carry as if implemented on the forex (daily).

    Parameters
    ----------
    trading_env : FXTradingEnvironment
    fwd_disc : pandas.DataFrame
        of forward discounts (positive if rf > rf in the us)
    map_fwd_disc : callable
        function to apply to `fwd_disc`, e.g. lambda x: x.rolling(5).mean()
    leverage : str
    kwargs : dict
        additional args to portfolio_construction.rank_sort()

    Returns
    -------

    """
    if fwd_disc is None:
        fwd_disc = trading_env.swap_points["bid"] / \
                   trading_env.spot_prices["bid"] * -1

    if map_fwd_disc is None:
        map_fwd_disc = lambda x: x

    # apply transformation to forward discounts
    fwd_disc = map_fwd_disc(fwd_disc)

    # signals
    portfolios = poco.rank_sort(fwd_disc, fwd_disc, **kwargs)

    # carry strategy
    strategy = FXTradingStrategy.from_long_short(portfolios, leverage=leverage)

    # trading
    trading = FXTrading(environment=trading_env, strategy=strategy)

    # backtest
    res = trading.backtest("unrealized")

    return res


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    trading_env = wrapper_prepare_environment(settings, bid_ask=True,
                                              spot_only=False)

    one_strat = saga_strategy(trading_env, 10, 10, fomc=True, leverage="net",
                              proxy_rate_pickle="overnight_rates.p",
                              e_proxy_rate_pickle=
                                "implied_rates_from_1m_ois_since.p",
                              meetings_pickle="meetings.p",
                              avg_impl_over=settings["avg_impl_over"],
                              avg_refrce_over=settings["avg_refrce_over"],
                              ffill=True)

    one_strat.plot()
    plt.show()

    # h_range = range(1, 2)
    # th_range = range(1, 5)
    #
    # def saga_strategy_for_mp(x):
    #     """Wrapper of saga_strategy to use with multiprocessing."""
    #     print(x)
    #
    #     s = saga_strategy(trading_env, x[0], x[1] / 100.0, fomc=True,
    #                       leverage="net",
    #                       proxy_rate_pickle="overnight_rates.p",
    #                       e_proxy_rate_pickle=
    #                           "implied_rates_from_1m_ois_since.p",
    #                       meetings_pickle="meetings.p",
    #                       avg_impl_over=settings["avg_impl_over"],
    #                       avg_refrce_over=settings["avg_refrce_over"],
    #                       ffill=True)
    #
    #     s.name = x
    #
    #     return s
    #
    # # multiprocess this!
    # with multiprocessing.Pool(4) as pool:
    #     strats = pool.map(saga_strategy_for_mp, product(h_range, th_range))
    #
    # # oncat -> will have a MultiIndex
    # res = pd.DataFrame(strats)

    # with open(path_to_data + "temp.p", mode="wb") as hangar:
    #     pickle.dump(res, hangar)

    # res.plot()
    # plt.show()

    # map_fwd_disc = lambda x: x.rolling(20).mean().shift(1)
    #
    # carry = retail_carry(trading_env, map_fwd_disc=map_fwd_disc,
    #                      leverage="net", n_portfolios=3)
    #
    # carry.plot()
    # plt.show()