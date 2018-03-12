from foolbox.fxtrading import *
from foolbox.data_mgmt import set_credentials as set_cred
from foolbox.wp_tabs_figs.wp_settings import settings
from foolbox.finance import get_pe_signals
import pickle
import pandas as pd
import re

path_to_data = set_cred.set_path("research_data/fx_and_events/")

# Set the output path, input data and sample
s_dt = pd.to_datetime(settings["sample_start"])
e_dt = pd.to_datetime(settings["sample_end"])

# s_dt = "2008-10-01"
# e_dt = "2008-10-08"

avg_impl_over = settings["avg_impl_over"]
avg_refrce_over = settings["avg_refrce_over"]

proxy_rate_pkl = "overnight_rates.p"
implied_rate_pkl = "implied_rates_from_1m_ois.p"
fx_pkl = "fx_by_tz_aligned_d.p"
fx_pkl_fomc = "fx_by_tz_sp_fixed.p"

no_good_curs = ["dkk", "jpy", "nok", "cad", "chf", "eur", "gbp", "nzd", "sek"]
no_good_curs_fomc = ["dkk", "jpy", "nok", "cad", "chf", "eur", "gbp", "nzd", "aud"]
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
    kwargs : any
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


def saga_strategy2(trading_env, trading_env_fomc, holding_period, threshold,
                   **kwargs):
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

    signals_fcast = get_pe_signals(curs, fcast_lag, threshold, path_to_data,
                                   fomc=False, **kwargs)

    signals_fcast_fomc = get_pe_signals(curs_fomc, fcast_lag, threshold,
                                        path_to_data, fomc=True, **kwargs)

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


def saga_strategy3(currencies, trading_env_fomc, holding_period, threshold,
                   **kwargs):
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
    curs_fomc = trading_env_fomc.currencies
    fcast_lag = holding_period + 2

    res_local = list()
    # Loop over currencies, compute balance series for each
    for curr in currencies:
        # Prepare the trading environment
        this_env = FXTradingEnvironment.from_scratch(
            spot_prices={
                "bid": fx_data["spot_bid"].loc[s_dt:e_dt, [curr]],
                "ask": fx_data["spot_ask"].loc[s_dt:e_dt, [curr]]},
            swap_points={
                "bid": fx_data["tnswap_bid"].loc[s_dt:e_dt, [curr]],
                "ask": fx_data["tnswap_ask"].loc[s_dt:e_dt, [curr]]}
            )

        # Remove outliers, reindex data, align and fillna
        this_env.remove_swap_outliers()
        this_env.reindex_with_freq('B')
        this_env.align_spot_and_swap()
        this_env.fillna(which="both", method="ffill")

        # Get the signals
        this_signal_fcast = get_pe_signals([curr], fcast_lag, threshold,
                                           path_to_data, fomc=False, **kwargs)

        # Reindex to be sure that no events are out of sample
        this_signal_fcast = this_signal_fcast.reindex(
            index=pd.date_range(s_dt, e_dt, freq='B'))

        # Make a strat
        this_strat_fcast = FXTradingStrategy.from_events(
            this_signal_fcast, blackout=1,
            hold_period=holding_period, leverage="unlimited")

        this_trading = FXTrading(environment=this_env,
                                 strategy=this_strat_fcast)

        # Get the results
        res = this_trading.backtest("balance")
        res_local.append(res)

    # Agrgregate over all currencies
    res_local = pd.concat(res_local, axis=1)
    res_local.columns = currencies


    signals_fcast_fomc = get_pe_signals(curs_fomc, fcast_lag, threshold,
                                        path_to_data, fomc=True, **kwargs)

    signals_fcast_fomc = signals_fcast_fomc.reindex(
        index=pd.date_range(s_dt, e_dt, freq='B'))

    # trading strategy FOMC--------------------------------------------------
    strategy_fcast_fomc = FXTradingStrategy.from_events(signals_fcast_fomc,
        blackout=1, hold_period=holding_period, leverage="net")

    # trading ---------------------------------------------------------------
    trading_fomc = FXTrading(environment=trading_env_fomc,
                             strategy=strategy_fcast_fomc)

    # backtest --------------------------------------------------------------
    res_fomc = trading_fomc.backtest("balance")

    return res_local, res_fomc


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
    fx_pkl = settings["fx_data_fixed"]
    no_good_curs = settings["drop_currencies"]

    # prepare environment ---------------------------------------------------
    # load data
    fx_pkl = pd.read_pickle(path_to_data + fx_pkl)

    # construct environment
    fx_data = dict()
    for key, panel in fx_pkl.items():
        fx_data[key] = panel.loc[:, :, settings["usd_fixing_time"]]

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
        swap_points = (fx_data["tnswap_bid"].loc[s_dt:e_dt, :] +
                       fx_data["tnswap_ask"].loc[s_dt:e_dt, :]) / 2

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


def retail_carry(trading_env, fwd_disc=None, map_signal=None,
                 leverage="net", **kwargs):
    """Construct carry as if implemented on the forex.

    Parameters
    ----------
    trading_env : FXTradingEnvironment
    fwd_disc : pandas.DataFrame
        of forward discounts (positive if rf > rf in the us)
    map_signal : callable
        function to apply to `fwd_disc`, e.g. lambda x: x.rolling(5).mean()
    leverage : str
    kwargs : dict
        additional args to portfolio_construction.rank_sort()

    Returns
    -------

    """
    if fwd_disc is None:
        fwd_disc = trading_env.mid_swap_points / \
                   trading_env.mid_spot_prices * -1

    if map_signal is None:
        map_signal = lambda x: x.shift(1)

    # apply transformation to forward discounts
    sig = map_signal(fwd_disc)
    sig = sig.reindex(columns=trading_env.currencies)

    s_dt = trading_env.mid_spot_prices.first_valid_index()
    e_dt = trading_env.mid_spot_prices.last_valid_index()
    sig = sig.loc[s_dt:e_dt, :]

    # signals
    portfolios = poco.rank_sort(sig, sig, **kwargs)

    # carry strategy
    strategy = FXTradingStrategy.from_long_short(portfolios, leverage=leverage)

    # trading
    trading = FXTrading(environment=trading_env, strategy=strategy)

    # backtest
    res = trading.backtest("unrealized")

    return res


def retail_value(trading_env, ppp, spot=None, map_signal=None,
                 leverage="net", **kwargs):
    """
    """
    if spot is None:
        spot = trading_env.mid_spot_prices

    if map_signal is None:
        map_signal = lambda x: x.shift(1)

    # reindex
    ppp_reix = ppp.shift(1).reindex(spot.index, method="ffill")

    # reer-like
    sig = ppp_reix / spot

    # apply transformation to forward discounts
    sig = map_signal(sig)
    sig = sig.reindex(columns=trading_env.currencies)

    s_dt = trading_env.mid_spot_prices.first_valid_index()
    e_dt = trading_env.mid_spot_prices.last_valid_index()
    sig = sig.loc[s_dt:e_dt, :]

    # signals
    portfolios = poco.rank_sort(sig, sig, **kwargs)

    # carry strategy
    strategy = FXTradingStrategy.from_long_short(portfolios, leverage=leverage)

    # trading
    trading = FXTrading(environment=trading_env, strategy=strategy)

    # backtest
    res = trading.backtest("unrealized")

    return res


def retail_momentum(trading_env, spot_ret=None, map_signal=None,
                    leverage="net", **kwargs):
    """Construct momentum strategy a la Menkhoff et al. (2012).

    Parameters
    ----------
    trading_env
    spot_ret
    map_signal
    leverage
    kwargs

    Returns
    -------

    """
    if spot_ret is None:
        spot_ret = np.log(trading_env.mid_spot_prices).diff()

    if map_signal is None:
        map_signal = lambda x: x.shift(1)

    # apply transformation to spot returns
    sig = map_signal(spot_ret)
    sig = sig.reindex(columns=trading_env.currencies)

    s_dt = trading_env.mid_spot_prices.first_valid_index()
    e_dt = trading_env.mid_spot_prices.last_valid_index()
    sig = sig.loc[s_dt:e_dt, :]

    # signals
    portfolios = poco.rank_sort(sig, sig, **kwargs)

    # the strategy
    strategy = FXTradingStrategy.from_long_short(portfolios, leverage=leverage)

    # trading
    trading = FXTrading(environment=trading_env, strategy=strategy)

    # backtest
    res = trading.backtest("unrealized")

    return res


def retail_vrp(trading_env, mfiv, rv=None, map_signal=None, leverage="net",
               **kwargs):
    """

    Parameters
    ----------
    trading_env
    mfiv
    rv
    map_signal
    leverage

    Returns
    -------

    """
    if map_signal is None:
        map_signal = lambda x: x.shift(1)

    if rv is None:
        ret = np.log(trading_env.mid_spot_prices).diff()
        rv = ret.rolling(22, min_periods=10).var() * 252

    sig = -1*(rv - mfiv)

    # apply transformation to forward discounts
    sig = map_signal(sig)
    sig = sig.reindex(columns=trading_env.currencies)

    s_dt = trading_env.mid_spot_prices.first_valid_index()
    e_dt = trading_env.mid_spot_prices.last_valid_index()
    sig = sig.loc[s_dt:e_dt, :]

    # signals
    portfolios = poco.rank_sort(sig, sig, **kwargs)

    # carry strategy
    strategy = FXTradingStrategy.from_long_short(portfolios, leverage=leverage)

    # trading
    trading = FXTrading(environment=trading_env, strategy=strategy)

    # backtest
    res = trading.backtest("unrealized")

    return res


def retail_dollar_carry(trading_env, us_rf, foreign_rf, map_signal=None,
                        leverage="net"):
    """

    Parameters
    ----------
    trading_env
    foreign_rf
    map_signal
    leverage
    kwargs

    Returns
    -------

    """
    if map_signal is None:
        map_signal = lambda x: x.shift(1)

    # signals
    mean_fwd_disc = foreign_rf.mean(axis=1)
    sig = map_signal(mean_fwd_disc).ge(map_signal(us_rf))
    sig = sig.astype(float) * 2.0 - 1.0
    sig = pd.concat([sig, ]*len(trading_env.currencies), axis=1)
    sig.columns = trading_env.currencies

    s_dt = trading_env.mid_spot_prices.first_valid_index()
    e_dt = trading_env.mid_spot_prices.last_valid_index()
    sig = sig.loc[s_dt:e_dt, :]

    # the strategy
    strategy = FXTradingStrategy.from_position_flags(sig, leverage=leverage)

    # trading
    trading = FXTrading(environment=trading_env, strategy=strategy)

    # backtest
    res = trading.backtest("unrealized")

    return res


def retail_slope(trading_env, rate_long, rate_short, map_signal=None,
                 leverage="net"):
    """

    Parameters
    ----------
    trading_env
    rate_long
    rate_short
    leverage

    Returns
    -------

    """
    if map_signal is None:
        map_signal = lambda x: x.shift(1)

    # signals
    sig = ((map_signal(rate_long) - map_signal(rate_short)) > 0) * 2 - 1

    sig.columns = trading_env.currencies

    s_dt = trading_env.mid_spot_prices.first_valid_index()
    e_dt = trading_env.mid_spot_prices.last_valid_index()
    sig = sig.loc[s_dt:e_dt, :]

    # the strategy
    strategy = FXTradingStrategy(actions=sig)

    # trading
    trading = FXTrading(environment=trading_env, strategy=strategy)

    # backtest
    res = trading.backtest("unrealized")

    return res


def wrapper_collect_strategies(trading_env):
    """

    Parameters
    ----------
    trading_env : TradingEnvironment

    Returns
    -------

    """
    # parameters ------------------------------------------------------------
    curs = trading_env.currencies
    s_dt = trading_env.mid_spot_prices.dropna().index[0]
    e_dt = trading_env.mid_spot_prices.dropna().index[-1]

    # data ------------------------------------------------------------------
    wmr_data = pd.read_pickle(path_to_data + "data_wmr_dev_d.p")
    fwd_disc_1m = wmr_data["fwd_disc"]
    fwd_disc_tn = -1*(trading_env.mid_swap_points/trading_env.mid_spot_prices)
    spot_ret_d = np.log(trading_env.mid_spot_prices).diff()
    ppp = pd.read_pickle(path_to_data + "ppp_1990_2017_y.p")
    ois_rates = pd.read_pickle(path_to_data + "ois_bloomi_1w_30y.p")
    on_rates = pd.read_pickle(path_to_data + "overnight_rates.p")
    mfiv = pd.read_pickle(path_to_data + "mfiv_1m.p")

    # signal mapper ---------------------------------------------------------
    def map_signal(w):
        return lambda x: x.rolling(w, min_periods=1).mean().shift(1)

    # strategies ------------------------------------------------------------
    strats = dict()

    for w in [5, 22]:
        # carry
        # strat = retail_carry(trading_env, fwd_disc=fwd_disc_tn,
        #                      map_signal=map_signal(w), leverage="net",
        #                      n_portfolios=3)
        # strats[("carry", "tomnext", w)] = strat
        #
        # strat = retail_carry(trading_env, fwd_disc=fwd_disc_1m,
        #                      map_signal=map_signal(w), leverage="net",
        #                      n_portfolios=3)
        # strats[("carry", "1m", w)] = strat
        #
        # # momentum
        # strat = retail_momentum(trading_env, spot_ret=-1*spot_ret_d,
        #                         map_signal=map_signal(w), leverage="net",
        #                         n_portfolios=3)
        # strats[("reversal", "standard", w)] = strat
        #
        # # vrp
        # strat = retail_vrp(trading_env, mfiv=mfiv, map_signal=map_signal(w),
        #                    leverage="net", n_portfolios=3)
        # strats[("vrp", "standard", w)] = strat
        #
        # # value
        # strat = retail_value(trading_env, ppp=ppp, map_signal=map_signal(w),
        #                      leverage="net", n_portfolios=3)
        # strats[("value", "standard", w)] = strat
        #
        # # dollar carry
        # strat = retail_dollar_carry(trading_env, us_rf=on_rates.loc[:, "usd"],
        #                             foreign_rf=on_rates.drop("usd", axis=1),
        #                             map_signal=map_signal(w),
        #                             leverage="net")
        #
        # strats[("dollar_carry", "overnight_rates", w)] = strat
        #
        # strat = retail_dollar_carry(
        #     trading_env, us_rf=ois_rates["1m"].loc[:, "usd"],
        #     foreign_rf=on_rates.drop("usd", axis=1),
        #     map_signal=map_signal(w), leverage="net")
        #
        # strats[("dollar_carry", "1m", w)] = strat

        strat = retail_slope(trading_env, ois_rates["1m"], on_rates,
                             map_signal(w))

        strats[w] = strat

    # strats = pd.concat(strats, axis=1)
    #
    # with pd.HDFStore(path_to_data + "strategies.h5", mode="w") as hangar:
    #     hangar.put("strats", strats)

    return strats


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from foolbox import tables_and_figures as taf
    import itertools as itools

    # trading environment ---------------------------------------------------
    trading_env = wrapper_prepare_environment(settings, bid_ask=False,
                                              spot_only=False)

    # drop_curs = [p for p in trading_env.currencies if p != "aud"]
    trading_env.drop(labels=settings["drop_currencies"],
                     axis="minor_axis", errors="ignore")

    # # start, end dates to align signals and available prices
    # s_dt = trading_env.mid_spot_prices.dropna().index[0]
    # e_dt = trading_env.mid_spot_prices.dropna().index[-1]

    # # saga strategy ---------------------------------------------------------
    # saga_strat = saga_strategy(trading_env, 10, 10, fomc=True,
    #                            leverage="net",
    #                            proxy_rate_pickle="overnight_rates.p",
    #                            e_proxy_rate_pickle=
    #                              "implied_rates_from_1m_ois_since.p",
    #                            meetings_pickle="events.p",
    #                            avg_impl_over=settings["avg_impl_over"],
    #                            avg_refrce_over=settings["avg_refrce_over"],
    #                            ffill=True)
    #
    # # to log returns
    # saga_strat = np.log(saga_strat).diff().replace(0.0, np.nan).dropna()\
    #     .rename("saga")
    #
    # with pd.HDFStore(path_to_data + "strategies.h5", mode="w") as hangar:
    #     hangar.put("saga", saga_strat)
    #
    # # carry -----------------------------------------------------------------
    # map_signal = lambda x: x.rolling(22, min_periods=1).mean().shift(1)
    #
    # carry_strat = retail_carry(trading_env, map_signal=map_signal,
    #                            leverage="net", n_portfolios=3)
    #
    # carry_strat = np.log(carry_strat).diff().replace(0.0, np.nan).dropna()\
    #     .rename("carry")
    #
    # with pd.HDFStore(path_to_data + "strategies.h5", mode="a") as hangar:
    #     hangar.put("carry", carry_strat)

    # # dollar carry ----------------------------------------------------------
    # mat = "1m"
    #
    # # import rate
    # rf = pd.read_pickle(path_to_data + "ois_bloomi_1w_30y.p")
    #
    # us_rf = rf[mat].loc[s_dt:e_dt, "usd"]
    # foreign_rf = rf[mat].loc[s_dt:e_dt, trading_env.currencies]
    #
    # map_signal = lambda x: x.rolling(22, min_periods=1).mean().shift(1)
    #
    # dol_carry_strat = retail_dollar_carry(trading_env, us_rf, foreign_rf,
    #                                       map_signal=map_signal)
    #
    # dol_carry_strat = np.log(dol_carry_strat).diff()\
    #     .replace(0.0, np.nan).dropna().rename("dol_carry")
    #
    # with pd.HDFStore(path_to_data + "strategies.h5", mode="a") as hangar:
    #     hangar.put("dol_carry", dol_carry_strat)
    #
    # # mfiv ------------------------------------------------------------------
    # # import mfiv
    # path_to_mfiv = set_cred.set_path("option_implied_betas_project/data/" +
    #                                  "estimates/")
    # hangar = pd.HDFStore(path_to_mfiv + "mfiv.h5", mode="r")
    # mfiv_dict = {
    #     re.sub('/', '', re.sub("usd", '', re.sub("m1m", '', k))): hangar.get(k)
    #     for k in hangar.keys() if "usd" in k and k.endswith("m1m")
    # }
    # hangar.close()
    #
    # mfiv = pd.DataFrame.from_dict(
    #     {k: v.loc[~v.index.duplicated(keep="last")]
    #      for k, v in mfiv_dict.items()}
    # )
    #
    # mfiv = mfiv.loc[s_dt:e_dt, trading_env.currencies]
    #
    # # strategy
    # map_signal = lambda x: x.rolling(22, min_periods=1).mean().shift(1)
    #
    # vrp_strat = retail_vrp(trading_env, mfiv, map_signal=map_signal,
    #                        leverage="net", n_portfolios=3)
    #
    # vrp_strat = np.log(vrp_strat).diff().replace(0.0, np.nan).dropna() \
    #     .rename("vrp")
    #
    # with pd.HDFStore(path_to_data + "strategies.h5", mode="a") as hangar:
    #     hangar.put("vrp", vrp_strat)
    #
    # # value -----------------------------------------------------------------
    # ppp = pd.read_pickle(path_to_data + "ppp_1990_2017_y.p")
    # ppp = ppp.loc[s_dt:e_dt, trading_env.currencies]
    #
    # ppp_strat = retail_value(trading_env, ppp, spot=None, map_signal=None,
    #                          leverage="net", n_portfolios=3)
    #
    # ppp_strat = np.log(ppp_strat).diff().replace(0.0, np.nan).dropna()\
    #     .rename("value")
    #
    # with pd.HDFStore(path_to_data + "strategies.h5", mode="a") as hangar:
    #     hangar.put("value", ppp_strat)
    #
    # # momentum --------------------------------------------------------------
    # map_signal = lambda x: x.rolling(22, min_periods=1).mean().shift(1)
    #
    # mom_strat = retail_momentum(trading_env, map_signal=map_signal,
    #                             leverage="net", n_portfolios=3)
    #
    # mom_strat.plot()
    # plt.show()

    # mom_strat = np.log(mom_strat).diff().replace(0.0, np.nan).dropna()\
    #     .rename("momentum")
    #
    # with pd.HDFStore(path_to_data + "strategies.h5", mode="a") as hangar:
    #     hangar.put("reversal", -1*mom_strat)

    # strats = dict()
    # for h in range(10, 11):
    #     for threshold in range(10, 11):
    #         res = saga_strategy2(tr_env, tr_env_fomc, h, threshold,
    #                              proxy_rate_pickle="overnight_rates.p",
    #                              e_proxy_rate_pickle=
    #                                "implied_rates_from_1m_ois_since.p",
    #                              meetings_pickle="meetings.p",
    #                              avg_impl_over=settings["avg_impl_over"],
    #                              avg_refrce_over=settings["avg_refrce_over"],
    #                              ffill=True)
    #         strats[(h, threshold)] = res
    #
    # res = pd.DataFrame(strats)

    # strats = dict()
    # for h in range(10, 11):
    #     for threshold in range(10, 12):
    #         res = saga_strategy2(tr_env, tr_env_fomc, h,
    #                              threshold,
    #                              **{"ffill": True,
    #                                 "avg_implied_over": avg_impl_over,
    #                                 "avg_refrce_over": avg_refrce_over})
    #         strats[(h, threshold)] = res
    #
    # res = pd.DataFrame(strats)

    # res2 = saga_strategy2(tr_env, tr_env_fomc, holding_period=10,
    #                       threshold=10,
    #                       **{"ffill": True,
    #                          "avg_implied_over": avg_impl_over,
    #                          "avg_refrce_over": avg_refrce_over})
    #
    # res3 = saga_strategy3(tr_env.currencies, tr_env_fomc, holding_period=10,
    #                       threshold=10,
    #                       **{"ffill": True,
    #                          "avg_implied_over": avg_impl_over,
    #                          "avg_refrce_over": avg_refrce_over})
    #
    # two = (1 + res2[0].pct_change() + res2[1].pct_change()).cumprod()
    #
    # three = (1 + res3[0].pct_change().sum(axis=1) +
    #          res3[1].pct_change()).cumprod()

    # # BROOMSTICKS==============================================================
    import time
    t0 = time.time()
    ix = pd.IndexSlice
    holding_range = range(1, 9)
    threshold_range = np.arange(1, 25, 3) #range(, 26)
    combos = list(itools.product(holding_range, threshold_range))
    cols = pd.MultiIndex.from_tuples(combos, names=["holding", "threshold"])
    out = pd.DataFrame(index=pd.date_range(s_dt, e_dt, freq='B'),
                       columns=cols)
    out_fomc = out.copy()

    for h in holding_range:
        for tau in threshold_range:
            print(h, tau)
            print("time elapsed {}".format((time.time()-t0)/60))
            res, res_fomc = \
                saga_strategy3(tr_env.currencies, tr_env_fomc,
                               holding_period=h,
                               threshold=tau,
                               **{"ffill": True,
                                  "avg_implied_over": avg_impl_over,
                                  "avg_refrce_over": avg_refrce_over})

            res = (1 + res.pct_change().sum(axis=1) +
                   res_fomc.pct_change()).cumprod()

            out.loc[:, ix[h, tau]] = res
            out_fomc.loc[:, ix[h, tau]] = res_fomc

    t1 = time.time()
    print(t1-t0)


    # with open(path_to_data+"broomstick_rx_data_v2.p", mode="wb") as fname:
    #     pickle.dump(out_fomc, fname)
    # with open(path_to_data+"broomstick_rx_data_fomc_v2.p", mode="wb") as fname:
    #     pickle.dump(out_fomc, fname)

    # END BROOMSTICKS==========================================================

    # import multiprocessing
    # from itertools import product

    # from foolbox.playground.to_del import cprofile_analysis

    # @cprofile_analysis(activate=True)
    # def broomstick_wrapper(x=(10,10)):
    #     print(x)
    #     h = x[0]
    #     tau = x[1]
    #     res = saga_strategy3(tr_env.currencies, tr_env_fomc, holding_period=h,
    #                          threshold=tau,
    #                          **{"ffill": True,
    #                             "avg_implied_over": avg_impl_over,
    #                             "avg_refrce_over": avg_refrce_over})
    #     res = (1 + res[0].pct_change().sum(axis=1) +
    #            res[1].pct_change()).cumprod()
    #     res.name = x
    #
    #     return res
    #
    # res = broomstick_wrapper()
    #
    # # p = multiprocessing.Pool(4)
    # # lol = p.map(broomstick_wrapper, product(range(9, 11), range(9, 11)))
    #
    # with multiprocessing.Pool(4) as pool:
    #     res = pool.map(broomstick_wrapper, product(range(9, 11), range(9, 11)))
    #
    # print(res)

    # strats = dict()
    # for h in range(2, 12):
    #     for threshold in range(10, 12):
    #         res = saga_strategy(tr_env, h, threshold/100.0, fomc=False)
    #         strats[(h, threshold)] = res
    #
    # res = pd.DataFrame(strats)

    res = wrapper_collect_strategies(trading_env)
    res[5].plot()
    res[22].plot()
    plt.show()

    # import seaborn as sns
    # sns.set_style("whitegrid")
    #
    # hangar = pd.HDFStore(path_to_data + "strategies.h5", mode="r")
    # strats = dict(hangar)
    # strats = strats["/strats"]
    #
    # for st in strats.columns.levels[0]:
    #     fig, ax = plt.subplots(figsize=(11, 8))
    #     strats[st].plot(ax=ax)
    #     fig.tight_layout()
    #     fig.savefig("c:/users/pozdeev/temp/" + st +".png")