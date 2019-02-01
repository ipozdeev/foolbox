from foolbox.api import *
from foolbox.fxtrading import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import matplotlib.lines as mlines
from foolbox.wp_tabs_figs.wp_settings import settings

# locators
minor_locator = mdates.YearLocator()
major_locator = mdates.YearLocator(2)

out_path = data_path + settings["fig_folder"]

def collect_data():
    """

    Returns
    -------

    """
    # fx data
    fx_pickle_name = "fx_data_tr_2000_2018_d.p"
    # fx_pickle_name = "fx_by_tz_aligned_d.p"

    fx = pd.read_pickle(data_path + fx_pickle_name)

    # events
    events_pickle_name = "events_reg_irreg_ann_eff.p"
    events_data = pd.read_pickle(data_path + events_pickle_name)
    events = events_data["scheduled"]["announced"]["joint_chg"]

    # ir prob
    irprob_pickle_name = "irprob_from_1m_ois_w_rp_subtracted.p"
    irprob = pd.read_pickle(data_path + irprob_pickle_name)

    # implied rates
    irprob_pickle_name = "irprob_from_1m_ois_w_rp_subtracted.p"
    irprob = pd.read_pickle(data_path + irprob_pickle_name)

    # align
    common_idx = pd.date_range(settings["sample_start"],
                               settings["sample_end"],
                               freq="B")

    fx = fx.reindex(index=common_idx)
    events = events.reindex(index=common_idx)
    irprob = irprob.reindex(index=common_idx)

    return fx, events, irprob


def construct_signals_from_irprob(irprob_df, events_df, horizon, threshold):
    """

    Parameters
    ----------
    irprob_df : pandas.DataFrame
        reindexed in a proper way (w/o large gaps, e.g. daily)
    events_df :
    pandas.DataFrame
        reindexed in a proper way (w/o large gaps, e.g. daily)
    horizon : int
    threshold : float

    Returns
    -------
    res : pandas.DataFrame

    """
    res = irprob_df\
        .gt(threshold)\
        .mul(np.sign(events_df.shift(-horizon))) \
        .shift(horizon)

    return res


if __name__ == "__main__":

    # settings --------------------------------------------------------------
    start_date = pd.to_datetime(settings["sample_start"])
    end_date = pd.to_datetime(settings["sample_end"])
    end_date = pd.to_datetime("2018-12-31")
    drop_curs = settings["drop_currencies"]
    avg_impl_over = settings["avg_impl_over"]
    avg_refrce_over = settings["avg_refrce_over"]
    base_lag = settings["base_holding_h"] + 2
    base_th = settings["base_threshold"]

    # data ------------------------------------------------------------------
    fx_data, events_data, irprob = collect_data()

    # trading environment ---------------------------------------------------
    fx_tr_env = FXTradingEnvironment.from_scratch(
        spot_prices={
            "bid": data_merged_tz["spot_bid"],
            "ask": data_merged_tz["spot_ask"]},
        swap_points={
            "bid": data_merged_tz["fwd_tn_bid"] - data_merged_tz["spot_bid"],
            "ask": data_merged_tz["fwd_tn_ask"] - data_merged_tz["spot_ask"]}
            )

    # tune trading environment
    fx_tr_env.drop("dkk", axis=1, level="currency", errors="ignore")
    fx_tr_env.remove_bid_ask_violation()
    fx_tr_env.remove_swap_outliers()
    fx_tr_env.reindex_with_freq('B')
    fx_tr_env.align_spot_and_swap()
    fx_tr_env.fillna(which="both", method="ffill")

    # timeline = fx_tr_env.spot_prices.index
    curs_extend = fx_tr_env.currencies
    curs = [p for p in curs_extend if p not in ["nok", "jpy"]]

    # signals ---------------------------------------------------------------
    # forecast
    # signals_fcast = get_pe_signals(curs, base_lag, base_th*100, data_path,
    #     fomc=False,
    #     avg_impl_over=avg_impl_over,
    #     avg_refrce_over=avg_refrce_over,
    #     bday_reindex=True)
    # signals = signals.replace(0.0, np.nan)
    signals_fcast = irprob\
        .reindex(index=timeline)\
        .gt(0.33)\
        .mul(np.sign(events.reindex(index=timeline).shift(-12)))\
        .drop("usd", axis=1)\
        .shift(12)

    signals_fcast.loc[:, "nok"] = np.nan
    signals_fcast.loc[:, "jpy"] = np.nan

    # signals_fomc = get_pe_signals(curs_extend, base_lag,
    #     base_th*100, data_path,
    #     fomc=True,
    #     avg_impl_over=avg_impl_over,
    #     avg_refrce_over=avg_refrce_over,
    #     bday_reindex=True)
    # signals_fomc = signals_fomc.replace(0.0, np.nan)

    signals_fcast = signals_fcast.reindex(index=timeline).replace(0.0, np.nan)
        # .fillna(signals_fomc)

    signals_fcast = signals_fcast.loc[start_date:end_date,:]
    # signals_perf.dropna(how="all")

    # perfect foresight
    signals_perf = np.sign(events)
    signals_perf = signals_perf.loc[:, curs_extend]
    # signals_perf = signals_perf.replace(0.0, np.nan)

    # signals_perf_us = -1*pd.concat(
    #     [np.sign(events.loc[:, "usd"]).rename(p) \
    #         for p in curs_extend], axis=1)

    # signals_perf_us = signals_perf_us.replace(0.0, np.nan)

    # signals_perf = signals_perf.reindex(index=timeline).replace(0.0, np.nan)\
    #     .fillna(signals_perf_us)
    #
    # # make forecast consistent
    signals_perf = signals_perf.where(signals_fcast.notnull())
    # signals_perf.dropna(how="all")

    signals_perf = signals_perf.reindex(index=timeline)\
        .loc[start_date:end_date, :]

    signals_perf = signals_perf.replace(0.0, np.nan)
    # signals_fcast = signals_fcast.replace(0.0, np.nan)

    # trading strategy ------------------------------------------------------
    # forecast
    strategy_fcast = FXTradingStrategy.from_events(signals_fcast,
        blackout=1, hold_period=10, leverage="net")

    # strategy_fcast.actions.to_clipboard()

    trading_fcast = FXTrading(environment=fx_tr_env, strategy=strategy_fcast)

    res_fcast = trading_fcast.backtest(method="unrealized_pnl")
    res_fcast.dropna().plot()
    taf.descriptives(np.log(res_fcast).diff().to_frame()*10000, scale=10)
    res_fcast.to_clipboard()

    # perfect
    strategy_perf = FXTradingStrategy.from_events(signals_perf,
        blackout=1, hold_period=10, leverage="net")

    trading_perf = FXTrading(environment=fx_tr_env, strategy=strategy_perf)

    res_perf = trading_perf.backtest(method="unrealized_pnl")
    # res_perf.to_clipboard()

    # plot ------------------------------------------------------------------
    fig, ax = plt.subplots()
    # Use time-indexed scale use the locators from settings
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_minor_locator(minor_locator)

    # Concatenate the data first
    to_plot = pd.concat((res_fcast.rename("fcst"), res_perf.rename("pfct")),
        axis=1) - 1

    to_plot = to_plot.dropna()

    # idx = signals_fcast.replace(0.0, np.nan).dropna(how="all").index
    # to_plot_add = to_plot.shift(1).reindex(index=idx).dropna(how="all")
    idx = strategy_fcast.position_flags.notnull().any(axis=1)
    tmp = res_fcast.where(idx).dropna()
    tmp.loc[tmp.index[0]-DateOffset(days=1)] = 1.0
    tmp = tmp.sort_index()
    # tmp.plot()

    to_descr = np.log(tmp).diff()

    descr = taf.descriptives(to_descr.to_frame("fcst")*10000, 10)

    # Plot it
    to_plot[["pfct"]].plot(ax=ax, color='k', linewidth=1.5, linestyle="--")
    to_plot[["fcst"]].plot(ax=ax, color='k', linewidth=1.5, linestyle="-")

    # Polish the layout
    ax.grid(which="both", alpha=0.33, linestyle=":")
    ax.set_xlabel("date", visible=True)
    ax.set_ylabel("cumulative return, in percent", visible=True)
    ax.legend_.remove()

    ax.annotate(r"$\mu={:3.2f}%$".format(descr.loc["mean", "fcst"]),
        xy=(0.9, 0.30), xycoords='axes fraction',
        horizontalalignment='right', fontsize=12)
    ax.annotate(r"$se={:3.2f}%$".format(descr.loc["se_mean", "fcst"]),
        xy=(0.9, 0.20), xycoords='axes fraction',
        horizontalalignment='right', fontsize=12)
    ax.annotate(r"$SR={:3.2f}%$".format(descr.loc["sharpe", "fcst"]),
        xy=(0.9, 0.10), xycoords='axes fraction',
        horizontalalignment='right', fontsize=12)

    # Add legend for lines
    solid_line = mlines.Line2D([], [], color='black', linestyle="-",
                               lw=2, label="forecast-based")
    dashed_line = mlines.Line2D([], [], color='black', linestyle="--",
                                lw=2, label="perfect foresight")
    ax.legend(handles=[solid_line, dashed_line], loc="upper left", fontsize=10)

    # Rotate the axis labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    fig.tight_layout()
    fig.savefig(out_path +
         "ueber_fcast_lag{:d}_thresh{:d}_ai{:d}_ar{:d}".\
         format(base_lag-2,int(base_th*10000),avg_impl_over,avg_refrce_over) +
         "_rx_bas_leverage_control" +
         "_time_new" +
         ".pdf")
