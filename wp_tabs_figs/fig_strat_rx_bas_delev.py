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

if __name__ == "__main__":

    # settings --------------------------------------------------------------
    start_date = pd.to_datetime(settings["sample_start"])
    end_date = pd.to_datetime(settings["sample_end"])
    drop_curs = settings["drop_currencies"]
    avg_impl_over = settings["avg_impl_over"]
    avg_refrce_over = settings["avg_refrce_over"]
    base_lag = settings["base_holding_h"] + 2
    base_th = settings["base_threshold"]

    # data ------------------------------------------------------------------
    # spot and swap data
    with open(data_path+"fx_by_tz_aligned_d.p", mode="rb") as fname:
        data_merged_tz = pickle.load(fname)

    # events
    with open(data_path+"events.p", mode="rb") as fname:
        data_events = pickle.load(fname)

    # trading environment ---------------------------------------------------
    fx_tr_env = FXTradingEnvironment.from_scratch(
        spot_prices={
            "bid": data_merged_tz["spot_bid"],
            "ask": data_merged_tz["spot_ask"]},
        swap_points={
            "bid": data_merged_tz["tnswap_bid"],
            "ask": data_merged_tz["tnswap_ask"]}
            )

    # tune trading environment
    fx_tr_env.drop(labels=["dkk"], axis="minor_axis", errors="ignore")
    fx_tr_env.remove_swap_outliers()
    fx_tr_env.reindex_with_freq('B')
    fx_tr_env.align_spot_and_swap()
    fx_tr_env.fillna(which="both", method="ffill")

    timeline = fx_tr_env.spot_prices.major_axis
    curs_extend = fx_tr_env.spot_prices.minor_axis
    curs = [p for p in  curs_extend if p not in ["nok", "jpy"]]

    # signals ---------------------------------------------------------------
    # forecast
    signals = get_pe_signals(curs, base_lag, base_th*100, data_path,
        fomc=False,
        avg_impl_over=avg_impl_over,
        avg_refrce_over=avg_refrce_over,
        bday_reindex=True)
    # signals = signals.replace(0.0, np.nan)

    signals.loc[:, "nok"] = np.nan
    signals.loc[:, "jpy"] = np.nan

    signals_fomc = get_pe_signals(curs_extend, base_lag,
        base_th*100, data_path,
        fomc=True,
        avg_impl_over=avg_impl_over,
        avg_refrce_over=avg_refrce_over,
        bday_reindex=True)
    # signals_fomc = signals_fomc.replace(0.0, np.nan)

    signals_fcast = signals.reindex(index=timeline).fillna(signals_fomc)

    signals_fcast = signals_fcast.loc[start_date:end_date,:]

    # perfect foresight
    signals_perf = np.sign(data_events["joint_cbs"])
    signals_perf = signals_perf.loc[:, curs_extend]
    # signals_perf = signals_perf.replace(0.0, np.nan)

    signals_perf_us = -1*pd.concat(
        [np.sign(data_events["joint_cbs"].loc[:, "usd"]).rename(p) \
            for p in curs_extend], axis=1)

    # signals_perf_us = signals_perf_us.replace(0.0, np.nan)

    signals_perf = signals_perf.reindex(index=timeline).fillna(signals_perf_us)

    # make forecast consistent
    signals_perf = signals_perf.where(signals_fcast)
    # signals_perf.dropna(how="all")

    signals_perf = signals_perf.loc[start_date:end_date,:]

    # trading strategy ------------------------------------------------------
    # forecast
    strategy_fcast = FXTradingStrategy.from_events(signals_fcast,
        blackout=1, hold_period=10, leverage="none")

    trading_fcast = FXTrading(environment=fx_tr_env, strategy=strategy_fcast)

    res_fcast = trading_fcast.backtest()

    # perfect
    strategy_perf = FXTradingStrategy.from_events(signals_perf,
        blackout=1, hold_period=10, leverage="none")

    trading_perf = FXTrading(environment=fx_tr_env, strategy=strategy_perf)

    res_perf = trading_perf.backtest()

    # plot ------------------------------------------------------------------
    fig, ax = plt.subplots()
    # Use time-indexed scale use the locators from settings
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_minor_locator(minor_locator)

    # Concatenate the data first
    to_plot = pd.concat((res_fcast.rename("fcst"), res_perf.rename("pfct")),
        axis=1) - 1
    to_plot = to_plot.dropna(how="all")

    idx = signals_fcast.replace(0.0, np.nan).dropna(how="all").index
    to_plot_add = to_plot.reindex(index=idx).dropna(how="all")
    descr = taf.descriptives(np.log(to_plot_add+1).diff()*10000, 1)

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
         "_rx_bas_leverage_control_time" +
         "_time" +
         ".pdf")
