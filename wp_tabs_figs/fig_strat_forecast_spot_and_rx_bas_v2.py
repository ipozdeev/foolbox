import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import matplotlib.lines as mlines
from foolbox.wp_tabs_figs.wp_settings import settings
from foolbox.finance import PolicyExpectation
from foolbox.fxtrading import *
from foolbox.data_mgmt import set_credentials as set_cred
from foolbox.finance import get_pe_signals
import pandas as pd
import tables_and_figures as taf

# Chose what to compute and plot: "spot" or "rx_bas"
mode = "rx_bas"

# Set the output path, input data and sample
path_to_data = set_cred.set_path("research_data/fx_and_events/")
out_path = path_to_data + settings["fig_folder"]
fx_pkl = settings["fx_data"]
fx_pkl_fomc = settings["fx_data_fixed"]
fomc_fixing = settings["usd_fixing_time"]
s_dt = settings["sample_start"]
e_dt = settings["sample_end"]

# Lists of currencies for which saga strategy to be computed
currenices_local = ["aud", "cad", "chf", "eur", "gbp", "nzd", "sek"]
currencies_fomc = ["aud", "cad", "chf", "eur", "gbp", "nzd", "sek",
                   "nok", "jpy"]

# Set up the parameters of trading strategies
holding_period = 10              # actually the holding period,
threshold = 10                   # threshold in bps

# Set up parameters of policy expectations
lag_expect = holding_period + 2  # forecast rate in advance of trading
avg_impl_over = settings["avg_impl_over"]
avg_refrce_over = settings["avg_refrce_over"]
pol_exp_args = {"ffill": True,
                "avg_implied_over": avg_impl_over,
                "avg_refrce_over": avg_refrce_over}

# Matplotlib settings: font and locators
plt.rc("font", family="serif", size=12)
minor_locator = mdates.YearLocator()
major_locator = mdates.YearLocator(2)

# Load the data
fx_data = pd.read_pickle(path_to_data + fx_pkl)
fx_data_fomc = pd.read_pickle(path_to_data + fx_pkl_fomc)

# Prepare trading environment for FOMC, local events will be conjured later
# Extract FOMC fixing time df's from panels
for key, panel in fx_data_fomc.items():
    fx_data_fomc[key] = panel.loc[:, :, settings["usd_fixing_time"]]

# Create the fomc environment
tr_env_fomc = FXTradingEnvironment.from_scratch(
    spot_prices={
        "bid": fx_data_fomc["spot_bid"].loc[s_dt:e_dt, :],
        "ask": fx_data_fomc["spot_ask"].loc[s_dt:e_dt, :]},
    swap_points={
        "bid": fx_data_fomc["tnswap_bid"].loc[s_dt:e_dt, :],
        "ask": fx_data_fomc["tnswap_ask"].loc[s_dt:e_dt, :]}
    )

# Clean-ups
tr_env_fomc.drop(labels=tr_env_fomc.currencies.difference(currencies_fomc),
                 axis="minor_axis", errors="ignore")
tr_env_fomc.remove_swap_outliers()
tr_env_fomc.reindex_with_freq('B')
tr_env_fomc.align_spot_and_swap()
tr_env_fomc.fillna(which="both", method="ffill")

# If mode is "spot", set both bid and ask to spot_mid, annihilate swap points
if mode is "spot":
    spot_mid = tr_env_fomc.mid_spot_prices
    tr_env_fomc.spot_prices["ask"] = spot_mid
    tr_env_fomc.spot_prices["bid"] = spot_mid
    tr_env_fomc.swap_points["ask"] *= 0
    tr_env_fomc.swap_points["bid"] *= 0

# Prepare trading environments for the local events
tr_env_local = dict()
for curr in currenices_local:
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

    if mode is "spot":
        spot_mid = this_env.mid_spot_prices
        this_env.spot_prices["ask"] = spot_mid
        this_env.spot_prices["bid"] = spot_mid
        this_env.swap_points["ask"] *= 0
        this_env.swap_points["bid"] *= 0

    # Append local environments
    tr_env_local[curr] = this_env


def saga_signals(currencies, forecast_lag, threshold, fomc=False,
                 **pol_exp_args):
    """Fetches foresight saga forecast signals and the corresponding perfect
    foresight signals (such that perfect foresight is available only if
    forecast can be made)

    Parameters
    ----------
    currencies: list
        of currencies for which signals are to be fetched
    forecast_lag: int
        how many days in advance of a meeting its outcome should be foretold
    threshold: float
        threshold for policy forecast in basis points
    fomc: bool
        if the signals should be constructed around FOMC meetings. Default is
        False, meaning that signals are constructed around events of local
        central banks
    pol_exp_args: dict
        with kwargs of 'get_pe_signals()' function

    Returns
    -------
    (signals_fcast, signals_pfct): tuple
        of pd.DataFrames where first df contains predicted signals and the
        second one contains 'perfect foresight' signals were forecast signals
        are available

    """
    signals_fcast = list()
    signals_pfct = list()

    if fomc:
        # FOMC event for every currency
        signals_fcast_fomc = get_pe_signals(currencies, forecast_lag,
                                            threshold, path_to_data, fomc=True,
                                            **pol_exp_args)
        # Get the FOMC meetings
        signals_pfct_fomc = PolicyExpectation.from_pickles(
                data_path=path_to_data, currency="usd")

        # Times -1, because FOMC
        signals_pfct_fomc = -1 * np.sign(signals_pfct_fomc.meetings)

        # Make signals from the meetings
        signals_pfct_fomc = pd.concat(
            [signals_pfct_fomc for curr in currencies], axis=1)
        signals_pfct_fomc.columns = currencies

        # Reindex
        signals_fcast_fomc = signals_fcast_fomc.reindex(
            index=pd.date_range(s_dt, e_dt, freq='B'))
        signals_pfct_fomc = signals_pfct_fomc.reindex(
            index=pd.date_range(s_dt, e_dt, freq='B'))

        # Leave only perfect foresight, that could have been forecast
        signals_pfct_fomc = \
            signals_pfct_fomc.where(signals_fcast_fomc.notnull(), np.nan)

        signals_fcast = signals_fcast_fomc
        signals_pfct = signals_pfct_fomc

    else:
        # Loop over the currenncies, and get local signals
        for curr in currencies:
            # Get the signals. First, forecast
            this_signal_fcast = get_pe_signals([curr], forecast_lag, threshold,
                                               path_to_data, fomc=False,
                                               **pol_exp_args)
            # Second, perfect foresight
            this_signal_pfct = PolicyExpectation.from_pickles(
                data_path=path_to_data, currency=curr)
            this_signal_pfct = \
                np.sign(this_signal_pfct.meetings.to_frame(curr))

            # Reindex to be sure that no events are out of sample
            this_signal_fcast = this_signal_fcast.reindex(
                index=pd.date_range(s_dt, e_dt, freq='B'))
            this_signal_pfct = this_signal_pfct.reindex(
                index=pd.date_range(s_dt, e_dt, freq='B'))

            # Finally, use perfect foresight on the same sample
            this_signal_pfct = \
                this_signal_pfct.where(this_signal_fcast.notnull(), np.nan)

            # Append the output
            signals_fcast.append(this_signal_fcast)
            signals_pfct.append(this_signal_pfct)

        # Aggregate
        signals_fcast = pd.concat(signals_fcast, axis=1)
        signals_pfct = pd.concat(signals_pfct, axis=1)

    return signals_fcast, signals_pfct


def saga_returns(trading_env_local, trading_env_fomc, signals_local,
                 signals_fomc, holding_period):
    """Wrapper around saga strategy, with NAV for individual currencies
    computed in a loop.

    Parameters
    ----------
    trading_env_local: dict
        with currency codes in keys and 'FXTradingEnvironment' instances as
        items
    trading_env_fomc:
        instance of 'FXTradingEnvironment' for pre-FOMC trading
    signals_local: pd.DataFrame
        of signals for local currencies
    signals_fomc: pd.DataFrame
        of pre-fomc trading signals
    holding_period: int
        holding period in days

    Returns
    -------
    strat_ret: pd.DataFrame
        with cumulative product of simple returns to the saga strategy

    """
    # Loop over currencies, compute balance series for each
    res_local = list()
    for curr in trading_env_local.keys():

        # Prepare the trading environment
        this_env = trading_env_local[curr]

        # Get the signal
        this_signal = signals_local[[curr]]

        # Make the strat
        this_strat = FXTradingStrategy.from_events(this_signal, blackout=1,
                                                   hold_period=holding_period,
                                                   leverage="unlimited")

        this_trading = FXTrading(environment=this_env, strategy=this_strat)

        # Get the results
        this_res = this_trading.backtest("balance")
        res_local.append(this_res)

    # Agrgregate over all currencies
    res_local = pd.concat(res_local, axis=1)

    # Repeat for the FOMC: Strategy...
    strat_fomc = FXTradingStrategy.from_events(signals_fomc, blackout=1,
                                               hold_period=holding_period,
                                               leverage="net")
    # Trading...
    trading_fomc = FXTrading(environment=trading_env_fomc, strategy=strat_fomc)

    # Result
    res_fomc = trading_fomc.backtest("balance")

    # Aggregate results of the local and FOMC trading
    strat_ret = res_local.pct_change().sum(axis=1) + res_fomc.pct_change()
    strat_ret = (1 + strat_ret).cumprod()

    return strat_ret


if __name__ == "__main__":

    # Get the signals. First, local
    signals_local_fcast, signals_local_pfct = \
        saga_signals(currencies=currenices_local,
                     forecast_lag=lag_expect, threshold=threshold,
                     fomc=False, **pol_exp_args)

    # Then FOMC
    signals_fomc_fcast, signals_fomc_pfct = \
        saga_signals(currencies=tr_env_fomc.currencies,
                     forecast_lag=lag_expect, threshold=threshold,
                     fomc=True, **pol_exp_args)

    # Procure cumulative simple returns
    strat_fcast = saga_returns(tr_env_local, tr_env_fomc,
                               signals_local_fcast, signals_fomc_fcast,
                               holding_period)

    strat_pfct = saga_returns(tr_env_local, tr_env_fomc,
                              signals_local_pfct, signals_fomc_pfct,
                              holding_period)

    # Compute returns
    ret_fcast = np.log(strat_fcast).diff().replace(0, np.nan).dropna()
    ret_pfct = np.log(strat_pfct).diff().replace(0, np.nan).dropna()

    # Compute descriptives of the forecast strategy
    descr_fcast = taf.descriptives(ret_fcast, 1)

    # Drop the period from June 2008 to June 2009, repeat descriptives
    tmp_df_descr = pd.concat([ret_fcast[:"2008-09"], ret_fcast["2009-09":]],
                             axis=0)
    tmp_df_descr = taf.descriptives(tmp_df_descr, 1)
    print(tmp_df_descr)

    # Plot the results --------------------------------------------------------
    # Figure 1: plot perfect foresight and forecas strats along time dimension
    fig1, ax = plt.subplots()
    # Concatenate the data first
    to_plot = pd.concat([ret_pfct.cumsum()*100,
                         ret_fcast.cumsum()*100], axis=1).ffill().fillna(0)
    to_plot.columns = ["pfct", "fcast"]

    # Plot it
    to_plot[["pfct"]].plot(ax=ax, color='k', linewidth=1.5, linestyle="--")
    to_plot[["fcast"]].plot(ax=ax, color='k', linewidth=1.5, linestyle="-")

    # Rotate the axis labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Use time-indexed scale use the locators from settings
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_minor_locator(minor_locator)

    # Polish the layout
    ax.grid(which="both", alpha=0.33, linestyle=":")
    ax.set_xlabel("date", visible=True)
    ax.set_ylabel("cumulative return in percent", visible=True)
    ax.legend_.remove()

    # Add some descriptives
    descr = descr_fcast
    ax.annotate(r"$\mu={:3.2f}%$".format(descr.loc["mean"].values[0]*10000),
        xy=(0.9, 0.30), xycoords='axes fraction',
        horizontalalignment='right', fontsize=12)
    ax.annotate(r"$se={:3.2f}%$".format(descr.loc["se_mean"].values[0]*10000),
        xy=(0.9, 0.20), xycoords='axes fraction',
        horizontalalignment='right', fontsize=12)
    ax.annotate(r"$SR={:3.2f}%$".format(descr.loc["sharpe"].values[0]),
        xy=(0.9, 0.10), xycoords='axes fraction',
        horizontalalignment='right', fontsize=12)

    # Add legend for lines
    solid_line = mlines.Line2D([], [], color='black', linestyle="-",
                               lw=2, label="forecast-based")
    dashed_line = mlines.Line2D([], [], color='black', linestyle="--",
                                lw=2, label="perfect foresight")
    ax.legend(handles=[solid_line, dashed_line], loc="upper left", fontsize=10)

    fig1.tight_layout()
    fig1.savefig(out_path +
                 "ueber_fcast_lag{:d}_thresh{:d}_ai{:d}_ar{:d}".\
                 format(holding_period, int(threshold*100),
                        avg_impl_over, avg_refrce_over) +
                 "_" + mode +
                 "_time" +
                 ".pdf")

    # Figure 2: forecast on the event axis
    fig2, ax = plt.subplots()
    pd.DataFrame(100*ret_fcast.dropna().values).cumsum().\
        plot(ax=ax, color='k', linewidth=1.5, linestyle="-")
    # Rotate the axis labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Use time-indexed scale use the locators from settings
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.xaxis.set_minor_locator(MultipleLocator(25))

    # Polish the layout
    ax.grid(which="both", alpha=0.33, linestyle=":")
    ax.set_xlabel("event number", visible=True)
    ax.set_ylabel("cumulative return in percent", visible=True)
    ax.legend_.remove()

    fig2.tight_layout()
    fig2.savefig(out_path +
                 "ueber_fcast_lag{:d}_thresh{:d}_ai{:d}_ar{:d}".\
                 format(holding_period, int(threshold*100),
                        avg_impl_over, avg_refrce_over) +
                 "_" + mode +
                 "_count" +
                 ".pdf")
